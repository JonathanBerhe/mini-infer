"""Profile the steps of an inference loop and reduce the trace to tables.

The method is the one of the DeepSeek-V4 decode profile
(`docs/benchmarks/2026-10-23-v4-flash-decode-profile.md`), written so that it
works for every model family:

1. `label_modules` puts each module call of a model in its own
   `torch.profiler.record_function` range. Layer and expert indices collapse
   to `*`, so all layers of one kind share one label. The decode methods that
   run outside `Module.__call__` (`DEFAULT_LABELLED_METHODS`) and the grouped
   expert matmuls get wrappers with the same labels. `restore()` removes all
   of it.
2. `profile_steps` runs a step function under `torch.profiler`, each step in
   a `step_label` range, then exports the Chrome trace and reduces it. Without
   CUDA only the CPU activity runs.
3. `summarize_step_trace` gives each kernel, copy and memset to the host call
   that launched it (same correlation id). That call gives the step, the
   innermost label and the innermost ATen op. The result holds per-step wall
   time, GPU busy time and counts, the GPU and host time by part of the model,
   the top labels and kernels, and the host sync sites.
4. `format_step_summary` prints those tables. On a CPU they show the wall time
   per step and the host time by part of the model.

`categorize` maps a label or a kernel name to a part of the model with
`DEFAULT_CATEGORY_RULES`; the first matching rule wins. A model with its own
module names passes its own rules.
"""

from __future__ import annotations

import contextlib
import functools
import json
import re
import tempfile
import threading
from bisect import bisect_right
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

# The profiler range around one step.
STEP_LABEL = "step"
# The range of the runner's packed forward inside a scheduler step.
RUNNER_LABEL = "forward_step_packed"
# The range of the sampling of a step.
SAMPLING_LABEL = "sampling"
# The range of the grouped expert matmuls, which run outside `Module.__call__`.
ROUTED_EXPERTS_LABEL = "routed_experts"

# Methods of model modules that run outside `Module.__call__`, so that module
# hooks miss them: the state-cache entry points of the hybrid families
# (DeepSeek-V4, Nemotron-H, Qwen3.5, Kimi Linear), the per-layer and per-block
# decode methods of DeepSeek-V4, the cached indexer of MiniMax-M3 and GLM, and
# the per-expert loop of the MoE blocks.
DEFAULT_LABELLED_METHODS: tuple[str, ...] = (
    "forward_prefill_with_cache",
    "forward_decode_with_cache",
    "forward_decode_with_cache_ragged",
    "decode_step_static",
    "forward_decode",
    "forward_decode_ragged",
    "forward_decode_step",
    "forward_decode_step_ragged",
    "forward_cached",
    "hc_pre",
    "hc_post",
    "_loop_experts",
)

# A rule maps the kernel name (`"kernel"`) or the innermost label (`"label"`)
# of a device event, or the label of a host range, to a part of the model.
CategoryRule = tuple[str, Literal["label", "kernel"], "re.Pattern[str]"]

# The parts of the model, by the module names of every family under
# `mini_infer.models`. A name that different classes share across the layers
# (`self_attn` of Kimi Linear, `mixer` of Nemotron-H, `mlp` of DeepSeek-V2,
# GLM, MiniMax-M3 and Inkling) carries its class, see `profile_label`, and the
# rules read that tag. The norms inside an attention block count as attention.
DEFAULT_CATEGORY_RULES: tuple[CategoryRule, ...] = (
    ("NCCL collectives", "kernel", re.compile(r"nccl", re.IGNORECASE)),
    ("sampling", "label", re.compile(rf"^{SAMPLING_LABEL}$")),
    ("runner glue", "label", re.compile(rf"^{RUNNER_LABEL}$|^decode_graph\.")),
    ("MoE shared expert", "label", re.compile(r"shared_experts")),
    ("MoE gate", "label", re.compile(r"\.gate(:|$)")),
    (
        "MoE routed experts",
        "label",
        re.compile(r"routed_experts|_loop_experts|\.experts(\.|$)|routed_expert_|mixer\[\w*MoE"),
    ),
    ("MoE other", "label", re.compile(r"block_sparse_moe|\.mlp\[\w*Mo[eE]")),
    (
        "linear attention",
        "label",
        re.compile(r"linear_attn|self_attn\[\w*Delta|mixer\[\w*Mamba"),
    ),
    (
        "attention",
        "label",
        re.compile(r"self_attn|mixer\[\w*Attention|attn_sconv|attn_res|attention_res"),
    ),
    ("hyper-connections", "label", re.compile(r"hc_attn|hc_ffn|hc_head")),
    ("MLP", "label", re.compile(r"\.mlp(\.|\[|:|$)|mlp_sconv|mlp_res")),
    ("norms", "label", re.compile(r"(^|\.)\w*norm\w*(:|$)")),
    ("embedding", "label", re.compile(r"embed")),
    ("lm_head", "label", re.compile(r"^lm_head")),
    ("RoPE tables", "label", re.compile(r"rotary")),
)
KERNEL_LAUNCHES = frozenset(
    {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"}
)
HOST_SYNCS = frozenset(
    {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy"}
)
DEVICE_EVENTS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

# A class, a method of it and the label of the wrapper around that method.
ClassMethod = tuple[type, str, str]


def _collapse(name: str) -> str:
    """The module name with layer and expert indices replaced by `*`."""
    return re.sub(r"\.\d+(?=\.|$)", ".*", name).removeprefix("model.")


def profile_label(model: Any, name: str, *, tagged: Collection[str] = ()) -> str:
    """A short profiler range name for the submodule `name` of `model`.

    Layer and expert indices become `*`, so that all layers share one name,
    and the leading `model.` goes. Each part named in `tagged` carries the
    class of its module, as in `self_attn[CSAAttention]`, for names that
    different classes share across the layers. The root module is `model`.
    """
    parts = []
    prefix: list[str] = []
    for part in name.split(".") if name else []:
        prefix.append(part)
        if part in tagged:
            part = f"{part}[{type(model.get_submodule('.'.join(prefix))).__name__}]"
        parts.append(part)
    label = re.sub(r"\.\d+(?=\.|$)", ".*", ".".join(parts)).removeprefix("model.")
    return label or "model"


def module_labels(model: Any, *, tagged: Collection[str] | None = None) -> dict[Any, str]:
    """The profiler label of every module of `model`, see `profile_label`.

    With `tagged=None`, the parts whose collapsed name holds more than one
    class across the layers carry their class, which keeps the kinds apart:
    the CSA, HCA and SWA attention of DeepSeek-V4, the delta and MLA attention
    of Kimi Linear, the Mamba, attention and MoE mixers of Nemotron-H, the
    dense and MoE `mlp` of DeepSeek-V2.
    """
    named = list(model.named_modules())
    if tagged is None:
        classes: dict[str, set[str]] = {}
        for name, module in named:
            if name:
                classes.setdefault(_collapse(name), set()).add(type(module).__name__)
        tagged = {label.rsplit(".", 1)[-1] for label, kinds in classes.items() if len(kinds) > 1}
    return {module: profile_label(model, name, tagged=tagged) for name, module in named}


def default_class_methods() -> list[ClassMethod]:
    """The grouped expert matmuls that run outside `Module.__call__`.

    Each expert pack runs its matmuls from a plain object that the MoE block
    holds, so no module hook sees them. Their wrappers share the label
    `ROUTED_EXPERTS_LABEL`.
    """
    from mini_infer.models.blocks.moe.fp8_expert import PackedFp8Experts
    from mini_infer.models.blocks.moe.mixtral_moe import PackedSwigluExperts
    from mini_infer.models.deepseek_v4.fp4_expert import PackedFp4Experts

    return [
        (cls, "run", ROUTED_EXPERTS_LABEL)
        for cls in (PackedSwigluExperts, PackedFp8Experts, PackedFp4Experts)
    ]


def label_modules(
    model: Any,
    *,
    methods: Sequence[str] = DEFAULT_LABELLED_METHODS,
    class_methods: Iterable[ClassMethod] | None = None,
    tagged: Collection[str] | None = None,
) -> Callable[[], None]:
    """Put each module call and decode method of `model` in its own profiler range.

    Global module hooks name every `Module.__call__`; each thread keeps its own
    stack of open ranges. The `methods` run outside `Module.__call__`, so each
    module that has one gets an instance wrapper named `label:method`. The
    `class_methods` (default `default_class_methods()`) get a class wrapper.
    `tagged` is passed to `module_labels`. Returns the function that removes
    all of it.
    """
    import torch

    labels = module_labels(model, tagged=tagged)
    local = threading.local()

    def open_ranges() -> list[contextlib.ExitStack]:
        ranges: list[contextlib.ExitStack] | None = getattr(local, "ranges", None)
        if ranges is None:
            ranges = local.ranges = []
        return ranges

    def enter(module: Any, args: Any) -> None:
        label = labels.get(module)
        if label is not None:
            opened = contextlib.ExitStack()
            opened.enter_context(torch.profiler.record_function(label))
            open_ranges().append(opened)

    def leave(module: Any, args: Any, output: Any) -> None:
        if module in labels:
            open_ranges().pop().close()

    def labelled(method: Callable[..., Any], label: str) -> Callable[..., Any]:
        @functools.wraps(method)
        def run(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(label):
                return method(*args, **kwargs)

        return run

    hooks = [
        torch.nn.modules.module.register_module_forward_pre_hook(enter),
        torch.nn.modules.module.register_module_forward_hook(leave),
    ]
    wrapped: list[tuple[Any, str]] = []
    for module, label in labels.items():
        for method in methods:
            if hasattr(type(module), method):
                setattr(module, method, labelled(getattr(module, method), f"{label}:{method}"))
                wrapped.append((module, method))
    class_runs = []
    for cls, method, label in default_class_methods() if class_methods is None else class_methods:
        original = getattr(cls, method)
        setattr(cls, method, labelled(original, label))
        class_runs.append((cls, method, original))

    def restore() -> None:
        for hook in hooks:
            hook.remove()
        for module, method in wrapped:
            del module.__dict__[method]
        for cls, method, original in class_runs:
            setattr(cls, method, original)

    return restore


def categorize(
    label: str | None, kernel: str, rules: Sequence[CategoryRule] = DEFAULT_CATEGORY_RULES
) -> str:
    """The part of the model that `label` or `kernel` belongs to, else `other`."""
    for category, field, pattern in rules:
        text = kernel if field == "kernel" else label
        if text is not None and pattern.search(text):
            return category
    return "other"


def short_kernel(name: str) -> str:
    """A kernel name without its return type, template arguments and parameters."""
    name = name.removeprefix("void ")
    return re.split(r"[<(]", name, maxsplit=1)[0].strip()[:72]


def _innermost(ranges: list[dict[str, Any]], points: list[float]) -> list[dict[str, Any] | None]:
    """The innermost of the nested `ranges` that holds each of the sorted `points`."""
    ranges = sorted(ranges, key=lambda r: (r["ts"], -r["dur"]))
    found: list[dict[str, Any] | None] = []
    stack: list[dict[str, Any]] = []
    index = 0
    for point in points:
        while index < len(ranges) and ranges[index]["ts"] <= point:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < ranges[index]["ts"]:
                stack.pop()
            stack.append(ranges[index])
            index += 1
        while stack and stack[-1]["ts"] + stack[-1]["dur"] < point:
            stack.pop()
        found.append(stack[-1] if stack else None)
    return found


def _self_times(ranges: list[dict[str, Any]]) -> dict[str, float]:
    """Microseconds inside each named range and outside its nested ranges."""
    totals: dict[str, float] = {}
    stack: list[dict[str, Any]] = []
    for item in sorted(ranges, key=lambda r: (r["ts"], -r["dur"])):
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= item["ts"]:
            stack.pop()
        totals[item["name"]] = totals.get(item["name"], 0.0) + item["dur"]
        if stack:
            totals[stack[-1]["name"]] -= item["dur"]
        stack.append(item)
    return totals


def _busy_us(intervals: list[tuple[float, float]]) -> float:
    """The length of the union of `(start, end)` intervals."""
    busy, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            busy += stop - max(start, end)
            end = stop
    return busy


def summarize_step_trace(
    events: list[dict[str, Any]],
    *,
    step_label: str = STEP_LABEL,
    category_rules: Sequence[CategoryRule] = DEFAULT_CATEGORY_RULES,
    top: int = 25,
    step_self_category: str | None = None,
) -> dict[str, Any]:
    """Reduce a Chrome trace of `step_label` ranges to per-step tables.

    Each kernel, copy and memset on the GPU belongs to the host call that
    launched it (same correlation id). That call gives the step, the innermost
    profiler range (the part of the model) and the innermost ATen op. Times
    are milliseconds per step, averaged over the profiled steps. The host time
    of a category is the time the host spends inside its ranges, outside the
    ranges nested in them. With `step_self_category`, the time of a step
    outside every nested range (the host work between the module calls) goes
    to that category.
    """
    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]
    annotations = [e for e in complete if e.get("cat") == "user_annotation"]
    steps = sorted((e for e in annotations if e["name"] == step_label), key=lambda e: e["ts"])
    ranges = [e for e in annotations if e["name"] != step_label]
    ops = [e for e in complete if e.get("cat") == "cpu_op"]
    host_calls = sorted(
        (e for e in complete if e.get("cat") in ("cuda_runtime", "cuda_driver")),
        key=lambda e: e["ts"],
    )
    if not steps:
        raise ValueError(f"the trace has no {step_label!r} ranges")
    n_steps = len(steps)
    step_starts = [s["ts"] for s in steps]

    def step_of(ts: float) -> int | None:
        index = bisect_right(step_starts, ts) - 1
        if index >= 0 and ts <= steps[index]["ts"] + steps[index]["dur"]:
            return index
        return None

    # Innermost range and op of each host call, per thread: ranges nest per thread.
    call_info: dict[int, tuple[int | None, str | None, str | None]] = {}
    per_step: list[dict[str, float]] = [
        {"launches": 0, "launch_ms": 0.0, "syncs": 0, "sync_ms": 0.0} for _ in steps
    ]
    sync_labels: dict[str, int] = {}
    for tid in {e["tid"] for e in host_calls}:
        calls = [e for e in host_calls if e["tid"] == tid]
        points = [e["ts"] for e in calls]
        in_range = _innermost([r for r in ranges if r["tid"] == tid], points)
        in_op = _innermost([o for o in ops if o["tid"] == tid], points)
        for call, label_range, op in zip(calls, in_range, in_op, strict=True):
            step = step_of(call["ts"])
            label = label_range["name"] if label_range else None
            correlation = call.get("args", {}).get("correlation")
            if correlation is not None:
                call_info[correlation] = (step, label, op["name"] if op else None)
            if step is None:
                continue
            if call["name"] in KERNEL_LAUNCHES:
                per_step[step]["launches"] += 1
                per_step[step]["launch_ms"] += call["dur"] / 1e3
            elif call["name"] in HOST_SYNCS:
                per_step[step]["syncs"] += 1
                per_step[step]["sync_ms"] += call["dur"] / 1e3
                where = f"{label or 'unlabelled'} ({op['name'] if op else 'no op'})"
                sync_labels[where] = sync_labels.get(where, 0) + 1

    busy: list[list[tuple[float, float]]] = [[] for _ in steps]
    counts: list[dict[str, int]] = [{} for _ in steps]
    by_category: dict[str, dict[str, float]] = {}
    by_label: dict[tuple[str, str], dict[str, float]] = {}
    by_kernel: dict[tuple[str, str], dict[str, Any]] = {}
    unattributed = 0
    for event in complete:
        if event.get("cat") not in DEVICE_EVENTS:
            continue
        launch = call_info.get(event.get("args", {}).get("correlation"))
        step_index = launch[0] if launch is not None else None
        if launch is None or step_index is None:
            unattributed += 1
            continue
        _, range_name, op_name = launch
        busy[step_index].append((event["ts"], event["ts"] + event["dur"]))
        counts[step_index][event["cat"]] = counts[step_index].get(event["cat"], 0) + 1
        category = categorize(range_name, event["name"], category_rules)
        ms = event["dur"] / 1e3
        cat_entry = by_category.setdefault(category, {"gpu_ms": 0.0, "kernels": 0, "host_ms": 0.0})
        cat_entry["gpu_ms"] += ms
        cat_entry["kernels"] += 1
        label_entry = by_label.setdefault(
            (range_name or "unlabelled", category), {"gpu_ms": 0.0, "kernels": 0}
        )
        label_entry["gpu_ms"] += ms
        label_entry["kernels"] += 1
        kernel_entry = by_kernel.setdefault(
            (short_kernel(event["name"]), op_name or event["cat"]),
            {"gpu_ms": 0.0, "kernels": 0, "categories": {}},
        )
        kernel_entry["gpu_ms"] += ms
        kernel_entry["kernels"] += 1
        kernel_entry["categories"][category] = kernel_entry["categories"].get(category, 0) + 1

    # Host time per category: each range's own time, outside its nested ranges.
    host_labels: list[dict[str, Any]] = []
    for label, us in _self_times(ranges + steps).items():
        if label == step_label:
            if step_self_category is None:
                continue
            category = step_self_category
        else:
            category = categorize(label, "", category_rules)
            host_labels.append({"key": label, "host_ms": us / 1e3 / n_steps, "category": category})
        entry = by_category.setdefault(category, {"gpu_ms": 0.0, "kernels": 0, "host_ms": 0.0})
        entry["host_ms"] += us / 1e3

    def per_step_rows(table: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = [
            {
                "key": key,
                "gpu_ms": entry["gpu_ms"] / n_steps,
                "kernels": entry["kernels"] / n_steps,
                **({"host_ms": entry["host_ms"] / n_steps} if "host_ms" in entry else {}),
                **(
                    {"category": max(entry["categories"], key=entry["categories"].get)}
                    if "categories" in entry
                    else {}
                ),
            }
            for key, entry in table.items()
        ]
        return sorted(rows, key=lambda row: (-row["gpu_ms"], -row.get("host_ms", 0.0)))

    return {
        "steps": [
            {
                "wall_ms": step["dur"] / 1e3,
                "gpu_busy_ms": _busy_us(busy[index]) / 1e3,
                "kernels": counts[index].get("kernel", 0),
                "memcpys": counts[index].get("gpu_memcpy", 0),
                "memsets": counts[index].get("gpu_memset", 0),
                **per_step[index],
            }
            for index, step in enumerate(steps)
        ],
        "categories": per_step_rows(by_category),
        "labels": per_step_rows(by_label)[:top],
        "kernels": per_step_rows(by_kernel)[:top],
        "host_labels": sorted(host_labels, key=lambda row: -row["host_ms"])[:top],
        "sync_sites": sorted(
            ((where, count / n_steps) for where, count in sync_labels.items()),
            key=lambda item: -item[1],
        ),
        "unattributed_device_events": unattributed,
    }


def new_profiler(on_cuda: bool) -> torch.profiler.profile:
    """A `torch.profiler` over the CPU activity, plus CUDA with `on_cuda`.

    No shapes and no stacks, so the host overhead stays low. The profiler
    records the thread that enters it, so enter it on the thread that runs
    the steps.
    """
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if on_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities, record_shapes=False)


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """The events of a Chrome trace file."""
    events: list[dict[str, Any]] = json.loads(Path(path).read_bytes())["traceEvents"]
    return events


def trace_events(
    profiler: torch.profiler.profile, trace_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Export the Chrome trace of a finished `profiler` and return its events.

    The trace file stays at `trace_path` when given, else it goes with its
    temporary directory. A trace of a real model is large (about 80 MB for a
    few DeepSeek-V4 steps), so keep it out of the repository.
    """
    if trace_path is None:
        with tempfile.TemporaryDirectory(prefix="step-profile-") as tmp:
            path = Path(tmp) / "trace.json"
            profiler.export_chrome_trace(str(path))
            return load_trace(path)
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    return load_trace(path)


def profile_steps(
    run_step: Callable[[], Any],
    *,
    steps: int,
    model: Any | None = None,
    on_cuda: bool = False,
    step_label: str = STEP_LABEL,
    methods: Sequence[str] = DEFAULT_LABELLED_METHODS,
    class_methods: Iterable[ClassMethod] | None = None,
    tagged: Collection[str] | None = None,
    category_rules: Sequence[CategoryRule] = DEFAULT_CATEGORY_RULES,
    top: int = 25,
    trace_path: str | Path | None = None,
    step_self_category: str | None = None,
) -> dict[str, Any]:
    """Profile `steps` calls of `run_step`, each in a `step_label` range.

    With `model`, every module call and decode method of the model gets its
    own range (`label_modules`) for the duration of the profile. The trace is
    reduced with `summarize_step_trace`; `trace_path` keeps the trace file.
    Warm up before this call: the first steps compile kernels and fill caches.
    """
    import torch

    restore = (
        label_modules(model, methods=methods, class_methods=class_methods, tagged=tagged)
        if model is not None
        else None
    )
    try:
        with new_profiler(on_cuda) as profiler:
            for _ in range(steps):
                with torch.profiler.record_function(step_label):
                    run_step()
            if on_cuda:
                torch.cuda.synchronize()
    finally:
        if restore is not None:
            restore()
    return summarize_step_trace(
        trace_events(profiler, trace_path),
        step_label=step_label,
        category_rules=category_rules,
        top=top,
        step_self_category=step_self_category,
    )


def format_step_summary(summary: dict[str, Any], *, gpu: bool | None = None) -> str:
    """The tables of a `summarize_step_trace` result as text.

    `gpu` adds the GPU columns and tables; by default they appear when the
    trace holds device events. Step rows that carry `phase`, `batch` and
    `tokens` (the scheduler profile) show them next to the wall time.
    """
    steps = summary["steps"]
    if gpu is None:
        gpu = any(row["kernels"] or row["memcpys"] or row["launches"] for row in steps)
    with_phase = all("phase" in row for row in steps)
    lines = ["profiled steps:"]
    header = f"{'step':>4} "
    if with_phase:
        header += f"{'phase':<10} {'batch':>5} {'tokens':>6} "
    header += f"{'wall ms':>8}"
    if gpu:
        header += (
            f" {'GPU busy':>8} {'kernels':>7} {'memcpy':>6} {'memset':>6} "
            f"{'launches':>8} {'launch ms':>9} {'syncs':>5} {'sync ms':>7}"
        )
    lines.append(header)
    for index, row in enumerate(steps):
        text = f"{index:>4} "
        if with_phase:
            text += f"{row['phase']:<10} {row['batch']:>5} {row['tokens']:>6} "
        text += f"{row['wall_ms']:8.1f}"
        if gpu:
            text += (
                f" {row['gpu_busy_ms']:8.2f} {row['kernels']:7d} {row['memcpys']:6d} "
                f"{row['memsets']:6d} {row['launches']:8d} {row['launch_ms']:9.1f} "
                f"{row['syncs']:5d} {row['sync_ms']:7.1f}"
            )
        lines.append(text)

    categories = summary["categories"]
    total_gpu = sum(row["gpu_ms"] for row in categories)
    total_host = sum(row["host_ms"] for row in categories)
    if gpu:
        lines.append(f"\nby part of the model, per step (GPU sum {total_gpu:.2f} ms):")
        lines.append(f"{'category':<20} {'GPU ms':>8} {'share':>6} {'kernels':>8} {'host ms':>8}")
        for row in categories:
            share = row["gpu_ms"] / total_gpu if total_gpu else 0.0
            lines.append(
                f"{row['key']:<20} {row['gpu_ms']:8.2f} {share:6.1%} "
                f"{row['kernels']:8.1f} {row['host_ms']:8.2f}"
            )
        lines.append("\ntop profiler ranges by GPU time, per step:")
        for row in summary["labels"]:
            label, category = row["key"]
            lines.append(
                f"{row['gpu_ms']:8.3f} ms {row['kernels']:7.1f} kernels  {label} [{category}]"
            )
        lines.append("\ntop kernels by GPU time, per step:")
        for row in summary["kernels"]:
            kernel, op = row["key"]
            count = f"{row['kernels']:7.1f}x"
            lines.append(f"{row['gpu_ms']:8.3f} ms {count}  {kernel} ({op}) [{row['category']}]")
        lines.append("\nhost syncs per step, from the trace (range, op):")
        for where, count in summary["sync_sites"]:
            lines.append(f"{count:6.2f}  {where}")
    else:
        lines.append(f"\nhost ms by part of the model, per step (host sum {total_host:.2f} ms):")
        lines.append(f"{'category':<20} {'host ms':>8} {'share':>6}")
        for row in sorted(categories, key=lambda row: -row["host_ms"]):
            share = row["host_ms"] / total_host if total_host else 0.0
            lines.append(f"{row['key']:<20} {row['host_ms']:8.2f} {share:6.1%}")
    lines.append("\ntop profiler ranges by host time, per step:")
    for row in summary["host_labels"]:
        lines.append(f"{row['host_ms']:8.3f} ms  {row['key']} [{row['category']}]")
    if gpu:
        lines.append(f"\nunattributed GPU events: {summary['unattributed_device_events']}")
    return "\n".join(lines)
