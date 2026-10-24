"""Greedy generation on real DeepSeek-V4-Flash on 2x B200 (Modal).

Closes the gap the load smoke (`scripts/modal_v4_flash_smoke.py`) explicitly
left open: that smoke runs ONE `model(input_ids)` forward and checks the
logits are finite; it does not generate. This script drives the full
single-request greedy path on the real checkpoint:

    load (from_checkpoint, per rank) -> prefill -> greedy decode -> detokenize

via `StateCacheGenerator`, the same path the CPU unit tests exercise on
synthetic configs.

What this validates:
  1. `DeepseekV4ForCausalLM.from_checkpoint` loads real V4-Flash onto each of
     2 B200s under tensor parallelism (config.json + meta construction +
     `load_weights` + rotary rebuild), reusing the exact path the unit tests
     round-trip on a synthetic checkpoint.
  2. The cache-aware prefill + decode generate COHERENT text from a real
     prompt (real weights + a layer that is bit-parity validated against the
     DeepSeek-V4 reference should continue a prompt sensibly, not emit noise).
  3. The two ranks generate IDENTICAL tokens. Under correct tensor
     parallelism the per-rank head shards all-reduce to the same logits, so
     any divergence between ranks is a TP bug.

Why no side-by-side reference comparison here: the DeepSeek-V4 reference
(`third_party/deepseek_v4_reference/`) is a full runnable model, but two
full V4-Flash models (~158 GB each) do not fit 2x B200 (384 GB HBM)
simultaneously. Bit-parity against the reference is already covered at the
attention-layer level by `tests/unit/test_v4_prefill_cache_aware.py` (cosine
sim > 0.999, prefill + decode). At the whole-model scale the practical bar
is coherent, deterministic, rank-consistent output.

`--compare-paths` runs the routed experts on both MoE paths of one loaded
model: the grouped path (ADR-036, ADR-040) and the per-expert loop. It
compares the last prefill logits of the prompt and of a long real passage,
times that prefill, and greedy-generates on each path.

`--profile-decode` shows where the time of one decode step goes. For two
contexts (the prompt, and the first `_PASSAGE_TOKENS` tokens of the passage)
each rank prefills, runs warm-up steps, then times plain decode steps. Rank 0
also lists the host syncs of a few steps and profiles a few more with
`torch.profiler`. The container reduces the trace to tables: GPU-busy time
against wall time, GPU time by part of the model, the top kernels, and the
kernel launches and host syncs per step. `--trace-dir` also saves the raw
traces; keep them out of the repository. With `--cuda-graph`, each context
also replays the ragged decode step from a CUDA graph (`DecodeCudaGraph`),
times and profiles those steps, and compares their tokens with the eager ones.

`--profile-batch` profiles the ragged decode step that the continuous-batching
server runs, at each batch size of `--batch-sizes` (default 1 and 16). Each
row prefills its own slice of the passage. The eager step runs with every
module labelled, so its trace splits the GPU time by part of the model, and
a CUDA graph of the same step is timed and profiled next to it.

Run with:
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_generate.py
    HF_TOKEN=<token> uv run modal run scripts/modal_v4_flash_generate.py \
        --prompt "The capital of France is" --max-new-tokens 48
    uv run modal run scripts/modal_v4_flash_generate.py --compare-paths
    uv run modal run scripts/modal_v4_flash_generate.py --profile-decode \
        --trace-dir /tmp/v4-traces
    uv run modal run scripts/modal_v4_flash_generate.py --profile-decode --cuda-graph
    uv run modal run scripts/modal_v4_flash_generate.py --profile-batch --batch-sizes 1,16
"""

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal

_HF_TOKEN = os.environ.get("HF_TOKEN")
_SECRETS = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []
_HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_GPU = "B200:2"
_PASSAGE = Path(__file__).parent / "data" / "technical_passage.md"
# Prefill length of the passage in `--compare-paths` and `--profile-decode`, in tokens.
_PASSAGE_TOKENS = 2048
# Decode steps of `--profile-decode`, per context. The sync check and the
# profile each cover one block of the CSA layers (compression ratio 4), so
# each sees one flush.
_WARMUP_STEPS = 8
_TIMED_STEPS = 16
_SYNC_CHECK_STEPS = 4
_PROFILED_STEPS = 4
# `--profile-batch`: the cache size of the continuous-batching server bench,
# and the prompt length of row 0 (row `b` adds 8 tokens per row).
_BATCH_MAX_SEQ_LEN = 1024
_BATCH_PROMPT_TOKENS = 24
# The profiler range around each profiled decode step.
_STEP_LABEL = "decode_step"
# Decode methods that run outside `Module.__call__`, so module hooks miss them.
_LABELLED_METHODS = (
    "forward_decode_with_cache",
    "decode_step_static",
    "forward_decode",
    "forward_decode_ragged",
    "forward_decode_step",
    "forward_decode_step_ragged",
    "hc_pre",
    "hc_post",
    "_loop_experts",
)
# The part of the model a kernel belongs to: the first rule that matches its
# kernel name (`"kernel"`) or its innermost profiler range (`"label"`).
_CATEGORY_RULES = (
    ("NCCL collectives", "kernel", re.compile(r"nccl", re.IGNORECASE)),
    ("sampling", "label", re.compile(r"^sampling$")),
    ("MoE routed experts", "label", re.compile(r"routed_experts|_loop_experts")),
    ("MoE shared expert", "label", re.compile(r"\.mlp\.shared_experts")),
    ("MoE gate", "label", re.compile(r"\.mlp\.gate")),
    ("MoE other", "label", re.compile(r"\.mlp\b")),
    ("attention", "label", re.compile(r"\.self_attn")),
    ("hyper-connections", "label", re.compile(r"hc_attn|hc_ffn|hc_head")),
    ("norms", "label", re.compile(r"(^|\.)(input_layernorm|post_attention_layernorm|norm)$")),
    ("embedding", "label", re.compile(r"^embed_tokens")),
    ("lm_head", "label", re.compile(r"^lm_head")),
    ("RoPE tables", "label", re.compile(r"rotary_emb")),
)
_KERNEL_LAUNCHES = frozenset(
    {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"}
)
_HOST_SYNCS = frozenset(
    {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy"}
)
_DEVICE_EVENTS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

app = modal.App("mini-infer-v4-flash-generate")

# Same image as the load smoke: Blackwell needs the cu128 torch wheel.
# FlashInfer covers attention math; the V4 attention path also has our SDPA
# reference plus the per-block HCA/CSA/SWA dispatchers.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=4.40",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
    )
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_python_source("mini_infer")
)


def _checkpoint(rank: int, msg: str) -> None:
    """Print + flush so a hang's location is visible in Modal logs immediately."""
    print(f"[rank {rank}] {msg}", flush=True)


def _run_one_rank(rank: int, world_size: int, prompt: str, max_new_tokens: int) -> dict:
    """Per-rank container work: load V4-Flash, greedy-generate."""
    import torch

    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    # from_checkpoint reads the raw config.json, builds on meta, runs
    # load_weights (per-rank TP slicing falls out here), and rebuilds the
    # rotary buffer. The download + load of ~158 GB is the slow step; if a
    # run hangs, it hangs inside this call.
    _checkpoint(rank, f"loading real V4-Flash via from_checkpoint onto {device}...")
    model = DeepseekV4ForCausalLM.from_checkpoint(_MODEL_NAME, device=device, dtype=torch.bfloat16)
    _checkpoint(rank, "load_weights done; building tokenizer + generator...")

    tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
    generator = StateCacheGenerator(model, tokenizer)

    prompt_ids = tokenizer.encode(prompt)
    _checkpoint(
        rank,
        f"prompt encoded ({len(prompt_ids)} tokens); greedy-generating {max_new_tokens} tokens...",
    )
    with torch.inference_mode():
        generated_ids = generator.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(generated_ids)
    _checkpoint(rank, f"generation done ({len(generated_ids)} tokens).")
    return {
        "rank": rank,
        "world_size": world_size,
        "n_prompt_tokens": len(prompt_ids),
        "n_generated": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_text": generated_text,
    }


def _compare_one_rank(
    rank: int,
    world_size: int,
    prompt: str,
    max_new_tokens: int,
    passage: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Per-rank work of `--compare-paths`: load once, run both MoE paths.

    `device` defaults to this rank's GPU. A CPU dry run passes `"cpu"`.
    """
    import time

    import torch

    from mini_infer.engine.state_cache_generator import StateCacheGenerator
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.blocks import grouped_experts
    from mini_infer.models.blocks.fp4_expert import PackedFp4Experts
    from mini_infer.models.blocks.fp4_grouped_kernel import supports_block_fp4_grouped_mm
    from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    if device is None:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    if not (
        grouped_experts.supports_grouped_experts(device) and supports_block_fp4_grouped_mm(device)
    ):
        raise RuntimeError("the grouped FP4 path does not run on this device")
    _checkpoint(rank, f"loading real V4-Flash via from_checkpoint onto {device}...")
    model = DeepseekV4ForCausalLM.from_checkpoint(_MODEL_NAME, device=device, dtype=torch.bfloat16)
    tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
    generator = StateCacheGenerator(model, tokenizer)
    moe_blocks = [m for m in model.modules() if isinstance(m, HashRoutedMoEFFN)]
    prompt_ids = tokenizer.encode(prompt)
    passage_ids = tokenizer.encode(passage)[:_PASSAGE_TOKENS]

    def last_logits(ids: list[int]) -> Any:
        cache = model.build_state_cache(
            max_seq_len=len(ids), batch_size=1, device=device, dtype=torch.bfloat16
        )
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            logits = model.forward_prefill_with_cache(input_ids, state_cache=cache)
        return logits[0, -1].float()

    def timed_ms(fn: Any) -> tuple[Any, float]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, (time.perf_counter() - start) * 1e3

    runs: dict[str, dict[str, Any]] = {}
    packed_layers = 0
    for path, disabled in (("grouped", False), ("loop", True)):
        grouped_experts._DISABLED_FOR_BENCH = disabled
        _checkpoint(rank, f"{path} path: prefill and generate...")
        short = last_logits(prompt_ids)
        last_logits(passage_ids)  # warm-up: Triton compiles its kernels here
        long, prefill_ms = timed_ms(lambda: last_logits(passage_ids))
        ids, generate_ms = timed_ms(
            lambda: generator.generate_ids(
                prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id
            )
        )
        if path == "grouped":
            packed_layers = sum(
                isinstance(block._expert_pack, PackedFp4Experts) for block in moe_blocks
            )
        runs[path] = {
            "short": short,
            "long": long,
            "prefill_ms": prefill_ms,
            "generate_ms": generate_ms,
            "ids": ids,
            "text": tokenizer.decode(ids),
        }
    grouped_experts._DISABLED_FOR_BENCH = False

    grouped, loop = runs["grouped"], runs["loop"]
    pairs = enumerate(zip(grouped["ids"], loop["ids"], strict=False))
    diverge = next((i for i, (a, b) in pairs if a != b), None)

    def agreement(key: str) -> dict[str, Any]:
        a, b = grouped[key], loop[key]
        return {
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "max_abs_diff": float((a - b).abs().max()),
            "same_argmax": bool(a.argmax() == b.argmax()),
        }

    return {
        "rank": rank,
        "moe_layers": len(moe_blocks),
        "packed_fp4_layers": packed_layers,
        "prompt_tokens": len(prompt_ids),
        "passage_tokens": len(passage_ids),
        "short_logits": agreement("short"),
        "long_logits": agreement("long"),
        "first_divergence": diverge,
        "generated": {path: run["ids"] for path, run in runs.items()},
        "text": {path: run["text"] for path, run in runs.items()},
        "prefill_ms": {path: run["prefill_ms"] for path, run in runs.items()},
        "generate_ms": {path: run["generate_ms"] for path, run in runs.items()},
    }


def _profile_label(model: Any, name: str) -> str:
    """A short profiler range name for the submodule `name` of `model`.

    Layer and expert indices become `*` so that all layers share one name, and
    each `self_attn` carries its class (CSA, HCA or SWA attention).
    """
    parts = []
    prefix: list[str] = []
    for part in name.split(".") if name else []:
        prefix.append(part)
        if part == "self_attn":
            part = f"self_attn[{type(model.get_submodule('.'.join(prefix))).__name__}]"
        parts.append(part)
    label = re.sub(r"\.\d+(?=\.|$)", ".*", ".".join(parts)).removeprefix("model.")
    return label or "model"


def _label_modules(model: Any) -> Callable[[], None]:
    """Put each module call and decode method of `model` in its own profiler range.

    Module hooks name every `Module.__call__`. The methods in
    `_LABELLED_METHODS` and the grouped expert matmuls run outside it, so
    instance and class wrappers name them. Returns the function that removes
    all of it.
    """
    import functools

    import torch

    from mini_infer.models.blocks.fp4_expert import PackedFp4Experts
    from mini_infer.models.blocks.mixtral_moe import PackedSwigluExperts

    labels = {module: _profile_label(model, name) for name, module in model.named_modules()}
    open_ranges: list[Any] = []

    def enter(module: Any, args: Any) -> None:
        label = labels.get(module)
        if label is not None:
            open_ranges.append(torch.profiler.record_function(label).__enter__())

    def leave(module: Any, args: Any, output: Any) -> None:
        if module in labels:
            open_ranges.pop().__exit__(None, None, None)

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
        for method in _LABELLED_METHODS:
            if hasattr(type(module), method):
                setattr(module, method, labelled(getattr(module, method), f"{label}:{method}"))
                wrapped.append((module, method))
    pack_runs = [(cls, cls.run) for cls in (PackedFp4Experts, PackedSwigluExperts)]
    for cls, run in pack_runs:
        cls.run = labelled(run, "mlp.routed_experts")

    def restore() -> None:
        for hook in hooks:
            hook.remove()
        for module, method in wrapped:
            del module.__dict__[method]
        for cls, run in pack_runs:
            cls.run = run

    return restore


def _category(label: str | None, kernel: str) -> str:
    """The part of the model a kernel belongs to, see `_CATEGORY_RULES`."""
    for category, field, pattern in _CATEGORY_RULES:
        text = kernel if field == "kernel" else label
        if text is not None and pattern.search(text):
            return category
    return "other"


def _short_kernel(name: str) -> str:
    """A kernel name without its return type, template arguments and parameters."""
    name = name.removeprefix("void ")
    return re.split(r"[<(]", name, maxsplit=1)[0][:72]


def _innermost(ranges: list[dict], points: list[float]) -> list[dict | None]:
    """The innermost of the nested `ranges` that holds each of the sorted `points`."""
    ranges = sorted(ranges, key=lambda r: (r["ts"], -r["dur"]))
    found: list[dict | None] = []
    stack: list[dict] = []
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


def _self_times(ranges: list[dict]) -> dict[str, float]:
    """Microseconds inside each named range and outside its nested ranges."""
    totals: dict[str, float] = {}
    stack: list[dict] = []
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


def _summarize_decode_trace(events: list[dict], top: int = 25) -> dict[str, Any]:
    """Reduce a Chrome trace of `_STEP_LABEL` ranges to per-step tables.

    Each kernel, copy and set on the GPU belongs to the host call that launched
    it (same correlation id). That call gives the decode step, the innermost
    profiler range (the part of the model) and the innermost ATen op. Times are
    milliseconds per step, averaged over the profiled steps.
    """
    from bisect import bisect_right

    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]
    annotations = [e for e in complete if e.get("cat") == "user_annotation"]
    steps = sorted((e for e in annotations if e["name"] == _STEP_LABEL), key=lambda e: e["ts"])
    ranges = [e for e in annotations if e["name"] != _STEP_LABEL]
    ops = [e for e in complete if e.get("cat") == "cpu_op"]
    host_calls = sorted(
        (e for e in complete if e.get("cat") in ("cuda_runtime", "cuda_driver")),
        key=lambda e: e["ts"],
    )
    if not steps:
        raise ValueError(f"the trace has no {_STEP_LABEL!r} ranges")
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
            if call["name"] in _KERNEL_LAUNCHES:
                per_step[step]["launches"] += 1
                per_step[step]["launch_ms"] += call["dur"] / 1e3
            elif call["name"] in _HOST_SYNCS:
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
        if event.get("cat") not in _DEVICE_EVENTS:
            continue
        launch = call_info.get(event.get("args", {}).get("correlation"))
        step_index = launch[0] if launch is not None else None
        if launch is None or step_index is None:
            unattributed += 1
            continue
        _, range_name, op_name = launch
        busy[step_index].append((event["ts"], event["ts"] + event["dur"]))
        counts[step_index][event["cat"]] = counts[step_index].get(event["cat"], 0) + 1
        category = _category(range_name, event["name"])
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
            (_short_kernel(event["name"]), op_name or event["cat"]),
            {"gpu_ms": 0.0, "kernels": 0, "categories": {}},
        )
        kernel_entry["gpu_ms"] += ms
        kernel_entry["kernels"] += 1
        kernel_entry["categories"][category] = kernel_entry["categories"].get(category, 0) + 1

    # Host time per category: each range's own time, outside its nested ranges.
    for label, us in _self_times(ranges).items():
        entry = by_category.setdefault(
            _category(label, ""), {"gpu_ms": 0.0, "kernels": 0, "host_ms": 0.0}
        )
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
        return sorted(rows, key=lambda row: -row["gpu_ms"])

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
        "sync_sites": sorted(
            ((where, count / n_steps) for where, count in sync_labels.items()),
            key=lambda item: -item[1],
        ),
        "unattributed_device_events": unattributed,
    }


def _time_steps(run_step: Callable[..., None], on_cuda: bool) -> list[float]:
    """Run `_WARMUP_STEPS` steps, then return the wall time of `_TIMED_STEPS` steps in ms.

    Each step ends when greedy sampling reads its tokens back to the host, so
    the time covers the whole step.
    """
    import time

    import torch

    for _ in range(_WARMUP_STEPS):
        run_step()
    if on_cuda:
        torch.cuda.synchronize()
    step_ms = []
    for _ in range(_TIMED_STEPS):
        start = time.perf_counter()
        run_step()
        step_ms.append((time.perf_counter() - start) * 1e3)
    return step_ms


def _profile_steps(
    run_step: Callable[..., None],
    entry: dict[str, Any],
    *,
    rank: int,
    on_cuda: bool,
    keep_trace: bool,
    model: Any = None,
) -> None:
    """Run `_PROFILED_STEPS` steps; rank 0 profiles them and adds the tables to `entry`.

    Other ranks run the same steps unprofiled, because each step all-reduces
    across the ranks. With `model`, every module call and decode method of
    the model gets its own profiler range (`_label_modules`).
    """
    import gzip
    import json
    import tempfile

    import torch

    if rank != 0:
        for _ in range(_PROFILED_STEPS):
            run_step()
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if on_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    restore = _label_modules(model) if model is not None else None
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            for _ in range(_PROFILED_STEPS):
                with torch.profiler.record_function(_STEP_LABEL):
                    run_step(labelled=True)
            if on_cuda:
                torch.cuda.synchronize()
    finally:
        if restore is not None:
            restore()
    _checkpoint(rank, "reducing the trace...")
    trace_path = Path(tempfile.mkdtemp(prefix="v4-decode-")) / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    trace_bytes = trace_path.read_bytes()
    entry["profile"] = _summarize_decode_trace(json.loads(trace_bytes)["traceEvents"])
    if keep_trace:
        entry["trace_gz"] = gzip.compress(trace_bytes)


def _profile_one_rank(
    rank: int,
    world_size: int,
    prompt: str,
    passage: str,
    keep_trace: bool,
    cuda_graph: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Per-rank work of `--profile-decode`: time and profile decode steps.

    For each context, prefill, run `_WARMUP_STEPS`, time `_TIMED_STEPS`, then
    run `_SYNC_CHECK_STEPS` with the CUDA sync debug mode on and
    `_PROFILED_STEPS` under the profiler. Only rank 0 checks syncs and
    profiles; rank 1 runs the same steps, as each step all-reduces across both.
    With `cuda_graph`, the same steps then replay the ragged decode step from
    a CUDA graph (`DecodeCudaGraph`) on a second cache, and their tokens are
    compared with the eager ones. `device` defaults to this rank's GPU. A CPU
    dry run passes `"cpu"`.
    """
    import functools
    import gc
    import warnings

    import torch

    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph
    from mini_infer.engine.sampler import SamplingParams, sample
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.blocks.fp4_expert import PackedFp4Experts
    from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    if device is None:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    on_cuda = torch.device(device).type == "cuda"
    _checkpoint(rank, f"loading real V4-Flash via from_checkpoint onto {device}...")
    model = DeepseekV4ForCausalLM.from_checkpoint(_MODEL_NAME, device=device, dtype=torch.bfloat16)
    tokenizer = Tokenizer.from_pretrained(_MODEL_NAME)
    greedy = SamplingParams(temperature=0.0)
    contexts = {
        "prompt": tokenizer.encode(prompt),
        "passage": tokenizer.encode(passage)[:_PASSAGE_TOKENS],
    }

    def synchronize() -> None:
        if on_cuda:
            torch.cuda.synchronize()

    def sample_into(tokens: list[int], logits: torch.Tensor, labelled: bool) -> None:
        if labelled:
            with torch.profiler.record_function("sampling"):
                tokens.append(sample(logits[0, -1], greedy))
        else:
            tokens.append(sample(logits[0, -1], greedy))

    # One decode step as `StateCacheGenerator` runs it: forward, sample, advance.
    def step(cache: Any, tokens: list[int], labelled: bool = False) -> None:
        token = torch.tensor([[tokens[-1]]], device=device, dtype=torch.long)
        with torch.inference_mode():
            step_logits = model.forward_decode_with_cache(
                token, start_pos=cache.start_pos, state_cache=cache
            )
            sample_into(tokens, step_logits, labelled)
        cache.advance_start_pos(1)

    # One decode step replayed from the graph; the host inputs are copied in.
    def graph_step(
        graph: DecodeCudaGraph, cache: Any, tokens: list[int], labelled: bool = False
    ) -> None:
        cache.check_capacity(cache.start_pos + 1)
        logits = graph.replay(torch.tensor([[tokens[-1]]]), torch.tensor([cache.start_pos]))
        sample_into(tokens, logits, labelled)
        cache.advance_start_pos(1)

    def prefill(ids: list[int], n_steps: int) -> tuple[Any, list[int]]:
        cache = model.build_state_cache(
            max_seq_len=len(ids) + n_steps + 1,
            batch_size=1,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            logits = model.forward_prefill_with_cache(
                torch.tensor([ids], device=device, dtype=torch.long), state_cache=cache
            )
        cache.advance_start_pos(len(ids))
        return cache, [sample(logits[0, -1], greedy)]

    def profiled(run_step: Callable[..., None], entry: dict[str, Any], labels: bool) -> None:
        _profile_steps(
            run_step,
            entry,
            rank=rank,
            on_cuda=on_cuda,
            keep_trace=keep_trace,
            model=model if labels else None,
        )

    n_steps = _WARMUP_STEPS + _TIMED_STEPS + _SYNC_CHECK_STEPS + _PROFILED_STEPS
    results: dict[str, Any] = {"rank": rank, "contexts": {}}
    for context, ids in contexts.items():
        _checkpoint(rank, f"{context}: prefill {len(ids)} tokens, then decode...")
        cache, tokens = prefill(ids, n_steps)
        run_step = functools.partial(step, cache, tokens)
        entry: dict[str, Any] = {
            "context_tokens": len(ids),
            "step_ms": _time_steps(run_step, on_cuda),
        }

        if rank == 0 and on_cuda:
            torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(_SYNC_CHECK_STEPS):
                    run_step()
        finally:
            if rank == 0 and on_cuda:
                torch.cuda.set_sync_debug_mode("default")
        sync_lines: dict[str, int] = {}
        for warning in caught:
            if "synchroniz" in str(warning.message):
                where = f"{warning.filename.rpartition('mini_infer/')[2]}:{warning.lineno}"
                sync_lines[where] = sync_lines.get(where, 0) + 1
        entry["sync_lines"] = sorted(
            ((where, n / _SYNC_CHECK_STEPS) for where, n in sync_lines.items()),
            key=lambda item: -item[1],
        )
        profiled(run_step, entry, labels=True)
        entry["text"] = tokenizer.decode(tokens)
        del run_step, cache

        if cuda_graph:
            # Capture first: warmup and capture write into the cache rows.
            _checkpoint(rank, f"{context}: capture the decode step in a CUDA graph...")
            graph_cache = model.build_state_cache(
                max_seq_len=len(ids) + n_steps + 1,
                batch_size=1,
                device=device,
                dtype=torch.bfloat16,
            )
            graph = DecodeCudaGraph(model, graph_cache, batch_size=1, device=device)
            prefilled, graph_tokens = prefill(ids, n_steps)
            graph_cache.copy_row_from(prefilled, src_row=0, dst_row=0)
            graph_cache.advance_start_pos(len(ids))
            del prefilled
            run_graph = functools.partial(graph_step, graph, graph_cache, graph_tokens)
            graph_entry: dict[str, Any] = {"step_ms": _time_steps(run_graph, on_cuda)}
            for _ in range(_SYNC_CHECK_STEPS):
                run_graph()
            profiled(run_graph, graph_entry, labels=False)
            pairs = enumerate(zip(tokens, graph_tokens, strict=True))
            graph_entry["first_divergence"] = next((i for i, (a, b) in pairs if a != b), None)
            graph_entry["text"] = tokenizer.decode(graph_tokens)
            entry["graph"] = graph_entry
            # The graph holds captured NCCL work; free it before the process
            # group goes away, which can otherwise hang.
            del run_graph, graph, graph_cache
            gc.collect()
            synchronize()
        results["contexts"][context] = entry

    moe_blocks = [m for m in model.modules() if isinstance(m, HashRoutedMoEFFN)]
    results["moe_layers"] = len(moe_blocks)
    results["packed_fp4_layers"] = sum(
        isinstance(block._expert_pack, PackedFp4Experts) for block in moe_blocks
    )
    return results


def _profile_batch_one_rank(
    rank: int,
    world_size: int,
    passage: str,
    batch_sizes: list[int],
    keep_trace: bool,
    device: str | None = None,
) -> dict[str, Any]:
    """Per-rank work of `--profile-batch`: the ragged decode step at each batch size.

    Row `b` of a batch prefills its own slice of the passage, of
    `_BATCH_PROMPT_TOKENS + 8 * b` tokens, so the rows sit at different
    positions. For each batch size, the eager ragged step runs with every
    module labelled, which splits its GPU time by part of the model. Then a
    `DecodeCudaGraph` of the same step runs on a second cache: it is timed and
    profiled, and its tokens are compared with the eager ones. The graph
    replays the kernels of the eager step, so the eager split also splits the
    graph. Rank 0 profiles; rank 1 runs the same steps. `device` defaults to
    this rank's GPU. A CPU dry run passes `"cpu"`.
    """
    import functools
    import gc

    import torch

    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph
    from mini_infer.engine.tokenizer import Tokenizer
    from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM

    if device is None:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    on_cuda = torch.device(device).type == "cuda"
    _checkpoint(rank, f"loading real V4-Flash via from_checkpoint onto {device}...")
    model = DeepseekV4ForCausalLM.from_checkpoint(_MODEL_NAME, device=device, dtype=torch.bfloat16)
    passage_ids = Tokenizer.from_pretrained(_MODEL_NAME).encode(passage)

    def new_cache(batch_size: int) -> Any:
        return model.build_state_cache(
            max_seq_len=_BATCH_MAX_SEQ_LEN,
            batch_size=batch_size,
            device=device,
            dtype=torch.bfloat16,
        )

    def prefill_rows(cache: Any, prompts: list[list[int]]) -> list[list[int]]:
        """Prefill each prompt in its own row; the first token of each row."""
        tokens = []
        for row, ids in enumerate(prompts):
            single = new_cache(1)
            with torch.inference_mode():
                logits = model.forward_prefill_with_cache(
                    torch.tensor([ids], device=device, dtype=torch.long), state_cache=single
                )
            cache.copy_row_from(single, src_row=0, dst_row=row)
            tokens.append([int(logits[0, -1].argmax())])
        return tokens

    def sample_rows(tokens: list[list[int]], logits: torch.Tensor, labelled: bool) -> None:
        if labelled:
            with torch.profiler.record_function("sampling"):
                next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
        else:
            next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
        for row, token in enumerate(next_tokens):
            tokens[row].append(token)

    # One ragged decode step, as the continuous-batching server runs it.
    def eager_step(
        cache: Any, tokens: list[list[int]], start: list[int], labelled: bool = False
    ) -> None:
        positions = [first + len(row) - 1 for first, row in zip(start, tokens, strict=True)]
        with torch.inference_mode():
            logits = model.forward_decode_with_cache_ragged(
                torch.tensor([[row[-1]] for row in tokens], device=device, dtype=torch.long),
                positions=torch.tensor(positions, device=device, dtype=torch.long),
                state_cache=cache,
            )
            sample_rows(tokens, logits, labelled)

    # The same step replayed from the graph; the host inputs are copied in.
    def graph_step(
        graph: DecodeCudaGraph,
        cache: Any,
        tokens: list[list[int]],
        start: list[int],
        labelled: bool = False,
    ) -> None:
        positions = [first + len(row) - 1 for first, row in zip(start, tokens, strict=True)]
        cache.check_capacity(max(positions) + 1)
        logits = graph.replay(
            torch.tensor([[row[-1]] for row in tokens], dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
        )
        sample_rows(tokens, logits, labelled)

    profile = functools.partial(_profile_steps, rank=rank, on_cuda=on_cuda, keep_trace=keep_trace)
    results: dict[str, Any] = {"rank": rank, "batches": {}}
    for batch_size in batch_sizes:
        prompts = [
            passage_ids[64 * row : 64 * row + _BATCH_PROMPT_TOKENS + 8 * row]
            for row in range(batch_size)
        ]
        start = [len(ids) for ids in prompts]
        _checkpoint(rank, f"batch {batch_size}: eager ragged step...")
        cache = new_cache(batch_size)
        tokens = prefill_rows(cache, prompts)
        run_eager = functools.partial(eager_step, cache, tokens, start)
        eager: dict[str, Any] = {"step_ms": _time_steps(run_eager, on_cuda)}
        profile(run_eager, eager, model=model)
        del run_eager, cache

        # Capture first: warmup and capture write into the cache rows.
        _checkpoint(rank, f"batch {batch_size}: capture the step in a CUDA graph...")
        graph_cache = new_cache(batch_size)
        graph = DecodeCudaGraph(model, graph_cache, batch_size=batch_size, device=device)
        graph_tokens = prefill_rows(graph_cache, prompts)
        run_graph = functools.partial(graph_step, graph, graph_cache, graph_tokens, start)
        graph_entry: dict[str, Any] = {"step_ms": _time_steps(run_graph, on_cuda)}
        profile(run_graph, graph_entry)
        graph_entry["same_tokens_as_eager"] = graph_tokens == tokens
        # The graph holds captured NCCL work; free it before the process group
        # goes away, which can otherwise hang.
        del run_graph, graph, graph_cache
        gc.collect()
        if on_cuda:
            torch.cuda.synchronize()
        results["batches"][batch_size] = {
            "prompt_tokens": start,
            "eager": eager,
            "graph": graph_entry,
        }
    return results


def _child_entry(
    rank: int, world_size: int, worker: Callable[..., dict], kwargs: dict, queue
) -> None:
    """Module-level entry-point for `mp.spawn` workers (must be picklable).

    Owns the process group: it sends the result before it destroys the group,
    so a teardown that hangs cannot hold back the result. `_spawn_ranks`
    stops a process that is still alive 10 s after both results arrived.
    """
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed

    try:
        _checkpoint(rank, "calling init_distributed (nccl)...")
        init_distributed(
            world_size=world_size,
            rank=rank,
            backend="nccl",
            master_addr="127.0.0.1",
            master_port=29500,
        )
        queue.put(("ok", worker(rank, world_size, **kwargs)))
    except Exception:
        import traceback

        queue.put(("err", f"rank {rank} failed:\n{traceback.format_exc()}"))
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _spawn_ranks(worker: Callable[..., dict], **kwargs: Any) -> list[dict]:
    """Run `worker(rank, world_size, **kwargs)` on 2 processes (one per B200), by rank."""
    import torch.multiprocessing as mp

    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    processes = [
        ctx.Process(target=_child_entry, args=(rank, world_size, worker, kwargs, queue))
        for rank in range(world_size)
    ]
    for p in processes:
        p.start()
    results = []
    try:
        while len(results) < world_size:
            results.append(queue.get(timeout=1680))
            if results[-1][0] == "err":
                break  # the other rank can wait in a collective until the timeout
    finally:
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    errors = [payload for status, payload in results if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    return sorted((r[1] for r in results), key=lambda entry: entry["rank"])


@app.function(image=image, timeout=300)
def versions() -> dict:
    """The image's torch and Triton, and whether torch has `grouped_mm` (no GPU)."""
    import torch
    from torch.nn import functional

    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "torch": torch.__version__,
        "triton": triton_version,
        "grouped_mm": hasattr(functional, "grouped_mm"),
    }


@app.function(
    image=image,
    gpu=_GPU,
    timeout=1800,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def compare_moe_paths(prompt: str, max_new_tokens: int, passage: str) -> dict:
    """Both MoE paths on one load of V4-Flash per rank; see `_compare_one_rank`."""
    per_rank = _spawn_ranks(
        _compare_one_rank, prompt=prompt, max_new_tokens=max_new_tokens, passage=passage
    )
    grouped_ids = per_rank[0]["generated"]["grouped"]
    return {
        "ranks_agree": all(entry["generated"]["grouped"] == grouped_ids for entry in per_rank),
        "per_rank": per_rank,
    }


@app.function(
    image=image,
    gpu=_GPU,
    # A warm-cache load plus the profile takes about 5 minutes; this caps a hang.
    timeout=900,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def decode_profile(prompt: str, passage: str, keep_trace: bool, cuda_graph: bool) -> list[dict]:
    """Decode-step timings of both ranks and the profile of rank 0; see `_profile_one_rank`."""
    return _spawn_ranks(
        _profile_one_rank,
        prompt=prompt,
        passage=passage,
        keep_trace=keep_trace,
        cuda_graph=cuda_graph,
    )


@app.function(
    image=image,
    gpu=_GPU,
    # A warm-cache load plus two batch sizes takes about 5 minutes; this caps a hang.
    timeout=900,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def batch_profile(passage: str, batch_sizes: list[int], keep_trace: bool) -> list[dict]:
    """Eager and graph ragged decode steps per batch size; see `_profile_batch_one_rank`."""
    return _spawn_ranks(
        _profile_batch_one_rank, passage=passage, batch_sizes=batch_sizes, keep_trace=keep_trace
    )


@app.function(
    image=image,
    gpu=_GPU,
    # Hard wall-clock ceiling: a warm-cache load plus a short greedy generation
    # completes well inside this; the cap only bounds a hung run.
    timeout=1800,
    secrets=_SECRETS,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def generate(prompt: str, max_new_tokens: int) -> dict:
    """Spawn 2 worker processes (one per B200), greedy-generate, check agreement."""
    per_rank = _spawn_ranks(_run_one_rank, prompt=prompt, max_new_tokens=max_new_tokens)
    # Tensor-parallel ranks must agree token-for-token; divergence is a TP bug.
    rank0_ids = per_rank[0]["generated_ids"]
    ranks_agree = all(entry["generated_ids"] == rank0_ids for entry in per_rank)
    return {
        "prompt": prompt,
        "ranks_agree": ranks_agree,
        "generated_text": per_rank[0]["generated_text"],
        "per_rank": per_rank,
    }


def _print_comparison(result: dict) -> None:
    rank0 = result["per_rank"][0]
    print(f"\nranks agree on the grouped tokens (TP consistency): {result['ranks_agree']}")
    print(f"MoE layers: {rank0['moe_layers']}, on packed FP4 experts: {rank0['packed_fp4_layers']}")
    print(f"prompt {rank0['prompt_tokens']} tokens, passage {rank0['passage_tokens']} tokens")
    for key in ("short_logits", "long_logits"):
        print(f"{key}, grouped against loop: {rank0[key]}")
    print(f"first diverging generated token: {rank0['first_divergence']}")
    for path in ("grouped", "loop"):
        print(
            f"{path}: passage prefill {rank0['prefill_ms'][path]:.1f} ms, "
            f"generate {rank0['generate_ms'][path]:.1f} ms, text {rank0['text'][path]!r}"
        )


def _print_profile(per_rank: list[dict], trace_dir: str) -> None:
    """Print the tables of `--profile-decode`, and save the traces to `trace_dir` if set."""
    import gzip
    import statistics

    rank0 = per_rank[0]
    print(
        f"\nMoE layers: {rank0['moe_layers']}, on packed FP4 experts: {rank0['packed_fp4_layers']}"
    )
    for context, entry in rank0["contexts"].items():
        print(f"\n=== {context}: {entry['context_tokens']} context tokens ===")
        for rank_entry in per_rank:
            times = rank_entry["contexts"][context]["step_ms"]
            print(
                f"rank {rank_entry['rank']} unprofiled step: median {statistics.median(times):.1f} "
                f"ms, min {min(times):.1f}, max {max(times):.1f} ({len(times)} steps)"
            )
        profile = entry["profile"]
        steps = profile["steps"]
        print("\nprofiled steps (rank 0):")
        print(
            f"{'wall ms':>8} {'GPU busy':>8} {'kernels':>7} {'memcpy':>6} {'memset':>6} "
            f"{'launches':>8} {'launch ms':>9} {'syncs':>5} {'sync ms':>7}"
        )
        for row in steps:
            print(
                f"{row['wall_ms']:8.1f} {row['gpu_busy_ms']:8.2f} {row['kernels']:7d} "
                f"{row['memcpys']:6d} {row['memsets']:6d} {row['launches']:8d} "
                f"{row['launch_ms']:9.1f} {row['syncs']:5d} {row['sync_ms']:7.1f}"
            )
        total_gpu = sum(row["gpu_ms"] for row in profile["categories"])
        print(f"\nby part of the model, per step (GPU sum {total_gpu:.2f} ms):")
        print(f"{'category':<20} {'GPU ms':>8} {'share':>6} {'kernels':>8} {'host ms':>8}")
        for row in profile["categories"]:
            share = row["gpu_ms"] / total_gpu if total_gpu else 0.0
            print(
                f"{row['key']:<20} {row['gpu_ms']:8.2f} {share:6.1%} "
                f"{row['kernels']:8.1f} {row['host_ms']:8.2f}"
            )
        print("\ntop profiler ranges by GPU time, per step:")
        for row in profile["labels"]:
            label, category = row["key"]
            print(f"{row['gpu_ms']:8.3f} ms {row['kernels']:7.1f} kernels  {label} [{category}]")
        print("\ntop kernels by GPU time, per step:")
        for row in profile["kernels"]:
            kernel, op = row["key"]
            count = f"{row['kernels']:7.1f}x"
            print(f"{row['gpu_ms']:8.3f} ms {count}  {kernel} ({op}) [{row['category']}]")
        print("\nhost syncs per step, from the trace (range, op):")
        for where, count in profile["sync_sites"]:
            print(f"{count:6.2f}  {where}")
        print("\nhost syncs per step, from the CUDA sync debug mode (source line):")
        for where, count in entry["sync_lines"]:
            print(f"{count:6.2f}  {where}")
        print(f"\nunattributed GPU events: {profile['unattributed_device_events']}")
        print(f"decoded text: {entry['text']!r}")
        traces = {"eager": entry}
        if "graph" in entry:
            graph = entry["graph"]
            traces["graph"] = graph
            print("\nCUDA graph of the ragged decode step (DecodeCudaGraph, batch 1):")
            for rank_entry in per_rank:
                times = rank_entry["contexts"][context]["graph"]["step_ms"]
                print(
                    f"rank {rank_entry['rank']} graph step: median "
                    f"{statistics.median(times):.1f} ms, min {min(times):.1f}, "
                    f"max {max(times):.1f} ({len(times)} steps)"
                )
            for row in graph["profile"]["steps"]:
                print(
                    f"profiled: wall {row['wall_ms']:.1f} ms, GPU busy {row['gpu_busy_ms']:.2f} "
                    f"ms, {row['kernels']} kernels, {row['memcpys']} memcpy, {row['syncs']} syncs"
                )
            for row in graph["profile"]["kernels"][:12]:
                kernel, op = row["key"]
                print(
                    f"{row['gpu_ms']:8.3f} ms {row['kernels']:7.1f}x  {kernel} [{row['category']}]"
                )
            print(f"first token that differs from eager: {graph['first_divergence']}")
            print(f"graph text: {graph['text']!r}")
        for mode, holder in traces.items():
            if trace_dir and "trace_gz" in holder:
                path = Path(trace_dir) / f"v4-flash-decode-{context}-{mode}.json.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(holder["trace_gz"])
                size = len(gzip.decompress(holder["trace_gz"])) >> 20
                print(f"{mode} trace saved to {path} ({size} MiB)")


def _print_batch_profile(per_rank: list[dict], trace_dir: str) -> None:
    """Print the tables of `--profile-batch`, and save the traces to `trace_dir` if set."""
    import statistics

    batches = per_rank[0]["batches"]
    sizes = sorted(batches, key=int)
    print("\nper step: eager ragged step against its CUDA graph (rank 0 profiles)")
    print(
        f"{'batch':>5} {'eager ms':>8} {'graph ms':>8} {'graph busy':>10} "
        f"{'eager ops':>9} {'graph ops':>9}  same tokens"
    )
    for size in sizes:
        eager, graph = batches[size]["eager"], batches[size]["graph"]

        def ops(entry: dict) -> float:
            steps = entry["profile"]["steps"]
            return statistics.mean(s["kernels"] + s["memcpys"] + s["memsets"] for s in steps)

        busy = statistics.mean(s["gpu_busy_ms"] for s in graph["profile"]["steps"])
        print(
            f"{size:>5} {statistics.median(eager['step_ms']):8.1f} "
            f"{statistics.median(graph['step_ms']):8.1f} {busy:10.2f} "
            f"{ops(eager):9.0f} {ops(graph):9.0f}  {graph['same_tokens_as_eager']}"
        )

    by_size = {
        size: {row["key"]: row for row in batches[size]["eager"]["profile"]["categories"]}
        for size in sizes
    }
    first, last = sizes[0], sizes[-1]
    print(f"\neager GPU ms per step by part of the model (batch {first} against {last}):")
    print(f"{'category':<20} " + " ".join(f"{'b' + str(s):>8}" for s in sizes) + f" {'delta':>8}")
    for category in sorted(by_size[last], key=lambda c: -by_size[last][c]["gpu_ms"]):
        values = [by_size[s].get(category, {"gpu_ms": 0.0})["gpu_ms"] for s in sizes]
        print(
            f"{category:<20} "
            + " ".join(f"{v:8.2f}" for v in values)
            + f" {values[-1] - values[0]:8.2f}"
        )
    for size in (first, last):
        print(f"\ntop profiler ranges by eager GPU time, batch {size}:")
        for row in batches[size]["eager"]["profile"]["labels"][:15]:
            label, category = row["key"]
            print(f"{row['gpu_ms']:8.3f} ms {row['kernels']:7.1f} ops  {label} [{category}]")
        print(f"\ntop kernels by eager GPU time, batch {size}:")
        for row in batches[size]["eager"]["profile"]["kernels"][:15]:
            kernel, op = row["key"]
            count = f"{row['kernels']:7.1f}x"
            print(f"{row['gpu_ms']:8.3f} ms {count}  {kernel} ({op}) [{row['category']}]")
        print(f"\ntop kernels by graph GPU time, batch {size}:")
        for row in batches[size]["graph"]["profile"]["kernels"][:10]:
            kernel, _ = row["key"]
            print(f"{row['gpu_ms']:8.3f} ms {row['kernels']:7.1f}x  {kernel}")
    for size in sizes:
        for mode in ("eager", "graph"):
            holder = batches[size][mode]
            if trace_dir and "trace_gz" in holder:
                path = Path(trace_dir) / f"v4-flash-batch{size}-{mode}.json.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(holder["trace_gz"])
                print(f"{mode} trace of batch {size} saved to {path}")


@app.local_entrypoint()
def main(
    prompt: str = "The capital of France is",
    max_new_tokens: int = 32,
    compare_paths: bool = False,
    profile_decode: bool = False,
    cuda_graph: bool = False,
    profile_batch: bool = False,
    batch_sizes: str = "1,16",
    trace_dir: str = "",
) -> None:
    if profile_batch:
        sizes = [int(size) for size in batch_sizes.split(",")]
        per_rank = batch_profile.remote(_PASSAGE.read_text(), sizes, bool(trace_dir))
        _print_batch_profile(per_rank, trace_dir)
        return
    if profile_decode:
        per_rank = decode_profile.remote(prompt, _PASSAGE.read_text(), bool(trace_dir), cuda_graph)
        _print_profile(per_rank, trace_dir)
        return
    if compare_paths:
        found = versions.remote()
        print(f"image: {found}")
        if not found["grouped_mm"] or found["triton"] is None:
            raise SystemExit("the image lacks grouped_mm or Triton; the grouped path cannot run")
        _print_comparison(compare_moe_paths.remote(prompt, max_new_tokens, _PASSAGE.read_text()))
        return
    result = generate.remote(prompt, max_new_tokens)
    print(f"\nprompt: {result['prompt']!r}")
    print(f"ranks agree (TP consistency): {result['ranks_agree']}")
    print(f"generated: {result['generated_text']!r}")
    for entry in result["per_rank"]:
        print(
            f"  rank {entry['rank']}: {entry['n_prompt_tokens']} prompt tokens -> "
            f"{entry['n_generated']} generated"
        )
