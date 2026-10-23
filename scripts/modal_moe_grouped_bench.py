"""MoE expert dispatch on one GPU: the per-expert path against grouped matmuls.

On Hopper and newer GPUs the MoE blocks run their routed experts as two
grouped matmuls (`grouped_experts.py`): `functional.grouped_mm` for bf16
experts, `fp8_grouped_kernel.block_fp8_grouped_mm` (Triton and Gluon kernels)
for block-FP8 ones.
Before, they looped over experts (Mixtral, GLM, Nemotron-H prefill) or
gathered each token's expert weights (Nemotron-H decode). Both paths are
still in the code, and `grouped_experts._DISABLED_FOR_BENCH` switches
between them, so one process measures before and after on the same GPU,
weights and inputs.

On one GPU it runs, in order:

0. The grouped-expert and FP8 kernel unit tests, on the GPU, to fail fast.
1. Layer benchmark: one MoE layer with random weights at real sizes
   (Nemotron 3.5 Lightning 30B-A3B, Mixtral 8x7B, GLM-5.3 with bf16 and
   with block-FP8 experts), from 1 to 16384 tokens. Wall milliseconds per
   forward for both paths, the relative difference of their outputs, and a
   CUDA graph replay at decode sizes where the path can be captured.
2. Profile: `torch.profiler` on the Nemotron and the FP8 GLM layers at 4
   and 2048 tokens. GPU ops, GPU busy time and host syncs per forward, the
   top kernels, and a Chrome trace of each run (open in
   https://ui.perfetto.dev). Then the FP8 layer with each tile configuration
   of the Triton kernel.
3. Real model: Nemotron 3.5 Lightning 30B-A3B from the `mini-infer-hf-cache`
   volume. Agreement of the two paths on a 512-token prefill, prefill time
   on a real technical passage (512 and 2048 tokens), and decode
   milliseconds per step at batch 1, 4 and 16, eager and CUDA graph.

Run with:
    MINI_INFER_TEST_GPU=H100 uv run modal run scripts/modal_moe_grouped_bench.py
    MINI_INFER_TEST_GPU=B200 uv run modal run scripts/modal_moe_grouped_bench.py
Add `--fp8-only` to run only the unit tests and the FP8 layer (steps 0 to
2), or `--fp8-kernel` to run the unit tests and then compare the FP8
kernel's current form with its first one (`fp8_kernel_bench`), or
`--fp8-tiles` for the same comparison without the loop, the profiles and
the bf16 target, plus this GPU's default tiles with and without swapped
operands. On an H100 or a B200, `--fp8-gluon` runs the unit tests and then
compares the Gluon kernels of `fp8_grouped_gluon.py` with the Triton one
(`fp8_gluon_bench`). `--fp8-expert-tiles` compares the Triton kernel's row
tiles with tiles of one expert each, on any GPU (`fp8_expert_tiles_bench`).
`--tests-only` runs only the unit tests of step 0.
Add `--out-dir <dir>` to keep the report and traces there (default: a new
temporary directory). Dry run with tiny sizes on this machine (needs
the Nemotron 3 Nano 4B tokenizer in the local Hugging Face cache; the FP8
layer needs Triton, so it is skipped here):
    uv run --with accelerate python scripts/modal_moe_grouped_bench.py
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal

app = modal.App("mini-infer-moe-grouped-bench")

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.jinja"]

# Modal fixes `gpu` when the decorator runs, so the GPU comes from the environment.
_TEST_GPU = os.environ.get("MINI_INFER_TEST_GPU", "H100")
# A cost cap, not an expected duration. A hung container bills the whole cap.
_TEST_TIMEOUT = int(os.environ.get("MINI_INFER_TEST_TIMEOUT", "1200"))

_REPO = Path(__file__).resolve().parent.parent
_LOCAL_FIXTURES = _REPO / "tests" / "fixtures"
_LOCAL_PASSAGE = _REPO / "scripts" / "data" / "technical_passage.md"
_REMOTE_FIXTURES = Path("/root/tests/fixtures")
_REMOTE_PASSAGE = Path("/root/data/technical_passage.md")
_GPU_TESTS = [
    "tests/unit/test_fp8_grouped_kernel.py",
    "tests/unit/test_fp8_grouped_gluon.py",
    "tests/unit/test_grouped_experts.py",
]
# A deadlocked kernel never returns. The tests stop after this many seconds.
_GPU_TESTS_TIMEOUT = 900

hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=5.14,<5.15",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_xet]",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "pytest",
    )
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
    .add_local_dir(str(_REPO / "tests"), remote_path="/root/tests")
    .add_local_file(str(_LOCAL_PASSAGE), remote_path=str(_REMOTE_PASSAGE))
)

LAYER_TOKENS = [1, 4, 16, 64, 256, 1024, 4096, 16384]
GRAPH_TOKENS = [1, 4, 16, 64]
PROFILE_TOKENS = [4, 2048]
# Tiles of the Triton FP8 kernel, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
# num_stages). The kernel splits the work per segment up to 256 rows (32
# tokens at top-8) and per expert tile above, so each list is tried at the
# sizes it serves.
FP8_SMALL_TILES = [
    (16, 32, 128, 4, 3),
    (16, 64, 64, 4, 3),
    (16, 64, 128, 4, 3),
    (16, 128, 64, 4, 3),
    (16, 128, 128, 8, 3),
    (32, 64, 128, 4, 3),
]
FP8_SMALL_TILE_TOKENS = [1, 4, 16]
FP8_LARGE_TILES = [
    (64, 128, 64, 4, 4),
    (64, 128, 128, 4, 3),
    (128, 64, 64, 4, 4),
    (128, 128, 64, 4, 4),
    (128, 128, 64, 8, 3),
    (128, 128, 128, 8, 3),
    (64, 256, 64, 8, 3),
    (128, 256, 64, 8, 3),
]
FP8_LARGE_TILE_TOKENS = [256, 1024, 4096, 16384]
PREFILL_TOKENS = [512, 2048]
DECODE_BATCHES = [1, 4, 16]
# Nemotron-H gathers expert weights (capturable) up to this many pairs and
# loops with host syncs above it; mirrors `nemotron_h._GATHER_EXPERT_LIMIT`.
_NEMOTRON_GATHER_LIMIT = 64


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_ms(
    fn: Callable[[], Any], device: str, *, budget_s: float = 0.4, max_iters: int = 50
) -> float:
    """Median wall milliseconds of `fn()` after warmup.

    Each call is timed from host to host with a device sync on both sides, so
    launch overhead and host syncs inside `fn` count, as they do in serving.
    """
    for _ in range(2):
        fn()
    _sync(device)
    times: list[float] = []
    deadline = time.perf_counter() + budget_s
    while len(times) < 5 or (len(times) < max_iters and time.perf_counter() < deadline):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)


@contextmanager
def _grouped(enabled: bool) -> Iterator[None]:
    """Run the MoE blocks with (True) or without (False) the grouped path."""
    from mini_infer.models.blocks import grouped_experts

    previous = grouped_experts._DISABLED_FOR_BENCH
    grouped_experts._DISABLED_FOR_BENCH = not enabled
    try:
        yield
    finally:
        grouped_experts._DISABLED_FOR_BENCH = previous


@contextmanager
def _build_on(device: str, dtype: Any) -> Iterator[None]:
    """Create modules directly on `device` in `dtype` (no fp32 host copy)."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous)


def _layer_specs(fixtures: Path, *, tiny: bool) -> list[dict[str, Any]]:
    nemotron = json.loads((fixtures / "nemotron_h" / "lightning_30b_a3b_config.json").read_text())
    glm = json.loads((fixtures / "glm53" / "config.json").read_text())
    specs: list[dict[str, Any]] = [
        {
            "name": "Nemotron 3.5 30B-A3B",
            "kind": "nemotron",
            "config": nemotron,
            "hidden": nemotron["hidden_size"],
            "inter": nemotron["moe_intermediate_size"],
            "experts": nemotron["n_routed_experts"],
            "top_k": nemotron["num_experts_per_tok"],
            "profile": True,
        },
        {
            "name": "Mixtral 8x7B",
            "kind": "mixtral",
            "hidden": 4096,
            "inter": 14336,
            "experts": 8,
            "top_k": 2,
        },
        {
            "name": "GLM-5.3",
            "kind": "glm",
            "hidden": glm["hidden_size"],
            "inter": glm["moe_intermediate_size"],
            "experts": glm["n_routed_experts"],
            "top_k": glm["num_experts_per_tok"],
            "shared": glm["n_shared_experts"],
            "scale": glm["routed_scaling_factor"],
        },
    ]
    # The same layer with e4m3 experts and 128x128 block scales, as the
    # GLM-5 FP8 checkpoints ship them. Random weights: the timing does not
    # depend on their values.
    specs.append({**specs[-1], "name": "GLM-5.3, FP8 experts", "kind": "glm_fp8", "profile": True})
    if tiny:
        # The FP8 grouped path needs Triton and 128-wide experts.
        specs = [spec for spec in specs if spec["kind"] != "glm_fp8"]
        for spec in specs:
            spec.update(hidden=64, inter=32, experts=min(spec["experts"], 16))
    return specs


def _build_layer(spec: dict[str, Any], device: str, dtype: Any) -> Any:
    import torch

    from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN
    from mini_infer.models.blocks.mixtral_moe import MoEFFN
    from mini_infer.models.nemotron_h import NemotronHConfig, _NemotronMoE

    with _build_on(device, dtype):
        if spec["kind"] == "nemotron":
            raw = dict(spec["config"])
            raw.update(
                hidden_size=spec["hidden"],
                moe_intermediate_size=spec["inter"],
                n_routed_experts=spec["experts"],
            )
            return _NemotronMoE(NemotronHConfig.from_hf(SimpleNamespace(**raw))).eval()
        if spec["kind"] == "mixtral":
            return MoEFFN(
                hidden_size=spec["hidden"],
                intermediate_size=spec["inter"],
                num_experts=spec["experts"],
                top_k=spec["top_k"],
            ).eval()
        fp8 = spec["kind"] == "glm_fp8"
        layer = GlmMoeFFN(
            hidden_size=spec["hidden"],
            moe_intermediate_size=spec["inter"],
            n_routed_experts=spec["experts"],
            top_k=spec["top_k"],
            n_shared_experts=spec["shared"],
            routed_scaling_factor=spec["scale"],
            expert_dtype="fp8" if fp8 else "bf16",
        ).eval()
        if fp8:
            with torch.no_grad():
                for expert in layer.experts:
                    for name in ("w1", "w2", "w3"):
                        weight = getattr(expert, name)
                        weight.copy_((torch.randn(weight.shape) * 0.5).to(torch.float8_e4m3fn))
                        getattr(expert, f"{name}_scale").uniform_(0.01, 0.03)
        return layer


def _old_path_capturable(spec: dict[str, Any], tokens: int) -> bool:
    """The loop syncs with the host; only Nemotron's small-batch gather captures."""
    return spec["kind"] == "nemotron" and tokens * spec["top_k"] <= _NEMOTRON_GATHER_LIMIT


def _graph_ms(layer: Any, x: Any, device: str) -> float:
    """Capture one forward of `layer` on `x` in a CUDA graph; time its replay."""
    import torch

    static_x = x.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(3):
            layer(static_x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        layer(static_x)
    return _time_ms(graph.replay, device)


def _profile(
    layer: Any, x: Any, device: str, *, wall_ms: float, name: str, traces: dict[str, bytes]
) -> list[str]:
    """Per-forward GPU ops, busy time and host syncs, plus the top kernels."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    iters = 5
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode():
        for _ in range(3):
            layer(x)
        _sync(device)
        with profile(activities=activities) as prof:
            for _ in range(iters):
                layer(x)
                _sync(device)
    events = prof.events()
    gpu_ops = [event for event in events if event.device_type == DeviceType.CUDA]
    busy_us = sum(event.time_range.elapsed_us() for event in gpu_ops)
    syncs = sum(1 for event in events if event.name == "cudaStreamSynchronize")
    by_kernel: Counter[str] = Counter()
    for event in gpu_ops:
        by_kernel[event.name] += event.time_range.elapsed_us()

    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        prof.export_chrome_trace(handle.name)
        traces[f"{name}.trace.json.gz"] = gzip.compress(Path(handle.name).read_bytes())

    busy_ms = busy_us / iters / 1000
    idle = max(0.0, 1 - busy_ms / wall_ms) if wall_ms > 0 else 0.0
    lines = [
        f"  {name}: {len(gpu_ops) / iters:.0f} GPU ops, {busy_ms:.3f} ms GPU busy of "
        f"{wall_ms:.3f} ms wall (GPU idle {100 * idle:.0f}%), "
        f"{syncs / iters:.0f} host syncs per forward"
    ]
    for kernel, total_us in by_kernel.most_common(5):
        lines.append(f"      {total_us / iters / 1000:8.3f} ms  {kernel[:110]}")
    return lines


def _fmt(value: float | str) -> str:
    return f"{value:9.3f}" if isinstance(value, float) else f"{value:>9}"


def _fp8_tile_sweep(
    layer: Any,
    hidden: int,
    device: str,
    dtype: Any,
    *,
    tile_list: list[tuple[int, int, int, int, int]],
    token_counts: list[int],
    title: str,
    swaps: tuple[bool, ...] = (True,),
) -> list[str]:
    """The grouped FP8 layer with each tile configuration of the Triton kernel.

    Each tile runs once per entry of `swaps`: with the weights as the left
    operand of the dot (True) or as the right one (False).
    """
    import torch

    from mini_infer.models.blocks import fp8_grouped_kernel

    lines = [
        "",
        f"## FP8 kernel tiles, {title} (wall ms per forward)",
        "   BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, swapped | "
        + " | ".join(f"{tokens:>7} tok" for tokens in token_counts),
    ]
    inputs = {
        tokens: torch.randn(1, tokens, hidden, device=device, dtype=dtype)
        for tokens in token_counts
    }
    for tiles in tile_list:
        for swap in swaps:
            cells = []
            hooks = _hooks(
                fp8_grouped_kernel,
                _TILES_FOR_BENCH=tiles,
                _SWAP_AB_FOR_BENCH=swap,
                _GLUON_FOR_BENCH=False,
            )
            try:
                with hooks, _grouped(True), torch.inference_mode():
                    for tokens in token_counts:
                        ms = _time_ms(partial(layer, inputs[tokens]), device)
                        cells.append(f"{ms:11.3f}")
            except Exception as exc:
                cells.append(f"failed: {exc!r}"[:120])
            label = f"{', '.join(map(str, tiles))}, {'yes' if swap else 'no'}"
            lines.append(f"   {label:>49} | " + " | ".join(cells))
    return lines


def _fp8_sweeps(layer: Any, hidden: int, device: str, dtype: Any) -> list[str]:
    """Triton tiles per segment at decode sizes and per expert tile at prefill sizes."""
    return _fp8_tile_sweep(
        layer,
        hidden,
        device,
        dtype,
        tile_list=FP8_SMALL_TILES,
        token_counts=FP8_SMALL_TILE_TOKENS,
        title="per segment, decode sizes",
    ) + _fp8_tile_sweep(
        layer,
        hidden,
        device,
        dtype,
        tile_list=FP8_LARGE_TILES,
        token_counts=FP8_LARGE_TILE_TOKENS,
        title="per expert tile, prefill sizes",
    )


@contextmanager
def _hooks(module: Any, **values: Any) -> Iterator[None]:
    """Set these bench hooks of `module` inside the block, and restore them after."""
    saved = {name: getattr(module, name) for name in values}
    for name, value in values.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _previous_fp8_kernel(enabled: bool, rows: int) -> Any:
    """Run the FP8 kernel as first measured: per row tile, weights on the right.

    That is the kernel's state before the swapped operands, the per-segment
    split and the tiles for each GPU family. `rows` is the number of
    (token, expert) pairs of the call, which picked one of its two tiles.
    """
    from mini_infer.models.blocks import fp8_grouped_kernel as kernel

    if not enabled:
        return _hooks(kernel)
    return _hooks(
        kernel,
        _SWAP_AB_FOR_BENCH=False,
        _PER_SEGMENT_FOR_BENCH=False,
        _TILES_FOR_BENCH=(16, 64, 64, 4, 3) if rows <= 256 else (64, 128, 64, 4, 4),
        _EXPERT_TILES_FOR_BENCH=False,
        _GLUON_FOR_BENCH=False,
    )


# Token counts that land in each tier of the kernel's tile table at top-8.
FP8_TIER_TOKENS = {"small": [1, 4, 16], "medium": [64, 256, 1024], "large": [4096, 16384]}


def _chosen_tiles_swap_sweep(layer: Any, hidden: int, device: str, dtype: Any) -> list[str]:
    """This GPU's default tile for each tier, with the operands swapped and not."""
    import torch

    from mini_infer.models.blocks import fp8_grouped_kernel as kernel

    lines: list[str] = []
    for tier, tiles in zip(FP8_TIER_TOKENS, kernel._tile_table(torch.device(device)), strict=True):
        lines += _fp8_tile_sweep(
            layer,
            hidden,
            device,
            dtype,
            tile_list=[tiles],
            token_counts=FP8_TIER_TOKENS[tier],
            title=f"{tier} batches, default tile",
            swaps=(True, False),
        )
    return lines


def fp8_kernel_bench(
    device: str,
    dtype: Any,
    fixtures: Path,
    *,
    traces: dict[str, bytes],
    shrink: bool = False,
    full: bool = True,
) -> list[str]:
    """The GLM-5.3 FP8 layer: the loop, the previous kernel and the new one.

    Then profiles of both kernels, the tile sweeps of the new one, and the
    same layer with bf16 experts on `grouped_mm` as the target. Without
    `full`, it skips the loop, the profiles and the target, and sweeps only
    this GPU's default tiles, with the operands swapped and not. `shrink`
    makes the layers tiny, for a dry run under Triton's interpreter.
    """
    import torch

    specs = {spec["kind"]: spec for spec in _layer_specs(fixtures, tiny=False)}
    token_counts, graph_tokens, profile_tokens = LAYER_TOKENS, GRAPH_TOKENS, PROFILE_TOKENS
    if shrink:
        for kind in ("glm", "glm_fp8"):
            specs[kind].update(hidden=128, inter=128, experts=8)
        token_counts, graph_tokens, profile_tokens = [1, 16, 64], [], [4]
    cuda = device.startswith("cuda")
    spec = specs["glm_fp8"]
    torch.manual_seed(0)
    layer = _build_layer(spec, device, dtype)
    lines = [
        f"## FP8 kernel, previous against new ({spec['name']}: hidden {spec['hidden']}, "
        f"expert width {spec['inter']}, {spec['experts']} experts, top-{spec['top_k']})",
        "wall ms per forward; graph columns are CUDA graph replays",
        "   tokens |      loop |  previous |       new | new vs previous | rel diff "
        "| graph previous | graph new",
    ]
    wall: dict[tuple[int, str], float] = {}
    paths = [("previous", True, True), ("new", True, False)]
    if full:
        paths.insert(0, ("loop", False, False))
    for tokens in token_counts:
        x = torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
        pairs = tokens * spec["top_k"]
        outs: dict[str, Any] = {}
        with torch.inference_mode():
            for name, grouped, previous in paths:
                with _grouped(grouped), _previous_fp8_kernel(previous, pairs):
                    wall[tokens, name] = _time_ms(partial(layer, x), device)
                    outs[name] = layer(x).float()
        base = outs["previous"]
        rel = float((outs["new"] - base).norm() / base.norm().clamp_min(1e-30))
        graph: dict[bool, float | str] = {True: "-", False: "-"}
        if cuda and tokens in graph_tokens:
            for previous in (True, False):
                try:
                    with _grouped(True), _previous_fp8_kernel(previous, pairs):
                        graph[previous] = _graph_ms(layer, x, device)
                except Exception as exc:
                    graph[previous] = "failed"
                    lines.append(f"   capture failed: {exc!r}"[:300])
        prev_ms, new_ms = wall[tokens, "previous"], wall[tokens, "new"]
        loop_ms: float | str = wall.get((tokens, "loop"), "-")
        lines.append(
            f"{tokens:>9} | {_fmt(loop_ms)} | {prev_ms:9.3f} | {new_ms:9.3f} | "
            f"{prev_ms / new_ms:14.2f}x | {rel:8.1e} | {_fmt(graph[True]):>14} | "
            f"{_fmt(graph[False]):>9}"
        )

    if not full:
        if cuda:
            lines += _chosen_tiles_swap_sweep(layer, spec["hidden"], device, dtype)
        return lines

    lines += ["", "## Profile, previous against new kernel"]
    for tokens in profile_tokens:
        x = torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
        for name, previous in (("previous", True), ("new", False)):
            with _grouped(True), _previous_fp8_kernel(previous, tokens * spec["top_k"]):
                ms = wall.get((tokens, name))
                if ms is None:
                    with torch.inference_mode():
                        ms = _time_ms(partial(layer, x), device)
                lines += _profile(
                    layer,
                    x,
                    device,
                    wall_ms=ms,
                    name=f"glm_fp8_{tokens}tok_{name}_kernel",
                    traces=traces,
                )
    if cuda:
        lines += _fp8_sweeps(layer, spec["hidden"], device, dtype)
    del layer
    if cuda:
        torch.cuda.empty_cache()

    target = specs["glm"]
    lines += [
        "",
        "## Target: the same layer with bf16 experts on grouped_mm (wall ms per forward)",
        "   tokens |     eager |     graph",
    ]
    torch.manual_seed(0)
    layer = _build_layer(target, device, dtype)
    with _grouped(True):
        for tokens in token_counts:
            x = torch.randn(1, tokens, target["hidden"], device=device, dtype=dtype)
            with torch.inference_mode():
                eager = _time_ms(partial(layer, x), device)
            replay: float | str = "-"
            if cuda and tokens in graph_tokens:
                try:
                    replay = _graph_ms(layer, x, device)
                except Exception as exc:
                    replay = f"failed: {exc!r}"[:80]
            lines.append(f"{tokens:>9} | {eager:9.3f} | {_fmt(replay)}")
    del layer
    if cuda:
        torch.cuda.empty_cache()
    return lines


# The Gluon kernels (sm90 and sm100) against the Triton kernel (`--fp8-gluon`).
FP8_GLUON_TOKENS = [32, 64, 256, 1024, 2048, 4096, 8192, 16384]
FP8_GLUON_KERNEL_TOKENS = [1024, 4096, 16384]
FP8_GLUON_TARGET_TOKENS = [1024, 4096, 16384]
# (BLOCK_W, BLOCK_T, BLOCK_K, warps, stages, warp specialization).
FP8_GLUON_TILES = [
    (128, 32, 64, 4, 4, True),
    (128, 64, 64, 4, 4, True),
    (128, 64, 128, 4, 3, True),
    (64, 128, 64, 4, 4, True),
    (128, 128, 64, 8, 4, True),
    (128, 128, 128, 8, 3, True),
    (128, 256, 64, 8, 3, True),
    (128, 64, 64, 4, 4, False),
    (128, 128, 64, 8, 4, False),
    (128, 256, 64, 8, 3, False),
]
# On sm100 `warps` counts the dequantization warps.
FP8_GLUON_BLACKWELL_TILES = [
    (128, 64, 64, 4, 4, True),
    (128, 64, 64, 8, 4, True),
    (128, 128, 64, 4, 4, True),
    (128, 128, 128, 4, 3, True),
    (128, 128, 128, 4, 4, True),
    (128, 128, 128, 8, 3, True),
    (128, 128, 128, 8, 4, True),
    (128, 256, 64, 4, 3, True),
    (128, 256, 128, 4, 2, True),
]
FP8_GLUON_SWEEP_TOKENS = [256, 1024, 4096, 16384]


@contextmanager
def _fp8_kernel(gluon: bool | None, tiles: tuple[Any, ...] | None = None) -> Iterator[None]:
    """Force the Triton kernel (False) or a Gluon one (True), or keep the default (None)."""
    from mini_infer.models.blocks import fp8_grouped_gluon, fp8_grouped_kernel

    with (
        _hooks(fp8_grouped_kernel, _GLUON_FOR_BENCH=gluon),
        _hooks(fp8_grouped_gluon, _TILES_FOR_BENCH=tiles),
    ):
        yield


# A label and a context that picks the FP8 kernel for that label.
KernelRuns = list[tuple[str, Callable[[], Any]]]


def _fp8_kernel_alone(layer: Any, spec: dict[str, Any], device: str, runs: KernelRuns) -> list[str]:
    """Each projection's grouped matmul alone, once for each of `runs`, with TFLOPS.

    Uses the layer's packed expert weights and random routing over all
    experts, as the layer benchmark does. TFLOPS count the useful products
    only, `2 * rows * N * K`.
    """
    import torch
    from triton.testing import do_bench

    from mini_infer.models.blocks.fp8_grouped_kernel import block_fp8_grouped_mm

    pack = layer._expert_pack
    lines = [
        "",
        "## FP8 grouped matmul alone (ms, median of do_bench; useful TFLOPS in brackets)",
        "   tokens | projection | " + " | ".join(f"{label:>18}" for label, _ in runs),
    ]
    for tokens in FP8_GLUON_KERNEL_TOKENS:
        pairs = tokens * spec["top_k"]
        experts = torch.randint(0, spec["experts"], (pairs,), device=device).sort().values
        ends = torch.bincount(experts, minlength=spec["experts"]).cumsum(0).to(torch.int32)
        for name, weight in (("gate_up", pack.gate_up), ("down", pack.down)):
            n_dim, k_dim = weight.weight.shape[1:]
            rows = torch.randn(pairs, k_dim, device=device, dtype=torch.bfloat16)
            flops = 2 * pairs * n_dim * k_dim
            cells = []
            for _label, choose in runs:
                with choose():
                    ms = do_bench(
                        partial(
                            block_fp8_grouped_mm,
                            rows,
                            weight.weight,
                            weight.scale,
                            block=weight.block,
                            row_experts=experts,
                            group_ends=ends,
                        ),
                        return_mode="median",
                    )
                cells.append(f"{ms:9.3f} ({flops / ms / 1e9:5.0f})")
            lines.append(f"{tokens:>9} | {name:>10} | " + " | ".join(cells))
    return lines


def _fp8_profiles(
    layer: Any, x: Any, device: str, runs: KernelRuns, *, traces: dict[str, bytes]
) -> list[str]:
    """A profile of the FP8 layer on `x` for each of `runs`."""
    import torch

    tokens = x.shape[1]
    lines = ["", f"## Profile at {tokens} tokens"]
    for label, choose in runs:
        with _grouped(True), choose():
            with torch.inference_mode():
                ms = _time_ms(partial(layer, x), device)
            name = f"glm_fp8_{tokens}tok_{label.replace(' ', '_')}"
            lines += _profile(layer, x, device, wall_ms=ms, name=name, traces=traces)
    return lines


def _bf16_target(spec: dict[str, Any], device: str, dtype: Any) -> list[str]:
    """The same layer with bf16 experts on `grouped_mm`, eager."""
    import torch

    lines = [
        "",
        "## Target: the same layer with bf16 experts on grouped_mm (eager wall ms per forward)",
    ]
    torch.manual_seed(0)
    layer = _build_layer(spec, device, dtype)
    with torch.inference_mode(), _grouped(True):
        for tokens in FP8_GLUON_TARGET_TOKENS:
            x = torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
            lines.append(f"{tokens:>9} | {_time_ms(partial(layer, x), device):9.3f}")
    del layer
    torch.cuda.empty_cache()
    return lines


def fp8_gluon_bench(
    device: str, dtype: Any, fixtures: Path, *, traces: dict[str, bytes]
) -> list[str]:
    """The GLM-5.3 FP8 layer with the Gluon kernel against the Triton kernel.

    It times the layer eager with Triton and with Gluon, on Hopper also
    without warp specialization. Then it times each projection alone,
    profiles both kernels at 4096 tokens, sweeps the Gluon tiles, and times
    the bf16 layer on `grouped_mm`.
    """
    import torch

    from mini_infer.device import is_datacenter_blackwell_device
    from mini_infer.models.blocks import fp8_grouped_gluon

    blackwell = is_datacenter_blackwell_device(device)
    specs = {spec["kind"]: spec for spec in _layer_specs(fixtures, tiny=False)}
    spec = specs["glm_fp8"]
    torch.manual_seed(0)
    layer = _build_layer(spec, device, dtype)
    lines = [
        f"## FP8 layer, Triton against Gluon ({spec['name']}: hidden {spec['hidden']}, "
        f"expert width {spec['inter']}, {spec['experts']} experts, top-{spec['top_k']})",
        "eager wall ms per forward; rel diff is Gluon against Triton",
        "   tokens |    triton |  gluon ws |  gluon no-ws | gluon tile             | "
        "speedup | rel diff",
    ]
    inputs = {
        tokens: torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
        for tokens in FP8_GLUON_TOKENS
    }
    for tokens in FP8_GLUON_TOKENS:
        x = inputs[tokens]
        pairs = tokens * spec["top_k"]
        tiles = fp8_grouped_gluon.default_tiles(pairs / spec["experts"], torch.device(device))
        runs: list[tuple[str, bool, tuple[Any, ...] | None]] = [
            ("triton", False, None),
            ("gluon", True, None),
        ]
        if not blackwell:
            runs.append(("gluon no-ws", True, (*tiles[:5], False)))
        wall: dict[str, float] = {}
        outs: dict[str, Any] = {}
        with torch.inference_mode(), _grouped(True):
            for label, gluon, forced in runs:
                with _fp8_kernel(gluon, forced):
                    wall[label] = _time_ms(partial(layer, x), device)
                    outs[label] = layer(x).float()
        base = outs["triton"]
        rel = float((outs["gluon"] - base).norm() / base.norm().clamp_min(1e-30))
        no_ws = wall.get("gluon no-ws")
        best = min(wall["gluon"], no_ws if no_ws is not None else wall["gluon"])
        no_ws_cell = f"{no_ws:12.3f}" if no_ws is not None else f"{'-':>12}"
        lines.append(
            f"{tokens:>9} | {wall['triton']:9.3f} | {wall['gluon']:9.3f} | "
            f"{no_ws_cell} | {', '.join(map(str, tiles)):<22} | "
            f"{wall['triton'] / best:6.2f}x | {rel:8.1e}"
        )

    kernels: KernelRuns = [
        ("triton", partial(_fp8_kernel, False)),
        ("gluon", partial(_fp8_kernel, True)),
    ]
    lines += _fp8_kernel_alone(layer, spec, device, kernels)
    lines += _fp8_profiles(layer, inputs[4096], device, kernels, traces=traces)

    lines += [
        "",
        "## Gluon tiles (eager wall ms per forward)",
        "   BLOCK_W, BLOCK_T, BLOCK_K, warps, stages, ws | "
        + " | ".join(f"{tokens:>7} tok" for tokens in FP8_GLUON_SWEEP_TOKENS),
    ]
    for tiles in FP8_GLUON_BLACKWELL_TILES if blackwell else FP8_GLUON_TILES:
        cells = []
        try:
            with torch.inference_mode(), _grouped(True), _fp8_kernel(True, tiles):
                for tokens in FP8_GLUON_SWEEP_TOKENS:
                    cells.append(f"{_time_ms(partial(layer, inputs[tokens]), device):11.3f}")
        except Exception as exc:
            cells.append(f"failed: {exc!r}"[:120])
        label = ", ".join(map(str, tiles))
        lines.append(f"   {label:>44} | " + " | ".join(cells))
    del layer
    torch.cuda.empty_cache()
    return lines + _bf16_target(specs["glm"], device, dtype)


# Two splits of the Triton kernel for many rows (`--fp8-expert-tiles`).
FP8_EXPERT_TILE_TOKENS = [64, 256, 1024, 2048, 4096, 8192, 16384]
FP8_EXPERT_TILES = [*FP8_LARGE_TILES, (32, 128, 64, 4, 4), (32, 256, 64, 8, 3)]
FP8_EXPERT_PROFILE_TOKENS = 4096


def _triton_split(expert_tiles: bool) -> Any:
    """Force the Triton kernel, with tiles of one expert each (True) or row tiles."""
    from mini_infer.models.blocks import fp8_grouped_kernel as kernel

    return _hooks(kernel, _EXPERT_TILES_FOR_BENCH=expert_tiles, _GLUON_FOR_BENCH=False)


def fp8_expert_tiles_bench(
    device: str, dtype: Any, fixtures: Path, *, traces: dict[str, bytes]
) -> list[str]:
    """The GLM-5.3 FP8 layer with the Triton kernel's two splits for many rows.

    Row tiles span several experts and run the K loop once for each. Expert
    tiles (the default) hold the rows of one expert. On Hopper the Gluon
    kernel runs too. Then it times each projection alone, profiles the
    kernels at 4096 tokens, sweeps the expert tiles, and times the bf16
    layer on `grouped_mm`.
    """
    import torch

    from mini_infer.device import is_hopper_device

    specs = {spec["kind"]: spec for spec in _layer_specs(fixtures, tiny=False)}
    spec = specs["glm_fp8"]
    torch.manual_seed(0)
    layer = _build_layer(spec, device, dtype)
    runs: KernelRuns = [
        ("row tiles", partial(_triton_split, False)),
        ("expert tiles", partial(_triton_split, True)),
    ]
    if is_hopper_device(device):
        runs.append(("gluon", partial(_fp8_kernel, True)))
    labels = [label for label, _ in runs]
    lines = [
        f"## FP8 layer, Triton row tiles against expert tiles ({spec['name']}: hidden "
        f"{spec['hidden']}, expert width {spec['inter']}, {spec['experts']} experts, "
        f"top-{spec['top_k']})",
        "eager wall ms per forward; rel diff is expert tiles against row tiles",
        "   tokens | "
        + " | ".join(f"{label:>12}" for label in labels)
        + " | expert vs row | rel diff",
    ]
    inputs = {
        tokens: torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
        for tokens in FP8_EXPERT_TILE_TOKENS
    }
    for tokens in FP8_EXPERT_TILE_TOKENS:
        wall: dict[str, float] = {}
        outs: dict[str, Any] = {}
        with torch.inference_mode(), _grouped(True):
            for label, choose in runs:
                with choose():
                    wall[label] = _time_ms(partial(layer, inputs[tokens]), device)
                    outs[label] = layer(inputs[tokens]).float()
        base = outs["row tiles"]
        rel = float((outs["expert tiles"] - base).norm() / base.norm().clamp_min(1e-30))
        cells = " | ".join(f"{wall[label]:12.3f}" for label in labels)
        lines.append(
            f"{tokens:>9} | {cells} | {wall['row tiles'] / wall['expert tiles']:12.2f}x | "
            f"{rel:8.1e}"
        )

    lines += _fp8_kernel_alone(layer, spec, device, runs)
    lines += _fp8_profiles(layer, inputs[FP8_EXPERT_PROFILE_TOKENS], device, runs, traces=traces)
    with _triton_split(True):
        lines += _fp8_tile_sweep(
            layer,
            spec["hidden"],
            device,
            dtype,
            tile_list=FP8_EXPERT_TILES,
            token_counts=FP8_LARGE_TILE_TOKENS,
            title="expert tiles, prefill sizes",
        )
    del layer
    torch.cuda.empty_cache()
    return lines + _bf16_target(specs["glm"], device, dtype)


def layer_bench(
    device: str,
    dtype: Any,
    fixtures: Path,
    *,
    tiny: bool,
    traces: dict[str, bytes],
    kinds: tuple[str, ...] | None = None,
) -> list[str]:
    """Steps 1 and 2: one MoE layer per model size, both paths, and profiles.

    `kinds` limits the run to those layer kinds (all when `None`).
    """
    import torch

    lines = ["## Layer benchmark (random weights, one MoE layer, wall ms per forward)"]
    cuda = device.startswith("cuda")
    graphs_ok = cuda
    token_counts = LAYER_TOKENS if not tiny else [1, 4, 64, 256]
    specs = _layer_specs(fixtures, tiny=tiny)
    for spec in specs if kinds is None else [s for s in specs if s["kind"] in kinds]:
        _log(f"layer bench: {spec['name']}")
        torch.manual_seed(0)
        layer = _build_layer(spec, device, dtype)
        lines.append("")
        lines.append(
            f"### {spec['name']}: hidden {spec['hidden']}, expert width {spec['inter']}, "
            f"{spec['experts']} experts, top-{spec['top_k']}"
        )
        lines.append(
            "   tokens |    before |     after |  speedup | rel diff | graph before | graph after"
        )
        wall: dict[tuple[int, bool], float] = {}
        for tokens in token_counts:
            x = torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
            with torch.inference_mode():
                for enabled in (False, True):
                    with _grouped(enabled):
                        wall[tokens, enabled] = _time_ms(partial(layer, x), device)
                with _grouped(False):
                    before_out = layer(x).float()
                with _grouped(True):
                    after_out = layer(x).float()
            rel = float((after_out - before_out).norm() / before_out.norm().clamp_min(1e-30))
            graph: dict[bool, float | str] = {False: "-", True: "-"}
            if graphs_ok and tokens in GRAPH_TOKENS:
                for enabled in (False, True):
                    if not enabled and not _old_path_capturable(spec, tokens):
                        graph[enabled] = "host sync"
                        continue
                    try:
                        with _grouped(enabled):
                            graph[enabled] = _graph_ms(layer, x, device)
                    except Exception as exc:
                        graph[enabled] = "failed"
                        graphs_ok = False
                        lines.append(f"   capture failed, graphs skipped from here: {exc!r}"[:300])
            before, after = wall[tokens, False], wall[tokens, True]
            lines.append(
                f"{tokens:>9} | {before:9.3f} | {after:9.3f} | {before / after:7.2f}x | "
                f"{rel:8.1e} | {_fmt(graph[False]):>12} | {_fmt(graph[True]):>11}"
            )
        if spec.get("profile"):
            lines.append("")
            lines.append(f"## Profile ({spec['name']} layer)")
            for tokens in PROFILE_TOKENS if not tiny else [4, 64]:
                x = torch.randn(1, tokens, spec["hidden"], device=device, dtype=dtype)
                for enabled in (False, True):
                    label = "after" if enabled else "before"
                    with _grouped(enabled):
                        ms = wall.get((tokens, enabled))
                        if ms is None:
                            with torch.inference_mode():
                                ms = _time_ms(partial(layer, x), device)
                        lines += _profile(
                            layer,
                            x,
                            device,
                            wall_ms=ms,
                            name=f"{spec['kind']}_layer_{tokens}tok_{label}",
                            traces=traces,
                        )
        if spec["kind"] == "glm_fp8" and cuda:
            lines += _fp8_sweeps(layer, spec["hidden"], device, dtype)
        del layer
        if cuda:
            torch.cuda.empty_cache()
    return lines


def _decode_ms(
    model: Any, device: str, dtype: Any, *, batch_size: int, cuda_graph: bool
) -> tuple[float, list[list[int]]]:
    """Milliseconds per greedy decode step at a full batch; also the tokens.

    Same procedure as the decode benchmark in `modal_nemotron_h_smoke.py`.
    """
    import torch

    from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph

    prompt_len, warmup, steps = 64, 4, 48
    cache = model.build_state_cache(
        max_seq_len=512, batch_size=batch_size, device=device, dtype=dtype
    )
    # Capture first: warmup and capture write into every row of the cache.
    graph = (
        DecodeCudaGraph(model, cache, batch_size=batch_size, device=device) if cuda_graph else None
    )
    torch.manual_seed(0)
    for row in range(batch_size):
        single = model.build_state_cache(max_seq_len=512, batch_size=1, device=device, dtype=dtype)
        ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
        with torch.inference_mode():
            model.forward_prefill_with_cache(ids, state_cache=single)
        cache.copy_row_from(single, src_row=0, dst_row=row)
    tokens = torch.zeros(batch_size, 1, dtype=torch.long)
    positions = torch.full((batch_size,), prompt_len, dtype=torch.long)
    generated: list[list[int]] = [[] for _ in range(batch_size)]

    def step() -> None:
        if graph is not None:
            logits = graph.replay(tokens, positions)
        else:
            with torch.inference_mode():
                logits = model.forward_decode_with_cache_ragged(
                    tokens.to(device), positions=positions.to(device), state_cache=cache
                )
        chosen = logits[:, -1].argmax(-1).tolist()
        for row, token in enumerate(chosen):
            generated[row].append(token)
            tokens[row, 0] = token
        positions.add_(1)

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    return 1000 * (time.perf_counter() - start) / steps, generated


def real_model_bench(model_dir: str, device: str, dtype: Any, passage: str) -> list[str]:
    """Step 3: the real Nemotron checkpoint, both paths."""
    import torch

    from mini_infer.engine.state_cache_generator import StateCacheGenerator

    lines = ["", "## Real model: Nemotron 3.5 Lightning 30B-A3B"]
    start = time.perf_counter()
    generator = StateCacheGenerator.from_pretrained(model_dir, device=device, dtype=dtype)
    model = generator.model
    ids = generator.tokenizer.encode(passage)
    lines.append(f"load {time.perf_counter() - start:.0f}s, passage {len(ids)} tokens")
    top_k = model.cfg.num_experts_per_tok

    def prefill(count: int) -> Any:
        cache = model.build_state_cache(
            max_seq_len=count + 8, batch_size=1, device=device, dtype=dtype
        )
        with torch.inference_mode():
            return model.forward_prefill_with_cache(
                torch.tensor([ids[:count]], device=device), state_cache=cache
            )[0]

    agree_len = min(512, len(ids))
    with _grouped(False):
        before = prefill(agree_len).float()
    with _grouped(True):
        after = prefill(agree_len).float()
    cos = float(
        torch.nn.functional.cosine_similarity(
            before.double().flatten(), after.double().flatten(), dim=0
        )
    )
    argmax = float((before.argmax(-1) == after.argmax(-1)).float().mean())
    lines.append(
        f"prefill {agree_len} tok, after vs before: cos={cos:.6f} "
        f"argmax_agree={argmax:.3f} max_abs={float((after - before).abs().max()):.3f}"
    )

    lines.append("prefill ms (real passage):")
    for count in PREFILL_TOKENS:
        if count > len(ids):
            lines.append(f"  {count} tok: skipped, passage has {len(ids)} tokens")
            continue
        ms = {}
        for enabled in (False, True):
            with _grouped(enabled):
                ms[enabled] = _time_ms(partial(prefill, count), device, budget_s=2.0, max_iters=10)
        lines.append(
            f"  {count:>5} tok: before {ms[False]:8.1f}  after {ms[True]:8.1f}  "
            f"({ms[False] / ms[True]:.2f}x)"
        )

    lines.append("decode ms/step (eager | CUDA graph):")
    for batch in DECODE_BATCHES:
        results: dict[tuple[bool, bool], float | str] = {}
        tokens: dict[bool, list[list[int]]] = {}
        for enabled in (False, True):
            with _grouped(enabled):
                ms, tokens[enabled] = _decode_ms(
                    model, device, dtype, batch_size=batch, cuda_graph=False
                )
                results[enabled, False] = ms
                capturable = enabled or batch * top_k <= _NEMOTRON_GATHER_LIMIT
                if not device.startswith("cuda"):
                    results[enabled, True] = "-"
                elif not capturable:
                    results[enabled, True] = "host sync"
                else:
                    try:
                        results[enabled, True], _ = _decode_ms(
                            model, device, dtype, batch_size=batch, cuda_graph=True
                        )
                    except Exception as exc:
                        results[enabled, True] = "failed"
                        lines.append(f"  capture failed: {exc!r}"[:300])
        same = sum(
            a == b
            for row_a, row_b in zip(tokens[False], tokens[True], strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
        total = sum(len(row) for row in tokens[False])
        lines.append(
            f"  batch {batch:>2}: before {_fmt(results[False, False])} | "
            f"{_fmt(results[False, True])}   after {_fmt(results[True, False])} | "
            f"{_fmt(results[True, True])}   greedy tokens equal {same}/{total}"
        )
    return lines


@app.function(
    image=image,
    gpu=_TEST_GPU,
    memory=32768,
    timeout=_TEST_TIMEOUT,
    volumes={"/hf-cache": hf_cache},
)
def moe_grouped_bench(
    fp8_only: bool = False,
    fp8_kernel: bool = False,
    fp8_tiles: bool = False,
    fp8_gluon: bool = False,
    fp8_expert_tiles: bool = False,
    tests_only: bool = False,
) -> dict[str, Any]:
    import subprocess
    import sys

    import torch
    import triton
    from huggingface_hub import snapshot_download

    assert torch.cuda.is_available(), "CUDA not available in Modal container"
    header = (
        f"gpu={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}"
    )
    _log(header)
    traces: dict[str, bytes] = {}
    lines = [header, ""]

    _log("step 0: unit tests on the GPU")
    try:
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", *_GPU_TESTS, "-q", "-rs", "-p", "no:cacheprovider"],
            cwd="/root",
            capture_output=True,
            text=True,
            timeout=_GPU_TESTS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        lines += [f"unit tests timed out after {_GPU_TESTS_TIMEOUT} s", *output.splitlines()[-60:]]
        return {"report": "\n".join(lines), "traces": traces}
    lines += ["## GPU unit tests", *tests.stdout.strip().splitlines()[-6:], ""]
    if tests.returncode != 0:
        lines.append("unit tests failed, benchmark skipped")
        lines += tests.stdout.splitlines()[-150:]
        return {"report": "\n".join(lines), "traces": traces}
    if tests_only:
        lines += tests.stdout.splitlines()[-40:]
        return {"report": "\n".join(lines), "traces": traces}

    if fp8_gluon:
        lines += fp8_gluon_bench("cuda", torch.bfloat16, _REMOTE_FIXTURES, traces=traces)
        return {"report": "\n".join(lines), "traces": traces}
    if fp8_expert_tiles:
        lines += fp8_expert_tiles_bench("cuda", torch.bfloat16, _REMOTE_FIXTURES, traces=traces)
        return {"report": "\n".join(lines), "traces": traces}
    if fp8_kernel or fp8_tiles:
        lines += fp8_kernel_bench(
            "cuda", torch.bfloat16, _REMOTE_FIXTURES, traces=traces, full=not fp8_tiles
        )
        return {"report": "\n".join(lines), "traces": traces}
    kinds = ("glm_fp8",) if fp8_only else None
    lines += layer_bench(
        "cuda", torch.bfloat16, _REMOTE_FIXTURES, tiny=False, traces=traces, kinds=kinds
    )
    if fp8_only:
        return {"report": "\n".join(lines), "traces": traces}
    try:
        model_dir = snapshot_download(
            MODEL_NAME, revision=REVISION, allow_patterns=ALLOW_PATTERNS, local_files_only=True
        )
    except Exception as exc:
        lines.append(f"real model skipped, weights not in the volume: {exc!r}")
    else:
        lines += real_model_bench(model_dir, "cuda", torch.bfloat16, _REMOTE_PASSAGE.read_text())
    return {"report": "\n".join(lines), "traces": traces}


def _save(result: dict[str, Any], out_dir: str) -> Path:
    target = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="moe-grouped-bench-"))
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.txt").write_text(result["report"] + "\n")
    for name, blob in result["traces"].items():
        (target / name).write_bytes(blob)
    return target


@app.local_entrypoint()
def main(
    out_dir: str = "",
    fp8_only: bool = False,
    fp8_kernel: bool = False,
    fp8_tiles: bool = False,
    fp8_gluon: bool = False,
    fp8_expert_tiles: bool = False,
    tests_only: bool = False,
) -> None:
    result = moe_grouped_bench.remote(
        fp8_only=fp8_only,
        fp8_kernel=fp8_kernel,
        fp8_tiles=fp8_tiles,
        fp8_gluon=fp8_gluon,
        fp8_expert_tiles=fp8_expert_tiles,
        tests_only=tests_only,
    )
    print(result["report"])
    print(f"\nreport and traces in {_save(result, out_dir)}")


if __name__ == "__main__":
    import sys

    import torch

    # Run the grouped path on the CPU (PyTorch's per-group fallback), so the
    # dry run covers both paths. Timings here mean nothing.
    from mini_infer.models.blocks import grouped_experts

    grouped_experts.is_hopper_or_newer_device = lambda device=None: True
    sys.path.insert(0, str(Path(__file__).parent))
    from modal_nemotron_h_smoke import _tiny_checkpoint

    dry_traces: dict[str, bytes] = {}
    dry_lines = layer_bench("cpu", torch.float32, _LOCAL_FIXTURES, tiny=True, traces=dry_traces)
    dry_lines += real_model_bench(
        _tiny_checkpoint(), "cpu", torch.float32, _LOCAL_PASSAGE.read_text()
    )
    dry_result = {"report": "\n".join(dry_lines), "traces": dry_traces}
    print(dry_result["report"])
    print(f"\nreport and traces in {_save(dry_result, sys.argv[1] if len(sys.argv) > 1 else '')}")
