"""Gluon block-FP8 grouped expert kernel (`blocks/fp8_grouped_gluon.py`) for Hopper.

Gluon does not run under Triton's interpreter, so two kinds of tests:

- Without a GPU: every default tile compiles for sm90, with no register
  spills and no write to a register that an in-flight `wgmma` still reads
  (`_wgmma_sass.py`). Needs Triton 3.8 or newer (the
  `mini-infer-triton-interp` Docker image has it), and skips otherwise.
- On an sm90 GPU: the kernel against fp32 math on the dequantized weights,
  against the Triton kernel, inside a CUDA graph, and under Triton's
  concurrency sanitizer, which fails on a race or a deadlock.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

pytest.importorskip("triton")

from mini_infer.device import is_hopper_device
from mini_infer.models.blocks import fp8_grouped_gluon, fp8_grouped_kernel
from mini_infer.models.blocks.fp8_grouped_gluon import Tiles
from mini_infer.models.blocks.fp8_grouped_kernel import block_fp8_grouped_mm
from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial
from tests.unit._wgmma_sass import wgmma_operand_hazards

needs_gluon = pytest.mark.skipif(
    not fp8_grouped_gluon._GLUON_AVAILABLE, reason="needs Triton 3.8 or newer with Gluon"
)

# The default tiles, each with and without warp specialization.
_ALL_TILES: list[Tiles] = [
    (*tiles[:5], ws) for _limit, tiles in fp8_grouped_gluon._TILES for ws in (True, False)
]


def _tile_id(tiles: Tiles) -> str:
    return "x".join(map(str, tiles[:5])) + ("-ws" if tiles[5] else "")


def _compile_sm90(tiles: Tiles, n_dim: int, k_dim: int) -> tuple[str, bytes]:
    """Compile the kernel for these tiles for sm90; return its name and cubin."""
    from triton import compile as triton_compile
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    block_w, block_t, block_k, warps, stages, warp_specialize = tiles
    rows = torch.empty(1024, k_dim, dtype=torch.bfloat16)
    weight = torch.empty(4 * n_dim, k_dim, dtype=torch.float8_e4m3fn)
    x_desc = TensorDescriptor.from_tensor(
        rows,
        [block_t, block_k],
        gl.NVMMASharedLayout.get_default_for([block_t, block_k], gl.bfloat16),
    )
    w_desc = TensorDescriptor.from_tensor(
        weight,
        [block_w, block_k],
        gl.NVMMASharedLayout.get_default_for([block_w, block_k], gl.float8e4nv),
    )
    if warp_specialize:
        kernel = fp8_grouped_gluon._block_fp8_grouped_mm_hopper_ws
    else:
        kernel = fp8_grouped_gluon._block_fp8_grouped_mm_hopper
    types = {
        "x_desc": x_desc.__mangle__(),
        "w_desc": w_desc.__mangle__(),
        "s_ptr": "*fp32",
        "group_ends_ptr": "*i32",
        "out_ptr": "*bf16",
    }
    constexprs = {
        "QBLOCK": 128,
        "NUM_STAGES": stages,
        "LOAD_REGS": fp8_grouped_gluon._LOAD_REGS,
        "num_warps": warps,
    }
    signature: dict[str, str] = {}
    constants: dict[tuple[int, ...], int] = {}
    for index, name in enumerate(inspect.signature(kernel.fn).parameters):
        if name in constexprs:
            signature[name] = "constexpr"
            constants[(index,)] = constexprs[name]
        else:
            signature[name] = types.get(name, "i32")
    options: dict[str, int] = {"num_warps": warps, "num_stages": 1}
    max_registers = fp8_grouped_gluon._max_registers(warps, warp_specialize)
    if max_registers is not None:
        options["maxnreg"] = max_registers
    compiled = triton_compile(
        GluonASTSource(fn=kernel, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", 90, 32),
        options=options,
    )
    return compiled.metadata.name, compiled.asm["cubin"]


@needs_gluon
@pytest.mark.parametrize("tiles", _ALL_TILES, ids=_tile_id)
@pytest.mark.parametrize(("n_dim", "k_dim"), [(4096, 6144), (6144, 2048)], ids=["gate_up", "down"])
def test_kernel_compiles_for_sm90_without_spills_or_wgmma_hazards(
    tiles: Tiles, n_dim: int, k_dim: int
) -> None:
    import triton

    name, cubin = _compile_sm90(tiles, n_dim, k_dim)
    cuobjdump = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/cuobjdump")
    if not os.path.exists(cuobjdump):
        pytest.skip("Triton ships no cuobjdump here to read the register use")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as file:
        file.write(cubin)
        file.flush()
        usage = subprocess.run(
            [cuobjdump, "--dump-resource-usage", file.name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        sass = subprocess.run(
            [cuobjdump, "-sass", file.name], capture_output=True, text=True, check=True
        ).stdout
    line = next(line for line in usage.splitlines() if "REG:" in line)
    assert "LOCAL:0 " in line, f"{name} spills registers: {line.strip()}"
    assert "HGMMA" in sass and "UTMALDG" in sass
    hazards = wgmma_operand_hazards(sass)
    assert not hazards, f"{name} overwrites registers of an in-flight wgmma: {hazards[:4]}"


def _hopper() -> None:
    if not torch.cuda.is_available() or not is_hopper_device("cuda"):
        pytest.skip("needs an sm90 GPU")
    if not fp8_grouped_gluon._GLUON_AVAILABLE:
        pytest.skip("needs Triton 3.8 or newer with Gluon")


def _fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * 0.5).to(torch.float8_e4m3fn)


def _problem(
    counts: list[int], off_rank: int, n_dim: int, k_dim: int, block: int = 128
) -> tuple[torch.Tensor, ...]:
    """Sorted rows with their experts, block-FP8 weights and group ends, on the CPU."""
    num_experts = len(counts)
    weight = _fp8((num_experts, n_dim, k_dim))
    scale = torch.rand(num_experts, -(-n_dim // block), -(-k_dim // block)) * 0.02 + 0.01
    row_experts = torch.cat(
        [
            torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts)),
            torch.full((off_rank,), num_experts),
        ]
    )
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32)
    rows = torch.randn(len(row_experts), k_dim).bfloat16()
    return rows, weight, scale, row_experts, group_ends


def _reference(
    rows: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    row_experts: torch.Tensor,
    block: int = 128,
) -> torch.Tensor:
    """fp32 math on the bf16-rounded dequantized weights; off-rank rows are zero."""
    num_experts = weight.shape[0]
    want = torch.zeros(len(row_experts), weight.shape[1])
    for expert in range(num_experts):
        picked = row_experts == expert
        if picked.any():
            deq = dequantize_block_fp8_to_bf16_partial(
                weight[expert], scale[expert], block_size=(block, block)
            )
            want[picked] = rows[picked].float() @ deq.float().T
    return want


def _gluon(monkeypatch: pytest.MonkeyPatch, tiles: Tiles | None) -> list[int]:
    """Force the Gluon kernel (with `tiles`); return a list that counts its calls."""
    calls: list[int] = []
    run = fp8_grouped_gluon.gluon_block_fp8_grouped_mm

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fp8_grouped_kernel, "_GLUON_FOR_BENCH", True)
    monkeypatch.setattr(fp8_grouped_gluon, "_TILES_FOR_BENCH", tiles)
    monkeypatch.setattr(fp8_grouped_gluon, "gluon_block_fp8_grouped_mm", counted)
    return calls


@pytest.mark.requires_cuda
@pytest.mark.parametrize("tiles", _ALL_TILES, ids=_tile_id)
@pytest.mark.parametrize(
    ("counts", "off_rank", "n_dim", "k_dim"),
    [
        # Empty and one-row experts, several row tiles, partial column and K
        # tiles (K = 352 is 5.5 steps of 64; the kernel runs an even count).
        ([3, 0, 300, 1, 70], 0, 320, 352),
        # Rows routed to other ranks sort last and give zeros.
        ([3, 0, 300, 1, 70], 9, 256, 384),
        # One expert holds every row.
        ([0, 700, 0], 0, 384, 256),
    ],
    ids=["mixed", "off-rank", "one-expert"],
)
def test_kernel_matches_dequantized_matmul(
    monkeypatch: pytest.MonkeyPatch,
    tiles: Tiles,
    counts: list[int],
    off_rank: int,
    n_dim: int,
    k_dim: int,
) -> None:
    _hopper()
    calls = _gluon(monkeypatch, tiles)
    torch.manual_seed(0)
    rows, weight, scale, row_experts, group_ends = _problem(counts, off_rank, n_dim, k_dim)
    got = block_fp8_grouped_mm(
        rows.cuda(),
        weight.cuda(),
        scale.cuda(),
        block=128,
        row_experts=row_experts.cuda(),
        group_ends=group_ends.cuda(),
    ).cpu()
    assert calls, "the Gluon kernel did not run"
    assert got.dtype == torch.bfloat16
    want = _reference(rows, weight, scale, row_experts)
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    ("k_dim", "block", "dtype", "takes"),
    [
        (6144, 128, torch.bfloat16, True),
        (352, 128, torch.bfloat16, True),  # 5.5 steps of 64 run as 6
        (320, 128, torch.bfloat16, False),  # 5 steps: the loop runs two at a time
        (6144, 64, torch.bfloat16, False),  # a 128-row weight tile spans two scales
        (6144, 128, torch.float16, False),
        (6152, 128, torch.bfloat16, False),  # TMA needs 16-byte rows
    ],
)
def test_supports_only_operands_the_kernel_handles(
    monkeypatch: pytest.MonkeyPatch, k_dim: int, block: int, dtype: torch.dtype, takes: bool
) -> None:
    monkeypatch.setattr(fp8_grouped_gluon, "_GLUON_AVAILABLE", True)
    monkeypatch.setattr(fp8_grouped_gluon, "is_hopper_device", lambda device: True)
    rows = torch.zeros(8, k_dim, dtype=dtype)
    weight = torch.zeros(2, 256, k_dim, dtype=torch.float8_e4m3fn)
    tiles: Tiles = (128, 128, 64, 8, 4, True)
    assert (
        fp8_grouped_gluon.supports_gluon_block_fp8_grouped_mm(rows, weight, block, tiles) is takes
    )


@pytest.mark.requires_cuda
def test_default_path_uses_gluon_for_many_rows_and_matches_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above 256 rows the default path is the Gluon kernel; it agrees with Triton."""
    _hopper()
    torch.manual_seed(0)
    num_experts, n_dim, k_dim = 16, 512, 768
    row_experts = torch.randint(0, num_experts, (3000,)).sort().values
    counts = torch.bincount(row_experts, minlength=num_experts)
    group_ends = counts.cumsum(0).to(torch.int32).cuda()
    weight = _fp8((num_experts, n_dim, k_dim)).cuda()
    scale = (torch.rand(num_experts, n_dim // 128, k_dim // 128) * 0.02 + 0.01).cuda()
    rows = torch.randn(len(row_experts), k_dim).bfloat16().cuda()
    args = (rows, weight, scale)
    kwargs = {"block": 128, "row_experts": row_experts.cuda(), "group_ends": group_ends}

    calls: list[int] = []
    run = fp8_grouped_gluon.gluon_block_fp8_grouped_mm
    monkeypatch.setattr(
        fp8_grouped_gluon,
        "gluon_block_fp8_grouped_mm",
        lambda *a, **k: calls.append(1) or run(*a, **k),
    )
    gluon = block_fp8_grouped_mm(*args, **kwargs)
    assert calls, "the default path did not take the Gluon kernel"
    monkeypatch.setattr(fp8_grouped_kernel, "_GLUON_FOR_BENCH", False)
    triton_out = block_fp8_grouped_mm(*args, **kwargs)
    assert len(calls) == 1
    rel = (gluon.float() - triton_out.float()).norm() / triton_out.float().norm()
    assert rel < 2e-3, f"Gluon and Triton differ by a relative {rel:.2e}"


@pytest.mark.requires_cuda
@pytest.mark.parametrize("warp_specialize", [True, False], ids=["ws", "no-ws"])
def test_kernel_runs_in_a_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, warp_specialize: bool
) -> None:
    """Captured once, the kernel follows new routing written into the same buffers."""
    _hopper()
    tiles: Tiles = (*fp8_grouped_gluon._TILES[1][1][:5], warp_specialize)
    _gluon(monkeypatch, tiles)
    torch.manual_seed(0)
    num_experts, n_dim, k_dim, num_rows = 8, 256, 256, 900
    weight = _fp8((num_experts, n_dim, k_dim)).cuda()
    scale = (torch.rand(num_experts, 2, 2) * 0.02 + 0.01).cuda()

    def routing() -> tuple[torch.Tensor, torch.Tensor]:
        experts = torch.randint(0, num_experts, (num_rows,)).sort().values
        ends = torch.bincount(experts, minlength=num_experts).cumsum(0).to(torch.int32)
        return experts, ends

    experts, ends = routing()
    rows = torch.randn(num_rows, k_dim).bfloat16().cuda()
    row_experts, group_ends = experts.cuda(), ends.cuda()

    def forward() -> torch.Tensor:
        return block_fp8_grouped_mm(
            rows, weight, scale, block=128, row_experts=row_experts, group_ends=group_ends
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()

    experts, ends = routing()
    row_experts.copy_(experts)
    group_ends.copy_(ends)
    rows.copy_(torch.randn(num_rows, k_dim).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, forward(), rtol=0, atol=0)
    want = _reference(rows.cpu(), weight.cpu(), scale.cpu(), experts)
    torch.testing.assert_close(captured.cpu().float(), want, rtol=1e-2, atol=1e-3)


_SANITIZER_SCRIPT = textwrap.dedent(
    """
    import torch
    from mini_infer.models.blocks import fp8_grouped_gluon

    torch.manual_seed(0)
    tiles = {tiles}
    counts = [3, 0, 150, 1, 70]
    num_experts, n_dim, k_dim = len(counts), 256, 384
    experts = torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts))
    rows = torch.randn(len(experts) + 5, k_dim).bfloat16().cuda()
    weight = (torch.randn(num_experts, n_dim, k_dim) * 0.5).to(torch.float8_e4m3fn).cuda()
    scale = torch.full((num_experts, 2, 3), 0.02).cuda()
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32).cuda()
    fp8_grouped_gluon.gluon_block_fp8_grouped_mm(
        rows, weight, scale, block=128, group_ends=group_ends, tiles=tiles
    )
    torch.cuda.synchronize()
    print("sanitizer run done")
    """
)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("tiles", _ALL_TILES, ids=_tile_id)
def test_kernel_passes_the_concurrency_sanitizer(tiles: Tiles) -> None:
    """Triton's sanitizer stops the kernel on a race on shared memory or a deadlock."""
    _hopper()
    env = dict(os.environ, TRITON_INSTRUMENTATION_MODE="consan", CUDA_LAUNCH_BLOCKING="1")
    result = subprocess.run(
        [sys.executable, "-c", _SANITIZER_SCRIPT.format(tiles=tiles)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0 and "sanitizer run done" in output, output[-3000:]
