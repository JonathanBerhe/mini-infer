"""Gluon packed-FP4 grouped expert kernels (`blocks/fp4_grouped_gluon.py`).

One kernel for sm90 (Hopper, `wgmma`) and one for sm100 (B200, `tcgen05`).
Gluon does not run under Triton's interpreter, so the tests are of two kinds:

- Without a GPU, each default tile compiles for its GPU family with the
  launcher's arguments and has no register spills. On sm90, the test makes
  sure that no instruction writes a register that a running `wgmma` still
  reads (`_wgmma_sass.py`). These tests need Triton 3.8, which the
  `mini-infer-triton-interp` Docker image has. With another Triton they
  skip.
- On a GPU of the case's family, the tests compare the kernel with fp32
  math on the dequantized weights and with the Triton kernel. They also
  capture it in a CUDA graph and run it under Triton's concurrency
  sanitizer, which fails on a race or a deadlock.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

pytest.importorskip("triton")

from mini_infer.device import is_datacenter_blackwell_device, is_hopper_device
from mini_infer.models.blocks import fp4_grouped_gluon, fp4_grouped_kernel
from mini_infer.models.blocks.fp4_grouped_kernel import block_fp4_grouped_mm
from mini_infer.models.blocks.fp8_grouped_gluon import GluonTiles
from mini_infer.quant.nvfp4 import dequantize_nvfp4_to_bf16
from tests.unit._wgmma_sass import wgmma_operand_hazards

needs_gluon = pytest.mark.skipif(
    not fp4_grouped_gluon._GLUON_AVAILABLE, reason="needs Triton 3.8 with Gluon"
)

Case = tuple[int, GluonTiles]

# (compute capability, tiles): the default tiles of each GPU family.
_CASES: list[Case] = [(90, tiles) for _limit, tiles in fp4_grouped_gluon._HOPPER_TILES] + [
    (100, tiles) for _limit, tiles in fp4_grouped_gluon._BLACKWELL_TILES
]


def _case_id(case: Case) -> str:
    arch, tiles = case
    return f"sm{arch}-" + "x".join(map(str, tiles[:5]))


def _compile(case: Case) -> tuple[str, bytes]:
    """Compile the kernel of this case for a DeepSeek-V4-Flash gate-up projection.

    The arguments go through the launcher's code and the JIT's own binding,
    so the compile sees the launcher's specialization.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.runtime.jit import create_function_from_signature

    arch, tiles = case
    num_rows, num_experts, n_dim, k_dim = 2048, 32, 4096, 4096
    rows = torch.empty(num_rows, k_dim, dtype=torch.bfloat16)
    packed = torch.empty(num_experts, n_dim, k_dim // 2, dtype=torch.int8)
    scale = torch.empty(num_experts, n_dim, k_dim // 32)
    group_ends = torch.empty(num_experts, dtype=torch.int32)
    out = torch.empty(num_rows, n_dim, dtype=torch.bfloat16)
    kernel, args, options = fp4_grouped_gluon._kernel_and_arguments(
        rows, packed, scale, out, group_ends=group_ends, tiles=tiles, blackwell=arch == 100
    )
    options = {**options, "debug": False, "instrumentation_mode": ""}
    target = GPUTarget("cuda", arch, 32)
    backend = make_backend(target)
    bind = create_function_from_signature(kernel.signature, kernel.params, backend)
    bound_args, specialization, compile_options = bind(*args, **options)
    compile_options, signature, constexprs, attrs = kernel._pack_args(
        backend, options, bound_args, specialization, compile_options
    )
    compiled = triton.compile(
        GluonASTSource(kernel, signature, constexprs, attrs),
        target=target,
        options=compile_options.__dict__,
    )
    return compiled.metadata.name, compiled.asm["cubin"]


@needs_gluon
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_kernel_compiles_without_spills_or_wgmma_hazards(case: Case) -> None:
    import triton

    name, cubin = _compile(case)
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
    assert "UTMALDG" in sass
    if case[0] == 100:
        # The tcgen05 multiply and the weight tile's store into tensor memory.
        assert "UTCHMMA" in sass and "STTM" in sass
    else:
        assert "HGMMA" in sass
        hazards = wgmma_operand_hazards(sass)
        assert not hazards, f"{name} overwrites registers of a running wgmma: {hazards[:4]}"


def _on_gpu(arch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if arch == 90 and not is_hopper_device("cuda"):
        pytest.skip("needs an sm90 GPU")
    if arch == 100 and not is_datacenter_blackwell_device("cuda"):
        pytest.skip("needs an sm100 GPU")
    if not fp4_grouped_gluon._GLUON_AVAILABLE:
        pytest.skip("needs Triton 3.8 with Gluon")


def _this_gpu() -> int:
    """The case family of the current GPU. Skips on other devices."""
    blackwell = torch.cuda.is_available() and is_datacenter_blackwell_device("cuda")
    arch = 100 if blackwell else 90
    _on_gpu(arch)
    return arch


def _fp4(
    num_experts: int, n_dim: int, k_dim: int, *, power_of_two: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random packed e2m1 weights and their scales, one per row and 32 columns."""
    packed = torch.randint(-128, 128, (num_experts, n_dim, k_dim // 2), dtype=torch.int8)
    scale = torch.rand(num_experts, n_dim, k_dim // 32) * 0.02 + 0.01
    if power_of_two:
        scale = torch.exp2(torch.round(torch.log2(scale)))
    return packed, scale


def _problem(
    counts: list[int], off_rank: int, n_dim: int, k_dim: int, *, power_of_two: bool = False
) -> tuple[torch.Tensor, ...]:
    """Sorted rows with their experts, FP4 weights and group ends, on the CPU."""
    num_experts = len(counts)
    packed, scale = _fp4(num_experts, n_dim, k_dim, power_of_two=power_of_two)
    row_experts = torch.cat(
        [
            torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts)),
            torch.full((off_rank,), num_experts),
        ]
    )
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32)
    rows = torch.randn(len(row_experts), k_dim).bfloat16()
    return rows, packed, scale, row_experts, group_ends


def _reference(
    rows: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, row_experts: torch.Tensor
) -> torch.Tensor:
    """fp32 math on the bf16-rounded dequantized weights. Off-rank rows are zero."""
    want = torch.zeros(len(row_experts), packed.shape[1])
    for expert in range(packed.shape[0]):
        picked = row_experts == expert
        if picked.any():
            deq = dequantize_nvfp4_to_bf16(packed[expert], scale[expert], block_size=32)
            want[picked] = rows[picked].float() @ deq.float().T
    return want


def _atol(want: torch.Tensor) -> float:
    """1 percent of the largest output: bf16 rounding grows with the output."""
    return 1e-2 * float(want.abs().max())


def _count_gluon_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """A list that gets one entry for each call of the Gluon launcher."""
    calls: list[int] = []
    run = fp4_grouped_gluon.gluon_block_fp4_grouped_mm

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fp4_grouped_gluon, "gluon_block_fp4_grouped_mm", counted)
    return calls


def _force_gluon(monkeypatch: pytest.MonkeyPatch, tiles: GluonTiles | None) -> list[int]:
    """Force the Gluon kernel with `tiles`, and count its calls."""
    monkeypatch.setattr(fp4_grouped_kernel, "_GLUON_FOR_BENCH", True)
    monkeypatch.setattr(fp4_grouped_gluon, "_TILES_FOR_BENCH", tiles)
    return _count_gluon_calls(monkeypatch)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@pytest.mark.parametrize(
    ("counts", "off_rank", "n_dim", "k_dim", "power_of_two"),
    [
        # Empty and one-row experts, several row tiles, partial column and K
        # tiles (K = 352 is 5.5 steps of 64 and 2.75 of 128).
        ([3, 0, 300, 1, 70], 0, 320, 352, False),
        # Rows routed to other ranks sort last and give zeros. Power-of-two
        # scales, as in the checkpoint.
        ([3, 0, 300, 1, 70], 9, 256, 512, True),
        # One expert holds every row.
        ([0, 700, 0], 0, 384, 256, False),
    ],
    ids=["mixed", "off-rank", "one-expert"],
)
# With two programs, each program walks many items.
@pytest.mark.parametrize("programs", [None, 2], ids=["all-sms", "two-programs"])
def test_kernel_matches_dequantized_matmul(
    monkeypatch: pytest.MonkeyPatch,
    case: Case,
    counts: list[int],
    off_rank: int,
    n_dim: int,
    k_dim: int,
    power_of_two: bool,
    programs: int | None,
) -> None:
    _on_gpu(case[0])
    calls = _force_gluon(monkeypatch, case[1])
    monkeypatch.setattr(fp4_grouped_gluon, "_PROGRAMS_FOR_TESTS", programs)
    torch.manual_seed(0)
    rows, packed, scale, row_experts, group_ends = _problem(
        counts, off_rank, n_dim, k_dim, power_of_two=power_of_two
    )
    got = block_fp4_grouped_mm(
        rows.cuda(),
        packed.cuda(),
        scale.cuda(),
        block=32,
        row_experts=row_experts.cuda(),
        group_ends=group_ends.cuda(),
    ).cpu()
    assert calls, "the Gluon kernel did not run"
    assert got.dtype == torch.bfloat16
    want = _reference(rows, packed, scale, row_experts)
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=_atol(want))


def _pretend_gpu(monkeypatch: pytest.MonkeyPatch, *, blackwell: bool) -> None:
    """Make `gluon_tiles` see Gluon and an sm100 (or sm90) GPU."""
    monkeypatch.setattr(fp4_grouped_gluon, "_GLUON_AVAILABLE", True)
    monkeypatch.setattr(fp4_grouped_gluon, "is_hopper_device", lambda device: not blackwell)
    monkeypatch.setattr(
        fp4_grouped_gluon, "is_datacenter_blackwell_device", lambda device: blackwell
    )


def _misaligned(rows: int, k_dim: int) -> torch.Tensor:
    """bf16 rows that start 2 bytes after a 16-byte boundary."""
    return torch.zeros(rows * k_dim + 1, dtype=torch.bfloat16)[1:].view(rows, k_dim)


@pytest.mark.parametrize(
    ("hopper", "k_dim", "block", "rows", "ends", "takes"),
    [
        (True, 4096, 32, None, torch.int32, True),
        (True, 384, 32, None, torch.int32, True),  # 6 steps of 64
        (True, 320, 32, None, torch.int32, False),  # 5 steps: the loop runs two at a time
        (False, 320, 32, None, torch.int32, True),  # the sm100 loop takes one step at a time
        (True, 4096, 16, None, torch.int32, False),  # the decode reads one scale per 32
        (True, 4112, 32, None, torch.int32, False),  # K must hold whole scale groups
        (True, 4096, 32, torch.float16, torch.int32, False),
        (True, 4096, 32, "misaligned", torch.int32, False),  # TMA needs 16-byte aligned data
        (True, 4096, 32, None, torch.int64, False),  # the kernels read int32 group ends
        (False, 4096, 32, "8 warps", torch.int32, False),  # the sm100 decode is one warpgroup
        (True, 4096, 32, "8 warps", torch.int32, True),
    ],
)
def test_gluon_takes_only_operands_the_kernel_handles(
    monkeypatch: pytest.MonkeyPatch,
    hopper: bool,
    k_dim: int,
    block: int,
    rows: torch.dtype | str | None,
    ends: torch.dtype,
    takes: bool,
) -> None:
    _pretend_gpu(monkeypatch, blackwell=not hopper)
    tiles: GluonTiles = (128, 128, 64, 8 if rows == "8 warps" else 4, 4, True)
    monkeypatch.setattr(fp4_grouped_gluon, "_TILES_FOR_BENCH", tiles)
    if rows == "misaligned":
        operand = _misaligned(8, k_dim)
    else:
        dtype = rows if isinstance(rows, torch.dtype) else torch.bfloat16
        operand = torch.zeros(8, k_dim, dtype=dtype)
    packed = torch.zeros(2, 256, k_dim // 2, dtype=torch.int8)
    scale = torch.ones(2, 256, k_dim // block)
    group_ends = torch.tensor([3, 8], dtype=ends)
    got = fp4_grouped_gluon.gluon_tiles(
        operand, packed, scale, block, group_ends, num_experts=2, by_default=False
    )
    assert (got is not None) is takes


def test_gluon_needs_contiguous_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_gpu(monkeypatch, blackwell=True)
    rows = torch.zeros(8, 256, dtype=torch.bfloat16)
    packed = torch.zeros(2, 128, 128, dtype=torch.int8)
    scale = torch.ones(2, 8, 128).transpose(1, 2)
    group_ends = torch.tensor([3, 8], dtype=torch.int32)
    got = fp4_grouped_gluon.gluon_tiles(
        rows, packed, scale, 32, group_ends, num_experts=2, by_default=False
    )
    assert got is None


@pytest.mark.parametrize(
    ("blackwell", "num_rows", "local_experts", "tiles"),
    [
        # sm90 runs the Gluon kernel at every size, with a tile for the mean
        # rows per expert.
        (False, 512, 256, fp4_grouped_gluon._HOPPER_TILES[0][1]),
        (False, 32768, 256, fp4_grouped_gluon._HOPPER_TILES[1][1]),
        (False, 98304, 256, fp4_grouped_gluon._HOPPER_TILES[2][1]),
        # Expert parallelism over 8 ranks: 32 local experts, but the rows
        # are the rows of all 256 experts.
        (False, 8192, 32, fp4_grouped_gluon._HOPPER_TILES[0][1]),
        # sm100 runs it above 64 mean rows per expert.
        (True, 16384, 256, None),
        (True, 98304, 256, fp4_grouped_gluon._BLACKWELL_TILES[1][1]),
        (True, 16384, 32, None),
    ],
)
def test_default_tiles_follow_the_mean_rows_of_all_experts(
    monkeypatch: pytest.MonkeyPatch,
    blackwell: bool,
    num_rows: int,
    local_experts: int,
    tiles: GluonTiles | None,
) -> None:
    _pretend_gpu(monkeypatch, blackwell=blackwell)
    rows = torch.zeros(num_rows, 256, dtype=torch.bfloat16)
    packed = torch.zeros(local_experts, 128, 128, dtype=torch.int8)
    scale = torch.ones(local_experts, 128, 8)
    group_ends = torch.zeros(local_experts, dtype=torch.int32)
    got = fp4_grouped_gluon.gluon_tiles(
        rows, packed, scale, 32, group_ends, num_experts=256, by_default=True
    )
    assert got == tiles


def test_forced_gluon_fails_when_no_kernel_takes_the_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced bench run must not time the Triton kernel under the Gluon name."""
    monkeypatch.setattr(fp4_grouped_kernel, "_GLUON_FOR_BENCH", True)
    monkeypatch.setattr(fp4_grouped_gluon, "_GLUON_AVAILABLE", False)
    with pytest.raises(ValueError, match="no Gluon kernel"):
        block_fp4_grouped_mm(
            torch.zeros(300, 256, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.int8),
            torch.ones(2, 128, 8),
            block=32,
            row_experts=torch.zeros(300, dtype=torch.long),
            group_ends=torch.tensor([150, 300], dtype=torch.int32),
        )


@pytest.mark.requires_cuda
def test_default_path_uses_gluon_for_many_rows_and_matches_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path takes the Gluon kernel here and agrees with Triton.

    3000 rows over 16 experts are over 256 rows and over 64 mean rows per
    expert, so sm90 and sm100 both take it.
    """
    _this_gpu()
    torch.manual_seed(0)
    num_experts, n_dim, k_dim = 16, 512, 768
    row_experts = torch.randint(0, num_experts, (3000,)).sort().values
    counts = torch.bincount(row_experts, minlength=num_experts)
    group_ends = counts.cumsum(0).to(torch.int32).cuda()
    packed, scale = _fp4(num_experts, n_dim, k_dim)
    rows = torch.randn(len(row_experts), k_dim).bfloat16().cuda()
    args = (rows, packed.cuda(), scale.cuda())
    kwargs = {"block": 32, "row_experts": row_experts.cuda(), "group_ends": group_ends}

    calls = _count_gluon_calls(monkeypatch)
    gluon = block_fp4_grouped_mm(*args, **kwargs)
    assert calls, "the default path did not take the Gluon kernel"
    monkeypatch.setattr(fp4_grouped_kernel, "_GLUON_FOR_BENCH", False)
    triton_out = block_fp4_grouped_mm(*args, **kwargs)
    assert len(calls) == 1
    rel = (gluon.float() - triton_out.float()).norm() / triton_out.float().norm()
    assert rel < 2e-3, f"Gluon and Triton differ by a relative {rel:.2e}"


_GRAPH_CASES: list[Case] = [
    (90, fp4_grouped_gluon._HOPPER_TILES[1][1]),
    (100, fp4_grouped_gluon._BLACKWELL_TILES[1][1]),
]


@pytest.mark.requires_cuda
@pytest.mark.parametrize("case", _GRAPH_CASES, ids=_case_id)
def test_kernel_runs_in_a_cuda_graph(monkeypatch: pytest.MonkeyPatch, case: Case) -> None:
    """Captured once, the kernel follows new routing written into the same buffers."""
    _on_gpu(case[0])
    calls = _force_gluon(monkeypatch, case[1])
    torch.manual_seed(0)
    num_experts, n_dim, k_dim, num_rows = 8, 256, 256, 900
    packed, scale = _fp4(num_experts, n_dim, k_dim)
    packed, scale = packed.cuda(), scale.cuda()

    def routing() -> tuple[torch.Tensor, torch.Tensor]:
        experts = torch.randint(0, num_experts, (num_rows,)).sort().values
        ends = torch.bincount(experts, minlength=num_experts).cumsum(0).to(torch.int32)
        return experts, ends

    experts, ends = routing()
    rows = torch.randn(num_rows, k_dim).bfloat16().cuda()
    row_experts, group_ends = experts.cuda(), ends.cuda()

    def forward() -> torch.Tensor:
        return block_fp4_grouped_mm(
            rows, packed, scale, block=32, row_experts=row_experts, group_ends=group_ends
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    assert calls, "the Gluon kernel did not run"

    experts, ends = routing()
    row_experts.copy_(experts)
    group_ends.copy_(ends)
    rows.copy_(torch.randn(num_rows, k_dim).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, forward(), rtol=0, atol=0)
    want = _reference(rows.cpu(), packed.cpu(), scale.cpu(), experts)
    torch.testing.assert_close(captured.cpu().float(), want, rtol=1e-2, atol=_atol(want))


# Two programs walk many items each: the stages, the weight buffers and the
# accumulators wrap around inside an item and from one item to the next.
_SANITIZER_SCRIPT = textwrap.dedent(
    """
    import torch
    from mini_infer.models.blocks import fp4_grouped_gluon

    torch.manual_seed(0)
    fp4_grouped_gluon._PROGRAMS_FOR_TESTS = 2
    tiles = {tiles}
    counts = [3, 0, 150, 1, 70, 40, 2]
    num_experts, n_dim, k_dim = len(counts), 512, 384
    experts = torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts))
    rows = torch.randn(len(experts) + 5, k_dim).bfloat16().cuda()
    packed = torch.randint(-128, 128, (num_experts, n_dim, k_dim // 2), dtype=torch.int8).cuda()
    scale = torch.full((num_experts, n_dim, k_dim // 32), 0.02).cuda()
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32).cuda()
    fp4_grouped_gluon.gluon_block_fp4_grouped_mm(
        rows, packed, scale, group_ends=group_ends, tiles=tiles
    )
    torch.cuda.synchronize()
    print("sanitizer run done")
    """
)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_kernel_passes_the_concurrency_sanitizer(case: Case) -> None:
    """Triton's sanitizer stops the kernel on a race on shared memory or a deadlock."""
    _on_gpu(case[0])
    env = dict(os.environ, TRITON_INSTRUMENTATION_MODE="consan", CUDA_LAUNCH_BLOCKING="1")
    result = subprocess.run(
        [sys.executable, "-c", _SANITIZER_SCRIPT.format(tiles=case[1])],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0 and "sanitizer run done" in output, output[-3000:]
