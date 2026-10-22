"""Gluon block-FP8 grouped expert kernels (`blocks/fp8_grouped_gluon.py`).

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
  sanitizer, which fails on a race or a deadlock. With two programs, each
  program walks many items, so the barrier rings wrap around.
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
from mini_infer.models.blocks import fp8_grouped_gluon, fp8_grouped_kernel
from mini_infer.models.blocks.fp8_grouped_gluon import GluonTiles
from mini_infer.models.blocks.fp8_grouped_kernel import block_fp8_grouped_mm
from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial
from tests.unit._wgmma_sass import wgmma_operand_hazards

needs_gluon = pytest.mark.skipif(
    not fp8_grouped_gluon._GLUON_AVAILABLE, reason="needs Triton 3.8 with Gluon"
)

Case = tuple[int, GluonTiles]

# (compute capability, tiles): the default tiles of each GPU family, the
# sm90 ones with and without warp specialization.
_CASES: list[Case] = [
    (90, (*tiles[:5], ws))
    for _limit, tiles in fp8_grouped_gluon._HOPPER_TILES
    for ws in (True, False)
] + [(100, tiles) for _limit, tiles in fp8_grouped_gluon._BLACKWELL_TILES]


def _case_id(case: Case) -> str:
    arch, tiles = case
    name = f"sm{arch}-" + "x".join(map(str, tiles[:5]))
    return name + ("-ws" if arch == 90 and tiles[5] else "")


def _compile(case: Case) -> tuple[str, bytes]:
    """Compile the kernel of this case for a GLM-5 gate-up projection.

    The arguments go through the launcher's code and the JIT's own binding,
    so the compile sees the launcher's specialization (constexpr strides,
    values divisible by 16).
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.runtime.jit import create_function_from_signature

    arch, tiles = case
    num_rows, num_experts, n_dim, k_dim = 2048, 32, 4096, 6144
    rows = torch.empty(num_rows, k_dim, dtype=torch.bfloat16)
    weight = torch.empty(num_experts, n_dim, k_dim, dtype=torch.float8_e4m3fn)
    scale = torch.empty(num_experts, n_dim // 128, k_dim // 128)
    group_ends = torch.empty(num_experts, dtype=torch.int32)
    out = torch.empty(num_rows, n_dim, dtype=torch.bfloat16)
    kernel, args, options = fp8_grouped_gluon._kernel_and_arguments(
        rows,
        weight,
        scale,
        out,
        block=128,
        group_ends=group_ends,
        tiles=tiles,
        blackwell=arch == 100,
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
    if not fp8_grouped_gluon._GLUON_AVAILABLE:
        pytest.skip("needs Triton 3.8 with Gluon")


def _this_gpu() -> int:
    """The case family of the current GPU. Skips on other devices."""
    blackwell = torch.cuda.is_available() and is_datacenter_blackwell_device("cuda")
    arch = 100 if blackwell else 90
    _on_gpu(arch)
    return arch


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
    """fp32 math on the bf16-rounded dequantized weights. Off-rank rows are zero."""
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


def _count_gluon_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """A list that gets one entry for each call of the Gluon launcher."""
    calls: list[int] = []
    run = fp8_grouped_gluon.gluon_block_fp8_grouped_mm

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fp8_grouped_gluon, "gluon_block_fp8_grouped_mm", counted)
    return calls


def _force_gluon(monkeypatch: pytest.MonkeyPatch, tiles: GluonTiles | None) -> list[int]:
    """Force the Gluon kernel with `tiles`, and count its calls."""
    monkeypatch.setattr(fp8_grouped_kernel, "_GLUON_FOR_BENCH", True)
    monkeypatch.setattr(fp8_grouped_gluon, "_TILES_FOR_BENCH", tiles)
    return _count_gluon_calls(monkeypatch)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@pytest.mark.parametrize(
    ("counts", "off_rank", "n_dim", "k_dim"),
    [
        # Empty and one-row experts, several row tiles, partial column and K
        # tiles (K = 352 is 5.5 steps of 64. The sm90 kernel runs an even count).
        ([3, 0, 300, 1, 70], 0, 320, 352),
        # Rows routed to other ranks sort last and give zeros.
        ([3, 0, 300, 1, 70], 9, 256, 384),
        # One expert holds every row.
        ([0, 700, 0], 0, 384, 256),
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
    programs: int | None,
) -> None:
    _on_gpu(case[0])
    calls = _force_gluon(monkeypatch, case[1])
    monkeypatch.setattr(fp8_grouped_gluon, "_PROGRAMS_FOR_TESTS", programs)
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


def _pretend_gpu(monkeypatch: pytest.MonkeyPatch, *, blackwell: bool) -> None:
    """Make `gluon_tiles` see Gluon and an sm100 (or sm90) GPU."""
    monkeypatch.setattr(fp8_grouped_gluon, "_GLUON_AVAILABLE", True)
    monkeypatch.setattr(fp8_grouped_gluon, "is_hopper_device", lambda device: not blackwell)
    monkeypatch.setattr(
        fp8_grouped_gluon, "is_datacenter_blackwell_device", lambda device: blackwell
    )


def _misaligned(rows: int, k_dim: int) -> torch.Tensor:
    """bf16 rows that start 2 bytes after a 16-byte boundary."""
    return torch.zeros(rows * k_dim + 1, dtype=torch.bfloat16)[1:].view(rows, k_dim)


@pytest.mark.parametrize(
    ("hopper", "k_dim", "block", "rows", "ends", "takes"),
    [
        (True, 6144, 128, None, torch.int32, True),
        (True, 352, 128, None, torch.int32, True),  # 5.5 steps of 64 run as 6
        (True, 320, 128, None, torch.int32, False),  # 5 steps: the loop runs two at a time
        (False, 320, 128, None, torch.int32, True),  # the sm100 loop takes one step at a time
        (True, 6144, 64, None, torch.int32, False),  # a 128-row weight tile spans two scales
        (True, 6144, 128, torch.float16, torch.int32, False),
        (True, 6136, 128, None, torch.int32, False),  # TMA needs 16-byte rows (96 even steps)
        (True, 6144, 128, "misaligned", torch.int32, False),  # TMA needs 16-byte aligned data
        (True, 6144, 128, None, torch.int64, False),  # the kernels read int32 group ends
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
    tiles: GluonTiles = (128, 128, 64, 8, 4, True)
    monkeypatch.setattr(fp8_grouped_gluon, "_TILES_FOR_BENCH", tiles)
    if rows == "misaligned":
        operand = _misaligned(8, k_dim)
    else:
        operand = torch.zeros(8, k_dim, dtype=rows or torch.bfloat16)
    weight = torch.zeros(2, 256, k_dim, dtype=torch.float8_e4m3fn)
    group_ends = torch.tensor([3, 8], dtype=ends)
    got = fp8_grouped_gluon.gluon_tiles(
        operand, weight, block, group_ends, num_experts=2, by_default=False
    )
    assert (got is not None) is takes


@pytest.mark.parametrize(
    ("blackwell", "num_rows", "local_experts", "tiles"),
    [
        # sm90 runs the Gluon kernel at every size, with a tile for the mean
        # rows per expert.
        (False, 512, 256, fp8_grouped_gluon._HOPPER_TILES[0][1]),
        (False, 32768, 256, fp8_grouped_gluon._HOPPER_TILES[1][1]),
        # Expert parallelism over 8 ranks: 32 local experts, but the rows
        # are the rows of all 256 experts.
        (False, 8192, 32, fp8_grouped_gluon._HOPPER_TILES[0][1]),
        # sm100 runs it above 64 mean rows per expert.
        (True, 16384, 256, None),
        (True, 32768, 256, fp8_grouped_gluon._BLACKWELL_TILES[1][1]),
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
    weight = torch.zeros(local_experts, 128, 256, dtype=torch.float8_e4m3fn)
    group_ends = torch.zeros(local_experts, dtype=torch.int32)
    got = fp8_grouped_gluon.gluon_tiles(
        rows, weight, 128, group_ends, num_experts=256, by_default=True
    )
    assert got == tiles


def test_forced_gluon_fails_when_no_kernel_takes_the_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced bench run must not time the Triton kernel under the Gluon name."""
    monkeypatch.setattr(fp8_grouped_kernel, "_GLUON_FOR_BENCH", True)
    monkeypatch.setattr(fp8_grouped_gluon, "_GLUON_AVAILABLE", False)
    with pytest.raises(ValueError, match="no Gluon kernel"):
        block_fp8_grouped_mm(
            torch.zeros(300, 256, dtype=torch.bfloat16),
            torch.zeros(2, 128, 256, dtype=torch.float8_e4m3fn),
            torch.ones(2, 1, 2),
            block=128,
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
    weight = _fp8((num_experts, n_dim, k_dim)).cuda()
    scale = (torch.rand(num_experts, n_dim // 128, k_dim // 128) * 0.02 + 0.01).cuda()
    rows = torch.randn(len(row_experts), k_dim).bfloat16().cuda()
    args = (rows, weight, scale)
    kwargs = {"block": 128, "row_experts": row_experts.cuda(), "group_ends": group_ends}

    calls = _count_gluon_calls(monkeypatch)
    gluon = block_fp8_grouped_mm(*args, **kwargs)
    assert calls, "the default path did not take the Gluon kernel"
    monkeypatch.setattr(fp8_grouped_kernel, "_GLUON_FOR_BENCH", False)
    triton_out = block_fp8_grouped_mm(*args, **kwargs)
    assert len(calls) == 1
    rel = (gluon.float() - triton_out.float()).norm() / triton_out.float().norm()
    assert rel < 2e-3, f"Gluon and Triton differ by a relative {rel:.2e}"


_GRAPH_CASES: list[Case] = [
    (90, (*fp8_grouped_gluon._HOPPER_TILES[1][1][:5], True)),
    (90, (*fp8_grouped_gluon._HOPPER_TILES[1][1][:5], False)),
    (100, fp8_grouped_gluon._BLACKWELL_TILES[1][1]),
]


@pytest.mark.requires_cuda
@pytest.mark.parametrize("case", _GRAPH_CASES, ids=_case_id)
def test_kernel_runs_in_a_cuda_graph(monkeypatch: pytest.MonkeyPatch, case: Case) -> None:
    """Captured once, the kernel follows new routing written into the same buffers."""
    _on_gpu(case[0])
    calls = _force_gluon(monkeypatch, case[1])
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
    assert calls, "the Gluon kernel did not run"

    experts, ends = routing()
    row_experts.copy_(experts)
    group_ends.copy_(ends)
    rows.copy_(torch.randn(num_rows, k_dim).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, forward(), rtol=0, atol=0)
    want = _reference(rows.cpu(), weight.cpu(), scale.cpu(), experts)
    torch.testing.assert_close(captured.cpu().float(), want, rtol=1e-2, atol=1e-3)


# Two programs walk 12 items each: the stages, the weight buffers and the
# accumulators wrap around inside an item and from one item to the next.
_SANITIZER_SCRIPT = textwrap.dedent(
    """
    import torch
    from mini_infer.models.blocks import fp8_grouped_gluon

    torch.manual_seed(0)
    fp8_grouped_gluon._PROGRAMS_FOR_TESTS = 2
    tiles = {tiles}
    counts = [3, 0, 150, 1, 70, 40, 2]
    num_experts, n_dim, k_dim = len(counts), 512, 384
    experts = torch.repeat_interleave(torch.arange(num_experts), torch.tensor(counts))
    rows = torch.randn(len(experts) + 5, k_dim).bfloat16().cuda()
    weight = (torch.randn(num_experts, n_dim, k_dim) * 0.5).to(torch.float8_e4m3fn).cuda()
    scale = torch.full((num_experts, 4, 3), 0.02).cuda()
    group_ends = torch.tensor(counts).cumsum(0).to(torch.int32).cuda()
    fp8_grouped_gluon.gluon_block_fp8_grouped_mm(
        rows, weight, scale, block=128, group_ends=group_ends, tiles=tiles
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
