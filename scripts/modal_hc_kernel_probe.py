"""Modal probe: bisect the HC Sinkhorn Triton kernel segfault.

Four attempts at running `hc_split_sinkhorn` as a Triton kernel on
torch 2.5.1 + triton 3.1.0 produced: SIGSEGV with a plain `range` loop
(twice, with different warp counts and eps plumbing), and a compile
hang with `tl.static_range`. The suspect is the axis-0 reduction
(`tl.sum(comb, axis=0)`) on a loop-carried 2D register tile, a codegen
path the repo's proven kernels never exercise (FlashAttention-style
loops reduce along axis 1 only).

This probe runs four kernel variants, EACH IN A SUBPROCESS so a
segfault or hang is contained and reported as an exit code rather
than killing the harness:

  1. noloop:      sinkhorn_iters=1 shape, no loop at all. Axis-0
                  reduction appears once, outside any loop. Tests
                  whether axis-0 reduction alone compiles.
  2. loop_axis0:  the current kernel shape. Plain `range` loop with
                  axis=1 and axis=0 sums inside. Expected to SIGSEGV
                  (reproduces the failure).
  3. loop_trans:  same loop, but every axis-0 sum is computed as
                  `tl.sum(tl.trans(comb), axis=1)`. The targeted fix:
                  same math, axis-1-only reduction codegen.
  4. unroll3:     `tl.static_range` with only 3 iterations. Tests
                  whether a SMALL forced unroll compiles (separates
                  "unroll is broken" from "19x unroll explodes").

Each variant that compiles also compares its output against a
sequential PyTorch oracle and prints max-abs-diff.

Exit-code legend in the verdict table: 0 = ok, -11 = SIGSEGV,
TIMEOUT = killed after 120s (compile hang), other = python error.

Run with:
    uv run modal run scripts/modal_hc_kernel_probe.py
"""

import os
import subprocess
import sys

import modal

app = modal.App("mini-infer-hc-kernel-probe")

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "L40S")

# Torch only. Triton arrives pinned by the torch wheel (3.1.0); the
# probe deliberately avoids importing mini_infer so the only variables
# are torch + triton + the kernel source under test.
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")

# Shared scaffolding for every variant: build inputs, run the kernel,
# compare against the sequential PyTorch oracle. The variant-specific
# kernel body is interpolated via %(kernel_src)s and %(iters)s.
_HARNESS_TEMPLATE = r"""
import torch
import triton
import triton.language as tl

HC = 4
ITERS = %(iters)s
EPS = 1e-6
N_ROWS = 32
MIX_HC = (2 + HC) * HC

%(kernel_src)s

def torch_oracle(mixes, hc_scale, hc_base):
    pre_f = mixes[..., :HC]
    post_f = mixes[..., HC : 2 * HC]
    comb_f = mixes[..., 2 * HC :].reshape(*mixes.shape[:-1], HC, HC)
    base_pre = hc_base[:HC]
    base_post = hc_base[HC : 2 * HC]
    base_comb = hc_base[2 * HC :].reshape(HC, HC)
    pre = torch.sigmoid(pre_f * hc_scale[0] + base_pre) + EPS
    post = 2.0 * torch.sigmoid(post_f * hc_scale[1] + base_post)
    comb = comb_f * hc_scale[2] + base_comb
    comb = comb.softmax(dim=-1) + EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    return pre, post, comb

device = torch.device("cuda", 0)
g = torch.Generator(device=device)
g.manual_seed(7)
mixes = torch.randn(N_ROWS, MIX_HC, dtype=torch.float32, device=device, generator=g)
hc_scale = torch.randn(3, dtype=torch.float32, device=device, generator=g)
hc_base = torch.randn(MIX_HC, dtype=torch.float32, device=device, generator=g)

pre = torch.empty((N_ROWS, HC), dtype=torch.float32, device=device)
post = torch.empty((N_ROWS, HC), dtype=torch.float32, device=device)
comb = torch.empty((N_ROWS, HC, HC), dtype=torch.float32, device=device)

print("launching...", flush=True)
kernel[(N_ROWS,)](mixes, hc_scale, hc_base, pre, post, comb, EPS)
torch.cuda.synchronize()
print("launched and synced", flush=True)

pre_r, post_r, comb_r = torch_oracle(mixes, hc_scale, hc_base)
print(f"max-abs-diff pre={ (pre - pre_r).abs().max().item():.3e}", flush=True)
print(f"max-abs-diff post={(post - post_r).abs().max().item():.3e}", flush=True)
print(f"max-abs-diff comb={(comb - comb_r).abs().max().item():.3e}", flush=True)
"""

# Kernel prologue shared by all variants (split + pre/post + first
# softmax step). Variants differ only in how they normalize columns
# and how they loop.
_KERNEL_PROLOGUE = r"""
@triton.jit
def kernel(mixes_ptr, hc_scale_ptr, hc_base_ptr, pre_ptr, post_ptr, comb_ptr, eps):
    row = tl.program_id(0)
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)
    offs_hc = tl.arange(0, 4)
    row_mix_ptr = mixes_ptr + row * 24

    pre_features = tl.load(row_mix_ptr + offs_hc)
    pre_base = tl.load(hc_base_ptr + offs_hc)
    pre = tl.sigmoid(pre_features * scale_pre + pre_base) + eps
    tl.store(pre_ptr + row * 4 + offs_hc, pre)

    post_features = tl.load(row_mix_ptr + 4 + offs_hc)
    post_base = tl.load(hc_base_ptr + 4 + offs_hc)
    post = 2.0 * tl.sigmoid(post_features * scale_post + post_base)
    tl.store(post_ptr + row * 4 + offs_hc, post)

    offs_j = tl.arange(0, 4)
    offs_k = tl.arange(0, 4)
    comb_offs = 8 + offs_j[:, None] * 4 + offs_k[None, :]
    comb_features = tl.load(row_mix_ptr + comb_offs)
    comb_base = tl.load(hc_base_ptr + comb_offs)
    comb = comb_features * scale_comb + comb_base

    row_max = tl.max(comb, axis=1)
    comb = tl.exp(comb - row_max[:, None])
    row_sum = tl.sum(comb, axis=1)
    comb = comb / row_sum[:, None] + eps
"""

_KERNEL_EPILOGUE = r"""
    tl.store(comb_ptr + row * 16 + offs_j[:, None] * 4 + offs_k[None, :], comb)
"""

VARIANTS: dict[str, dict[str, str]] = {
    # 1. No loop at all (iters=1 shape). One axis-0 sum, outside a loop.
    "noloop": {
        "iters": "1",
        "kernel_src": _KERNEL_PROLOGUE
        + r"""
    col_sum = tl.sum(comb, axis=0)
    comb = comb / (col_sum[None, :] + eps)
"""
        + _KERNEL_EPILOGUE,
    },
    # 2. Current kernel shape: plain range loop, axis-0 sums inside.
    "loop_axis0": {
        "iters": "20",
        "kernel_src": _KERNEL_PROLOGUE
        + r"""
    col_sum = tl.sum(comb, axis=0)
    comb = comb / (col_sum[None, :] + eps)
    for _ in range(19):
        row_sum = tl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + eps)
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + eps)
"""
        + _KERNEL_EPILOGUE,
    },
    # 3. Targeted fix: axis-0 sums expressed as trans + axis-1 sum.
    "loop_trans": {
        "iters": "20",
        "kernel_src": _KERNEL_PROLOGUE
        + r"""
    col_sum = tl.sum(tl.trans(comb), axis=1)
    comb = comb / (col_sum[None, :] + eps)
    for _ in range(19):
        row_sum = tl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + eps)
        col_sum = tl.sum(tl.trans(comb), axis=1)
        comb = comb / (col_sum[None, :] + eps)
"""
        + _KERNEL_EPILOGUE,
    },
    # 4. Small forced unroll: static_range with 3 trips.
    "unroll3": {
        "iters": "4",
        "kernel_src": _KERNEL_PROLOGUE
        + r"""
    col_sum = tl.sum(comb, axis=0)
    comb = comb / (col_sum[None, :] + eps)
    for _ in tl.static_range(3):
        row_sum = tl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + eps)
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + eps)
"""
        + _KERNEL_EPILOGUE,
    },
    # 5. Scale-vector Sinkhorn: loop carries 1D row/col scale vectors;
    #    the matrix stays loop-invariant; every reduction operates on a
    #    body-local fresh value (the FlashAttention loop shape). Column
    #    sums via plain axis-0.
    "scalevec_axis0": {
        "iters": "20",
        "kernel_src": _KERNEL_PROLOGUE
        + r"""
    cs0 = tl.sum(comb, axis=0)
    col_scale = 1.0 / (cs0 + eps)
    row_scale = tl.full((4,), 1.0, dtype=tl.float32)
    for _ in range(19):
        scaled = comb * row_scale[:, None] * col_scale[None, :]
        rs = tl.sum(scaled, axis=1)
        row_scale = row_scale / (rs + eps)
        scaled2 = comb * row_scale[:, None] * col_scale[None, :]
        cs = tl.sum(scaled2, axis=0)
        col_scale = col_scale / (cs + eps)
    comb = comb * row_scale[:, None] * col_scale[None, :]
"""
        + _KERNEL_EPILOGUE,
    },
    # 6. Same as 5 but column sums via trans + axis-1, in case axis-0
    #    reductions inside loop bodies are themselves broken.
    "scalevec_trans": {
        "iters": "20",
        "kernel_src": _KERNEL_PROLOGUE
        + r"""
    cs0 = tl.sum(comb, axis=0)
    col_scale = 1.0 / (cs0 + eps)
    row_scale = tl.full((4,), 1.0, dtype=tl.float32)
    for _ in range(19):
        scaled = comb * row_scale[:, None] * col_scale[None, :]
        rs = tl.sum(scaled, axis=1)
        row_scale = row_scale / (rs + eps)
        scaled2 = comb * row_scale[:, None] * col_scale[None, :]
        cs = tl.sum(tl.trans(scaled2), axis=1)
        col_scale = col_scale / (cs + eps)
    comb = comb * row_scale[:, None] * col_scale[None, :]
"""
        + _KERNEL_EPILOGUE,
    },
}


@app.function(image=image, gpu=_BENCH_GPU, timeout=900)
def probe() -> str:
    import torch

    lines: list[str] = []
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.append(f"PyTorch: {torch.__version__}")
    import triton

    lines.append(f"Triton: {triton.__version__}")
    lines.append("")

    for name, spec in VARIANTS.items():
        src = _HARNESS_TEMPLATE % spec
        # Write to a real file: @triton.jit calls inspect.getsource() on
        # the kernel function, which fails with "could not get source
        # code" under `python -c`. A file on disk gives inspect a source.
        src_path = f"/tmp/hc_probe_variant_{name}.py"
        with open(src_path, "w") as f:
            f.write(src)
        print(f"=== variant {name}: starting (timeout 120s) ===", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, src_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            verdict = f"exit={result.returncode}"
            detail = result.stdout.strip().replace("\n", " | ")
            if result.returncode != 0 and result.stderr:
                tail = result.stderr.strip().splitlines()[-1]
                detail = f"{detail} | stderr: {tail}"
        except subprocess.TimeoutExpired as exc:
            verdict = "TIMEOUT (compile hang)"
            partial = exc.stdout or b""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            detail = partial.strip().replace("\n", " | ")
        line = f"{name:>12}: {verdict:<24} {detail}"
        lines.append(line)
        print(line, flush=True)

    report = "\n".join(lines)
    print()
    print("=== VERDICT TABLE ===")
    print(report)
    return report


@app.local_entrypoint()
def main() -> None:
    report = probe.remote()
    print()
    print(report)
