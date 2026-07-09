# TPU v5e: Pallas attention kernels, on-hardware validation

Date: 2026-07-09
Hardware: Google Colab TPU v5e (single chip), jax 0.7.2 with matched libtpu
Code: branch `tpu-pallas-backend`, commit `83321f8`
Runner: `scripts/run_tpu_pallas_kernels.py` via `scripts/colab/run_tpu.ipynb`, `interpret=False`

## What this is

The first full on-hardware run of the TPU backend (ADR-023, ADR-024). Every
kernel executes with real Mosaic lowering (not interpret mode) and is checked
against a NumPy reference at the project parity bar (cosine similarity > 0.99).
This is a correctness gate with indicative timings, not a tuned performance
benchmark: shapes are small, block sizes are conservative, and there is no DMA
pipelining.

## Results

| Kernel | Config | Cosine | Max abs err | ms/call |
|---|---|---|---|---|
| Dense attention | 4 heads, seq 128, d 128, non-causal | 0.9999949 | 3.56e-03 | 68.0 |
| Dense attention | same, causal | 0.9999963 | 7.72e-03 | 72.7 |
| Paged decode | 8 seqs, MHA 8 heads, d 128, page 16 | 0.9999966 | 6.64e-03 | 62.2 |
| Paged decode | GQA 8q/2kv | 0.9999950 | 3.86e-03 | 66.4 |
| Paged prefill | q_len 8, MHA 8 heads | 0.9999957 | 8.27e-03 | 65.1 |
| Paged prefill | q_len 8, GQA 8q/2kv | 0.9999959 | 7.80e-03 | 63.8 |

Overall: ALL PASS, exit code 0.

## Reading the numbers

- The ~1e-3 max abs error vs the fp32 reference (vs ~1e-6 in interpret mode) is
  the TPU's default matmul precision (bf16 accumulation passes), not a logic
  difference; cosines sit at 0.99999x against the > 0.99 bar.
- ms/call at these toy shapes is dominated by per-call overhead and says little
  about throughput at real model shapes; it is recorded to confirm the kernels
  execute, not to claim speed. No performance conclusions should be drawn until
  a shaped, roofline-framed benchmark exists.

## What it took to get here (hardware findings)

Three environment/lowering issues were found and fixed en route, none of them
kernel-math bugs; full detail in ADR-024's amendment:

1. Colab's preinstalled libtpu lagged jaxlib and rejected valid Mosaic
   ("Unsupported version: expected <= 7 but got 8"); the notebook now installs a
   matched `jax[tpu]` and restarts once.
2. Mosaic's block-shape rule (last two dims divisible by (8, 128) or equal to
   the array dims) rejected the original pool layout; pools are now heads-first
   `(num_kv_heads, num_pages, page_size, head_dim)`, decode q carries a
   singleton axis, prefill q is carried transposed. Interpret mode does not
   check this rule.
3. The clone-once bootstrap silently re-ran a stale checkout on a persisted
   Colab VM; both bootstraps now sync to the branch head and print the running
   commit.

## Reproduce

Open `scripts/colab/run_tpu.ipynb` in Colab (Runtime -> TPU -> Run all; run the
cell twice if the one-time libtpu alignment restart triggers). The log must
print the running commit and end with ALL PASS.
