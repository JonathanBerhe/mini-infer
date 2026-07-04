# MSA block-sparse decode kernel: parity + microbench + end-to-end A/B

Date: 2026-07-03. GPU: NVIDIA A10G (Modal), torch 2.6.0 cu124, bf16 pool.
Script: `scripts/modal_msa_kernel_bench.py`. Kernel:
`mini_infer/cache/msa_paged_attention.py` (`msa_paged_decode_triton`).

## Parity (kernel vs pure-torch reference, bf16 paged data)

The torch reference is itself CPU-validated against the dense-mask oracle
(`tests/unit/test_msa_paged_attention.py`). Sweeps cover ragged batches,
partial last blocks, and short contexts where the selection carries `-1`
padding.

| case | cosine | max_abs |
|---|---|---|
| kv=4, topk=16, seq=[4096] | 1.000000 | 1.2e-4 |
| kv=4, topk=16, seq=[1000, 3333] (ragged) | 1.000000 | 6.1e-5 |
| kv=1, topk=4, seq=[700] | 1.000000 | 0.0 |
| kv=4, topk=16, seq=[130] (-1 padding) | 1.000000 | 0.0 |

Sparsity probe: randomizing K/V ONLY outside the selected blocks leaves the
kernel output bit-identical in all four cases (the kernel provably reads just
the selected blocks).

## Op microbench (M3 head geometry: 64q/4kv/head_dim 128, topk 16, index block 128)

Kernel vs the shipped torch path (materialize full K/V + dense block mask +
SDPA), 20 timed iterations after warmup:

| context | B | kernel (us) | torch (us) | speedup |
|---|---|---|---|---|
| 1,024 | 1 | 463 | 1,064 | 2.30x |
| 1,024 | 8 | 2,213 | 7,624 | 3.45x |
| 4,096 | 1 | 461 | 2,374 | 5.15x |
| 4,096 | 8 | 2,229 | 18,208 | 8.17x |
| 16,384 | 1 | 430 | 7,848 | 18.27x |
| 16,384 | 8 | 2,211 | 61,055 | 27.62x |
| 65,536 | 1 | 464 | 30,035 | 64.78x |
| 65,536 | 8 | 2,219 | 234,954 | 105.88x |

Kernel time is flat in context (O(topk * block) reads); the torch path grows
linearly (O(context) materialize + mask + attention). The 4-block-per-program
grid underfills the A10 at B=1, hence the flat ~460us floor; larger batches
amortize it.

## End-to-end A/B (synthetic M3-shaped model, 114M params, 16q/1kv/128, 4 layers)

Chunked prefill to 32,768 tokens, then 64 timed greedy decode steps per arm:

| arm | tok/s |
|---|---|
| torch path | 35.38 |
| kernel path | 33.74 |

Speedup 0.95x (a slight LOSS) with token identity PASS. At toy scale the
per-step host work (selection glue, small-tensor launches) plus the indexer's
O(context) re-scoring dominate; the attention op the kernel accelerates is a
sliver of the step. Consequence (per ADR-020's rule): the kernel stays
**off by default**; the ship decision falls to the end-to-end A/B on the real
428B checkpoint, where per-layer attention traffic is materially larger
(57 sparse layers, 64 heads) while the host overhead per step is unchanged.

## Notes

- First run failed on a missing `transformers` in the image (module-level
  import in `paged_kv_cache.py`); the same failure mode as the first V4 kernel
  bench. The bench images must install transformers even when no HF model is
  used.
- An earlier kernel revision produced NaN whenever the selection carried `-1`
  padding: dead entries ran through the online-softmax rescale while the
  running max was still `-inf` (`exp(-inf - -inf) = nan`). Fixed by skipping
  dead entries outside the softmax update; the [130]-context parity case now
  guards this regression.
