# ADR-008: Paged FlashAttention varlen — investigated, kept as a tunable, not the default

Date: 2026-04-28
Status: Accepted (with the unexpected finding that paged FA underperforms materialized FA on our reference hardware)

## Context

ADR-007 showed a 30% throughput drop against the decode-only Triton kernel
of ADR-005. The suspected cause was materialization: every step copies the
K/V of each layer from `BlockPool` storage into a contiguous buffer for
`flash_attn_varlen_func`. flash-attn 2.7+ added a `block_table` argument, so
the kernel can read paged K/V with no copy. The plan was to use it and win
back the throughput.

## Decision

1. Add the paged path, `_packed_attention_paged_flash` in
   `cache/packed_attention.py`.
2. Keep the materialized path, `_packed_attention_materialized_flash`. FA
   paged needs `block_size % 256 == 0`, too strict for every configuration.
   The materialized path is also faster.
3. `packed_attention_forward` uses paged FA on CUDA when
   `cache._pool.block_size % 256 == 0`. Otherwise it uses materialized FA on
   CUDA, or the PyTorch reference on other devices.
4. The default `block_size` stays 16, so `ModelRunner.from_pretrained()`
   uses materialized FA. `block_size=256` is the opt-in for paged FA.
5. The `[cuda]` extra requires `flash-attn>=2.8`. Paged varlen needs 2.7+,
   and we tested 2.8.3.
6. `PagedKVCache` gets `block_table_padded(device)`, `seq_lens_tensor(device)`
   and `pool_storage_for_layer(layer_idx)` for paged kernels.

## Empirical finding

Paged FA was slower almost everywhere, on A10 (Ampere, SM_86) and H100
(Hopper, SM_90).
Materialized FA runs at `block_size=16`, paged FA at `block_size=256`.
Throughput is in tok/s at C concurrent requests.

| Workload | A10 materialized | A10 paged | Δ | H100 materialized | H100 paged | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Short, C=1 | 36.6 | 17.5 | -52% | 57.2 | 43.9 | -23% |
| Short, C=4 | 88.0 | 49.9 | -43% | 144.9 | 132.3 | -9% |
| Short, C=8 | 122.2 | 122.7 | ≈0 | 203.0 | 209.8 | +3% |
| Long (~3.9k), C=1 | 14.5 | 12.0 | -17% | 21.4 | 16.8 | -22% |
| Long, C=4 | 19.7 | 16.9 | -14% | 28.4 | 24.3 | -14% |

Paged FA is close only for short prompts at C=8, and the +3% on H100 is
within run-to-run noise. Elsewhere materialized FA is faster by 9-52%. The
likely causes, most likely first:

1. `block_size=256`, the FA minimum, wastes bandwidth on short prompts. A
   5-token prompt takes a 256-slot block, and `cache_seqlens` masks the 251
   unused slots.
2. Materialization is cheap at our scale: about 3 MB per layer per step for
   0.5B with ~3k contexts, small next to the matmul work.
3. Each tile pays for the
   `block_table[batch_idx][block_idx] → physical_block` lookup.
4. Paged FA pays off at 70B+ models with 32k+ contexts, where the gather
   reaches gigabytes per step.

## Alternatives considered

- Make paged FA the default, with `block_size=256` forced on CUDA. Rejected:
  the loss is consistent and large, and every CUDA run gets more memory
  pressure.
- Drop the paged path. Rejected: the abstraction is correct, and with both
  paths in place the dispatch follows measured data.

## Consequences

- The headline goal, to close the materialization gap, failed.
- Two FA paths need upkeep, but both are short (about 80 lines to revert).
- On M1 (CPU, fp32), 75 unit and 3 golden tests pass. On Modal A10, the smoke
  test passes on both paths, with bf16 tail drift on the longest prompt only.
  `tests/unit/cache/test_packed_attention.py::test_paged_flash_matches_torch_reference`
  (`requires_cuda`) covers paged FA.
- Full numbers: `docs/benchmarks/2026-04-27-packed-forward.md` ("Paged vs
  materialized FA" section).

## Follow-ups

- Bench again on a 70B+ model. The gather cost scales linearly with
  `(num_layers × kv_len × num_kv_heads × head_dim)`, and our 0.5B model has
  the smallest value of all four factors.
- A custom Triton varlen-paged kernel with `block_size=16`. It avoids the FA
  indirection and the wasted bandwidth of `block_size=256`. It is Phase 3b
  stretch work, only if the gather dominates step time.
