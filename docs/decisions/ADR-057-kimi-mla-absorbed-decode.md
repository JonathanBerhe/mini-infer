# ADR-057: Run the Kimi MLA decode in the absorbed form

Date: 2026-11-06
Status: Accepted

## Context

Kimi Linear 48B has 7 MLA layers among its 27 layers, and Kimi K3 has 24.
Each MLA layer keeps one shared row of 576 values per token in its state:
512 latent values and 64 key values that are never rotated (NoPE).

Before this change, every decode step ran `kv_b_proj` over the whole history
in every MLA layer. That rebuilt the keys and values of 32 heads for each
history token, about 4.2M multiply-adds per token, and ran `kv_a_layernorm`
over the history again.

The step read the buffer up to the end of its longest row, not up to its
capacity. So the cost grew with the history, at any capacity.

## Decision

1. The state keeps the latent after `kv_a_layernorm`. The norm acts on each
   token alone, so one norm at write time gives the same values as a norm of
   the history at every read.
2. A decode step, one new token per row, runs the absorbed form
   (`_KimiMlaAttention._attend_absorbed`, DeepSeek-V2 section 2.1.2). Split
   `kv_b_proj` per head into `W_UK` and `W_UV`. The query becomes
   `[W_UK^T q_nope, q_rope]`, and it scores each 576-wide state row with one
   dot product. The output is `W_UV` times the weighted sum of the latents.
   Per history token this is about 35K multiply-adds instead of 4.2M. The
   math runs in fp32, and the output rounds to the model dtype once.
3. A span (prefill, chunked prefill) keeps the decompressed form. Above
   about 170 queries per span, it needs fewer operations than the absorbed
   form.
4. Both forms share the causal mask and the DSA selection of Kimi K3.1, so
   the indexer picks the same keys in both. A step still reads the buffer up
   to the end of its longest row.
5. A decode step of the layer reads one value back to the host (the end of
   the longest row), not two.
6. If the `kv_b_proj` weight is not floating point, for example an
   `Int8Linear`, the layer keeps the decompressed form.
7. The class flag `_KimiMlaAttention.absorbed_decode` selects the form, so
   the tests and the profile script compare both forms on one set of weights.

## Alternatives Considered

- The FlashInfer MLA decode kernel. Its head sizes (512, 64) match Kimi,
  and it reads only the filled positions of each row. It needs a page
  layout of the state and has no sparse mask for the K3.1 selection. The
  PyTorch absorbed form is its oracle.
- A loop over the rows with each row's own length: one host read per row
  and layer, and more launches.
- A state of rebuilt keys and values, like the reference `KimiDynamicCache`.
  It holds 10,240 values per token and layer instead of 576.
- The decompressed form for short histories. In eager mode on the H100 it
  is 0.05 to 0.09 ms faster per layer at 1,024 tokens and below. A switch
  by length changes the rounding of a request in the middle of its output,
  and a decode CUDA graph removes the launch cost behind the difference.

## Consequences

One MLA layer at the Kimi Linear 48B widths, decompressed form against
absorbed form (H100, bf16) and main against this change (M1 Pro CPU, fp32).
The full tables are in the
[profile report](../benchmarks/2026-11-06-kimi-mla-decode-profile.md).

| Batch | History | H100 before | H100 after | CPU before | CPU after |
|---|---|---:|---:|---:|---:|
| 1 | 1,024 | 0.46 ms | 0.51 ms | 13.1 ms | 4.9 ms |
| 1 | 4,096 | 0.59 ms | 0.49 ms | 37.2 ms | 5.5 ms |
| 1 | 16,384 | 1.42 ms | 0.48 ms | 141.9 ms | 6.9 ms |
| 8, ragged | longest 4,096 | 4.61 ms | 0.63 ms | 507.4 ms | 14.6 ms |

- The layer time is now almost constant in the history. On the H100 the
  absorbed layer is bound by kernel launches at every length measured.
- In fp32 the two forms agree to 1e-5 on Kimi Linear, K3 and K3.1, and the
  reference parity tests pass with the absorbed decode. In bf16 both forms
  stay within the bf16 tolerance of fp32.
- The MLA layers are not the main cost of the 2.1 tokens per second of Kimi
  Linear 48B on one B200. On the H100 a 4-layer slice with 256 experts
  takes 30 ms per eager step for 8 ms of GPU work, and 768 of its 769 host
  syncs come from the `torch.where` of each expert in the MoE. For the full
  model that is about 6,700 syncs per step (an estimate).
