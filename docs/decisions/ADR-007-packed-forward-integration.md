# ADR-007: Packed-varlen forward — one model.forward per scheduler step

Date: 2026-04-27
Status: Accepted

## Context

ADR-006 shipped chunked prefill with two forwards per step: one
`model.forward` for the prefill chunk and one for the batched decoders. The
Q/K/V projections and the MLP, the largest cost, ran twice over the same
weights. This slice (B) merges both into one packed forward per step. It
uses the varlen attention of FlashAttention (FA), which takes sequences of
different lengths in one call.

## Decision

1. One forward per step. The scheduler builds a packed sequence
   `(1, total_q, hidden)`, where `total_q = sum(per-request q_lens)`. A
   prefilling request adds `chunk_size` tokens and a decoding request adds 1.
   cu_seqlens_q and cu_seqlens_k mark the request boundaries.
2. `cache/packed_attention.py` (commit `f140c20`) holds the attention
   abstraction. `packed_attention_forward(...)` calls
   `flash_attn_varlen_func` on CUDA and a per-request SDPA loop elsewhere.
3. `PagedKVCache.materialize_packed_kv(layer_idx)` gathers the K/V history of
   every request along dim 0, the shape FA varlen wants. It returns the
   offsets as `cu_seqlens_k`.
4. A packed-forward path replaces the Qwen2 patch: projections and RoPE on
   the packed sequence, `cache.append_kv_packed(...)`,
   `materialize_packed_kv(...)`, then `packed_attention_forward(...)`.
5. `runner.forward_step(cache, packed_input_ids, cu_seqlens_q, position_offsets)`
   is the one entry point. `prefill`, `decode`, `decode_batch` and
   `prefill_chunk` become 3-line wrappers.
6. The scheduler `_step` runs: admit, sample decoders, reap DONE, then
   `_packed_forward(alive)`. A new request gets a slot in `_batched_cache` at
   admission, so `RunningRequest` loses its temporary prefill cache.
7. The patch applies on every device. FA varlen on CUDA and the SDPA loop on
   CPU and MPS give the same outputs.

## Alternatives considered

- Keep the two-forward design behind a flag. It doubles the test surface,
  and commit `9ea3ff4` keeps it for a clean revert.
- A custom Triton varlen-paged kernel, as in vLLM. The gather is only about
  3 MB per layer per step for 0.5B on A10.
- `flash_attn_with_kvcache`, the paged API of FA 2.7+. It needs one `q_len`
  for the whole batch, so our mixed batches need padding.
- Sample decoders after the forward. Finished slots then stay in the cache
  during the forward.

## Consequences

- The matmul cost of each step spreads over all in-flight q-tokens.
- `runner.forward_step` is the single primitive, so the runner API is
  smaller.
- The gather runs per layer per step and can limit very long contexts at
  high concurrency.
- The packed Qwen2 patch is longer than the old one.
- flash-attn is a runtime dependency on CUDA, through the optional `[cuda]`
  extra.
- A custom kernel can replace FA behind `packed_attention_forward` with no
  scheduler or patch change.
- On M1 (CPU path), all 75 unit tests pass, golden tests included. Two
  stress tests also pass: 8 concurrent mixed-length requests, and a 32-block
  pool.

## Update: variable q-length per decoder (2026-07-29)

The decision is unchanged, but two points changed.

- Point 1: a decoding request now adds a list of tokens, and `cu_seqlens_q`
  advances by its length. Plain sampling still adds one. A speculative
  verify step feeds the whole drafted block at once.
- Point 5: commit `0d5f4d4` moved the packed-logits body into
  `forward_step_packed`, and `forward_step` is now a last-position slice over
  it. `_packed_forward` calls the primitive and slices each last position at
  `cu_seqlens_q[index + 1] - 1`, because the verify step needs logits at
  every position. `forward_step` still serves every other caller, such as the
  prefill worker and the draft model in `engine/speculative.py`.

`tests/unit/scheduler/test_scheduler_variable_tokens.py` covers the packing and
slicing without a model.
