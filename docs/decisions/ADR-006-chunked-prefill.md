# ADR-006: Chunked prefill (head-of-line-blocking fix)

Date: 2026-04-27
Status: Accepted (with packed-forward optimization deferred to a follow-up ADR)

## Context

After continuous batching (ADR-005), a long prompt that arrives while short
requests decode blocks the engine for its full prefill. A
`model.forward(...)` over a 4k-token prompt is one large GPU call, and no
decoder advances during it. This is head-of-line blocking (one long job
stalls the jobs behind it).

Chunked prefill runs the prefill in chunks of 256 tokens by default, between
decode steps. A 4k-token prefill then takes 16 engine steps. This slice uses
two forwards per step: one `model.forward` for the prefill chunk and one for
the batched decode. Decoders still wait, but only for one chunk. One packed
forward per step over both gives more throughput. ADR-007 covers it.

## Decision

1. State machine. `RequestState` gains `CHUNKED_PREFILLING`, and
   `RunningRequest` gains `tokens_prefilled: int` and
   `prefill_cache: PagedKVCache | None`. The lifecycle is
   `WAITING -> PREFILLING -> CHUNKED_PREFILLING -> DECODING -> DONE`.
   `PREFILLING` lasts one step.
2. Per-request prefill cache. Each prefilling request owns a temporary
   `PagedKVCache(batch_size=1)`. Each call to
   `prefill_chunk(cache, chunk_tokens, position_offset)` adds the K/V of one
   chunk. After the last chunk, `merge_request` moves the blocks into the
   shared cache, as in ADR-005.
3. The default chunk size of 256 is also the default of vLLM and SGLang
   (`ContinuousScheduler(chunk_size=...)`).
4. `PagedKVCache.append_kv_packed(packed_k, packed_v, cu_seqlens_q_new, layer_idx)`
   writes ragged appends per slot, and a slot can append zero tokens.
   `append_kv` now calls it with uniform `cu_seqlens`. The packed-forward
   path uses it.
5. `PagedKVCache.get_mask_sizes(query_length, layer_idx)` is overridden.
   Without it, `create_causal_mask` of HF reads `kv_length` from
   `DynamicCache.layers`, which our paged cache does not fill. The mask then
   covers only the new chunk, and the output silently differs from the
   un-chunked path. The override returns
   `(max(self._num_tokens) + query_length, 0)`.
   `test_chunked_prefill_matches_unchunked` found this bug before release.
6. No change to the patched Qwen2 attention. Decode keeps the paged fast
   path. Prefill chunks go through the stock HF attention via
   `cache.update`, as the single-shot prefill of ADR-005 does.

## Alternatives considered

- One packed forward per step now. `flash_attn_varlen_func` does not read
  paged K/V, and `flash_attn_with_kvcache` does not take variable-length
  queries across requests. So it needs a K/V copy or a custom Triton
  kernel. Two forwards first let us test the state machine, the cache
  transfer and the `get_mask_sizes` fix alone. A kernel change in the same
  slice can hide such a bug.
- Keep only `RequestState.PREFILLING`. `_step` treats both states the same,
  but `CHUNKED_PREFILLING` makes the state machine clear for the cost of one
  enum value.
- Chunk sizes per request. One default is enough to prove correctness.

## Consequences

- Head-of-line blocking is gone for prompts up to about
  chunk_size × max_step_time / decode_time. Decoders advance about once per
  chunk.
- When prefill and decode are both pending, each step pays two
  `model.forward` calls. The gain of the packed forward is deferred.
- Any later cache subclass faces the same `get_mask_sizes` invariant.
- `prefill_cache` has meaning only during prefill. A
  `dict[RequestID, PagedKVCache]` in the scheduler avoids this, but needs
  more code.
- A revert is about 80 lines (`prefill_chunk`, `_advance_chunked_prefill`).
  `append_kv_packed` and `get_mask_sizes` stay either way.
- On M1 (CPU), `test_chunked_prefill_matches_unchunked` runs a long prompt
  with `chunk_size = prompt_len // 3` and matches the single-chunk greedy
  output token-for-token. 4 new tests cover `append_kv_packed`.
- 2 stress tests are in `tests/stress/test_chunked_prefill_load.py`. One
  runs 8 concurrent requests of 5–144 tokens with chunk_size=32 and matches
  a serial reference token-for-token. One uses a 32-block pool: all 6
  requests complete, and the pool gets all blocks back.
- CI runs 73 tests, without the stress tests (`@pytest.mark.slow`). The
  Modal GPU run is deferred to the packed-forward slice.
