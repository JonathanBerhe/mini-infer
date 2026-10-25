# ADR-027: DSpark drafter port

Date: 2026-07-26
Status: Accepted (drafter ported and validated against the reference;
confidence scheduling remains deferred, and multi-request batching landed
in [ADR-031](ADR-031-multi-request-dspark.md))

## Context

DeepSeek released DSpark ("Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation", 2026-06-27) with the MIT-licensed `DeepSpec`
training and eval code. `docs/plans/dspark-evaluation.md` stages the work.
This ADR is the Stage A output: it resolves the parity-critical unknowns
before the Stage B code.

ADR-011 found the failure mode that DSpark targets. On H100, a fixed draft
length won 1.29x on a slow prompt and lost 0.83x on fast ones, for 1.00x
overall. ADR-011 already lists "adaptive K" as a follow-up.

DSpark has a semi-autoregressive drafter and a batch-global confidence
scheduler. This ADR covers the drafter (Stages B and C). The scheduler is
Stage D, deferred.

The released `dspark_qwen3_4b_block7` drafter (BF16, ~1.39B params,
block_size=7, 5 draft layers, `target_layer_ids=[1,9,17,25,33]`) pairs with
Qwen3-4B. Drafter training stays out of scope, because the `DeepSpec`
pipeline needs a ~38 TB cache of target hidden states.

## Decision

Port the Qwen3 DSpark drafter from scratch. Each mechanic below comes from
the `DeepSpec` code, not from the paper prose.

### 1. Module location: `src/mini_infer/engine/dspark/`, not `models/`

The registry (`ModelRegistry` + `load_model(name)`) maps one
`architectures[0]` string to one class that runs standalone. The drafter can
never run alone, because every forward needs injected target hidden states.
`SpeculativeRunner` already coordinates two coupled models for ADR-011, so
the drafter fits that layer. `engine/dspark/drafter.py` holds the
`Qwen3DSparkDrafter`, a plain `nn.Module`. One drafter forward proposes a
whole `gamma`-token block, not K serial draft forwards. In `models/`, only
`Qwen3ForCausalLM.forward` changes (point 2).

### 2. Target hidden-state taps: optional kwarg, zero-cost when unused

`Qwen3ForCausalLM.forward` gets `tap_layers: frozenset[int] | None = None`
and a sink. One guarded line in the layer loop records the post-block hidden
state `x` of each tapped layer. `forward()` still returns only `logits`, so
`ContinuousScheduler`, the golden tests and the two-model V1 see no change.
A forward that always returns a dict was rejected, because it changes the
return shape of every caller.

### 3. Drafter attention: its own SDPA call, not the shared `block_mask` path

The `block_mask` path of `packed_attention_torch` reads K/V only from a
`PagedKVCache`, with no hook for the projected target context, and it is
torch-only. So the drafter calls `F.scaled_dot_product_attention` directly,
with `is_causal=False`.

At batch 1 the drafter needs no mask. The `DeepSpec` training mask
has two rules: context keys come before the block anchor, and draft
positions see only their own block. With one block in flight, both rules
always hold. The reference also passes `attention_mask=None` at inference.
The mask returns in Stage D, with more than one block in flight. Until then,
`packed_attention_torch(block_mask=...)` is the unit-test oracle for the
unmasked call.

### 4. Parity-critical mechanics, read from the `DeepSpec` source

- Position ids and RoPE: positions are global, from one running counter over
  the whole generation, never reset per round. The anchor is at offset 0, and
  the `gamma-1` mask positions follow
  (`deepspec/modeling/dspark/common.py::create_position_ids`). The context
  (the newly committed tokens) and the draft block have contiguous positions.
  So `deepspec/eval/dspark/draft_ops.py::forward_dspark_draft_block` takes
  both in one slice,
  `position_ids[:, past_key_values_draft.get_seq_length():start+block_size]`,
  for one `rotary_emb` call. `k = cat([k_ctx, k_noise])` is rotated with the
  full cos/sin, and `q` uses the last `q_len` rows. The port must copy this
  contiguous-slice scheme exactly.
- Bidirectionality. Attention within a block is bidirectional. Only the
  Markov head (`VanillaMarkov.sample_block_tokens`) adds causal order: it
  conditions each step on the token sampled at the step before.
- Hidden-state scope. Round 1 injects the target hidden states for the whole
  prompt. Each later round injects only the committed window
  (`<= block_size + 1` rows), as a reassignment, not an append. No hidden
  states are re-derived over a growing prompt window, so the stop gate of the
  plan does not trigger.
- Draft cache. `k_ctx`/`v_ctx` (the projected context) accumulate, and
  `past_key_values_draft.crop(start)` drops `k_noise`/`v_noise` (the block)
  every round. So this cache grows at the same linear rate as any KV cache.
  It is small (5 layers, batch 1), so it stays unpaged, with a `truncate_to`
  contract like `PagedKVCache.truncate_to`.
- Confidence head. A bare `Linear(in, 1)` on the post-final-norm hidden
  state. Callers apply the sigmoid. With `confidence_head_with_markov=true`,
  its extra input is `W1[x_{k-1}]` from the same `nn.Embedding` object as the
  Markov head.
- Markov head. `B = W1 @ W2`, rank 256, added to the base logits with no
  scale or temperature.
- Frozen embed and LM head. The checkpoint holds full, independent
  `embed_tokens.weight` and `lm_head.weight` tensors
  (`tie_word_embeddings: false`). Training value-copied them from the target
  and froze them. The port loads them as ordinary weights, with no tying.
- Anchor and bonus, with no catch-up step. The anchor of round N+1 is
  `verification.next_token` from round N, as in the mini-infer convention.
  The ADR-011 `SpeculativeRunner` needs a catch-up draft forward to resync
  its draft cache after full acceptance. DSpark needs no catch-up step,
  because the drafter discards its block K/V every round.

### 5. Batch size 1 only, for the Stage B/C scope

The `DeepSpec` reference asserts `batch_size == 1`, so no batched path exists
to port. Multi-request batching is Stage D, with its own design, because
masking a bidirectional block across requests is a different problem.

## Alternatives Considered

- Register `Qwen3DSparkModel` in the model registry. Rejected: the registry
  loads one standalone checkpoint, and the drafter never runs alone.
- Reuse the `block_mask` path for drafter attention. Rejected in point 3,
  but it stays the parity-test oracle.
- Tie the drafter embed and LM head to the target to save ~1.48 GiB.
  Rejected: the released checkpoint ships independent copies.
- Port DeepSeek-V4-DSpark. Rejected: its draft module is inside the 284B/1.6T
  V4 checkpoints, and it taps the final decoder layers, which the `DeepSpec`
  harness asserts against.
- Port EAGLE-3. Rejected in the plan: DSpark is newer, and its accepted
  length beats EAGLE-3 by 27-31% on the same training data.
- N-gram/PLD speculation. Out of scope, a separate ADR-011 follow-up.

## Consequences

- Greedy verification stays lossless by the ADR-011 argument, so the Stage B
  golden tests do not change. Confidence truncation at T=0 is also lossless.
  Only the Stage D scheduler at T>0 needs the non-anticipating argument of
  the paper.
- The draft cache is one more hand-rolled state object to keep in sync. Its
  `truncate_to` contract lowers that risk.
- `tap_layers` defaults to `None`, so no existing caller sees a change.
- The position-id and RoPE scheme (point 4) is the detail most likely to give
  wrong logits with no crash. A wrong slice rotates `k_ctx` with the wrong
  positions. The Stage B parity contract compares logits per position, not
  only tokens, to catch it.
- Stage C adds confidence-threshold truncation and an accepted-length
  benchmark. Stage D covers temperature > 0, multi-request batching and the
  batch-global confidence scheduler (Algorithm 1).
  [ADR-031](ADR-031-multi-request-dspark.md) landed the first two in
  `ContinuousScheduler`. The confidence scheduler and `PDScheduler` support
  remain deferred.

## Validation (done)

Two layers pass. On CPU, `tests/unit/test_dspark_drafter_parity.py` compares
per-position base, Markov-corrected and confidence logits with
re-transcribed reference formulas, on random weights. On GPU,
`scripts/modal_dspark_parity.py` runs our drafter and the `DeepSpec` one on
real weights (Qwen3-4B + `dspark_qwen3_4b_block7`, one L4), with the same
injected context.

On the GPU, one round gives identical draft tokens (7/7 positions). So do all
four rounds of a real greedy loop (accepting 1, 2, 3, 2 of 7). The worst
element of each bf16 tensor differs by ~1 bf16 ULP, and `mean_abs_diff` is
20-100x below `max_abs_diff`. In fp32 the deltas shrink by 1600-6700x, to
`rel_l2` ~2e-6. A wrong formula does not shrink with mantissa width. Raw
numbers: `docs/benchmarks/data/dspark-parity-{raw,fp32}.json`.

`DeepSpec` also runs under transformers 5.14.1, not only its pinned 5.10.2.
The Qwen3 target forward against HF ran in bf16 only, because two fp32 copies
of the 4B target do not fit an L4.
