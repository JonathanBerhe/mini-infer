# ADR-027: DSpark drafter port (Stage A design study)

Date: 2026-07-26
Status: Proposed

## Context

DeepSeek released DSpark ("Confidence-Scheduled Speculative Decoding
with Semi-Autoregressive Generation", DeepSeek-AI + Peking University,
2026-06-27) alongside the MIT-licensed `DeepSpec` training/eval
codebase. `docs/plans/dspark-evaluation.md` scopes a staged evaluation;
this ADR is Stage A's deliverable: resolve the parity-critical
unknowns and decide how a from-scratch port fits mini-infer, before
any code lands (Stage B).

ADR-011 shipped greedy, fixed-K=4, two-model speculative decoding and
found the exact failure mode DSpark targets: on H100, a fixed draft
length won 1.29x on a slow prompt and lost 0.83x on fast ones, netting
1.00x, because the draft/verify overhead isn't worth paying on every
step. DSpark's confidence-scheduled truncation is the principled fix,
and ADR-011 already lists "adaptive K" as a follow-up.

DSpark separates into two contributions: a semi-autoregressive
drafter (parallel block proposal + a sequential Markov correction) and
a batch-global confidence scheduler. This ADR covers only the drafter
(Stage B/C scope); the scheduler is Stage D, deferred.

Released artifacts make the drafter portable without training:
`dspark_qwen3_4b_block7` (BF16, ~1.39B params, block_size=7,
5 draft layers, `target_layer_ids=[1,9,17,25,33]`) pairs with
Qwen3-4B as the target. Drafter training itself stays out of scope
(the `DeepSpec` pipeline needs a ~38 TB target hidden-state cache).

## Decision

Port the Qwen3 DSpark drafter as a from-scratch module living
alongside the engine's orchestration code, not in the model registry,
with the following mechanics, each resolved against `DeepSpec`'s
actual code (not the paper's prose) and adversarially re-verified:

### 1. Module location: `src/mini_infer/engine/dspark/`, not `models/`

`src/mini_infer/models/` is built around `ModelRegistry` +
`load_model(name)`: one HF `config.json`'s `architectures[0]` string
dispatches to one class that loads and runs standalone
(`src/mini_infer/models/__init__.py:24-103`). The drafter fails that
contract on both ends: it can never run standalone (every forward
needs injected target hidden states), and nothing else in the engine
is scheduled by pairing two architecture strings. `SpeculativeRunner`
(`src/mini_infer/engine/speculative.py`) already owns the "orchestrate
two coupled models + their caches" job for ADR-011's two-model V1;
the DSpark drafter fits that same layer, just with a smaller, coupled
drafter instead of a second standalone model.

Layout:
- `engine/dspark/drafter.py`: the `Qwen3DSparkDrafter` module (5-layer
  backbone, KV injection, Markov head, confidence head). Plain
  `nn.Module` with a `from_pretrained(drafter_name, target_config)`
  loader; not `BaseCausalLM`, not `@register_model`.
- `engine/dspark/draft_cache.py`: the drafter's own small unpaged
  cache (see point 5).
- `engine/dspark/attention.py` (or inline if small): the bespoke
  bidirectional-block attention.
- `engine/dspark_speculative.py` (or a new class in `speculative.py`):
  the orchestration loop, replacing the K-serial-draft-forward phase
  of ADR-011's `SpeculativeRunner` with one drafter forward producing
  a whole `gamma`-token block.
- One additive change to `models/qwen3.py`'s `Qwen3ForCausalLM.forward`
  (point 2). No other file in `models/` changes.

### 2. Target hidden-state taps: optional kwarg, zero-cost when unused

Add `tap_layers: frozenset[int] | None = None` and a sink (dict or
callback) to `Qwen3ForCausalLM.forward`, defaulting to `None`. Inside
the existing layer loop (`qwen3.py:128-129`), one guarded line records
`x` (the post-block hidden state) when the current layer index is in
`tap_layers`. This keeps every existing caller's contract
untouched: `forward()` still returns exactly `logits`
(`(1, total_q, vocab_size)`, per `BaseCausalLM`'s documented contract
in `models/base.py:62-70`), so `ContinuousScheduler`, the golden
tests, and the two-model V1 path see no change. An
always-collect-and-return-a-dict design was rejected: it would change
the return shape for every caller and tax every plain Qwen3 request
with hidden-state bookkeeping it never asked for.

### 3. Drafter attention: bespoke additive mask, not the shared `block_mask` path

`packed_attention_torch`'s per-request `block_mask` (added for
MiniMax-M3 MSA) looks tailor-made for the drafter's bidirectional
block attention, but it's the wrong fit: the dispatcher sources K/V
exclusively from a `PagedKVCache`
(`cache/packed_attention.py:124-135`) with no hook for the drafter's
second K/V source (the projected, injected target context), and it's
torch-backend-only. The drafter builds its own additive mask on a
plain `F.scaled_dot_product_attention` call instead.
`packed_attention_torch(block_mask=...)` stays useful as the unit-test
oracle, the same role it plays for MSA's parity tests
(`tests/unit/test_minimax_m3_parity.py`).

### 4. Confirmed mechanics (bit-parity-critical, verified against `DeepSpec` source and adversarially re-checked)

- **Position ids and RoPE**: sequential, anchor at offset 0, the
  `gamma-1` mask positions at offsets 1..gamma-1
  (`deepspec/modeling/dspark/common.py::create_position_ids`).
  Positions are global and absolute, one running counter over the
  whole generation, never reset per round. The elegant part: because
  "context" (the just-verified, newly-committed tokens) and "draft"
  (the new block) occupy *contiguous* positions in that global
  counter, `deepspec/eval/dspark/draft_ops.py::forward_dspark_draft_block`
  gets both in a single slice,
  `position_ids[:, past_key_values_draft.get_seq_length():start+block_size]`,
  and feeds that one slice into a single `rotary_emb` call. Inside
  attention (`qwen3/modeling.py`'s `Qwen3DSparkAttention.forward`),
  `k = cat([k_ctx, k_noise])` is rotated in one `apply_rotary_pos_emb`
  call using that shared cos/sin (`k_embed = k * cos`, unsliced,
  which is exactly why the slice must span both segments); `q` (draft
  positions only) uses the last `q_len` rows of the same cos/sin. I
  verified this by direct calculation: the elementwise multiply in
  `apply_rotary_pos_emb` only type-checks if cos/sin's length equals
  k's length (`ctx_len + q_len`), which forced tracing the exact slice
  math to confirm it produces that length. Not a shortcut worth
  reproducing loosely; the port's position-id construction should
  mirror this contiguous-slice trick exactly, since getting it wrong
  silently rotates `k_ctx` with the wrong absolute positions.
- **Bidirectionality**: attention within a block is genuinely
  bidirectional (`is_causal=False`, and the training-time
  `dspark_mask_mod` grants same-block visibility with no q-vs-kv
  ordering term). Causal structure is reintroduced only by the Markov
  head's sequential sampling loop
  (`VanillaMarkov.sample_block_tokens`), which conditions each step's
  bias on the *previous step's sampled token*, not on the base logits'
  own (bidirectional) computation.
- **Hidden-state scope and the side-cache's actual shape**: round 1
  injects the target's hidden states for the *entire prompt*
  (`deepspec/eval/dspark/evaluator.py::_init_context`, over the
  target's full prefill `output_hidden_states`). Every later round
  injects only the just-verified window (`<= block_size + 1` rows,
  `_update` does a full *reassignment*, not an append). This
  resolves the plan's stop-gate question: the side state does **not**
  require re-deriving hidden states over a growing prompt window each
  round (that would have been the "architecturally ugly" case the
  plan's gate was checking for). What actually grows with sequence
  length is the drafter's own K/V cache: `k_ctx`/`v_ctx` (the fc-projected
  target injections) accumulate permanently in
  `past_key_values_draft` (a `DynamicCache`), while `k_noise`/`v_noise`
  (this round's mask-token K/V) get discarded every round via
  `past_key_values_draft.crop(start)`, *unconditionally*, regardless
  of how many draft tokens were accepted. This is a small, unpaged,
  per-layer, per-request K/V cache that grows at exactly the same
  linear-in-generated-length rate the target's own `PagedKVCache`
  already does, just uncompressed by paging (5 layers, batch 1, small
  post-fc hidden size). The gate does not trigger; the "side cache" in
  the plan is this drafter-owned K/V cache, and it needs a
  `truncate_to`-shaped API mirroring `PagedKVCache.truncate_to`'s
  contract, implemented as plain tensor slicing rather than block
  tables.
- **Confidence head**: consumes the *post-final-norm* hidden state,
  the same tensor the drafter's own `lm_head` and Markov head consume
  (no pre-norm variant exists). It is a bare `Linear(in, 1)` with *no*
  sigmoid inside the module (sigmoid is applied only at each call
  site: `BCEWithLogitsLoss` in training, explicit `.sigmoid()` at
  inference truncation). When `confidence_head_with_markov=true` (the
  released config), the confidence head's extra input is `W1[x_{k-1}]`
  using the *literal same* `nn.Embedding` object the Markov head owns,
  not a second table of the same shape.
- **Markov head**: `B = W1 @ W2`, rank 256, applied strictly additively
  to the base logits with no scale or temperature factor anywhere in
  the bias path (the `temperature` that does appear is ordinary
  sampling temperature on the corrected logits, unrelated).
- **Frozen embed/LM-head**: resolves the plan's open question
  cleanly. The released checkpoint's safetensors *contain* full,
  independent `embed_tokens.weight` / `lm_head.weight` tensors
  (`tie_word_embeddings: false`); `initialize_embeddings_and_head` is a
  one-time value **copy** at training-build time
  (`.copy_()` into the drafter's own pre-allocated parameters, then
  frozen), not a tied/shared Parameter. At load time there is no
  tying step at all, just an ordinary weight load. The port should do
  the same: load the drafter's own embed/lm-head tensors as ordinary
  weights, with no cross-model tensor-sharing plumbing.
- **Anchor/bonus convention and no catch-up step needed**: the anchor
  fed into round N+1 is exactly `verification.next_token` from round
  N (the resampled correction at the first rejection, or the bonus
  token if every draft token was accepted), matching mini-infer's own
  bonus-token convention. Unlike ADR-011's V1 `SpeculativeRunner`,
  which needs an explicit catch-up draft forward to resync the draft
  cache after full acceptance (because its draft is an ordinary
  causal model whose KV cache must contain an entry for the bonus
  token), DSpark's drafter needs **no catch-up step**: its own
  self-attention KV (`k_noise`/`v_noise`) is unconditionally discarded
  every round before the next one starts, independent of the
  acceptance outcome. This is structurally simpler than the V1 loop it
  sits alongside.

### 5. Batch size 1 only, matching Stage B/C scope

`DeepSpec`'s own reference asserts `batch_size == 1` throughout
(`generate_decoding_sample`, `build_dspark_proposal`); there is no
padded-batch or multi-request path to port from. This matches the
plan: Stage B/C are batch-1; multi-request integration is Stage D,
deferred, and will need its own design (padding/masking a bidirectional
block attention across requests is a different problem than the
verify-side packing `ContinuousScheduler` already does).

## Alternatives Considered

- **Register `Qwen3DSparkModel` in the model registry** like every
  other architecture. Rejected: the registry's whole contract
  (`load_model` loading one self-describing checkpoint standalone)
  doesn't apply to a module that can never run without a paired
  target; forcing it in would make `REGISTRY.lookup` need a carve-out
  no other entry needs.
- **Reuse `packed_attention_forward`'s `block_mask` path** for the
  drafter's attention. Rejected: it only reads K/V from a
  `PagedKVCache`, with no injection point for the second K/V source
  the drafter needs, and it's torch-backend-only by construction.
  Kept as the parity-test oracle instead.
- **Tie the drafter's embed/LM-head to the target's own tensors** at
  load time, to save the ~1.48 GiB the duplicated copies cost.
  Rejected: the released checkpoint doesn't do this (ships full,
  independent copies, `tie_word_embeddings: false`), and tying would
  be a design decision diverging from what was actually
  trained/shipped, for a memory saving that doesn't matter at this
  checkpoint's scale.
- **Port DeepSeek-V4-DSpark instead of the Qwen3 drafter.** Rejected,
  per the plan: the draft module rides inside the 284B/1.6T V4
  checkpoints (unportable at this project's scale), and its config
  taps the final decoder layers, which `DeepSpec`'s own OSS harness
  explicitly asserts against, an unresolved inconsistency not worth
  inheriting.
- **Port EAGLE-3 instead of DSpark.** Considered in the original
  evaluation plan and rejected there: no open serving engine has a
  confidence-scheduler implementation for either method, but DSpark
  is the more recent, better-differentiated target, and its offline
  results already beat EAGLE-3's accepted length by 27-31% on the
  same training data.
- **N-gram/PLD speculation.** Out of scope for this ADR; already
  listed as a separate, complementary follow-up in ADR-011.

## Consequences

- **Positive**: no catch-up-step bookkeeping (simpler than ADR-011's
  V1 loop in that respect). The frozen-weight question resolves to
  "just load it," removing a planned cross-model tensor-tying
  mechanism before it was ever built. Temperature-0 verification stays
  provably lossless by the same argument as ADR-011 (greedy accept is
  argmax equality), so Stage B's golden tests are untouched and
  truncating the confidence-scheduled verification length at T=0 is
  trivially lossless too, only Stage D's batch-global scheduler at
  T>0 needs the paper's non-anticipating argument.
- **Negative**: the drafter's own K/V cache is a new, hand-rolled
  unpaged cache type, not reusable from `PagedKVCache`, so it is one
  more state object to keep in sync manually (mitigated by giving it
  the same `truncate_to`-shaped contract). The hidden-state tap kwarg
  on `Qwen3ForCausalLM.forward` is new surface area; if a second
  family ever needs the same trick, extract a shared helper then
  rather than duplicating the guarded-loop pattern.
- **Risk**: the position-id/RoPE scheme (point 4) is genuinely subtle
  and the single detail most likely to produce silently-wrong logits
  rather than a crash if implemented from memory instead of from the
  cited slice logic. Stage B's parity contract (per-position logit
  comparison, not just end-to-end tokens) exists specifically to catch
  this class of bug.
- **Reversibility**: additive only. The `tap_layers` kwarg on Qwen3 is
  optional and defaults to `None`; nothing in `models/`, the scheduler,
  or `PagedKVCache` is modified in a way existing callers would notice.

## Validation

Per `docs/plans/dspark-evaluation.md` Stage B: random-weight
micro-config CPU unit tests comparing per-position base logits,
Markov-biased logits, and confidence logits against `DeepSpec`'s
reference code, followed by real-checkpoint temperature-0 token-parity
fixtures (Qwen3-4B + `dspark_qwen3_4b_block7`) generated on one short
Modal GPU run. Existing Qwen3 golden tests are unaffected (point 2).

## Pointers

- Plan: `docs/plans/dspark-evaluation.md`
- Prior art in this repo: `docs/decisions/ADR-011-speculative-decoding.md`
  (the two-model V1 this port extends alongside)
- Reference code: `deepspec/modeling/dspark/qwen3/modeling.py`,
  `deepspec/modeling/dspark/common.py`,
  `deepspec/modeling/dspark/markov_head.py`,
  `deepspec/eval/dspark/draft_ops.py`,
  `deepspec/eval/dspark/evaluator.py`,
  `deepspec/eval/base_evaluator.py`
- Checkpoint: `deepseek-ai/dspark_qwen3_4b_block7` (HF)
- MSA mask-oracle precedent: `src/mini_infer/cache/packed_attention.py`,
  `tests/unit/test_minimax_m3_parity.py`

## Follow-ups

- Stage C: confidence-threshold truncation + accepted-length benchmark
  against `DeepSpec`'s bundled eval sets.
- Stage D (deferred): temperature > 0 rejection sampling, multi-request
  batching, and the batch-global confidence scheduler (Algorithm 1),
  which also needs to decide its relationship to `ContinuousScheduler`'s
  ADR-024 OOM preemption and to `PDScheduler`.
