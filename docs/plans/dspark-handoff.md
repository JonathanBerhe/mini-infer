# DSpark work: session handoff

Written 2026-07-28 at the end of a long session. Everything below is either
committed or stated as not-done. Read this, then
`docs/plans/dspark-multi-request.md`, and you can start implementing.

## Where things stand

- **Branch `dspark-multi-request`**, clean tree, one commit ahead of main:
  `19765b2 Design multi-request DSpark in ContinuousScheduler` (a plan doc
  only, no code).
- **`main` has the whole drafter port already**, merged as
  `74b9e8c` (PR #25, squashed, 17 commits). Branch deleted.
- **All tests pass**: 702 CPU (what CI gates) + 77 `requires_model` including
  the golden suite = 779, zero failures. 38 `gpu`/`requires_cuda` tests are
  unverifiable without a GPU, never run locally.

## What is DONE and merged (Stages A, B, C, and D's first item)

**The drafter** lives in `src/mini_infer/engine/dspark/`: `drafter.py`
(5-layer backbone, KV injection, from_hf/load_weights), `attention.py`
(two-source K/V + asymmetric RoPE), `markov_head.py`, `confidence_head.py`,
`draft_cache.py` (unpaged, truncatable), `proposal.py` (confidence
truncation), `sampling.py` (rejection sampling), `speculative.py` (the
batch-1 `DSparkSpeculativeRunner`, which the multi-request work generalizes).

**Design decisions already made and justified in
`docs/decisions/ADR-027-dspark-drafter-port.md` (Accepted):**
- Lives in `engine/`, NOT the model registry: the drafter can never load
  standalone, so it fails `load_model`'s contract.
- Its own SDPA call, not the shared `packed_attention` `block_mask` path.
  At batch-1 it needs NO mask, because both of the reference's mask conditions
  are vacuous with one block in flight. The mask returns with batching.
- No catch-up step (unlike ADR-011's V1): the drafter discards its own block
  K/V every round regardless of acceptance.
- Loads the checkpoint's own untied embed/lm_head; no tensor tying.

**Validated two ways.** Micro-config CPU tests against re-transcribed
reference formulas, plus `scripts/modal_dspark_parity.py` running DeepSeek's
ACTUAL DeepSpec implementation side by side on real weights: draft tokens
identical in bf16 and fp32, single round and across four accept/reject rounds.
An fp32 rerun shrank the bf16 deltas 1600-6700x to rel_l2 ~2e-6, proving they
were rounding, not a formula error.

**Stage C measured** (`docs/benchmarks/2026-07-27-dspark-accepted-length.md`):
accepted length 6.24 math / 5.10 code / 3.90 chat against the paper's ~5.57 /
~5.12 / ~3.49. Code exact, math and chat 12% above. Both of the paper's
structural claims reproduce: position-1 survival 0.94 on math (beats DFlash's
0.88) and per-step conditional acceptance flat to position 7 (no tail decay).
Confidence threshold 0.50 removes 38% of verify width for 6% of tau.

**Rejection sampling** (temperature > 0) is implemented and tested
algebraically, by Monte Carlo, and adversarially.

## Hard-won facts. Do not rediscover these.

**Three hypotheses about the Stage C gap were tested and REFUTED. Do not
re-propose them.** Chat templating alone LOWERS tau (-3/-12/-6%); tripling the
generation budget buys 3%; temperature 1.0 costs 1-4%.

**The real cause was a harness bug**: Qwen3's chat template defaults to
`enable_thinking=True`, so the target emitted `<think>` blocks while the
drafter is trained on `--disable-thinking` responses. DeepSpec's evaluator
hardcodes `enable_thinking=False`. One flag moved tau +65/+62/+27%. It also
explains why templating had looked harmful: raw text never invokes the
template, so it never triggered thinking mode. `Tokenizer.encode_chat` now
takes `enable_thinking` and forwards it only when set.

**A retracted argument**: `1 - TV` does NOT generally exceed
`P(argmax match)`. A well-matched drafter does BETTER under greedy, because
greedy pays full credit whenever its argmax is right (~94% at position 1 on
math). The Stage C writeup retracts this explicitly; do not resurrect it.

**bf16 spec-vs-baseline divergence is expected, not a bug.** Verifying at
`q_len > 1` versus decoding at `q_len == 1` uses different matmul shapes and
reduction orders; one flipped near-tie cascades into a different suffix. fp32
is the strict oracle (ADR-011's stance, and how vLLM/SGLang treat it).
`tests/unit/test_dspark_speculative_real_model.py` encodes this: fp32 exact
across all-rejected / partial-accept / full-accept, bf16 characterized only.

**Benchmark harness gotchas.** Modal re-imports the module remotely WITHOUT
the caller's env, so env-var knobs silently revert to container defaults; pass
them as function arguments (there is now a check comparing the signature
against the call site, because this bit three times). Subsetting the baseline
makes any raw seconds ratio compare different workloads, so normalize
per-prompt. Results checkpoint to a Modal Volume because a dropped client
heartbeat used to discard a 90-minute run; use `modal run --detach`.

**Modal runs need explicit approval each time** per the user's standing
preference; debug locally on CPU first. A meta-device state-dict key diff
against the checkpoint's safetensors header costs nothing and catches loader
mismatches before any GPU spend.

## What is NEXT: multi-request in ContinuousScheduler

`docs/plans/dspark-multi-request.md` is the design. It was adversarially
reviewed against the code; the first draft had **six blockers** and the plan is
the revised version. Read it in full, but the load-bearing part is:

**One invariant governs everything:**

```
target_slot_seq_len == draft_cache_len + context_rows        (INV)
```

because the drafter's RoPE rotates keys against an UNSLICED cos/sin table
(`engine/dspark/attention.py:48`, `k_embed = (k * cos)`). Three of the six
blockers are (INV) violated by different routes, and none fails cleanly: two
kill the engine thread (only `OutOfMemoryError` is caught), one silently
rotates the injected context against wrong absolute positions.

**The six blockers and their fixes, all folded into the plan:**
1. Prefix-cache hits shorten the tap context. `__init__` raises when
   `dspark is not None and prefix_cache is not None`.
2. An OOM retry derives `start` from counters a partial `append_kv_packed`
   already inflated. Snapshot lengths, roll back both caches in
   `_preempt_on_oom` BEFORE `_reap_done` (pre-shift indices), clear `offered`.
3. Re-drafting after rollback. Guard on `drafted_for_start == start`; sound
   because the proposal is a pure function of `(anchor, context, start)`.
4. The tap sink is ONE packed tensor for the whole batch. Slice by
   `cu_seqlens_q`, iterate `target_layer_ids` in CONFIG ORDER (fc was trained
   against that concat order), keep the batch dim, `.clone()` (a view pins the
   whole activation).
5. The verify logits need the same per-request offset. **This is the most
   dangerous item**: without it, verification reads another request's logits
   and emits plausible wrong tokens with NO error.
6. The parity gate as first written is unachievable at bf16, and there is real
   feedback coupling (one request's `total_q` depends on others' accepted
   counts). Split into exact-bookkeeping / exact-fp32 / statistical-bf16.

**A correction worth remembering**: admission headroom must NOT be multiplied
by gamma. A round appends more than it commits and truncation returns the
difference, so blocks per EMITTED token are unchanged; only the intra-step
transient is new. Correct form is
`decode_headroom_blocks + ceil((gamma + 1) / block_size) + 1`. Multiplying
would throttle the concurrency the benchmark exists to measure.

**An EOS conflict that would diverge silently**: the scheduler DROPS the EOS
token (`_finish` then `continue`, `continuous_scheduler.py:356-358`) while
`run_greedy` APPENDS it, with a test pinning that
(`tests/unit/test_dspark_speculative.py:294`). Follow the scheduler's
convention in the new path.

**Verified true against the code** (I checked each personally, do not
re-verify): the two-phase `append_kv_packed`; `truncate_to` is shrink-only and
idempotent so rollback is unconditionally safe; `test_scheduler_oom.py` CANNOT
reach the partial-append state because its fake runner raises before any
append, so a test placed there would pass vacuously; `forward_step` returns
last-position logits only, so verification must use `forward_step_packed`.

**One proof of safety**: `truncate_to`'s published-prefix-block guard provably
cannot trip on the spec path, since committed length only ever grows past the
prompt. No need to design around it.

### Staging (each with its own gate)

1. **Variable tokens per step, NO drafter.** Per-request token lists in
   `_sample_decoders` and `_packed_forward`; switch to `forward_step_packed`.
   Gate: plain path unchanged including the golden suite. Do this first
   precisely so the golden suite validates the risky shared-code change in
   isolation.
2. **Drafter state and the draft/verify phases**, with the point-5 slices and
   the point-8 emission rules (clamp `max_tokens`, drop EOS, truncate to
   `start + len(emitted)`, skip truncation for finished requests).
3. **OOM rollback and idempotence**, with a NEW test (two speculators, raise
   mid-append) since the existing one cannot reach it.
4. **Benchmark**: accepted length and throughput vs concurrency on the
   open-loop harness, which doubles as the SPS(B) curve the batch-global
   scheduler needs. Needs stats plumbing first.

### Deferred, with reasons in the plan

Batch-global confidence scheduler (Algorithm 1) and SPS(B) profiling; batched
drafter forwards (v1 loops per request, and staging 4 should instrument
per-phase time to decide it by measurement); top-k/top-p under speculation;
transactional K/V append; `PDScheduler`, which has the same one-token
assumption; trace/stats plumbing, including that
`DSparkStats.confidence_observations` is unbounded.

## Conventions to respect

No em dashes anywhere. No `Co-Authored-By` trailers. Commit titles under ~70
chars, bodies 2-5 lines, no "Phase N"/"Slice N" references, no Tests/Cost
sections. Validation order: `uv run ruff format .`, `ruff check .`,
`mypy src/`, `pytest tests/unit/ -v`, golden tests if engine logic changed.
Benchmarks need real long prompts, not `paragraph * N`. No budget figures in
commits, ADRs, or docs.
