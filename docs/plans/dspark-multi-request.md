# Plan: multi-request DSpark speculative decoding in ContinuousScheduler

Date: 2026-07-28
Revised: 2026-07-28 after an adversarial review against the code found six
blockers in the first draft; the review notes are folded in below and the
things it refuted are marked so they do not get re-proposed.
Status: Proposed

Stage D of [dspark-evaluation.md](dspark-evaluation.md), continuing
[ADR-027](../decisions/ADR-027-dspark-drafter-port.md). ADR-011 listed
"spec-decode as an option inside `ContinuousScheduler`'s step loop" as a
follow-up; this is that, for the DSpark drafter.

It is the prerequisite for the piece that differentiates: the paper's
batch-global confidence scheduler sets verification lengths ACROSS in-flight
requests against a profiled capacity curve, so it has nothing to optimize
until more than one request is in flight. vLLM merged the drafter and
explicitly excluded that scheduler; no engine has an open implementation.

## What blocks it today

`ContinuousScheduler` assumes one token per request per step in four places:

1. `_sample_decoders` samples one token from `last_logits` and returns
   `dict[id(req), int]`.
2. `_packed_forward`'s decode branch appends one token, advancing
   `cu_seqlens_q` by 1.
3. It calls `runner.forward_step`, which returns only each request's
   LAST-position logits. Verification needs every position.
4. `_admit_waiting` reserves 8 blocks of decode runway, sized for one token
   per step.

There is also a sequencing mismatch. Today the engine samples at the START of
step N+1 from logits produced by step N, so emission precedes the forward.
Speculative decoding cannot work that way: the draft and its verification
belong to the same step, and what to emit is only known after the verify.

## The invariant everything hinges on

The drafter requires, at every draft call:

```
len(position_ids) == context_rows + block_size
```

because `apply_dspark_rotary_pos_emb` rotates keys of length
`ctx_len + q_len` against an unsliced cos/sin table
(`engine/dspark/attention.py:47-48`). `_propose` builds that span as
`arange(draft_cache.get_seq_length(), start + block_size)`, so equivalently:

```
target_slot_seq_len == draft_cache_len + context_rows          (INV)
```

Three of the six blockers below are the same violation of (INV) reached by
different routes, and none of them fails cleanly: two crash the engine
thread, one silently rotates the injected context with wrong absolute
positions. Every design point that touches cache lengths exists to preserve
(INV), and it is worth asserting at the top of the draft call rather than
trusting the arithmetic.

## Design

### 1. Opt-in, and refuse the prefix cache outright

`ContinuousScheduler(..., dspark: DSparkSpec | None = None)`. With `None`,
every path taken is today's; spec branches guard on a per-request field.

**`__init__` raises when `dspark is not None` and
`runner.block_pool.prefix_cache is not None`.** A prefix-cache hit
pre-populates the slot and sets `tokens_prefilled` non-zero
(`continuous_scheduler.py:331-339`), so those prompt positions never pass
through a forward and never produce taps. `start` is still `len(prompt)` while
`context_rows` is short, violating (INV) and killing the engine thread, since
only `OutOfMemoryError` is caught. The evaluation plan already required
"keep the drafter path prefix-cache-off in v1" and the first draft of this
design silently dropped it.

Raising at construction is chosen over per-request downgrade because it is the
loudest. Injecting only the post-prefix suffix would also satisfy (INV) but
feeds the drafter less context than it trained on, so it is a measurement, not
a fallback.

### 2. Per-request state in one field

`RunningRequest` gains `spec: SpecDecodeState | None`, keeping
technique-specific fields out of a type every scheduler shares.

```python
@dataclasses.dataclass
class SpecDecodeState:
    draft_cache: DSparkDraftCache
    context: torch.Tensor | None      # tapped target states, cloned
    anchor: int | None
    offered: list[int] | None         # None = no round in flight
    draft_probs: torch.Tensor | None  # needed by _verify at temperature > 0
    confidence: torch.Tensor | None
    start: int | None                 # target seq_len this round was drafted at
    drafted_for_start: int | None     # round identity, for OOM idempotence
    stats: DSparkStats
```

`start` and `drafted_for_start` are not bookkeeping niceties; they are what
makes points 6 and 7 work.

### 3. Eligibility is a predicate, and mixed batches are mandatory

Speculation is refused per request when
`top_k > 0 or top_p < 1.0`, because the spec path builds distributions through
`logits_to_probs` (`engine/dspark/sampling.py`) which applies temperature only.
Threading top-k/top-p through both the draft and target sides is deferred;
silently ignoring them would change a caller's sampling semantics.

That makes **mixed spec / non-spec batches mandatory for correctness, not
convenience**: an ineligible request must still be servable in the same batch.
So `_sample_decoders` keeps its current behavior for non-spec decoders and the
spec phases run alongside it. Decided once at admission, logged at INFO on
downgrade.

### 4. The restructured step

```
1. _cancel_pending                    unchanged
2. _admit_waiting                     headroom per point 5
3. _draft_speculators                 for spec DECODING requests:
                                        - if spec.anchor is None: sample from
                                          last_logits through the request's own
                                          SamplingParams, emit, run EOS /
                                          max_tokens checks, set anchor
                                        - if spec.drafted_for_start == start:
                                          reuse the existing proposal (point 7)
                                        - else draft; clamp len(offered) to the
                                          remaining max_tokens room
   _sample_decoders                   non-spec decoders only, unchanged
4. _reap_done                         unchanged
5. _packed_forward                    prefill chunks + 1 token per non-spec
                                      decoder + (1 + len(offered)) per spec
                                      decoder; forward_step_packed for
                                      all-position logits; taps collected when
                                      any request speculates
6. _verify_speculators                per request with spec.offered is not None:
                                      slice its logits, accept prefix, pick
                                      bonus, clamp+emit, truncate its slot,
                                      refresh context and anchor, clear offered
```

Steps 3 and 6 are no-ops when nothing speculates.

**Membership is keyed off `spec.offered is not None`, never off `state`.** The
distribution loop in `_packed_forward` flips `PREFILLING -> DECODING`
mid-iteration (`continuous_scheduler.py:410-419`), so a request that just
finished prefill is DECODING by the time step 6 runs while having no proposal.
Keying off `state` would make `_verify` read the argmax at that request's
chunk's FIRST position and emit a garbage token.

### 5. Slicing the packed forward per request

Both slices below were unspecified in the first draft and are the two most
dangerous items in the review, because neither fails loudly.

**Taps.** `hidden_state_sink[layer]` is one `(1, total_q, hidden)` tensor
covering EVERY request's q-tokens (`models/qwen3.py:145-147`). Per request:

```python
lo, hi = cu_seqlens_q[b], cu_seqlens_q[b + 1]
rows = torch.cat([sink[i][:, lo:hi, :] for i in cfg.target_layer_ids], dim=-1).clone()
assert rows.shape[1] == hi - lo
```

Two constraints: iterate `target_layer_ids` in CONFIG ORDER, since `fc` was
trained against that concatenation order and a sorted or set-ordered concat
silently feeds it permuted features; and keep the leading batch dim. The
`.clone()` is load-bearing, an un-cloned slice is a view that pins the whole
packed activation (the same defect `last_logits` has today at
`model_runner.py:160-164`).

**Verify logits.** Per request:

```python
lo = cu_seqlens_q[b]
mine = packed[:, lo : lo + 1 + len(offered), :]
assert mine.shape[1] == 1 + len(offered)
```

Without the offset, verification reads another request's logits and emits
plausible-looking wrong tokens with no error at all.

**Context lifecycle.** Round 1 accumulates the full prompt's taps across
chunks; every later round REPLACES context with the freshly verified window,
so the buffer shrinks to at most `gamma + 1` rows after the first verify and
stays there. Both destinations are fed from the same forward in a mixed batch,
which is why the destination is chosen per request from its phase snapshot
rather than from a single rule.

**Bound.** Tap memory lives OUTSIDE the block pool's accounting, so its
exhaustion is a plain host/GPU OOM that `_preempt_on_oom` cannot recover.
At Qwen3-4B shapes (hidden 2560, 5 taps, bf16) a row is 25.6 KB, so a
512-token prompt is ~13 MB and 16 concurrent prompts ~210 MB; the operative
bound is not that example but the longest prompt the pool admits. v1 caps the
prompt length eligible for speculation and states in point 8 that a
tap-buffer OOM is unrecoverable.

### 6. Abandon-round rollback on OOM

`append_kv_packed` advances `_num_tokens` and allocates for every slot at
`layer_idx == 0` before writing any K/V (`paged_kv_cache.py:519-529`), so a
raise at slot j leaves slots `0..j-1` advanced with no data. `_preempt_on_oom`
rolls nothing back and `_engine_loop` re-enters `_step` from the top.

For a speculator that is not merely stale K/V: the retry derives `start` from
the inflated `seq_lens` while `context` and `draft_cache` are unchanged,
violating (INV) and raising a `RuntimeError` that is not caught. **A
recoverable OOM becomes a dead engine thread**, and this has no batch-1
analogue.

Fix, as a numbered design point rather than a footnote: `_packed_forward`
snapshots `cache_seq_lens` (it already computes them at line 387) and each
`SpecDecodeState`'s pre-draft `draft_cache.get_seq_length()`. `_preempt_on_oom`
then, BEFORE `_reap_done` so indices are still pre-shift, truncates both caches
back for every surviving spec request and clears `offered`. `truncate_to` is
shrink-only and idempotent (`paged_kv_cache.py:144-171`) and the target is
always above `published_threshold`, so this is safe to apply unconditionally.

This is far cheaper than the transactional append that would fix the
underlying two-phase behavior, which stays out of scope.

### 7. Re-drafting on retry must be idempotent

After a rollback the retry would draft a second time for the same round.
Guard step 3 on `spec.drafted_for_start == start and spec.offered is not None`
and reuse the existing proposal, clearing it only in step 6. This is sound
because the proposal is a pure function of `(anchor, context, start)`.

### 8. Emission rules, stated because the two paths disagree

- Clamp to `max_tokens - len(tokens_generated)`; without this a spec request
  overshoots `max_tokens` by up to `gamma`.
- Stop BEFORE the first EOS and do NOT emit the EOS token. This is the
  scheduler's convention (`continuous_scheduler.py:356-358`) and the OPPOSITE
  of `run_greedy`, which appends EOS to its output and has a test pinning that
  (`tests/unit/test_dspark_speculative.py:288-292`). A verbatim port would
  make the spec path emit a token the plain path drops, a mismatch invisible
  in decoded text.
- Truncate the slot to `start + len(emitted)`, not `start + len(committed)`.
- Skip truncation entirely for a request that finished, its blocks are about
  to be freed wholesale.
- Clamp `len(offered)` at draft time too, so the verify forward does not
  append K/V for positions that can never commit.

### 9. Admission headroom: the first draft's reasoning was wrong

The first draft multiplied headroom by `gamma + 1`, arguing the runway drains
`gamma` times faster. **That is incorrect.** A round appends `1 + offered`
positions but commits only `accepted + 1`, and truncation returns the
difference, so steady-state blocks per EMITTED token are unchanged by
speculation. What is new is only an intra-step transient of the in-flight
chunk. Multiplying headroom would reserve a large constant per request and cut
achievable concurrency, which is the axis staging step 4 exists to measure.

Correct form:

```
decode_headroom_blocks + ceil((gamma + 1) / block_size) + 1
```

Any slack beyond that should be argued as a probability of reaching ADR-024's
cancel path, which cancels its victim outright, not as a drain rate.

## Staging

1. **Variable tokens per step, no drafter.** Per-request token LISTS in
   `_sample_decoders` and `_packed_forward`, switch to `forward_step_packed`.
   Gate: plain path unchanged, including the golden suite.
2. **Drafter state and the draft/verify phases**, with the slices of point 5
   and the emission rules of point 8.
3. **OOM rollback and idempotence.** `tests/unit/test_scheduler_oom.py` today
   OOMs the first forward of a single request that has no `last_logits`
   (lines 144-186), so it cannot reach the partial-append state; a test there
   would pass vacuously. Needs two speculators in flight with a mid-append
   raise.
4. **Benchmark.** Accepted length and throughput against concurrency on the
   open-loop harness, which doubles as the SPS(B) curve the batch-global
   scheduler needs. Requires stats plumbing (point below) and a `num_blocks`
   chosen so headroom is not the binding constraint.

## Gate, split three ways

The first draft's gate ("two concurrent requests each produce exactly what
they produce alone at temperature 0") is unachievable at bf16, and would have
fired its own stop-condition on benign numerics. Two reasons: a request's
logits depend on the batch's `total_q` shape, which ADR-011 and the Stage C
benchmark both measured; and worse, there is genuine feedback coupling, since
one request's `total_q` contribution depends on OTHER requests' accepted
counts, so the batch composition is not even stable across runs.

1. **Bookkeeping, exact.** Against a deterministic fake runner: committed
   tokens, cache lengths, (INV), and emission clamps.
2. **Numerics, exact.** CPU fp32, real small model, two concurrent requests
   versus each alone.
3. **GPU bf16, statistical only.** Accepted length and coherence, explicitly
   NOT per-token identity, referencing
   `tests/unit/test_dspark_speculative_real_model.py` for the caveat.

## Deferred, with the reason

- Batch-global confidence scheduler (Algorithm 1) and SPS(B) profiling. This
  plan is its prerequisite.
- Batched drafter forwards. v1 loops the drafter once per request: batching it
  needs the block mask that batch-1 avoids (ADR-027 point 3) plus padded
  ragged contexts, and the target's verify dominates the step. The review
  correctly notes this leaves a serial latency floor, so staging step 4 should
  instrument per-phase time to decide it by measurement.
- Top-k / top-p under speculation (point 3).
- Transactional K/V append (point 6).
- `PDScheduler`, which has the same one-token assumption.
- Trace/stats: `_classify_phase` has no notion of a verify step and there is
  no channel for `DSparkStats`, whose `confidence_observations` is unbounded
  and must not accumulate per request indefinitely. Needed before step 4.
- Dropping `spec` state at completion: `RequestHandle` keeps the
  `RunningRequest` alive, so the drafter cache would outlive the request.
