# Plan: multi-request DSpark speculative decoding in ContinuousScheduler

Date: 2026-07-28

Revised: 2026-07-28. A review against the code tried to break the first draft
and found six blockers. Its notes are part of the text below. The ideas that
it disproved are marked, so that nobody proposes them again.

Revised: 2026-10-18. Rewritten in plain English. Staging steps 1, 2, and 3
are done.

Status: In progress. Step 4, the benchmark, is open.

Line numbers in the design part refer to the code at commit `74b9e8c`, before
this work started. The findings of each step describe the code after it.

This plan is Stage D of [dspark-evaluation.md](dspark-evaluation.md), and it
continues [ADR-027](../decisions/ADR-027-dspark-drafter-port.md). ADR-011
listed "spec-decode as an option inside `ContinuousScheduler`'s step loop" as
a follow-up. This plan is that follow-up, for the DSpark drafter.

This work is a prerequisite for the part that sets this engine apart. That
part is the batch-global confidence scheduler of the paper. It sets the verification lengths across all
requests in flight, against a measured capacity curve. So it has nothing to
optimize until more than one request is in flight. vLLM merged the drafter but
left that scheduler out on purpose, and no engine has an open implementation.

## Words used in this plan

The plan uses these terms with one meaning each:

- Speculative decoding: a small model guesses tokens, and the large model verifies them.
- Target: the large model whose output the user receives.
- Drafter: the small DSpark model that guesses the next tokens.
- Round: one drafted block plus its verification.
- Anchor: the last committed token, where a round starts.
- Proposal, or offered tokens: the drafted tokens that a round sends for verification.
- Verify forward: one target forward over the anchor and the offered tokens.
- Accepted prefix: the offered tokens up to the first one that verification rejects.
- Bonus token: the next token of the target after the accepted prefix.
- Commit: verification keeps a token, as an accepted or a bonus token. EOS or max_tokens can still stop its emission.
- Emit: send a token to the client.
- Speculator: a request that uses speculative decoding.
- Plain path: normal decoding, one token per request per step.
- Prefill: the forwards over the prompt, in chunks.
- K/V: the keys and values that attention keeps for each position.
- Slot: the part of the shared KV cache that holds one request.
- Block pool: the fixed memory, in blocks, that holds every slot.
- Packed forward: one target forward over the tokens of all requests.
- Taps: the hidden states of the target at a fixed set of layers.
- Injected context: the taps that the drafter receives as input.
- OOM: out of memory. Mostly the block pool has no free block, and the engine can recover. Point 5 also uses it for host or GPU memory, where the engine cannot recover.
- Preemption: the engine cancels one request, the victim, to free blocks.
- Reap: remove the finished requests and free their blocks.
- Headroom: blocks that admission reserves beyond the prompt.
- Idempotent: safe to run twice, with the same result.
- EOS: the token that ends a sequence.
- gamma: the block length, the number of drafted tokens per round.
- tau: committed tokens per round, the bonus token included.
- Gate: the tests that must pass before the next step starts.
- Mutation: a small deliberate break in the code, to test the tests.

## What blocked it before this work

Before step 1, `ContinuousScheduler` assumed one token per request per step in
four places:

1. `_sample_decoders` sampled one token from `last_logits` and returned
   `dict[id(req), int]`.
2. The decode branch of `_packed_forward` appended one token, so it advanced
   `cu_seqlens_q` by 1.
3. It called `runner.forward_step`, which returns only the logits at the last
   position of each request. Verification needs every position.
4. `_admit_waiting` reserved 8 blocks of decode headroom, sized for one token
   per step.

The order of a step did not fit either. The engine samples at the start of
step N+1 from the logits of step N, so it emits before the forward.
Speculative decoding cannot work that way. The draft and its verification
belong to the same step. The engine knows what to emit only after the verify
forward.

## The invariant everything depends on

An invariant is a rule that must hold at a fixed point in the code. At every draft call, the
drafter requires:

```
len(position_ids) == context_rows + block_size
```

The reason is the position encoding. `apply_dspark_rotary_pos_emb` rotates
keys of length `ctx_len + q_len` against an unsliced cos/sin table
(`engine/dspark/attention.py:47-48`). `_propose` builds that span as
`arange(draft_cache.get_seq_length(), start + block_size)`. So the same rule,
in terms of cache lengths, is:

```
target_slot_seq_len == draft_cache_len + context_rows          (INV)
```

Three of the six blockers below break (INV), each by a different route. None
of them fails cleanly. Two crash the engine thread, and one silently rotates
the injected context with wrong absolute positions.

Every design point that touches cache lengths exists to keep (INV) true. The design asserts it at the
top of the draft call, and does not trust the arithmetic. (Step 2 moved this
assertion. Its findings explain why.)

## Design

### 1. Opt-in, and refuse the prefix cache

`ContinuousScheduler(..., dspark: DSparkSpec | None = None)` turns the feature
on. With `None`, every path is the same as before. Each speculative branch
runs only for a request whose `spec` field is set.

When `dspark is not None` and `runner.block_pool.prefix_cache is not None`,
`__init__` raises. A prefix cache stores the K/V of
prompt beginnings for reuse. A hit in it fills the slot ahead of time and sets
`tokens_prefilled` above zero (`continuous_scheduler.py:331-339`). So those
prompt positions never go through a forward, and they never produce taps.

`start` is still `len(prompt)`, but `context_rows` is too short. That breaks
(INV) and kills the engine thread, because only `OutOfMemoryError` is caught.
The evaluation plan already required "keep the drafter path prefix-cache-off
in v1". The first draft of this design dropped that rule without a word.

An error at construction is the loudest failure, so the design prefers it to a
downgrade per request. Another option is to inject only the suffix after the
cached prefix. That also keeps (INV) true, but it gives the drafter less
context than it saw in training. So it is a measurement, not a fallback.

### 2. Per-request state in one field

`RunningRequest` gets one new field, `spec: SpecDecodeState | None`. Fields
that belong to one technique then stay out of a type that every scheduler
shares.

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

`start` and `drafted_for_start` are not decoration. Points 6 and 7 depend on
them.

### 3. Eligibility is a predicate, and mixed batches are required

When `top_k > 0 or top_p < 1.0`, a request does not speculate. The reason is
that the speculative path builds its distributions with `logits_to_probs`
(`engine/dspark/sampling.py`), which applies temperature only. Support for
top-k and top-p on the draft side and the target side is deferred. To ignore
them silently changes the sampling that the caller asked for.

So a batch that mixes speculators and plain requests is required for
correctness, not only for convenience. A request that cannot speculate must
still run in the same batch. `_sample_decoders` keeps its behavior for plain
decoders, and the speculative phases run next to it. The scheduler decides
once, at admission, and it logs each downgrade at INFO level.

### 4. The new order of a step

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

If no request speculates, steps 3 and 6 do nothing.

The verify phase selects its requests by `spec.offered is not None`, never by
`state`. The distribution loop in `_packed_forward` changes
`PREFILLING -> DECODING` in the middle of the loop
(`continuous_scheduler.py:410-419`). So when step 6 runs, a request that just finished
prefill is DECODING, but it has no proposal. A selection by `state`
makes `_verify` read the argmax at the first position of that chunk, and it
then emits a garbage token.

### 5. Slicing the packed forward per request

The first draft did not specify the two slices below. The review rated them
the two most dangerous items, because neither fails loudly.

The first slice is for taps. `hidden_state_sink[layer]` is one
`(1, total_q, hidden)` tensor that covers the q-tokens of every request
(`models/qwen3.py:145-147`). For each request:

```python
lo, hi = cu_seqlens_q[b], cu_seqlens_q[b + 1]
rows = torch.cat([sink[i][:, lo:hi, :] for i in cfg.target_layer_ids], dim=-1).clone()
assert rows.shape[1] == hi - lo
```

Two rules apply here. First, iterate `target_layer_ids` in the order of the
configuration. `fc` was trained on that order of concatenation, and a sorted
or set order feeds it permuted features without any error. Second, keep the
leading batch dimension.

The `.clone()` is necessary. A slice without it is a
view that keeps the whole packed activation in memory. (`last_logits` has the
same defect at `model_runner.py:160-164`.)

The second slice is for the verify logits. For each request:

```python
lo = cu_seqlens_q[b]
mine = packed[:, lo : lo + 1 + len(offered), :]
assert mine.shape[1] == 1 + len(offered)
```

Without the offset, verification reads the logits of another request. It then
emits wrong tokens that look plausible, and nothing reports an error.

The injected context changes over the life of a request. Round 1 collects the
taps of the full prompt across all chunks. Every later round replaces the
context with the window that the target just verified. So after the first
verify forward, the buffer holds at most `gamma + 1` rows. In a mixed batch,
one forward feeds both kinds of update. So each request picks its update from
its own phase, not from one rule for the batch.

The tap buffer costs memory outside the accounting of the block pool. Its
exhaustion is a plain host or GPU OOM, and `_preempt_on_oom` cannot recover
from it. At Qwen3-4B shapes (hidden 2560, 5 taps, bf16), a row is 25.6 KB. So
a 512-token prompt needs about 13 MB, and 16 concurrent prompts need about
210 MB. The real limit is not that example but the longest prompt that the
pool admits. So v1 (this first version) limits the prompt length that can
speculate, because an OOM of the tap buffer cannot be recovered.

### 6. Roll back the round on OOM

`append_kv_packed` advances `_num_tokens` and allocates blocks for every slot
at `layer_idx == 0`, before it writes any K/V (`paged_kv_cache.py:519-529`).
So a raise at slot j leaves slots `0..j-1` advanced with no data. Before step
3, `_preempt_on_oom` rolled nothing back, and `_engine_loop` started `_step`
again from the top.

For a speculator, this is worse than stale K/V. The retry takes `start` from
the inflated `seq_lens`, but `context` and `draft_cache` stay the same. That
breaks (INV) and raises a `RuntimeError` that nothing catches. A recoverable
OOM becomes a dead engine thread, and the batch-1 path has no such case.

The fix is a numbered design point, not a footnote. `_packed_forward` saves
`cache_seq_lens`, which it already computes at line 387. It also saves the
pre-draft `draft_cache.get_seq_length()` of each `SpecDecodeState`. Then
`_preempt_on_oom` truncates both caches back for every surviving speculator,
and it clears `offered`. It does this before `_reap_done`, while the indices
have not shifted yet.

`truncate_to` only shrinks, and it is idempotent
(`paged_kv_cache.py:144-171`). The target is always above
`published_threshold`. So this is safe to apply to every request.

This costs
far less than a transactional append (one that applies fully or not at all),
which fixes the two-phase behavior at its root. That fix stays out of scope.
(Step 3 rolls back only the target slots and keeps `offered`. Its findings
explain why.)

### 7. A redraft on retry must be idempotent

Without a guard, the retry after a rollback drafts a second time for the same
round. When `spec.drafted_for_start == start and spec.offered is not None`,
step 3 reuses the existing proposal. Only step 6 of the step order clears it.
This is sound, because the proposal is a pure function of
`(anchor, context, start)`.

### 8. Emission rules, stated because the two paths disagree

The speculative path follows these rules:

- Clamp the commit to `max_tokens - len(tokens_generated)`. Without this, a
  speculator overshoots `max_tokens` by up to `gamma`.
- Stop before the first EOS, and do not emit the EOS token. This is the
  convention of the scheduler (`continuous_scheduler.py:356-358`).
- Note that `run_greedy` does the opposite: it appends EOS to its output, and a
  test pins that (`tests/unit/test_dspark_speculative.py:288-294`).
- Do not port `run_greedy` literally. If the speculative path copies it, it
  emits the EOS token that the plain path drops. The decoded text hides that.
- Truncate the slot to `start + len(emitted)`, not `start + len(committed)`.
- Skip the truncation for a request that finished. The reap frees all of its
  blocks next.
- Also clamp `len(offered)` at draft time. Then the verify forward does not
  append K/V for positions that can never commit.

### 9. Admission headroom: the first draft was wrong

The first draft multiplied the headroom by `gamma + 1`. Its argument was that a
speculator uses up its reserve `gamma` times faster. That argument is
incorrect. A round appends `1 + offered` positions but commits only
`accepted + 1`, and the truncation returns the difference. So the blocks per
emitted token stay the same with speculation.

The only new cost is a short peak inside a step, for the window in flight. A
multiplied headroom reserves a large constant per request. That cuts the
concurrency that the engine can reach, and concurrency is exactly what staging
step 4 measures. The correct form is:

```
decode_headroom_blocks + ceil((gamma + 1) / block_size) + 1
```

Any slack beyond that must be argued as a probability, not as a drain rate. It
is the probability of reaching the cancel path of ADR-024, which cancels its
victim outright. (Step 2 found that a request without room for this headroom
must downgrade, not get rejected. Its findings explain why.)

## Staging

The work has four steps:

1. Variable tokens per step, with no drafter. Each request gives
   `_sample_decoders` and `_packed_forward` a token list, and the scheduler
   switches to `forward_step_packed`. Gate: the plain path is unchanged,
   including the golden suite. Done.
2. Drafter state and the draft and verify phases, with the slices of point 5
   and the emission rules of point 8. Done, see the findings of step 2. Gates
   1 and 2 live in `tests/unit/test_dspark_scheduler.py`.
3. OOM rollback and idempotence. Both older tests in
   `tests/unit/test_scheduler_oom.py` raise from the fake runner before it
   touches the cache. So neither one reaches the state of a partial append,
   and a test modeled on them passes vacuously (for the wrong reason). This
   step needs two speculators in flight and a raise from inside
   `append_kv_packed`. Step 1 already did the emission half of idempotence.
   Done, see the findings of step 3. The failure injection is
   `tests/unit/_oom_injection.py`.
4. Benchmark. Measure the accepted length and the throughput against
   concurrency. See the design of step 4 below for why this runs inside the
   engine and not on the open-loop benchmark. That also gives the SPS(B) curve
   that the batch-global scheduler needs. This step requires the stats
   plumbing (see the deferred items) and a `num_blocks` value so large that
   headroom is not the limit.

## Findings from staging step 1

Step 1 landed as described. `_sample_decoders` and `_packed_forward` carry a
token list for each request, and `forward_step_packed` is the forward of the
scheduler. The plain path is unchanged, step for step. The third finding
below then changed what the plain path does after a recoverable OOM. An audit
of the touched code found three things that matter for later steps.

The fully fused TurboQuant attention path requires q_len == 1 for every
request. One of the conditions that `packed_attention.py` sets for it is
`q.shape[0] == cu_seqlens_q.shape[0] - 1` (total_q == batch_size). So any
step that carries a verify window falls back to the materialized V2a path.
That fallback is documented and correct. But a `kv_quant="turbo3"` run loses
the fused kernel on every speculative step. If step 4 compares a speculative
turbo3 run with a plain turbo3 baseline, it must say so.

The benchmark scripts count stream items as tokens. `http_openloop_bench.py`
adds one to `n_tokens` for each non-empty text chunk. It computes ITL
(inter-token latency) as `decode_span / (n_tokens - 1)`.
`modal_packed_bench.py` takes the mean time between consecutive stream items.

A verify step that commits several tokens emits them as one `GenerationStep`.
So both metrics show fewer tokens, and the tokens seem slower. The stats
plumbing of step 4 must fix this. It can emit one step per committed token,
or it can put a token count on each step.

On the plain path, an OOM retry emitted a token twice. That is now fixed.
`_sample_decoders` appends and emits before the forward, and only a forward
that completes overwrites `last_logits`. So on the retry, every surviving
decoder sampled the same `last_logits` again. It then appended a copy of the
token that it had already streamed.

A rollback cannot fix this. The token is already on the output queue, and
possibly at the client. So `RunningRequest.pending_decode_tokens` holds the
sampled tokens, and the retry feeds them again instead of sampling again. The
forward that consumes them clears them. This is the idempotence guard of point
7, for the plain path. The speculative path uses the same shape: hold the
round, do not derive it again.

That fix did not roll back cache lengths, so point 6 still applied after step
1. The regression test is
`test_scheduler_oom.py::test_oom_retry_does_not_re_emit_an_already_emitted_token`.
It needs two requests: a prefiller takes the preemption, so the decoder
survives. It runs `_step` directly, because the interleaving through the
engine thread is a race.

## Findings from staging step 2

Step 2 landed. `engine/dspark/spec.py` holds `DSparkSpec`, `SpecDecodeState`,
and the shared functions of a round. The batch-1 runner now delegates to
them, so its exact fp32 tests keep gating the same math. `ContinuousScheduler`
has the draft and verify phases, the eligibility decision at admission, the
slices of point 5, and the emission rules of point 8. `SpecDecodeState` also
gained `context_parts`, the prompt taps of each chunk, which round 1 joins
into `context`.

The reap now drops the `spec` state of a finished request. This matters
because `RequestHandle` keeps the `RunningRequest` alive after completion, and
the drafter cache must not outlive the request. The design needed three
corrections, and step 2 taught one lesson.

(INV) is a precondition of proposing, not of the draft phase. The plan said to
assert it "at the top of the draft call". But `propose_block` folds the
injected context into the drafter cache and truncates it back to `start`. So
once a proposal exists, `draft_cache_len + context_rows` is larger than the
slot length, and that is correct. An assertion at the top fails on the reuse
path of point 7 with no real error. So the assertion lives inside the branch
that drafts.

Speculation must never turn away a request that the engine can serve. The
headroom of point 9 went into `required_blocks` before the test for permanent
rejection. So once dspark was configured, a prompt that fits the pool on the
plain path was cancelled outright. An optimization must not make the engine
fail work that it serves without it. Now such a request downgrades to the
plain path, like the top-k, top-p, and long-prompt downgrades. Only a request
that does not fit even on the plain path gets rejected.

The first version of the headroom test asserted the cancellation. So it
encoded the bug as the expected behavior. It now pins the constant from both
sides: admit, and downgrade.

A target other than Qwen3, with dspark configured, killed the engine thread.
Only the Qwen3 forward accepts the tap arguments. So any other model family
raised TypeError inside the first forward that carried a speculator, and
`_engine_loop` does not catch that. `ModelRunner.supports_hidden_state_taps`
inspects the signature of the forward. The scheduler refuses the combination
at construction, next to the refusal of the prefix cache.

The lesson: the numerics gate cannot see bugs in where the context comes from.
Greedy correctness does not depend on the drafter. A context sliced from the
rows of another request changes what the drafter proposes, never what the
target commits. So tests of token equality pass with a fully corrupted context
buffer. A mutation of the refresh slice survived every token assertion.

Only assertions on values catch such a bug. The bookkeeping suite scripts the
tap sink to carry row positions. It pins the contents of the buffer, with a
speculator placed behind another request. Step 4 has the same blind spot: a
bug of this kind shows up as a lower tau, not as wrong output.

## Findings from staging step 3

Step 3 landed. `_packed_forward` saves the length of every slot right before
the forward. After the forward returns, it clears the copy. `_preempt_on_oom`
first calls `_roll_back_partial_append`, which truncates every slot back to
the saved length. Only then does it pick the victim and reap.

This also fixes the plain path. Before, a plain decoder in slot 0 resumed at its inflated
length after the OOM. It then wrote its token past a hole of unwritten K/V,
and nothing reported an error.

Points 6 and 7 of this design contradict each other on one detail: whether
`_preempt_on_oom` clears `offered`. Point 6 rolls the drafter cache back to
its pre-draft length and clears `offered`, so the retry drafts again. Point 7
keeps the proposal and reuses it, and it clears `offered` only in the verify
phase. Each design is consistent on its own. A mix of the two breaks (INV).
Drafting folds the injected context into the drafter cache and truncates it
to `start`.

If `offered` is cleared while the drafter cache stays, the redraft breaks
(INV) at once, because the context is counted twice. If the drafter cache
rolls back while the proposal stays, the next round breaks (INV). The code
follows point 7. Only the target slots roll back, and each speculator keeps
its anchor, its proposal, and its drafter cache. This saves a second drafter
forward, and a round at a temperature above zero is not sampled twice.

The tests pin this choice. A faithful point 6 drafts again, which only the
draft count of the scripted OOM test detects (6 proposals instead of 4). A
mixed version, which clears `offered` and keeps the drafter cache, fails the
speculator OOM tests with an (INV) assertion.

The rollback is complete because only the forward can raise. The
`allocate` method of the pool is the only source of `OutOfMemoryError`. Its
only callers are `append_kv_packed` and `append_stream_packed`, and both run
inside the forward. Both allocate on layer 0 before any write. Publication
into the prefix cache happens on the last layer, after every write. So a
failed forward writes nothing and publishes nothing, and a truncation to the
saved length never lands inside a published block.

One slot keeps extra blocks until the retry. A slot adds each new block to its
list before it advances its count. So the slot whose allocation raised can
hold blocks that its unchanged length does not need, and `truncate_to` leaves
them alone. They do not leak. The retry feeds that slot the same tokens, so it
reuses exactly those blocks. If the slot is the victim, the reap frees
them.

Token equality cannot see this bug on the plain path. With a random micro
model, a missing rollback still produced the same tokens. Even with the hole
in its K/V, the model picked the same argmax. Only an exact assertion on the
slot lengths after recovery caught the bug. This is the same lesson as in
step 2: for state that does not change the greedy argmax, assert the state
itself.

The prefill branch now asserts that the slot length equals `tokens_prefilled`
before it feeds a chunk. The speculators had this protection already, through
their start-vs-slot assertion. A prefiller with an inflated slot now fails
loudly instead of storing its chunk past a hole. The OOM tests cover a
surviving prefiller and a slot that keeps extra blocks. A rollback that skips
prefilling slots fails the test with the surviving prefiller.

## Design of staging step 4

Step 4 runs inside the engine process, not through the HTTP open-loop
benchmark that the staging list names. That benchmark has six problems for
this purpose:

- The server cannot enable dspark.
- Its default prompt is longer than `max_spec_prompt_tokens`, so every
  request quietly runs the plain path.
- Its client counts stream chunks as tokens.
- Its GPU image uses library versions that the DSpark path never ran on.
- Its arrival rate does not control how many requests are in flight.
- No channel carries the speculative statistics out of the engine.

The in-process driver is `mini_infer.bench.dspark_sweep`. A point submits a
fixed list of prompts at once to a scheduler whose `max_concurrent` is the
level under test. While a prompt still waits, the scheduler keeps that many
requests in flight, as a closed loop. Every level uses the same prompts, so a
change between levels is a change of concurrency, not of workload.

After the last admission the batch drains, and the two arms drain
differently. Plain requests of one length finish together, but speculative
requests finish at different rounds. So a point reports two speeds.

The whole run counts every token. The steady part counts only the steps in
which a prompt still waited, so every slot was busy. A point also reports its
mean in-flight count, weighted by time, which shows how far the run fell
short of its label. The steady speedup is the fair comparison between the
arms.

Two pieces of plumbing serve both drivers. `GenerationStep.num_tokens` says
how many tokens a step carries, so a client can count tokens instead of
steps. `GenerationResult.spec_summary` carries a fixed-size `SpecSummary` of
the speculative rounds. `_finish` builds it, because the reap drops the full
statistics right after. A request that did not speculate reports None.

The sweep refuses a point that does not measure what its label says. It
raises in two cases: a dspark point holds a request that did not speculate,
or a request was cancelled at admission. A timed pass puts a GPU synchronize at
every phase boundary and records every forward's width and duration. Those
barriers slow the step, so the throughput numbers come from the untimed
points only. Each stored point carries a fingerprint of its prompts and
configuration, so a rerun with another configuration refuses the old results.

Three caveats apply to the results. The attention backend is `torch`, the
only one that the DSpark path ran on with a GPU. It rebuilds the packed K/V
for every layer of every forward, so it can favor speculation, which needs
fewer forwards. Token agreement in bf16 is statistical, not exact.

The third caveat is about tau. Near `max_tokens`, the scheduler offers
fewer tokens than the drafter proposed, so a request's last rounds commit
less. The batch-1 runner never shortens a proposal for room. So `tau` is
biased low against batch-1, and `tau_unclamped` leaves those rounds out to
make the comparison fair. An EOS inside a committed block still drops the
tokens from it on. So the throughput numbers count the tokens that each
request really returned.

The HTTP open-loop benchmark stays useful for latency under live traffic.
It needs a server setting for dspark and a client that sums `num_tokens`.

## Gate, split three ways

The first gate of this design was: "two concurrent requests each produce
exactly what they produce alone at temperature 0". At bf16, that gate cannot
pass, and harmless rounding trips it. There are two reasons. First, the logits
of a request depend on the `total_q` shape of the batch, as ADR-011 and the
Stage C benchmark both measured.

Second, and worse, the requests are coupled. The `total_q` share of one
request depends on the accepted counts of the other requests. So the
composition of the batch is not stable even across runs. The gate therefore
has three parts:

1. Bookkeeping, exact. Against a deterministic fake runner: committed tokens,
   cache lengths, (INV), and emission clamps.
2. Numerics, exact. CPU fp32, a real small model, two concurrent requests
   against each one alone.
3. GPU bf16, statistical only. Accepted length and coherence, explicitly not
   identity per token. `tests/unit/test_dspark_speculative_real_model.py`
   explains the reason.

## Deferred, with the reason

These items are deferred. Some of them return in step 4, as their entries say:

- The batch-global confidence scheduler (Algorithm 1) and SPS(B) profiling.
  This plan is what they need first.
- Batched drafter forwards. v1 runs the drafter once per request, in a loop.
  A batched drafter needs the block mask that batch-1 avoids (ADR-027 point 3),
  and padded contexts of different lengths. The verify forward of the target
  also takes most of the time of a step. The review notes correctly that the
  loop leaves a serial floor on latency. So step 4 must measure the time of
  each phase, and the decision follows the measurement.
- Top-k and top-p under speculation (point 3).
- A transactional K/V append (point 6).
- `PDScheduler`, which has the same one-token assumption.
- Trace and stats. `_classify_phase` has no idea of a verify step, and no
  channel carries `DSparkStats`. Its `confidence_observations` list has no
  bound, and it must not grow without limit for each request. Step 4 needs
  this first.
