# DSpark evaluation plan

Date: 2026-07-02
Updated: 2026-07-04, re-checked against main after the MiniMax-M3 (#18)
and open-loop bench + OOM recovery (#19) merges
Updated: 2026-07-26, re-checked against main after the TPU backend
(#20), Inkling (#21), and Kimi Linear (#23) merges (none touch the
Qwen3/spec-decode/packed-attention/scheduler surfaces this plan
depends on); transformers dev pin note refreshed for the 5.14.x bump.
Status: Proposed; Stage A (design study) starting on
`dspark-drafter-port`.

Paper: "DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation" (DeepSeek-AI + Peking University,
2026-06-27). Distributed as `DSpark_paper.pdf` inside the MIT-licensed
[deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) repo;
there is no arXiv entry (arXiv 2606.19348 is the V4 report it cites).

## What DSpark is (two separable contributions)

1. **A semi-autoregressive drafter.** A 5-layer parallel backbone
   (modified DFlash) drafts a whole block of gamma tokens in ONE forward
   pass: input is the previous round's committed token plus gamma-1 mask
   tokens, with target-model context injected by projecting hidden
   states from a fixed set of target layers into every draft layer's
   K/V. Attention is bidirectional within the block. A lightweight
   sequential head (default: rank-256 low-rank "Markov" bigram bias)
   then makes each position's distribution exactly causal:
   `p_k = softmax(U_k + B(x_{k-1}, .))`, so lossless rejection sampling
   still applies. A confidence head (linear + sigmoid) predicts each
   draft token's conditional survival probability, supervised by the
   analytical acceptance rate `c*_k = 1 - 0.5 * ||p_draft - p_target||_1`.
2. **Confidence-scheduled verification.** A batch-global scheduler
   picks a per-request verification length each step by maximizing
   expected tokens-per-second against a profiled engine capacity curve
   SPS(B): under light load verify long drafts, under heavy load prune
   low-confidence suffixes. Appendix A proves the scheduler must be
   non-anticipating (early-stopping) or losslessness breaks.

Verification itself is standard rejection sampling; at temperature 0 it
collapses to argmax equality, and truncating the verification length at
temperature 0 is trivially lossless.

Reported results: offline accepted length +26.7-30.9% vs EAGLE-3 and
+16.3-18.4% vs DFlash on Qwen3-4B/8B/14B and Gemma4-12B (all drafters
retrained on identical data); in production, per-user speed +60-85%
(V4-Flash) and +57-78% (V4-Pro) at matched aggregate throughput vs an
MTP-1 baseline, and +51/52% fleet throughput at fixed SLA. Caveats the
paper itself concedes: the production fleet runs below the GPU
compute-saturation threshold (the regime friendliest to speculation),
hardware is undisclosed, and the baseline is a 1-token drafter. No
independent replication exists yet.

## Why this maps to mini-infer

- ADR-011 shipped V1 speculative decoding (greedy, two-model, fixed
  K=4, single request) and its benchmark found exactly the failure mode
  DSpark targets: fixed-K drafting won 1.29x on a slow prompt and lost
  0.83x on fast ones (H100 aggregate 1.00x). Confidence-scheduled
  drafting is the principled fix, and ADR-011 already lists "adaptive
  K" as a follow-up.
- The mechanical primitives exist: `forward_step_packed` (multi-token
  verify), `PagedKVCache.truncate_to` (rollback), packed-varlen
  attention across all families. Since the MiniMax-M3 merge, packed
  attention also has a per-request additive `block_mask` path
  (`packed_attention_torch`) and a tested mask-construction pattern
  (`MiniMaxM3Indexer.build_block_mask`) that the drafter's mask tests
  can lean on as an oracle.
- Released artifacts make a port testable without training: BF16
  drafter checkpoints for Qwen3-4B/8B/14B and Gemma4-12B
  (`dspark_qwen3_4b_block7` = 1.39B params, block size 7), plus the
  DeepSpec eval harness as a bit-parity reference (batch-1, plain
  PyTorch rejection sampling, supports temperature 0).
- Differentiation: vLLM merged drafter support (PR #46995, 2026-07-01)
  but explicitly excluded confidence scheduling; SGLang and
  TensorRT-LLM PRs are still open. No engine has an open
  implementation of the scheduler (Algorithm 1). A readable,
  parity-tested port is exactly this project's niche.

## Out of scope (explicit)

- **Drafter training.** DeepSpec's pipeline needs a ~38 TB target
  hidden-state cache and an 8-GPU node. We consume released
  checkpoints only.
- **V4-DSpark.** The production draft module rides inside the 284B /
  1.6T checkpoints; also its config taps the final decoder layers,
  which the OSS harness explicitly forbids, an unresolved
  inconsistency. Qwen3-4B is the evaluation vehicle.
- **Native MTP-head speculation** (V4/M3/GLM MTP weights we currently
  drop at load). Different technique; do not bundle.
- **Tree-attention verification.** DSpark is chain-based; convenient,
  nothing to build.

## Evaluation stages and gates

### Stage A: design study (CPU-only, no code shipped), done

Done (2026-07-26), on `dspark-drafter-port`: see
`docs/decisions/ADR-027-dspark-drafter-port.md`. All five
parity-critical unknowns resolved against `DeepSpec`'s actual code
(not the paper's prose) and adversarially re-verified:

- Position ids are sequential and global; RoPE inside the draft block
  and on the injected target context share one `rotary_emb` call over
  a single contiguous slice, because context and draft positions are
  adjacent in the running counter. Subtle enough that the ADR spells
  out the exact mechanism.
- Round 1 injects the full prompt's target hidden states; every later
  round injects only the just-verified window (a full reassignment,
  not an append). The **gate below does not trigger**: what grows
  with sequence length is the drafter's own small unpaged K/V cache
  (linear in generated length, same rate as any ordinary KV cache),
  not a re-derivation over a growing prompt window.
- Confidence head consumes the post-final-norm hidden state and
  shares the Markov head's embedding table exactly (one object, not
  two tables of the same shape).
- Anchor for round N+1 is `verification.next_token` from round N
  (bonus token or corrected token, matching our own convention). No
  catch-up step needed, unlike ADR-011's V1: the drafter's own
  self-attention KV is discarded every round regardless of acceptance.
- Frozen embed/LM-head: the checkpoint ships full, untied copies
  (`tie_word_embeddings: false`). Load as ordinary weights; no tying
  plumbing needed.

Deliverable: ADR-027, with alternatives considered (registry
registration, reusing `packed_attention`'s `block_mask`, tying
embed/LM-head, V4-DSpark, EAGLE-3, n-gram/PLD).

**Gate** (did not trigger): if the hidden-state side cache had been
architecturally ugly (e.g. unbounded prompt-length memory with no
clean paging story), stop and document why in the ADR.

### Stage B: drafter port, batch-1, temperature 0

Implement `Qwen3DSparkModel` in mini-infer: KV injection of tapped
target hidden states, bidirectional block attention, Markov head,
confidence head. Add hidden-state taps to the Qwen3 target forward and
a plain-tensor side cache with truncate (the drafter's own cache stays
unpaged; the reference uses a cropped DynamicCache and 5 small layers
do not justify paging).

Drafter attention: a plain SDPA call inside the drafter module. The
shared packed-attention `block_mask` path added for MiniMax-M3 MSA is
not a structural fit (the dispatcher sources K/V exclusively from a
`PagedKVCache` and has no injection point for the projected
target-hidden context keys, and the drafter cache is unpaged), but
`packed_attention_torch(block_mask=...)` is the right unit-test oracle,
mirroring how the MSA paged kernel is validated against the dense-mask
reference. The target's verify forward keeps its fast causal backends
untouched. As built, batch-1 needs no mask at all (both of the
reference's training-time mask conditions are vacuous with one block in
flight); see ADR-027 point 3.

Parity contract, in order:

1. Random-weight micro-config CPU unit tests comparing per-position
   base logits U_1..U_gamma, Markov-biased logits, and confidence
   logits against the DeepSpec reference code (same pattern as
   `tests/unit/test_minimax_m3_parity.py`).
2. Real-checkpoint temperature-0 token-parity fixtures generated with
   the DeepSpec harness (Qwen3-4B target + `dspark_qwen3_4b_block7`) on
   a single short Modal GPU run (L4/A10 class, 24 GB fits both in
   BF16). DeepSpec's `requirements.txt` and its checkpoints' own
   `config.json` both pin transformers 5.10.2 exactly; the repo's dev
   group has since moved to 5.14.x (bumped for Inkling, following an
   earlier bump to 5.12.x for MiniMax-M3). Both bumps forced explicit
   numerics realignment in existing families (GLM-MoE-DSA's indexer
   RoPE convention broke twice, across the 5.10 to 5.12 and 5.12 to
   5.14 jumps; MiniMax-M3's block selection broke on the second one),
   so treat DeepSpec's pin as load-bearing rather than a stale
   default. Before setting up an isolated venv, try the cheap
   validation first: run DeepSpec's eval harness under the repo's
   current dev-pinned transformers against one bundled eval set (e.g.
   GSM8K) and check the accepted-length number lands near the paper's
   reported ~5.57 for Qwen3-4B. If it matches, generate fixtures
   in-repo and skip the isolated venv; if it diverges, that confirms
   the isolated 5.10.2 venv is warranted and fixtures should record
   the transformers version they were generated under.

Existing golden tests are untouched: greedy verification emits the
target's exact argmax by construction.

**Gate:** bit-parity at micro-config level and token parity on real
checkpoints. If parity fails and cannot be root-caused, stop and
document.

### Stage C: confidence-scheduled truncation, batch-1, done

Done (2026-07-27). Results:
[docs/benchmarks/2026-07-27-dspark-accepted-length.md](../benchmarks/2026-07-27-dspark-accepted-length.md).

tau lands at 3.79 math / 3.14 code / 3.07 chat (templated, 50 prompts) against
the paper's ~5.57 / ~5.12 / ~3.49. Threshold 0.30 cuts verified tokens per
round 18% for a 0.7% tau cost. Per-step conditional acceptance is flat out to
position 7, reproducing the paper's claim for the sequential head.

**Gate: partially met, and the two hypotheses this plan named were both
tested and refuted.** Chat templating was predicted to close the gap; it
lowers tau in all three domains (-3/-12/-6%). Truncation at 128 tokens was
predicted to bias the long-answer domains; a 3x longer budget buys 3%. What
remains, and is the leading explanation, is that the paper measures at
temperature 1.0 with rejection sampling (acceptance `1 - TV`) while this is
greedy (acceptance `P(argmax match)`), a strictly harsher bar. The
confidence head's calibration error points the same way, since it is trained
against exactly the temperature-1.0 acceptance rate. Confirming this needs
Stage D's rejection sampling.

Original plan text follows.

Add threshold-mode truncation mirroring DeepSpec's
`_confident_prefix_length` (truncate the draft at the first position
where sigmoid(confidence) < threshold). Measure on DeepSpec's bundled
eval sets (`eval_datasets/`: gsm8k, humaneval, mt-bench, alpaca) for
direct Table 1 comparability:

- Accepted length tau per round (primary metric; hardware-independent,
  unlike wall-clock, which ADR-011 showed washes out on fast GPUs with
  small models).
- Per-position conditional acceptance (the paper's position-decay
  claim: parallel drafters hold up at deep positions where EAGLE-style
  drafters decay).
- Confidence calibration (ECE / AUROC via the metrics in DeepSpec's
  `confidence_head.py`; released checkpoints ship no calibration, so we
  measure the raw head, paper claims ECE 3-8% uncalibrated).
- Threshold sweep reproducing the acceptance-vs-tokens-per-step
  trade-off (paper: chat acceptance 45.7% to 95.7% as threshold rises).
- A worst-case set: low-acceptance, long multi-turn prompts, to
  quantify the paper's admitted unrecoverable parallel-draft overhead.

Baselines, all runnable in-repo: no-spec decode; V1 two-model spec
(Qwen3-4B target + Qwen3-0.6B draft, K=4); DSpark drafter without
truncation; DSpark with threshold. EAGLE-3/DFlash comparisons are out
(their released checkpoints target Qwen3-8B, tight on a 24 GB card).

Deliverable: a docs/benchmarks entry. This would be the first
third-party measurement of any DSpark claim.

The offline batch-1 metrics need no server. Any serving-level run goes
through the open-loop harness (`scripts/http_openloop_bench.py`, Modal
entrypoint `scripts/modal_openloop_bench.py`) with the KV pool sized
via `MINI_INFER_NUM_BLOCKS` / `MINI_INFER_BLOCK_SIZE`; the 1024 x 16
dev default holds about 16K token slots and OOMs under load. Direct
`ModelRunner` construction (fixtures, `SpeculativeRunner` benches)
takes the same values as kwargs, the env vars only cover the HTTP
path.

**Gate:** does our tau land in the paper's ballpark (about 3.5 chat,
5.6 math on Qwen3-4B at block 7)? A large gap means either a port bug
or a paper problem; both are findings worth writing up.

### Stage D: deferred, design-only until C justifies it

- Temperature > 0 rejection sampling with the corrected-distribution
  resample (already an ADR-011 follow-up; the Appendix A
  non-anticipating constraint only bites here).
- Multi-request speculative decoding inside `ContinuousScheduler`
  (rated LARGE: `_admit_waiting`, `_sample_decoders`, `_packed_forward`
  all assume one token per request per step, unchanged by #19). This
  now also interacts with mid-step OOM preemption (ADR-024): a verify
  chunk appends up to gamma K/V positions per request per step instead
  of one, so mid-step allocation bursts grow by roughly gamma, the
  8-block decode headroom drains roughly gamma times faster, and
  preemption fires more often. The victim is cancelled outright, not
  paused, so a preempted decoder loses its whole generation, and a
  mid-append OOM leaves earlier slots with advanced counters and
  allocated blocks but no written K/V, which `_preempt_on_oom` does not
  roll back. Re-tune admission headroom and revisit the victim policy
  as part of this work; see
  `docs/decisions/ADR-024-engine-oom-preemption.md` and
  `tests/unit/test_scheduler_oom.py`.
- SPS(B) profiling plus the batch-global Algorithm 1 scheduler. This is
  the "first open implementation" opportunity, but it only demonstrates
  its value under multi-request load, so it cannot precede scheduler
  integration. The open-loop harness (results in
  `docs/benchmarks/2026-04-30-open-loop-rate-sweep.md`) already
  measures the engine capacity curve under offered load; SPS(B)
  profiling should extend that harness rather than build a new driver.
- Sequencing note: `ContinuousScheduler` is not the only multi-request
  scheduler. PDScheduler (`src/mini_infer/workers/pd_scheduler.py`,
  plan in `docs/plans/pd-scheduler.md`) drives the disaggregated path
  and its decode step is also one token per request per step
  (`DecodeSession.step`). Stage D targets `ContinuousScheduler` only;
  whether and when the PD path inherits spec decode is a decision to
  record in the Stage A ADR so the scheduler rework does not fork.

## Risks

- **Position-id conventions** between anchor, mask tokens, and RoPE are
  the classic parity killer; that is why Stage B compares per-position
  logits, not just end-to-end tokens.
- **`truncate_to` published-prefix-block edge case**: rollback into a
  published prefix-cache block is refused; at block size 7 a draft
  spans block boundaries more often than at K=4. Keep the drafter path
  prefix-cache-off in v1 and add a unit test for the boundary.
- **Checkpoint licensing**: the DeepSpec repo and V4 checkpoints are
  MIT, but the drafter checkpoint pages carry no license tag; confirm
  before committing derived fixtures.
- **Local memory**: 16 GB MPS will not hold Qwen3-4B BF16 + 1.39B
  drafter comfortably, and no smaller drafter exists. Local work is
  CPU micro-config only; real checkpoints run on Modal.
- **Environment split** for fixture generation (transformers 5.10.2 vs
  the repo's current 5.14.x dev pin), unless the cheap validation in
  Stage B shows DeepSpec's reference tolerates the newer version.

## References

- Paper: `DSpark_paper.pdf` in https://github.com/deepseek-ai/DeepSpec
- Reference code: `deepspec/modeling/dspark/qwen3/`,
  `deepspec/eval/dspark/draft_ops.py`,
  `deepspec/eval/base_evaluator.py` (rejection sampling),
  `deepspec/modeling/dspark/markov_head.py`
- Checkpoints: `deepseek-ai/dspark_qwen3_4b_block7` (HF)
- Engine landscape: vLLM PR #46995 (drafter merged, scheduler
  explicitly out of scope), SGLang issue #29488 (open),
  TensorRT-LLM PR #15808 (open)
- In-repo: ADR-011, `src/mini_infer/engine/speculative.py`,
  `src/mini_infer/cache/paged_kv_cache.py::truncate_to`,
  `docs/benchmarks/2026-04-29-speculative-decoding.md`
