# MiniMax-M3 428B real-model gate: coherence, TP consistency, kernel A/B

Date: 2026-07-03. Hardware: 4x H200 (Modal), bf16 + block-FP8-resident routed
experts, tensor + expert parallelism (world_size 4). Scripts:
`scripts/modal_m3_stage_weights.py` (staging) and `scripts/modal_m3_generate.py`
(gate). Checkpoint: `MiniMaxAI/MiniMax-M3` @ `bfd6c97f...`, 854 GB bf16,
59 shards.

## Setup

- **Staging**: the bf16 checkpoint was rewritten shard-by-shard on a CPU
  container with the 21,888 routed-expert weights quantized to block-FP8
  e4m3 (`[128,128]` blocks, `weight_scale_inv` scales, the exact format the
  loader's `Fp8Expert` path consumes), shrinking the volume to ~450 GB and
  roughly halving GPU load time. Everything else stays bf16.
- **Load**: `load_weights_streaming` (per-shard `safe_open`, peak host RAM
  ~ one shard per rank); each rank materializes its TP/EP slice directly on
  its GPU. Peak GPU memory: 121.3 GB per H200 (of 141 GB).
- **Decode**: the engine's own sampler at the model's shipped generation
  defaults (`temperature 1.0, top_p 0.95`), RNG seeded identically per rank
  so trajectories stay rank-comparable. Prompt via the model's chat template.

## Results

**Coherence gate: PASS.** Prompt: "What is the capital of France? Name two
famous landmarks there." Output (rank 0, first 96 tokens):

> `<mm:think>`The user is asking a simple factual question. The capital of
> France is Paris. Two famous landmarks there are the Eiffel Tower and the
> Louvre Museum. I can also mention other famous landmarks like Notre-Dame
> Cathedral, Arc de Triomphe, or Sacre-Coeur Basilica. Let me provide a
> clear, helpful answer.`</mm:think>`The capital of France is **Paris**.
> Two famous landmarks there are: 1. **The Eiffel Tower** (an iconic iron
> lattice tower completed in 1889 ...

A well-formed reasoning block, correct facts, clean formatting.

**Rank consistency: PASS.** All four ranks produced byte-identical text
(TP all-reduces + expert-parallel dispatch + seeded sampling agree).

**Decode-kernel A/B (the ADR-022 ship gate): 0.98x, token identity PASS.**
Chunked prefill of a 16,384-token document, then 32 timed decode steps per
arm on identical cache state:

| arm | tok/s |
|---|---|
| torch path (materialize + dense mask) | 3.00 |
| block-sparse kernel (`msa_paged_decode`) | 2.95 |

No end-to-end win at 16K context: each decode step moves ~13 GB of expert
weights (top-4 of 128 experts x 57 layers, FP8) versus ~2 GB of K/V for the
torch path's materialization, so attention is a thin slice of the step and
the kernel's op-level 18x (at 16K) cannot move the total. Consistent with the
A10 microbench scaling and with the V4 decode-kernel finding (ADR-020).
**Verdict: the kernel stays off by default.** Unlike V4's kernel, the arms
are token-identical here, so the flag remains available for long-context
experiments where attention's share grows.

## The bug this gate caught

The first gate run produced fluent-but-degenerate output (correct "Paris."
then repetition; babble under sampling). Root cause: the port ran FULL RoPE
while the checkpoint is trained PARTIAL (first 64 of 128 dims). The tiny-model
parity harness had passed an explicit `rope_parameters` dict without
`partial_rotary_factor`, which silently degenerates HF itself to full rope,
so parity was green on both sides of the same wrong convention. At short
context the erroneously-rotated half moves by nearly-identity angles (low
frequencies), which is why top-1 stayed fluent while the distribution tail
was corrupted. Fix: width-`rotary_dim` tables + `apply_rotary_pos_emb_partial`
everywhere, and deployment-shaped harness configs (flat fields copied from
the real config.json). The full unit ladder plus this gate now pin both.

## Failure log (methodology notes)

- Two ephemeral `modal run --detach` attempts died at container boot
  ("Runner has been shutting down for too long"); `modal deploy` +
  `Function.spawn` is the reliable pattern for long GPU runs.
- One run crashed on `apply_chat_template` returning a `tokenizers.Encoding`
  (transformers 5.x) where ints were assumed; templates are now rendered to
  text and re-encoded, validated against the real tokenizer locally first.
