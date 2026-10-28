# ADR-047: DeepSeek-V4.1-Flash port design

Date: 2026-10-25
Status: Accepted (design only). Stage A of the V4.1-Flash plan is done, and
its gate did not trigger. Stage B builds the port.

> Amendment (2026-10-28): section 8 now makes exact decoder replay the
> Stage E default. The short replay of the tech report becomes an opt-in flag.

## Context

DeepSeek-V4.1-Flash keeps the V4 attention core but changes how layers share
their caches. It also adds Engram lookup tables, a new hyper-connection order,
FP4 caches and a DSpark drafter inside the checkpoint. The plan lists eight
details where a port can silently drift from the reference. This ADR settles
each one against the reference code, not the tech report's prose.

The reference is DeepSeek's `inference/` folder at Hugging Face revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`, copied to
`third_party/deepseek_v41_reference/` (not tracked by git). The 1017 weight
refresh (revision `1bae0e10`) has the same configuration and tensor names.
The findings come from three methods:

1. A line-by-line read of `model.py`, `kernel.py` and `engram.py`.
2. A CPU run of the reference on a probe model of 6 layers with random
   weights. PyTorch versions replaced its TileLang kernels.
3. A rebuild of the Engram hash layout from the configuration and the real
   tokenizer.

DeepInfer
(`deepseek-ai/DeepInfer`, the engine that serves V4.1) was the second source
for decoder replay.

## Decision

### 1. Single-pass mHC: the input mix comes from the previous sublayer

In the reference `Block.forward`, the post and comb coefficients of a
sublayer come from its own input. Its input mix comes from one sublayer
earlier. The attention uses the `ffn_pre` of the previous block, and the FFN
uses the `attn_pre` of its own block. The head mixes the copies with the
last `ffn_pre`. V4.1 has no `hc_head_*` weights. The Sinkhorn split is the V4
one, and the probe ran it with our `hc_split_sinkhorn` (ADR-018).

Port change: `HyperConnections.hc_pre` splits into "compute the
coefficients" and "apply a given input mix". The decoder layer carries the
mix to the next sublayer.

### 2. Layer roles and shared state

| Role | Layers | Owns |
| :--- | :--- | :--- |
| Sliding window only | 0, 1, DSpark 40 to 42 | window KV |
| Full | 2, 8, 14, 20 (`kv_source_layer_ids`) | window KV, compressor, compressed KV, indexer with `wk` and `k_norm` (index K) |
| Reindex | 24, 28, 32, 36 (`index_source_layer_ids` minus the Full layers) | window KV, indexer query (`wq_b`, `weights_proj`) |
| Reuse | every other compressed layer | window KV |

A Reindex layer reads the compressed KV and index K of the latest Full layer
and runs its own top-k. A Reuse layer reads the latest compressed KV and the
latest top-k. Layer 20 builds the candidate pool for the layers after it.

Decision: one attention block class with a static role and the ids of its
source layers. A small per-forward object carries the shared views down the
stack, in place of the reference's process-wide slot.

### 3. Compressor

Ratio 2 (layers 2, 8 and 14) pools each group of 2 tokens with a softmax
over the gate scores. It has no overlap window and no position bias. Ratio 1
(layer 20) is a plain BF16 projection and an RMS norm, one entry per token.

Port change: `TokenLevelCompressor` gets a V4.1 mode without overlap and
without position bias, plus a ratio-1 path.

### 4. Indexer

The indexer has no Hadamard rotation, which V4 had. The candidate pool keeps
the 2,048 best blocks of 8 positions, and top-k takes
`min(512, end_pos // ratio)` picks.

On tied scores, `torch.topk` picks indices that depend on the row width, so
the prefill and the decode of the reference pick different positions. The
ReLU in the score makes ties common. With quantization off, the two paths
differed by up to 0.54 in the logits. With a stable top-k where the lower
index wins a tie, they agreed within 2.4e-6.

Decision: the port breaks ties toward the lower index at both top-k sites,
as the V4 indexer already does (`_topk_lower_index_on_ties`). The Stage B
harness patches the reference's `topk` the same way.

### 5. Quantization numerics

| Tensor | V4.1 format |
| :--- | :--- |
| Window KV | FP8 E4M3, power-of-two scale per 32, whole vector |
| Compressed KV | FP4 E2M1, one E4M3 scale per 16, after RoPE |
| Indexer query and key | FP4 E2M1, E8M0 scale per 32, no Hadamard |
| Attention output before `wo_a` | FP8 E4M3, power-of-two scale per 32 |
| Inputs of FP8 and FP4 GEMMs | FP8, block 32 |

Float noise of about 1e-7 can move a value across a rounding boundary. With
the round trips on, the reference's own prefill and decode stored index-key
rows one FP4 step apart. That changed 7 of 120 index picks and moved the
logits by up to 0.32. So no port can match the quantized reference exactly
over a long run.

Decision: the parity contract in section 10 separates quantization-off
equality from quantization-on agreement. The first port stores compressed KV
and index K in BF16, and the `simulate_act_quant` mode applies the round
trips where the reference does. Real FP4 storage waits for Stage E. BF16
costs 6.7 GB for 16 requests of 128K tokens.

### 6. Engram hashing

The rebuild matched the checkpoint exactly: 99,092 compressed vocabulary
entries, and 24 hash columns per module with their own prime bucket counts.
The table rows (384,006,168 and 384,016,682) equal the
`engram_num_embeddings` values.

Decision: an `Engram` module and a per-request hash state that keeps only the
compressed ids of the last 3 tokens. The table has two backends: a device
tensor (CPU tests, and 4x B200) and a pinned host table with a CPU gather
(2x B200). The load fails if the vocabulary size does not match.

### 7. Router and experts

The router uses `sqrtsoftplus` of FP32 logits. The bias picks the experts,
but the routing weights are the raw scores of the picks. Image tokens use
`bias_vl` in place of `bias`. There are no hash-routed layers.

### 8. Encoder-decoder boundary and decoder replay

The port derives the boundary as `num_hidden_layers // 2` (20). Layer 20
computes the decoder's compressed KV and index K from its own input, one
token at a time. Each layer computes its window KV the same way. So decoder
replay can run the decoder layers on the end of the prompt only.

The tech report (arXiv 2609.19969) and DeepInfer (`KEEP_TOKENS`) replay
about the last 128 prompt tokens. This changes outputs by design, because
the decoder windows miss older tokens. Each decoder layer needs exact
inputs for 127 more tokens than the layer above it. So an exact replay
covers the last 128 + 19 × 127 = 2,541 tokens, DSpark taps included.

Decision: Stage B runs the full prefill, as the reference `model.py` does.
Stage E makes exact decoder replay the default. Layer 20 still projects its
compressed KV and index K for every token. Layers 20 to 39 run attention
and MoE on the last 2,541 tokens only, and the result must match the full
prefill under section 10. A short replay (default 128 tokens) is an opt-in
flag for agreement with DeepInfer and SGLang. On a 128K prompt, the decoder
then runs on under 2% of the tokens (an estimate, not measured).

### 9. DSpark drafter

The draft input is the last token and 4 noise tokens. Attention is
sliding-window only, over a ring of projected target taps. The confidence
head reads the hidden state before the final norm, unlike our Qwen3 drafter
(ADR-027). The reference has no verify loop, so the port reuses the verify
loop and scheduling of ADR-027 and ADR-031.

### 10. Model class, state cache and parity contract

A new `DeepseekV41ForCausalLM` in `models/deepseek_v41.py` reuses the
unchanged V4 blocks. `StateLayerSpec` gets the id of the layer that owns the
compressed KV and index K, and a borrowing layer allocates zero-width
buffers. Prefix sharing (ADR-020) works unchanged. The prefill/decode split
path (ADR-016) does not serve `StateCache` models, so the gate did not
trigger. The port also needs chunked prefill and multi-token verification,
which the reference lacks. Both must equal the token-by-token decode.

Parity contract for Stage B:

1. Quantization off, stable top-k: every block and the whole model match the
   patched reference within the tolerances of the V4 parity tests.
2. Quantization on: each round trip and each block matches on equal inputs.
   For the whole model, report agreement (top-k overlap, logit cosine,
   greedy tokens), with no hard equality.
3. Engram: equal hash ids on the checkpoint's encoding fixtures, and equal
   module outputs with a small random table.
4. DSpark: equal base logits, Markov-corrected logits and confidence per
   draft position.

## Alternatives Considered

- Role flags on `DeepseekV4ForCausalLM`. Rejected. V4.1 changes almost every
  V4 path, so flags put the V4 results from the GPU runs at risk.
- Keep the reference's process-wide shared slot. Rejected. An explicit
  per-forward object shows the data flow and does not leak state.
- Store FP4 caches from the start. Deferred to Stage E. It needs a
  dequantize-on-load path in the decode kernel, and BF16 fits at our scale.
- Keep the raw `torch.topk` tie order. Rejected. The reference's own prefill
  and decode disagree under ties.
- Make the short replay the default, as DeepInfer does. Rejected. It
  changes outputs, so the parity contract of section 10 cannot hold.
- Engram on the GPUs on 2x B200. Not possible. The other weights take
  307 GB and the tables 203 GB, against 384 GB of HBM.

## Consequences

- Stage B can start: the eight unknowns have answers, and the parity
  contract is fixed.
- Parity tests must patch the reference's `topk`. A test that compares the
  raw reference across prefill and decode fails by design.
- With the round trips on, prefix sharing can differ from a fresh prefill at
  rounding boundaries. With them off and stable ties, it stays exact.
- Every checkout needs `third_party/deepseek_v41_reference/`, or the parity
  tests skip. No Hugging Face golden model exists for V4.1 (transformers
  5.20.2 has no V4.1 class).
- Follow-ups: Stage D makes sure that a Modal B200 container can get about
  210 GB of host RAM for the Engram tables. Stage E adds FP4 cache storage,
  exact decoder replay and the batched DSpark drafter.
