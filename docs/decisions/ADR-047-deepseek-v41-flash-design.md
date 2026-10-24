# ADR-047: DeepSeek-V4.1-Flash port design

Date: 2026-10-25
Status: Accepted (design only). Stage A of
[the V4.1-Flash plan](../plans/deepseek-v41-flash.md) is done, and its gate
did not trigger. Stage B builds the port.

## Context

DeepSeek-V4.1-Flash keeps the V4 attention core but changes how layers share
their caches. It also adds Engram lookup tables, a new hyper-connection order,
FP4 caches and a DSpark drafter inside the checkpoint. The plan lists eight
details where a port can silently drift from the reference. This ADR settles
each one against the reference code, not the tech report's prose.

The reference is DeepSeek's `inference/` folder at Hugging Face revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. A copy lives in
`third_party/deepseek_v41_reference/`, with the tokenizer and a `REVISION`
file. `third_party/` is not tracked by git, so each checkout fetches it once.
The 1017 weight refresh (revision `1bae0e10`) has the same configuration and
the same tensor names.

Three methods produced the findings:

1. A line-by-line read of `model.py`, `kernel.py` and `engram.py`.
2. A CPU run of the reference itself. A probe replaced the reference's
   TileLang kernels with PyTorch versions:
   - our `hc_split_sinkhorn` and our `fp8_e4m3_roundtrip`
   - a new FP4 round trip with E4M3 scales
   - a gather plus softmax with a sink for `sparse_attn`.

   The probe model has 6 layers with random weights. It uses every V4.1
   layer role:
   - layer 0: sliding window only
   - layers 1 and 2: a ratio-2 Full layer and its Reuse layer
   - layer 3: a ratio-1 Full layer that also builds the candidate pool
   - layer 4: a Reindex layer that uses the pool
   - layer 5: a final Reuse layer.

   The probe scripts stayed in a scratch folder. They are not part of this
   change.
3. A rebuild of the Engram hash layout from the configuration and the real
   tokenizer.

DeepInfer (`deepseek-ai/DeepInfer`, the engine that serves V4.1) was the
second source for bounded replay.

## Decision

### 1. Single-pass mHC: the input mix comes from the previous sublayer

The reference `Block.forward` does this:

1. `residual = x`. Compute `(attn_pre, attn_post, attn_comb)` from `x` with
   the attention's `hc_attn_*` weights.
2. Mix the four copies into the attention input with `pre_mix`. This is
   the `ffn_pre` that the previous block returned. Layer 0 uses a one-hot mix
   that takes copy 0.
3. Run the attention norm and the attention. Expand back to four copies
   with `attn_post` and `attn_comb`.
4. Compute `(ffn_pre, ffn_post, ffn_comb)` from the new stream with the
   `hc_ffn_*` weights.
5. Mix the FFN input with `attn_pre` from step 1. Run the FFN norm and the
   MoE. Expand with `ffn_post` and `ffn_comb`.
6. Return the stream and `ffn_pre`.

So the post and comb coefficients of a sublayer come from its own input, but
its input mix comes from one sublayer earlier. After the last block, the head
mixes the copies with the last `ffn_pre`, applies the final norm and computes
the logits. V4.1 has no `hc_head_*` weights.

The Sinkhorn split is unchanged from V4: `pre = sigmoid + eps`,
`post = 2 * sigmoid`, and 20 Sinkhorn iterations for `comb`. The probe ran
the reference with our `hc_split_sinkhorn` (ADR-018) as its kernel.

Engram runs before the block, on all four copies (section 6). The DSpark
taps sit before layers 37, 38 and 39: the mean of the four copies, joined in
that order into a 15,360-wide input.

Port change: `HyperConnections.hc_pre` now computes and applies its mix in
one call. V4.1 splits it into "compute the coefficients" and "apply a given
input mix", and the decoder layer carries the mix to the next sublayer.

### 2. Layer roles and shared state

The roles come from the configuration:

| Role | Layers | Owns |
| :--- | :--- | :--- |
| Sliding window only | 0, 1, DSpark 40 to 42 | window KV |
| Full | 2, 8, 14, 20 (`kv_source_layer_ids`) | window KV, compressor, compressed KV, indexer with `wk` and `k_norm` (index K) |
| Reindex | 24, 28, 32, 36 (`index_source_layer_ids` minus the Full layers) | window KV, indexer query (`wq_b`, `weights_proj`) |
| Reuse | every other compressed layer | window KV |

A Reindex layer reads the compressed KV and the index K of the latest Full
layer, and runs its own top-k. A Reuse layer reads the latest compressed KV
and the latest top-k indices. "Latest" means the most recent source in layer
order. The reference keeps one process-wide slot per shared tensor and
overwrites it as the layers run. So layers 3 to 7 reuse layer 2, layers 21
to 23 reuse layer 20, and layers 25 to 27 reuse the top-k of layer 24.

Every top-k index already points into the joined `[window ; compressed]` KV
of one attention call. The offset is the length of the window part: the whole
chunk during prefill, and the window size (128) during decode. All layers of
one forward pass use the same offset, so reused indices stay valid.

Layer 20 builds the candidate pool. Only index layers after it use the pool
(24, 28, 32 and 36). The encoder indexers (2, 8 and 14) do not.

Decision: one attention block class with a static role and the ids of its
source layers. A small per-forward object carries the shared views (compressed
KV, index K, top-k, candidate mask) down the stack, in place of the
reference's process-wide slot.

### 3. Compressor

- Ratio 2 (layers 2, 8 and 14): `wkv` and `wgate` are `[512, 5120]`. The
  checkpoint stores them in BF16, and the reference runs them in FP32. A
  group of 2 tokens pools into one entry with a softmax over the gate scores,
  per channel. Then come an RMS norm and RoPE on the last 64 dims, at the
  position of the group's first token. There is no overlap window and no
  position bias. These are the V4 CSA parts that V4.1 drops.
- Ratio 1 (layer 20): a plain BF16 projection and an RMS norm. There is no
  gate and no accumulator, and one entry per token.
- Prefill keeps a trailing partial group in the accumulator. In decode, a
  group closes when `(start_pos + 1) % 2 == 0`, and the entry takes RoPE
  position `start_pos + 1 - ratio`.
- A query at position `t` sees compressed entries `j < (t + 1) // ratio`.

Port change: `TokenLevelCompressor` gets a V4.1 mode without overlap and
without position bias, plus a ratio-1 path.

### 4. Indexer

- Query: `wq_b(qr)` gives 32 heads of 128 dims. RoPE goes on the last 64
  dims at the query positions. Then an FP4 round trip with one power-of-two
  (E8M0) scale per 32 values.
- Key (Full layers only): `k_norm(wk(latent))`, where `latent` is the
  compressed entry before RoPE and `wk` is `[128, 512]` in BF16. RoPE goes on
  the last 64 dims at the entry's position. Then the same FP4 round trip.
- There is no Hadamard rotation. V4 rotated both query and key. DeepInfer's
  fused kernel still has a Hadamard option, but no V4.1 code path sets it.
- Weights: `weights_proj(x)` (BF16, one per head), times
  `index_head_dim ** -0.5 * n_heads ** -0.5`.
- Score of position `j`: the sum over heads of `relu(q_h . k_j) * w_h`.
  Under tensor parallelism, each rank sums its heads and an all-reduce adds
  the ranks.
- Candidate pool (layer 20): each block of 8 positions scores as its best
  position. The block that holds the newest visible position always stays in
  (its score is set to plus infinity). The pool keeps the 2,048 best blocks
  and drops picks that no position can reach. Reindex layers set every
  position outside the pool to minus infinity before their own top-k.
- Top-k: `min(512, end_pos // ratio)` picks, sorted by position. A pick
  that the query cannot see becomes -1.

Tie order is a real problem. The reference calls `torch.topk`, and on tied
scores `torch.topk` picks indices that depend on the row width. Prefill and
decode use different widths, so they pick different positions. The ReLU makes
exact zeros common, which makes ties common. In the probe, with quantization
off, `prefill(n)` and `prefill(n0)` plus single-token decodes differed by up
to 0.54 in the logits. With a stable top-k where the lower index wins a tie,
the logits agreed within 2.4e-6. In a run with 32 indexer heads, all 120
index picks of the two paths matched.

Decision: the port breaks ties toward the lower index at both top-k sites,
as the V4 indexer already does (`_topk_lower_index_on_ties`). The Stage B
harness patches the reference's `topk` the same way.

### 5. Quantization numerics

| Tensor | V4 | V4.1 |
| :--- | :--- | :--- |
| Window KV | FP8 E4M3, power-of-two scale per 64, non-RoPE dims only | FP8 E4M3, power-of-two scale per 32, whole 512-wide vector, RoPE dims included |
| Compressed KV | FP8 E4M3 per 64, non-RoPE dims only | FP4 E2M1 with one E4M3 scale per 16, whole vector, after RoPE |
| Indexer query and key | Hadamard, then FP4 E2M1 with E8M0 scale per 32 | FP4 E2M1 with E8M0 scale per 32, no Hadamard |
| Attention output before `wo_a` | not quantized | FP8 E4M3, power-of-two scale per 32 |
| Inputs of FP8 and FP4 GEMMs | FP8, block 128 | FP8, block 32 |

The new compressed-KV format computes its scale as `e4m3(amax / 6)`, with
`amax` at least `6 * 2^-9`. The value is then divided by that scale, clamped
to plus or minus 6, rounded to E2M1 and multiplied back. The scale is not a
power of two, so the division is not exact. The reference divides, so the
port divides too.

Quantization makes the model's output jump at rounding boundaries. Float
noise of about 1e-7, from a different summation order, can move a value
across a boundary. In the probe, with the round trips on, prefill and decode
of the reference itself stored index-key rows that differed by 0.25 (one FP4
step). That changed 7 of 120 index picks, and the logits moved by up to 0.32.
So no port can match the quantized reference exactly over a long run.

Decision: the parity contract in section 10 separates quantization-off
equality from quantization-on agreement. The first port stores compressed
KV and index K in BF16 (working precision). The `simulate_act_quant` mode
applies the round trips where the reference applies them. Real FP4 storage
waits for Stage E. At our scale, BF16 costs about 3.2 KB per token, against
890 bytes for FP4: 6.7 GB for 16 requests of 128K tokens.

### 6. Engram hashing

The rebuild matched the checkpoint exactly:

- The compressed vocabulary has 99,092 entries, the value in the
  configuration. The tokenizer normalizer chain is NFKC, NFD, strip accents,
  lowercase, collapse whitespace, keep a single space through a sentinel,
  strip. A token that decodes to a broken UTF-8 byte keeps its raw token
  string as its key. This used `tokenizers` 0.22.2.
- Each module has 3 n-gram sizes (2, 3 and 4) times 8 heads, so 24 hash
  columns. Each column has its own prime bucket count. The primes start
  after 15,999,999, grow strictly and are never reused. Layer 1 uses 16,000,057
  to 16,000,463, and layer 14 uses 16,000,477 to 16,000,889. The table rows
  of a module equal the sum of its 24 primes: 384,006,168 and 384,016,682,
  the `engram_num_embeddings` values.
- The multipliers come from `numpy.random.default_rng(10007 * layer_id)`:
  four integers below `(int64 max // 99092) // 2`, each doubled plus one.
- The hash of an n-gram is the XOR of `token * multiplier` over the last n
  tokens, modulo the column's prime, plus the column's offset. A token before
  the sequence start, or on the far side of an image span, becomes the
  compressed id of the pad token (id 2).

The lookup reads 24 FP8 rows of 256 values with one E8M0 scale per 32, and
dequantizes them to BF16 (6,144 values). `wkv` (FP8, `[25600, 6144]`) turns
them into a key per copy (4 x 5,120) and one value (5,120). For each copy,
the gate is `sigmoid(signed_sqrt(dot))`. Here `dot` is the sum of
`h * q_weight * k_weight * key`, scaled by `5120 ** -0.5` and by the RMS
reciprocals of `h` and `key`. Each copy becomes `h + gate * value`. Image
tokens get a gate of 0.

Decision: an `Engram` module and a per-request hash state. The state needs
only the compressed ids of the last 3 tokens. The reference keeps the whole
sequence. The table sits behind one interface with two backends: a device
tensor (CPU tests, and 4x B200) and a pinned host table with a CPU gather
(2x B200). The hash constants are built at load time from the configuration
and the tokenizer, and the load fails if the vocabulary size does not match.

### 7. Router and experts

`sqrtsoftplus` of FP32 logits (`x.float() @ W.float().T`). The bias picks
the experts, but the routing weights are the raw scores of the picks. The
weights are normalized by their sum plus 1e-20 and multiplied by 1.5. Tokens
inside an image span use `bias_vl` in place of `bias`. There are no
hash-routed layers. The SwiGLU clamp is the V4 one (up branch within plus or
minus 10, gate branch at most 10, in FP32). The DSpark layers use 128 experts
with 3 active.

### 8. Encoder-decoder boundary and bounded replay

The Hugging Face configuration has no boundary field. DeepInfer reads
`n_prefill_layer` from its own configuration. The port derives the boundary
as `num_hidden_layers // 2` (20). It asserts that this equals both the last
`kv_source_layer_ids` entry and `candidate_source_layer_id`.

DeepInfer's decoder bounded replay has three modes. `off` runs the full
prefill. `mask` keeps every query but narrows each decoder query's window.
`compact` drops the decoder queries outside the kept suffix. In both `mask`
and `compact`, the window of a decoder query is the normal window
intersected with the last `KEEP_TOKENS` positions of the prompt. Layer 20
still projects the global KV of every prompt token from the encoder output.
`KEEP_TOKENS` must be at least the window plus the maximum DSpark rollback
(draft length plus 1). The README default is 140. `compact` requires every
Engram layer in the encoder (true) and every DSpark tap on one side of the
boundary (true).

Decision: the reference `model.py` runs the full prefill, and so does the
port. Bounded replay is a Stage E option that is off by default. It changes
outputs by design.

### 9. DSpark drafter

- The draft input is the last token followed by 4 noise tokens (id
  128,799). A shared embedding and head serve the draft and the target.
- Attention is sliding-window only. The ring holds the projected target
  taps (`main_norm(main_proj(taps))`), one entry per target position. The 5
  draft positions also see each other in both directions. Prefill fills the
  ring with the last 128 prompt positions.
- The Markov head (rank 256) adds a bias from the previous token to each
  position's logits. Sampling runs position by position.
- The confidence head reads the hidden state before the final norm, joined
  with the Markov embedding of the previous token. Our Qwen3 drafter
  (ADR-027) reads the state after the final norm. This is a real difference.
- The reference has no verify loop and adds one ring entry per target step.
  The port reuses the verify loop and the multi-request scheduling of
  ADR-027 and ADR-031, and writes one ring entry per verified token.

### 10. Model class, state cache and parity contract

- A new `DeepseekV41ForCausalLM` in `models/deepseek_v41.py`, with its own
  configuration class that reads `text_config`. It reuses the unchanged V4
  blocks: attention sink, grouped output projection, RoPE tables with YaRN,
  RMS norm, the Sinkhorn split, the MoE experts with W4A8 and the shared
  expert.
- `StateLayerSpec` gets the id of the layer that owns the compressed KV and
  index K. A borrowing layer allocates zero-width compressed buffers and no
  indexer state. `max_seq_len` skips borrowing layers. The index K state
  needs no compressor accumulator in V4.1. Prefix sharing (ADR-020) works
  unchanged, because it snapshots and restores each layer, and zero-width
  copies cost nothing. The prefill/decode split path (ADR-016) does not
  serve `StateCache` models today, so V4.1 needs no layout work there.
  **Gate:** not triggered.
- The reference runs only a prefill from position 0 and single-token decode
  steps. The port also needs chunked prefill and multi-token verification.
  Both must equal the token-by-token decode. The probe shows that the
  reference's prefill and decode agree within 2.4e-6 when ties are stable
  and quantization is off, so this contract is well defined.

Parity contract for Stage B:

1. Quantization off, stable top-k: every block and the whole model match
   the patched reference within the tolerances of the V4 parity tests. This
   covers prefill, single-token decode, chunked prefill and multi-token
   decode.
2. Quantization on: each round trip and each block matches when both sides
   get the same inputs. For the whole model, report agreement (top-k overlap,
   logit cosine, greedy tokens), with no hard equality.
3. Engram: equal hash ids for every position of the token streams in the
   checkpoint's encoding fixtures, and equal module outputs with a small
   random table.
4. DSpark: equal base logits, Markov-corrected logits and confidence per
   draft position.

## Alternatives Considered

- **Role flags on `DeepseekV4ForCausalLM`.** Rejected. V4.1 changes the
  layer roles, the hyper-connection order, the compressor, the indexer and
  the head, and it adds Engram, DSpark and vision. Flags touch almost every
  V4 path and put the GPU-validated V4 results at risk. Shared blocks
  get small additive options instead.
- **Keep the reference's process-wide shared slot.** Rejected. An explicit
  per-forward object shows the data flow at the call site, and it does not
  leak state between calls.
- **Store FP4 caches from the start.** Deferred to Stage E. FP4 storage needs
  a dequantize-on-load path in the decode kernel, and BF16 fits at our
  scale.
- **Keep the raw `torch.topk` tie order for bit parity with the reference.**
  Rejected. The reference's own prefill and decode disagree under ties, so
  that order is not a stable target.
- **Engram on the GPUs on 2x B200.** Not possible. The other weights take
  307 GB and the tables 203 GB, against 384 GB of HBM.
- **Bounded replay on by default.** Rejected for now. Parity comes first,
  and the reference runs the full prefill.

## Consequences

- Stage B can start: the eight unknowns have answers, and the parity contract
  is fixed.
- Parity tests must patch the reference's `topk`. A test that compares the
  raw reference across prefill and decode fails by design.
- With the quantization round trips on, prefix sharing replays the suffix
  with decode steps, so it can differ from a fresh prefill at rounding
  boundaries. With the round trips off and stable ties, it stays exact.
- Every checkout needs the reference in `third_party/deepseek_v41_reference/`,
  or the parity tests skip.
- The engine has no Hugging Face golden model for V4.1 (transformers 5.20.2
  has no V4.1 class). The reference `model.py` is the only golden source.

## Pointers

- Reference: `third_party/deepseek_v41_reference/` (`model.py:429` Compressor,
  `model.py:488` Indexer, `model.py:584` `select_candidate_blocks`,
  `model.py:614` Attention, `model.py:909` Block, `model.py:1034` DSpark,
  `engram.py`, `kernel.py:127` FP4 round trip).
- DeepInfer: `deepinfer/environ.py` (bounded replay settings),
  `deepinfer/model_executor/input_metadata/yoco.py`,
  `deepinfer/model_executor/models/deepseek.py` (`_init_yoco`).
- In-repo: `src/mini_infer/cache/state_cache.py`,
  `src/mini_infer/cache/state_prefix_cache.py`,
  `src/mini_infer/models/blocks/v4/compressor.py`,
  `src/mini_infer/models/blocks/v4/lightning_indexer.py`,
  `src/mini_infer/models/blocks/hyper_connections.py`,
  `tests/unit/_v4_reference_helpers.py`.

## Follow-ups

- Stage B: the text-only port with the parity contract above.
- Stage D: make sure that a Modal B200 container can get about 210 GB of
  host RAM for the Engram tables.
- Stage E: FP4 cache storage, bounded replay, and the batched DSpark drafter.
