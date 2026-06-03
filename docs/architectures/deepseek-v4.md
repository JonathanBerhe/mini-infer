# DeepSeek-V4 walkthrough

A line-by-line correspondence between the DeepSeek-V4 paper, the upstream
[V4-Pro inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference),
and mini-infer's from-scratch port.

This is the realization of mini-infer's research-paper-inference-engine
niche for V4: every paper primitive is implemented in clean Python with
a pointer back to the reference's tilelang / PyTorch source, and a
bit-parity test against the reference where applicable.

> **Audience**: someone with the V4 paper open who wants to follow the
> implementation. Pair this doc with [ADR-014](../decisions/ADR-014-deepseek-v4-hybrid-attention.md)
> (decisions made along the way, why we deviated where we did).

## TL;DR

V4 introduces five novel primitives that distinguish it from V3 / MLA-family
models:

1. **Hybrid attention** with per-layer dispatch (SWA / CSA / HCA).
2. **Token-level compressor** for long-range KV access.
3. **Lightning Indexer** for sparse attention top-k selection.
4. **Attention Sink** for numerical stability of softmax with long contexts.
5. **Grouped output projection** that collapses many small heads into few groups.
6. **Hash-routed MoE** with per-token-id expert assignment (vs score-topk).
7. **Hyper-Connections** for cross-layer residual mixing via Sinkhorn-normalized doubly-stochastic matrices.
8. **YaRN long-context RoPE** with wave-frequency correction past the original train length.

Bit-parity is validated for HCA + CSA forward (prefill + decode), the
MoE gate / expert routing matrix, and `Block.hc_pre` / `Block.hc_post`
against the reference at FP32 with cosine-sim 1.0000 and max abs diff
< 1e-9.

## The map

| Paper primitive | Reference (`third_party/deepseek_v4_reference/model.py`) | Our code |
|---|---|---|
| Backbone | `class Transformer` (L769) | `DeepseekV4ForCausalLM` in `src/mini_infer/models/deepseek_v4.py` (L586) |
| Decoder layer | `class Block` (L647) | `Block` in `src/mini_infer/models/blocks/deepseek_v4_decoder_layer.py` |
| Token-level compressor | `class Compressor` (L279) | `TokenLevelCompressor` in `blocks/v4/compressor.py` (L43) |
| Lightning Indexer | `class Indexer` (L380) | `LightningIndexer` in `blocks/v4/lightning_indexer.py` (L61) |
| Attention Sink | inside `Attention` (L436) | `AttentionSink` in `blocks/v4/sink.py` |
| Grouped output projection | `Attention.wo_a` / `wo_b` (L520-540) | `GroupedOutputProjection` in `blocks/v4/grouped_output.py` |
| HCA (full long-range attention) | `Attention.forward` with `m'` (L436) | `HCAAttention` in `blocks/hca.py` (L204) |
| CSA (chunked SWA + Indexer) | `Attention.forward` with overlap (L436) | `CSAAttention` in `blocks/csa.py` |
| SWA (ratio=0 fallback) | implicit in `Attention` | `SWAAttention` in `blocks/swa.py` |
| Hash-routed gate | `class Gate` (L546) | `HashRoutedGate` in `blocks/hash_routed_gate.py` |
| Hash-routed MoE FFN | `class MoE` (L609) | `HashRoutedMoEFFN` in `blocks/hash_routed_moe_ffn.py` |
| Hyper-Connections | `Block.forward` mixing (L676-700) | `HyperConnections` in `blocks/hyper_connections.py` (L137); core kernel `hc_split_sinkhorn` (L56) |
| HC head reduction | `Transformer.forward` final mix (L800-820) | `HCHeadReduction` in `blocks/hyper_connections.py` (L265) |
| YaRN long-context RoPE | `precompute_freqs_cis` (L200) | `RotaryEmbedding` in `blocks/rotary.py` |
| Per-request state | `Attention` mutable buffers (single-request) | `StateCache` in `cache/state_cache.py` + `build_state_cache_layer_specs` |

Bit-parity validation is live in `tests/unit/` for every primitive in
the table above; the test layout is intentionally not enumerated here
because it churns faster than the architecture itself. Run
`uv run pytest tests/unit/ -k 'v4 or hca or csa or hyper or hash_routed'`
to see which primitives are exercised.

## §2.1 — Backbone

The V4 backbone is a stack of `Block`s with three primitives per block
(attention, FFN, residual mixing) plus a head reduction at the end.

**Reference (`model.py:769`)**:
```python
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        # embedding + N x Block + RMSNorm + ParallelHead
```

**Our code (`src/mini_infer/models/deepseek_v4.py:586`)**:
```python
class DeepseekV4ForCausalLM(BaseCausalLM):
    # embed_tokens + _DeepseekV4InnerModel + lm_head
    # _DeepseekV4InnerModel is the stacked Block list (line 525)
```

**Key dispatch**: each layer reads `config.compress_ratios[layer_idx]`
and instantiates one of `{SWAAttention, CSAAttention, HCAAttention}`.
The reference does this in `Transformer.__init__` (L769-790); we do
it in `_DeepseekV4InnerModel.__init__`. See ADR-014 for the dispatch
rationale.

## §2.2 — Hybrid attention dispatch

V4's signature design: each layer picks one of three attention modes
based on its compress-ratio `m`.

```
compress_ratio m = 0  → SWA  (sliding window only, ratio=0 layers in V4-Flash head)
compress_ratio m = 4  → CSA  (chunked SWA + Lightning Indexer + token-compressor)
compress_ratio m = 128 → HCA (full long-range attention over compressed tokens)
```

**Reference**: `class Attention` (L436) has all three modes in one class,
branched on `args.compress_ratios[layer_idx]`. The forward pass picks
between `local_attention` (SWA-style) and a compressed-history path.

**Our code**: three separate classes for the three modes.
- `SWAAttention` (`blocks/swa.py:14`)
- `CSAAttention` (`blocks/csa.py:25`)
- `HCAAttention` (`blocks/hca.py:204`)

Three files because the per-mode forward + per-mode KV-cache layout
differ enough that a single class would obscure the math. The shared
primitives (Q-LoRA, partial RoPE, AttentionSink, GroupedOutputProjection)
are factored out into `blocks/v4/`.

### Token-level compressor (paper §2.2)

Compresses every `m` consecutive tokens into one compressed token via a
small learned projection + softmax. The compressed sequence is what HCA
attends to for the long-range context (allowing 128× fewer K/V slots
to cover the full history).

**Reference (`model.py:279`, `class Compressor`)**:
```python
def forward(self, hidden_states, start_pos, ratio):
    # Project to (kv, score) of `m * (kv_head_dim + 1)` features
    # Softmax over m-token windows
    # Sum-pool weighted by scores
```

**Our code (`blocks/v4/compressor.py:43`, `class TokenLevelCompressor`)**:
- `forward` (L113): the textbook prefill compressor.
- `forward_prefill_with_cache` (L165): cache-aware prefill (incremental
  prompt processing for chunked-prefill scheduling).
- `forward_decode_step` (L288): single-token-step incremental compressor
  for the decode loop.

The overlap mode (paper §2.2.1) lets compressed tokens share boundaries
across batches. Our `_overlap_transform` (L97) matches the reference's
overlap rotation precisely; this is exercised by the CSA bit-parity
test in `tests/unit/`.

**Key V4-specific detail**: the compressor projections are done in the
input dtype (BF16 for production, FP32 for tests), then cast to FP32
for the softmax + position-bias math, then cast back to the input dtype.
The "cast inside the matmul vs outside" decision was a bug fix; see the
`b30d3fe` commit and its ADR-014 follow-up note.

### Lightning Indexer (paper §2.3)

A small per-head dot-product top-k selector that picks which K positions
each Q should attend to. Replaces full attention with sparse attention
over the indexer's top-k indices.

**Reference (`model.py:380`, `class Indexer`)**:
```python
def forward(self, hidden_states, prev_kv, start_pos):
    # Project Q (per-indexer-head) and K (shared)
    # ReLU dot product
    # Top-k along K axis
    # Returns (topk_indices, scores)
```

**Our code (`blocks/v4/lightning_indexer.py:61`, `class LightningIndexer`)**:
- `forward` (L134): standalone prefill (full-attention indexer for
  small contexts).
- `forward_prefill_with_cache` (L216): cache-aware prefill where the
  indexer also maintains a per-request sub-cache.
- `forward_decode_step` (L316): per-token incremental indexer for decode.

### Attention Sink (paper §2.3)

A learnable per-head logit that's added to the softmax denominator. Lets
the model "attend to nothing" without producing degenerate distributions
when no key matches the query well — critical for long-context stability.

**Reference**: defined inside `class Attention` (L436) as a parameter
`attn_sink: shape (n_heads,)`. Added to the softmax denominator via
`exp(sink) + sum(exp(scores))`.

**Our code (`blocks/v4/sink.py`, `class AttentionSink`)**: same math,
small class so it's reusable across HCA / CSA / SWA. Per-head learnable
logit, added to softmax denom.

### Grouped Output Projection (paper §2.3)

V4 uses many attention heads (`n_h`) but reduces them through groups
before the output projection. The output projection has two parts:
`wo_a` collapses heads to groups, `wo_b` projects groups to hidden_size.

**Reference (`model.py:520-540`)**: `Attention` has `wo_a` and `wo_b`
projections.

**Our code (`blocks/v4/grouped_output.py`, `class GroupedOutputProjection`)**:
same two-stage projection; exposed as a reusable block.

## §2.3 — HCA: full long-range attention

HCA (Hybrid Compressed Attention with `m' = 128`) layers attend to the
*entire* compressed history (every K position is a compressed token
representing 128 raw tokens). No indexer is used here because the
compressed sequence is already short.

**Reference (`Attention.forward` with `ratio=128`)**: the K side of
attention uses the compressor's output as the K cache. Q stays at full
resolution.

**Our code (`blocks/hca.py:204`, `class HCAAttention`)**:
- `forward` (L284): single-shot prefill.
- `forward_prefill_with_cache` (L365): cache-aware prefill.
- `forward_decode_step` (later in file): per-token decode.

**Helper functions** for K-side index computation:
- `_build_window_topk_idxs` (L84): for SWA windowing.
- `_build_window_decode_topk_idxs` (L107): decode-time variant.
- `_build_hca_topk_idxs` (L169): the K positions HCA attends to (the
  full compressed history).
- `_build_hca_decode_topk_idxs` (L142): decode variant.

## §2.3 (cont.) — CSA: chunked SWA + Indexer

CSA (Compressed Sparse Attention with `m = 4`) layers do sliding-window
attention over the raw tokens *and* sparse attention over the indexer-
selected compressed history. Two attention paths fused.

**Reference**: same `class Attention`, but with `ratio=4` and the
indexer active.

**Our code (`blocks/csa.py:25`, `class CSAAttention`)**: same shape as
HCA, plus the indexer wiring + the overlap-mode compressor for the
in-flight chunk.

## §2.3 (cont.) — SWA: ratio=0 fallback

V4-Flash starts its 43-layer stack with 2 SWA layers (compress_ratio=0)
before transitioning to CSA + HCA. SWA is pure sliding-window attention
with no compressor and no indexer.

**Reference**: implicit — `ratio=0` is handled inside `class Attention`
by skipping the compressed path.

**Our code (`blocks/swa.py:14`, `class SWAAttention`)**: separate file
to keep the V4-Flash head clean. The "no compressor, no indexer" path
is much simpler than HCA / CSA and benefits from its own implementation
rather than a branch inside a shared class.

## §2.2 — Hash-routed MoE

V4's MoE has two routing modes:
1. **Hash routing**: the first `num_hash_routed_layers` (3 in V4-Flash)
   use a `tid2eid` table — a fixed lookup from token-id to expert-id.
   Deterministic, no learnable gate.
2. **Score-topk routing**: every other layer uses learned scores +
   top-k expert selection. Three score functions: `softmax`, `sigmoid`,
   `softplus_sqrt`.

### Gate

**Reference (`model.py:546`, `class Gate`)**: handles both routing modes
in one class via a `routing_mode` flag.

**Our code (`blocks/hash_routed_gate.py`, `class HashRoutedGate`)**:
same two-mode design. The `score_func` parameter picks between the three
score functions (paper §2.2 specifies different ones for different
layers).

### FFN

**Reference (`model.py:609`, `class MoE`)**: routed experts + shared
experts. Shared experts are collapsed into a single MLP (V2/V3 trick).

**Our code (`blocks/hash_routed_moe_ffn.py`, `class HashRoutedMoEFFN`)**:
same shape. Routed experts use the standard Mixtral-shape expert MLP
(`blocks/mixtral_moe.py::MixtralExpert`); the shared experts are
collapsed into one MLP whose intermediate_size is `n_shared * shared_intermediate`.

## §2.5 — Hyper-Connections

V4's most novel residual scheme. Each block has `hc_mult` (= 4 in
V4-Flash) "copies" of the hidden state that are mixed via a doubly-
stochastic matrix produced by Sinkhorn normalization.

### The kernel

**Reference**: a tilelang kernel `hc_split_sinkhorn` (in the reference
repo's kernel.py / model.py mixing math). Performs `n` Sinkhorn
iterations on a learnable mixing matrix to produce a doubly-stochastic
matrix.

**Our code (`blocks/hyper_connections.py`, `hc_split_sinkhorn`)**:
two implementations behind one dispatcher.

- `_hc_split_sinkhorn_torch`: line-by-line transcription of the
  reference's tilelang kernel into pure PyTorch. Same fp32
  accumulator, same iteration count (`hc_sinkhorn_iters = 20`), same
  `eps = 1e-6`. The numerical oracle and the CPU / MPS path. See
  ADR-014 for the original transcription decision.
- `hc_split_sinkhorn_triton` (`blocks/hc_sinkhorn_kernel.py`): fused
  Triton kernel, CUDA fast path. One launch replaces ~50 PyTorch op
  dispatches per call; measured ~14x per-call latency reduction on
  L40S at V4's `hc=4, iters=20`. Restored to kernel form because the
  reference itself is a kernel. The Sinkhorn loop uses a scale-vector
  formulation (loop carries 1D vectors, not the matrix) to dodge a
  Triton 3.1 compiler crash on reductions over loop-carried tiles;
  ADR-018 records the bisect and the parity contract.

### The block

**Reference (`Block.forward`, L676-700)**: each forward step does
`hc_pre` → attention → `hc_post` → FFN → `hc_post` (the "pre/post"
mixing happens before and after each sub-block).

**Our code (`blocks/hyper_connections.py:137`, `class HyperConnections`)**:
- `hc_pre` (L185): mixes the `hc_mult` copies before a sub-block.
  Returns `(branch_input, residual, mix_matrix)` so the sub-block runs
  on the mixed input, and `hc_post` can combine its output with the
  residual using the same `mix_matrix`.
- `hc_post` (L222): combines a sub-block's output with the residual
  via the mix matrix from `hc_pre`.

### The head reduction

**Reference (`Transformer.forward`, L800-820)**: at the end of the
stack, the `hc_mult` copies are collapsed back to one hidden state
before the LM head.

**Our code (`blocks/hyper_connections.py:265`, `class HCHeadReduction`)**:
the collapse step. `(B, T, hc_mult, dim) → (B, T, dim)`. Same math as
the reference; explicit class because the reduction is shared between
the LM-head path and any future probe of the per-copy state.

## §3.1 — YaRN long-context RoPE

YaRN is the long-context RoPE variant V4 uses. It interpolates RoPE
frequencies based on the position's distance past the original train
length, with a wave-frequency correction that preserves the
short-range fidelity of standard RoPE while extending the context.

**Reference (`model.py:200`, `precompute_freqs_cis`)**: builds the
inv-freq table with a linear ramp between `beta_fast` and `beta_slow`
that interpolates between extrapolation and linear-scaling.

**Our code (`blocks/rotary.py`, `RotaryEmbedding`)**: same primitive.
YaRN kicks in when the requested `seq_len > yarn_original_seq_len`.
The same `RotaryEmbedding` block serves V2 / V3 / V4 (V2 / V3 don't
use YaRN by default; V4 does).

## Decoder-layer assembly

**Reference (`model.py:647`, `class Block`)**:
```python
def forward(self, h, start_pos, freqs_cis):
    attn_in, attn_residual, mix = self.hc_pre(h)
    attn_out = self.attn(attn_in, ...)
    h = self.hc_post(attn_out, attn_residual, mix)
    ffn_in, ffn_residual, mix = self.hc_pre(h)
    ffn_out = self.ffn(ffn_in)
    h = self.hc_post(ffn_out, ffn_residual, mix)
    return h
```

**Our code (`blocks/deepseek_v4_decoder_layer.py`, `class Block`)**:
same shape. Wraps the per-layer attention type (SWA/CSA/HCA) +
HashRoutedMoEFFN (or SwiGLU for the dense layers) + HyperConnections.

Importantly: when `use_hyper_connections=False` (configs where HC is
disabled), `Block` degenerates to the pre-norm pattern (h = h +
attn(norm(h)); h = h + ffn(norm(h))). Same code path, HC is gated.

## Cache structure

V4 needs a richer per-request cache than V2/V3's MLA because each layer
type has its own state:

- **SWA layers**: sliding-window K/V buffer (circular).
- **CSA layers**: SWA buffer + compressor in-flight accumulator + indexer
  sub-cache.
- **HCA layers**: compressed-history K/V.

**Reference**: per-request state lives in the `Attention` instance via
mutable buffers (the reference is single-request; mini-infer is multi-
request).

**Our code**: `StateCache` in `src/mini_infer/cache/state_cache.py`
generalizes this to a multi-request engine. Each request has a `StateCacheRequest`
holding the per-layer state spec. `build_state_cache_layer_specs(config, ...)`
in `models/deepseek_v4.py:500` constructs the spec list from a V4 config.

## Validation

Bit-parity tests live under `tests/unit/` and follow the contract:

1. Construct our primitive + the equivalent reference primitive on the
   same synthetic input + same weights.
2. Run forward through both at FP32.
3. Assert cosine-sim 1.0000 (i.e. `1.0 - cos_sim < 1e-6`) and max abs
   diff < 1e-9.

Coverage spans the primitives in the map at the top of this doc. The
specific test file layout is intentionally not enumerated here: tests
churn faster than the architecture, and a static doc that lists test
files goes stale on every refactor. Run `uv run pytest tests/unit/`
with a relevant `-k` filter to see what's exercised for any given
primitive.

Integration-level tests (full forward end-to-end vs HF reference + the
V4-Pro inference reference) sit alongside the unit tests under the
same `tests/unit/` directory; the V4 model-level test is the
end-to-end gate.

## Where we diverged + why

Pulled out into [ADR-014](../decisions/ADR-014-deepseek-v4-hybrid-attention.md):

1. **Three attention classes instead of one**. Reference has one
   `class Attention` that switches on `ratio`. We split into
   `SWAAttention` / `CSAAttention` / `HCAAttention`. Cleaner per-mode
   forward; the shared primitives are in `blocks/v4/`.
2. **Tilelang → PyTorch transcription for `hc_split_sinkhorn`**. The
   reference uses a tilelang kernel; we transcribed it into plain
   PyTorch. We're FP32 / 20-iteration / `eps=1e-6` identical to the
   reference. A Triton port is a follow-up if profiling demands it.
3. **`StateCache` abstraction for per-request state**. Reference is
   single-request; mini-infer is multi-request. The per-layer state
   spec is generated at model build time.
4. **Compressor dtype handling**. Our compressor does the projection
   matmul in the input dtype, then casts to FP32 for the softmax +
   position-bias math, then casts back. The reference does this
   implicitly through tilelang's mixed-precision contract; we made the
   cast explicit. Bug commit `b30d3fe` documents how we got this wrong
   the first time.

## Pointers

- **Reference** (gold for bit-parity): `third_party/deepseek_v4_reference/model.py`.
  Original at the V4-Pro HF repo's `inference/` directory.
- **Paper**: `docs/papers/deepseek-v4.pdf`.
- **ADR-014**: design decisions made along the way.
- **Our model class**: `src/mini_infer/models/deepseek_v4.py`.
- **Per-primitive blocks**: `src/mini_infer/models/blocks/v4/` for the
  V4-specific primitives, `blocks/{hca,csa,swa,hyper_connections,
  hash_routed_gate,hash_routed_moe_ffn}.py` for the larger pieces.
- **Loader**: `DeepseekV4ForCausalLM.load_weights` handles the
  V4-Flash safetensors format (block-FP8 + NVFP4 dequant, reference-
  compact key rename, expert-parallel sharding, meta-device + per-slice
  GPU streaming). Validated against the real safetensors index.

## What's still open

- **Live V4-Flash forward on 2× B200**. Load is proven; one Modal run
  remains to close the end-to-end story. Blocked on Modal spend cycle.
  See `roadmap-2026.md` § Immediate.
- **Triton port of `hc_split_sinkhorn`**. PyTorch transcription is
  bit-parity correct but slower than the tilelang reference. Future
  follow-up when the V4 path is profile-bound.

## Adding the next architecture

Once a new architecture's reference is published, the steps to add a
walkthrough doc like this one:

1. Drop the reference into `third_party/<family>_reference/`.
2. Implement primitives in `src/mini_infer/models/blocks/<family>/`
   with bit-parity tests against the reference.
3. Compose into a `<family>ForCausalLM` class in
   `src/mini_infer/models/<family>.py`.
4. Register with `@register_model`.
5. Add the walkthrough doc here at `docs/architectures/<family>.md`,
   following this doc's structure.
6. Cross-link from `README.md` and `roadmap-2026.md`.

The 30-day target from `roadmap-2026.md` is from upstream-reference
availability to a green bit-parity + a published walkthrough doc.
