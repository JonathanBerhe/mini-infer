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
| YaRN long-context RoPE | `precompute_freqs_cis` (L200) | `RotaryEmbedding` in `blocks/rope.py`, two tables per model (see §3.1) |
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

Every expert, routed and shared, clamps its two projections before the
SiLU when `swiglu_limit` is positive. The gate projection is capped at
`swiglu_limit`, and the up projection is clamped to `[-swiglu_limit,
swiglu_limit]`. Released V4 configs set 10. The port reads the value
from the HF config into `DeepseekV4Config.moe_swiglu_limit` and applies
it through `activations.clamped_swiglu`. A value of 0 keeps plain
SwiGLU.

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

**Our code (`blocks/rope.py`, `RotaryEmbedding`)**: same primitive.
YaRN kicks in when the requested `seq_len > yarn_original_seq_len`.
The same `RotaryEmbedding` block serves V2 / V3 / V4 (V2 / V3 don't
use YaRN by default; V4 does).

**Two tables per model.** The reference builds the table per attention
layer (`Attention.__init__`). A compressed layer (CSA or HCA) rotates
with `compress_rope_theta` and YaRN. A pure sliding-window layer
rotates with `rope_theta` and no YaRN. V4-Flash sets 160000 and 10000.
`DeepseekV4ForCausalLM` holds `window_rotary_emb` and
`compress_rotary_emb` (built by `build_rotary_tables`), and
`rotary_for_layer` picks one per layer. The compressed entries and the
Lightning Indexer of a compressed layer use that layer's table. An
earlier version used one `rope_theta` table with YaRN for every layer.
On an 8,192-token prompt that version scored a perplexity of 247
instead of 4.4 (`docs/benchmarks/2026-10-19-v4-projection-dtype.md`).
`tests/unit/test_v4_rope_tables.py` compares each layer's table with
the reference's `precompute_freqs_cis`.

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

## Activation quantization modes

DeepSeek trained V4 with quantization-aware training (QAT): training
rounded some attention activations to low-precision formats. The
reference inference code repeats that rounding. It quantizes a tensor
and converts it straight back (a round trip), so attention sees the
rounding that the model learned with. Three points get a round trip:

| Where (reference) | Reference call | Round trip |
|---|---|---|
| Window KV after RoPE (`Attention.forward`) | `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` | FP8 e4m3 on the non-RoPE dims, one scale per 64 values |
| Main compressor output after RoPE (`Compressor.forward`, `rotate=False`) | same call | same |
| Indexer keys and indexer queries (`Compressor.forward` with `rotate=True`, `Indexer.forward`) | `rotate_activation(x)`, then `fp4_act_quant(x, 32, True)` | Hadamard rotation, then FP4 e2m1 on all dims, one scale per 32 values |

The RoPE dims of the KV stay unquantized. With `scale_fmt="ue8m0"`,
which every released V4 config sets, each FP8 scale is rounded up to a
power of two. FP4 scales are always powers of two.

A released checkpoint adds a second kind of rounding. Its linear
weights are FP8 (one scale per 128x128 tile), and its routed experts
are FP4 (one scale per 32 values). For these weights, the reference's
`linear()` quantizes the GEMM input to FP8 with one scale per 128
values. It then calls `fp8_gemm` or `fp4_gemm`. These kernels multiply
the dequantized input by the dequantized weight and sum in FP32. The
GEMMs that take an FP8 input are:

| Reference linear | Port module | Weight in a release |
|---|---|---|
| Attention `wq_a`, `wq_b`, `wkv`, `wo_b` | `q_a_proj`, `q_b_proj`, `swa_kv_proj`, `grouped_output.wo_b` | FP8 |
| Indexer `wq_b` | `indexer.wq_b` | FP8 |
| Shared expert `w1`, `w2`, `w3` | `mlp.shared_experts.w1` / `w2` / `w3` | FP8 |
| Routed experts `w1`, `w2`, `w3` | `mlp.experts.N.w1` / `w2` / `w3` (`FP4Expert`) | FP4 |

The gate, the compressors, the indexer `weights_proj` and `wo_a` hold
BF16 or FP32 weights in the reference. Their inputs stay unquantized.

Two config flags select one of three modes:

- Both off (the default). The port skips all of this. Window KV,
  compressed KV, the Lightning Indexer and every GEMM input run at the
  working precision (FP32 on CPU, BF16 on GPU). This is more precise
  than the reference's real inference path. The identity-stub parity
  tests pin this mode: they replace the reference quantizers with
  functions that return their input.
- `simulate_act_quant=True`. The port applies the QAT round trips at
  the same points as the reference. This covers prefill, cache-aware
  prefill, decode and ragged decode, in SWA, CSA and HCA layers. This
  mode matches a reference run on a BF16 checkpoint.
  `act_quant_scale_fmt` mirrors the reference's `scale_fmt` (`"ue8m0"`
  or `None`).
- `simulate_act_quant=True` and `simulate_fp8_gemm_inputs=True`. The
  port also gives each GEMM in the table an FP8 round trip of its input,
  one scale per 128 values. This mode matches a reference run on a
  released FP8 / FP4 checkpoint. The port dequantizes the weights at
  load, so the round trip followed by the normal matmul multiplies the
  same numbers as the kernels. The second flag needs the first, because
  the reference never quantizes GEMM inputs without the QAT round trips.

The code is in `blocks/v4/act_quant.py`. It transcribes the tilelang
kernels and the Hadamard transform into PyTorch. Like the kernels, it
computes in FP32 and casts back to the input dtype. Each attention
block, the indexer and the MoE FFN take `act_quant: SimulatedActQuant
| None`, and `None` means off. For the GEMM inputs, the constructors
register a forward pre-hook on each linear in the table, so the forward
code does not change. `FP4Expert` has no linear modules and applies
the round trip in its `forward`.

With GEMM inputs quantized, the routed experts also change their order
of operations. The port normally scales each expert's output by its
routing weight. The reference scales the activation before `w2`
instead. Without quantization the two orders give the same result. With
quantization they do not, because `w2` then rounds the scaled input. So
in this mode the port scales before `w2`, like the reference.

The modes need some dimensions to split into whole blocks.
`kv_head_dim - rope_head_dim` must be a multiple of 64, and
`index_head_dim` must be a power of two and a multiple of 32. V4 meets
both (448 and 128). Each GEMM input width in the table must be a
multiple of 128. The reference's `act_quant` asserts this itself, so
every released config meets it. The block constructors check all of
these.

The QAT round trips matter most for the indexer. FP8 moves each KV
value by a small relative amount, because e4m3 keeps 3 mantissa bits.
FP4 is coarse: each e2m1 value is 0, 0.5, 1, 1.5, 2, 3, 4 or 6 times
its block scale. That rounding changes the indexer scores enough to
change which compressed blocks the top-k selects. On the synthetic
parity config, the default mode against the quantized reference fell
to cosine 0.955 on one CSA decode step. With the QAT round trips on,
the port stayed within a relative error of 2e-7.

The GEMM-input round trips matter as much. Against a reference built
with FP8 weights, the QAT-only mode missed by a relative error of 3e-2
to 3.6e-1, and CSA fell to cosine 0.951. With both flags on, the port
stayed within 9e-8 for attention and 1.1e-7 for the MoE. Scaling the
routed experts after `w2` instead missed by 1.6e-2.

The quantized-mode parity tests (`pytest tests/unit -k act_quant`) run
the reference with real PyTorch mirrors of its quantizer and GEMM
kernels in place of the identity stubs. The GEMM tests build the
reference with FP8 linears and FP8 or FP4 experts, and load the same
weights into the port the loader's way. In both quantized modes, the
port matches the reference's quantized KV dims and indexer keys bit for
bit, and its indexer top-k picks exactly. The RoPE dims differ only by
last-bit RoPE rounding.

Some limits apply:

- The kernels sum each 128-wide K tile (32-wide for FP4), then scale
  and add the tiles. The port runs one matmul on the dequantized values.
  When an input spans several tiles, the two sums agree to FP32
  rounding but not bit for bit.
- The CPU tests compare against PyTorch mirrors of the kernels, not
  against a GPU run. The CUDA Hadamard kernel sums in a different
  order. A value that falls exactly on an FP4 rounding boundary can
  then round the other way, so bit equality with a GPU run is not
  guaranteed.

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
