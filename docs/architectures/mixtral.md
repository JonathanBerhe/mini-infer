# Mixtral walkthrough

A line-by-line correspondence between the Mixtral 8x7B / 8x22B
release, HuggingFace transformers' `MixtralForCausalLM`, and
mini-infer's from-scratch port.

Mixtral is the **simplest MoE in mini-infer's registry**. Top-k
sparse routing (top-2 of 8), per-expert MLPs, no shared experts, no
routed scaling, no hash routing. Reading this walkthrough first makes
the V2/V3/V4 MoE variants (shared experts, hash routing, scoring
functions) easier to follow.

## TL;DR

A Llama-shape decoder where the dense FFN is replaced by a top-k
sparse MoE block. Everything else (GQA attention, RoPE, RMSNorm,
SwiGLU expert MLPs) is Llama-baseline.

The one primitive that makes Mixtral *Mixtral*:

1. **Top-k sparse MoE FFN.** A small linear router produces per-expert
   logits per token; the top-k experts are selected (Mixtral defaults
   to top-2 of 8), their softmax weights are renormalised to sum to 1
   over the chosen experts, and each chosen expert processes ONLY the
   tokens routed to it. Per-expert outputs are weighted-summed back.

That's it. The rest is Llama.

This is the FFN primitive every later MoE family in the registry
(V2 / V3 / Kimi-K2 / V4) extends.

## The map

| Primitive | Reference (HF `modeling_mixtral.py`) | Our code |
|---|---|---|
| Backbone | `MixtralModel` + `MixtralForCausalLM` | `MixtralForCausalLM` + `_MixtralInnerModel` in `src/mini_infer/models/mixtral.py` (L100, L74) |
| Decoder layer | `MixtralDecoderLayer` | shared `LlamaDecoderLayer` shape; MoE FFN swaps in for SwiGLU |
| GQA attention | `MixtralAttention` | `GQAAttention` in `blocks/gqa.py` (shared with Llama) |
| Top-k sparse MoE FFN | `MixtralSparseMoeBlock` | `MoEFFN` in `blocks/mixtral_moe.py` (L65) |
| Single expert MLP | `MixtralBLockSparseTop2MLP` | `MixtralExpert` in `blocks/mixtral_moe.py` (L46) |
| Router (gate) linear | `block_sparse_moe.gate` (no bias) | `MoEFFN.gate` (`nn.Linear`, no bias) |

Bit-parity is exercised against HF transformers' `MixtralSparseMoeBlock`
on synthetic input at FP32. Mixtral 8x7B (47B parameters) is too
large for M1 fp16 dev hardware; integration validation runs against
HF reference output at temperature=0 in CI for the parts that fit
locally, and on Modal for the full forward (~$1 per smoke).

## Top-k sparse MoE FFN

The whole story.

### The flow

For each token's hidden state of shape `(hidden_size,)`:

1. **Router** projects to per-expert logits: `gate(x) → (num_experts,)`.
2. **Softmax** over the `num_experts` axis. Renormalises the routing
   probabilities.
3. **Top-k** picks the `k = top_k` highest-probability experts.
4. **Per-expert dispatch**: for each chosen expert `j`, gather the
   tokens that picked `j` into a single tensor, run them through
   `expert[j]`, multiply each token's output by its expert weight,
   and scatter-add back into the output buffer at the original
   token positions.
5. **Sum** over the `top_k` contributions per token.

Mixtral defaults: `num_experts = 8`, `top_k = 2`. So every token goes
through exactly 2 of the 8 experts, and the result is a weighted
combination.

### The expert MLP

**Reference**: `MixtralBLockSparseTop2MLP` is a SwiGLU-shaped FFN.

**Our code**: `MixtralExpert` in `blocks/mixtral_moe.py:46`.

```python
class MixtralExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up

    def forward(self, x):
        return self.w2(functional.silu(self.w1(x)) * self.w3(x))
```

Same math as a plain `SwiGLU` block (`blocks/swiglu.py`). The naming
difference is the Mixtral checkpoint convention: `w1 / w2 / w3`
instead of Llama's `gate_proj / up_proj / down_proj`. Mapping:
`w1 = gate`, `w2 = down`, `w3 = up`.

This is why mini-infer carries a separate `MixtralExpert` class
rather than reusing `SwiGLU`: the parameter names need to match the
checkpoint exactly for `load_state_dict` to find them.

### Renormalization

The top-k softmax weights are renormalised so they sum to 1 *over the
chosen k experts*. Critical detail — the alternative (use the original
softmax weights directly) leaves the per-token output as a weighted
sum of a small fraction of the experts, which biases the output
distribution.

**Reference**: `routing_weights = routing_weights /
routing_weights.sum(dim=-1, keepdim=True)` after `topk`.

**Our code**: same step in `MoEFFN.forward` (`blocks/mixtral_moe.py:141`).

### Dispatch implementation

The per-expert loop is the V1 dispatch:

```python
for expert_idx in range(num_experts):
    tokens_for_expert = (expert_assignments == expert_idx).nonzero()
    if tokens_for_expert.numel() == 0:
        continue
    expert_input = hidden_states[tokens_for_expert]
    expert_output = self.experts[expert_idx](expert_input)
    weights = routing_weights[tokens_for_expert]
    weighted_output = expert_output * weights
    output.index_add_(0, tokens_for_expert, weighted_output)
```

Simple, correct, slow at large `num_experts`. The fused
grouped-GEMM path (one matmul that computes all expert outputs
simultaneously, then scatter) is a follow-up. Mixtral 8x7B has
only 8 experts so the loop overhead is bounded.

## Expert parallelism

Tied to `mini-infer`'s tensor-parallelism infrastructure (ADR-015).
The MoE block is **expert-parallel**: each rank owns
`num_experts // world_size` contiguous experts (rank `r` holds
experts `[r*N/ws, (r+1)*N/ws)`).

The dispatch:

1. **Router is replicated**. Every rank computes the same top-k
   per-token (the gate weights are replicated, the input is replicated
   for this prefill step).
2. **Per-rank loop** runs only over the local experts. Tokens routed
   to off-rank experts produce no contribution on this rank.
3. **All-reduce** sums the partial routed accumulators across ranks
   to recover the full result.

This is the simplest correct form of expert parallelism. The
all-to-all dispatch optimisation (lower comm cost when tokens cluster
on a few experts) is a follow-up; for 8 experts on 2-4 ranks the
current path is fine.

**Reference**: HF transformers runs MoE on a single device, no EP.

**Our code**: `MoEFFN.__init__` shards experts based on
`get_world_size()`; `MoEFFN.forward` calls `all_reduce_sum` at the end
to combine partial sums across ranks. At `world_size=1` the
all-reduce is a no-op and the path is bit-equivalent to the
single-device implementation.

## Decoder layer assembly

Same shape as Llama: pre-norm + GQA attention + post-attention norm +
MoE FFN + residual.

**Reference (`MixtralDecoderLayer.forward`)**: standard pre-norm
decoder, attention block followed by MoE block, both with residual
connections.

**Our code**: mini-infer reuses the Llama-style decoder shape;
`_MixtralInnerModel` instantiates the MoE FFN per layer instead of
SwiGLU.

## Where Mixtral differs from V2/V3/V4 MoE (preview)

Read this section as a bridge to the MLA + V4 walkthroughs.

| | Mixtral | V2 / V3 / Kimi-K2 | V4 |
|---|---|---|---|
| Shared experts | No | Yes (collapsed into one MLP) | Yes (same trick) |
| Routed scaling | No | `routed_scaling_factor` | Per-mode |
| Renormalization | Always after top-k | Optional (`norm_topk_prob`) | Per-mode |
| Routing function | `softmax` of router logits | `softmax` of router logits | One of `softmax` / `sigmoid` / `sqrtsoftplus` |
| Hash routing | No | No | First N layers use `tid2eid` lookup (deterministic) |
| Layer dispatch | All MoE | First `first_k_dense_replace` dense, rest MoE | Per-layer attention type + per-layer routing mode |

Mixtral's MoE is the cleanest baseline; every later family adds one
or more knobs on top.

## Validation contract

Bit-parity against HF transformers' `MixtralSparseMoeBlock` on
synthetic input at FP32 with cosine-sim > 0.999. The Mixtral 8x7B
checkpoint is too large for M1 fp16 dev hardware; the test exercises
the MoE block on synthetic configs at scale-appropriate sizes.
Integration validation against HF golden output runs on Modal when
budget permits.

Tests under `tests/unit/`; layout intentionally not enumerated.

## Where we diverged + why

1. **`MixtralExpert` class instead of reusing `SwiGLU`**. Parameter
   names (`w1/w2/w3`) need to match the Mixtral checkpoint
   convention. `SwiGLU` uses `gate_proj/up_proj/down_proj`. Two
   classes; one matches each checkpoint convention.
2. **Expert parallelism**: built into `MoEFFN` itself, gated on
   `world_size`. Single-device behaviour is bit-equivalent (no
   collectives); multi-device is correct via the simple per-rank
   loop + final all-reduce.

## Pointers

- **Reference**: HF transformers
  `transformers/models/mixtral/modeling_mixtral.py`.
- **Our model class**: `src/mini_infer/models/mixtral.py`.
- **MoE block**: `src/mini_infer/models/blocks/mixtral_moe.py`.
- **Expert MLP**: `blocks/mixtral_moe.py::MixtralExpert`.

## What's still open

- **Fused grouped-GEMM dispatch.** The per-expert Python loop is
  simple and correct but quadratic in the number of experts when most
  receive a few tokens. A fused path (one matmul that computes all
  expert outputs simultaneously, then scatter) is the optimisation
  follow-up. Aligned with the niche only if it's the same fused
  primitive the reference implementations use.
- **All-to-all dispatch for expert parallelism.** When tokens cluster
  on a few experts, the current per-rank-local loop with a final
  all-reduce moves more data than needed. An all-to-all dispatch
  (each rank sends only the tokens its experts will process) is the
  production answer; deferred until profiling shows it matters at the
  rank counts mini-infer targets.
