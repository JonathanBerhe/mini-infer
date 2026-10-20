# ADR-035: Nemotron-H port (Mamba-2 state-space layers, non-gated MoE)

Date: 2026-10-20
Status: Accepted

## Context

NVIDIA's Nemotron 3.5 family ships as `NemotronHForCausalLM` (`nemotron_h`).
Lightning 30B-A3B (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
OpenMDW 1.1) has 52 layers, each one RMSNorm and one mixer:

- 23 Mamba-2 layers: 64 heads of dimension 64, state size 128, 8 groups
  for B and C, conv kernel 4, chunk 128.
- 23 MoE layers: 128 routed experts, top 6, with a sigmoid router and an
  aux-free selection bias, plus one shared expert. The experts are
  non-gated MLPs with a squared-ReLU activation.
- 6 attention layers: 32 query heads, 2 KV heads, head dimension 128, and
  no positional encoding.

The checkpoint is 65.8 GB in BF16 and also holds one multi-token-prediction
layer. Nemotron 3 Nano 4B uses the same class with dense MLP layers in
place of the MoE, and Bolt and Super add a latent projection around the
experts.

A Mamba-2 layer is a selective state-space model: per head, a fixed-size
state decays by `exp(dt * A)` each token and adds `dt * x B^T`, and the
output reads it with `C`. Mini-infer had no layer of this kind. The pinned
transformers 5.14 implements `nemotron_h`, so it is the parity oracle with no
upgrade.

## Decision

1. **One mixer per layer, kinds from the config.** `layers_block_type`, its
   transformers spellings and the older one-letter `hybrid_override_pattern`
   all map to `mamba`, `attention`, `moe` and `mlp`. Unsupported values
   (another activation, a sliding window) are rejected at load.
2. **Mamba-2 written from scratch in fp32** (`models/blocks/mamba2.py`).
   Prefill uses the chunked state-space-duality scan, decode uses the
   token-by-token recurrence, and a test pins the two against each other.
   The KDA and Gated DeltaNet code is not reused, because Mamba-2 has no
   delta rule: its update is purely additive.
3. **Follow the CUDA kernels, not the CPU fallback, on two points.** The SSM
   math stays in fp32, and the step size is not clamped (`time_step_limit`
   is `(0, inf)`). The transformers CPU fallback also clamps the step size at
   `time_step_min`, which changes only steps below 1e-3.
4. **Parity oracle for long spans: the reference's token-by-token path.**
   Up to transformers 5.14, the CPU fallback for Mamba-2 in Nemotron-H and
   Zamba2 sums the inter-chunk state recurrence over the target chunk
   instead of the source chunk. It is exact for one chunk and wrong beyond
   it: on a test model its one-shot logits differ from its own token-by-token
   logits by 1.6e-5, while ours match the token-by-token ones within 7e-8.
   transformers 5.15.0 fixed it as part of a convolution refactor
   (huggingface/transformers#47452), and from 5.18 all Mamba-2 models share
   one chunk scan. Running a reproduction on each release confirmed the
   bisection. The repo pins 5.14, so spans of one chunk are compared one
   shot and longer spans against the reference fed one token at a time,
   which is valid on every version. A test asserts the behavior that matches
   the installed version, so a transformers upgrade does not fail it.
5. **MoE: reuse the GLM router, stack the experts.** `GlmNoAuxTcGate` is
   the same sigmoid, aux-free-bias, grouped router. The routed experts are
   stacked into two tensors that the loader fills slot by slot. Up to 64
   token-expert pairs (any decode step up to batch 10), the layer gathers
   the chosen experts' weights, with no host sync. Longer prefills loop
   over the experts. Expert outputs accumulate in fp32, as in the reference.
   The latent projection of Bolt and Super is supported.
6. **Serving on the state-cache path.** `NemotronHStateCache` holds an fp32
   SSM state and a conv tail per Mamba layer and K/V buffers per attention
   layer; MoE and MLP layers hold nothing. The decode step has no host sync,
   so it declares `SUPPORTS_DECODE_CUDA_GRAPH` and uses ADR-033's graph.
7. **Streaming load in 1 GiB batches.** `iter_safetensors_shards` gains a
   `max_bytes` option that reads a file tensor by tensor. The Nano 4B is one
   8 GB file, and loading it whole next to the model does not fit a 16 GB
   laptop. Only loaders that place each tensor on its own may use the option,
   because a batch can split an FP8 weight from its scale.

## Alternatives Considered

- **Reuse `kda_chunkwise`.** Rejected: the delta rule subtracts the state's
  prediction before the update, and Mamba-2 does not.
- **Depend on `mamba_ssm` and `causal-conv1d` kernels.** Rejected for the
  port: the repo writes its kernels from scratch, and the PyTorch path runs
  on CPU and MPS. A fused decode kernel, like ADR-034's, can follow.
- **Match the transformers CPU fallback exactly.** Rejected: it is not what
  the model is served with, and in the pinned 5.14 its chunked path is
  wrong past one chunk.
- **Upgrade transformers to 5.15 or later for this port.** Deferred: every
  other parity suite in the repo is pinned to 5.14, and the token-by-token
  oracle makes the upgrade unnecessary here.

## Consequences

- Tests: the chunked scan against the recurrence, prefill and greedy decode
  against transformers, chunked prefill, ragged decode after a row copy,
  the dense-MLP and latent-MoE layer kinds, both MoE paths, the streaming
  loader, the real Lightning config, and CUDA graph against eager.
- Real weights on a laptop: `tests/unit/test_nemotron_h_real_weights.py`
  loads Nemotron 3 Nano 4B (21 Mamba-2, 17 MLP, 4 attention layers) and
  completes "Paris" on an M1 Pro, MPS fp16, in about 2 minutes. It runs only
  with `--run-large-models`.
- Real 30B-A3B on one A100-SXM4-80GB, bf16, against transformers on the
  same weights (`scripts/modal_nemotron_h_smoke.py`, four prompts of 6 to
  60 tokens): the run passed. Prefill logit cosine was 0.9949 to 0.9989,
  with the same last-token top-1 on every prompt. Greedy output matched the
  reference for 31, 7, 2 and 32 of 32 tokens, and both sides stay fluent.
  The first prompt completes as " Paris.". The CUDA graph gives the same
  tokens as eager, and two runs give the same tokens.
- That agreement is looser than the dense Qwen3.8's (cosine of at least
  0.99985). On synthetic models our bf16 port and the reference's bf16 path
  drift from the fp32 result by the same amount (cosine 0.99998), so this is
  not a port defect that the fp32 tests miss. Two bf16 effects plausibly
  widen the gap on the real model: MoE top-6 routing, where a near-tie picks
  another expert, and the reference's CPU path, which rounds the step size
  to bf16 where the CUDA kernels and this port do not. A layer-by-layer
  routing comparison on the real weights would settle it.
- Decode on the 30B-A3B, milliseconds per step: 42.3 eager and 13.6 with
  the CUDA graph at batch 1 (3.1x), 43.5 and 29.5 at batch 4 (1.5x). End to
  end, batch 1 serves 48.4 tokens per second, against 20.4 eager. At batch
  4 the expert gather copies about 0.5 GB per MoE layer, which a grouped
  expert matmul would avoid. About 3.2B parameters (6.4 GB in bf16) are
  active per token, so the batch-1 bound on an A100 is near 3 ms.
- Mini-infer's tokenizer prepends BOS when a tokenizer defines one, a
  repo-wide rule from the Gemma 4 port. Transformers does not for this
  tokenizer (`add_bos_token` is false), so the real-weight gate feeds both
  sides the same ids.
- The multi-token-prediction layer is dropped at load. Tensor parallelism
  and the vision towers of Bolt and Super are not ported.
