# ADR-035: Nemotron-H port (Mamba-2 state-space layers, non-gated MoE)

Date: 2026-10-20
Status: Accepted

## Context

NVIDIA's Nemotron 3.5 family ships as `NemotronHForCausalLM` (`nemotron_h`).
Lightning 30B-A3B (65.8 GB in BF16) has 52 layers:

- 23 Mamba-2 layers with 64 heads, state size 128 and chunk 128.
- 23 MoE layers with 128 routed experts (top 6) and one shared expert. The
  experts are non-gated squared-ReLU MLPs.
- 6 attention layers with 32 query heads, 2 KV heads and no positional
  encoding.

Nemotron 3 Nano 4B has dense MLPs in place of the MoE. Bolt and Super add
a latent projection around the experts.

A Mamba-2 layer keeps a fixed-size state per head that decays and grows
with each token. Mini-infer had no such layer. The pinned transformers 5.14
implements `nemotron_h` and is the parity oracle.

## Decision

1. One mixer per layer, with kinds from `layers_block_type` or
   `hybrid_override_pattern`. Unsupported values fail at load.
2. Mamba-2 written from scratch in fp32 (`models/blocks/mamba2.py`).
   Prefill uses a chunked scan and decode a token-by-token recurrence. A
   test pins the two together.
3. Follow the CUDA kernels, not the CPU fallback: fp32 SSM math and no
   step-size clamp (`time_step_limit` is `(0, inf)`).
4. Up to transformers 5.14, the Mamba-2 CPU fallback is wrong for spans
   longer than one chunk. On a test model, its one-shot logits differ from
   its own token-by-token logits by 1.6e-5. Ours match them within 7e-8.
   transformers 5.15.0 fixed it (huggingface/transformers#47452). The repo
   pins 5.14, so long spans are compared against the reference fed one
   token at a time.
5. MoE reuses `GlmNoAuxTcGate`. Up to 64 token-expert pairs (any decode
   step up to batch 10), the layer gathers the chosen expert weights with
   no host sync. Longer prefills loop over the experts.
6. `NemotronHStateCache` holds an fp32 SSM state and a conv tail per Mamba
   layer. With no host sync, decode uses ADR-033's CUDA graph.
7. `iter_safetensors_shards` gains a `max_bytes` option that streams 1 GiB
   batches. The Nano 4B file (8 GB) does not fit next to the model on a
   16 GB laptop.

## Alternatives Considered

- Reuse `kda_chunkwise`. Rejected, because Mamba-2 has no delta rule.
- Depend on `mamba_ssm` and `causal-conv1d`. Rejected, because the PyTorch
  path runs on CPU and MPS.
- Match the transformers CPU fallback. Rejected, because it is wrong past
  one chunk in 5.14.
- Upgrade transformers to 5.15 or later. Deferred, because every other
  parity suite is pinned to 5.14.

## Consequences

- `tests/unit/test_nemotron_h_real_weights.py` loads Nano 4B and completes
  "Paris" on an M1 Pro in about 2 minutes (`--run-large-models` only).
- The real 30B-A3B on one A100-SXM4-80GB passed against transformers
  (`scripts/modal_nemotron_h_smoke.py`). Prefill logit cosine was 0.9949
  to 0.9989, with the same top-1 on all four prompts. Greedy output matched
  for 31, 7, 2 and 32 of 32 tokens.
- That agreement is looser than the dense Qwen3.8's (cosine of at least
  0.99985). On synthetic models, our bf16 port and the bf16 reference drift
  from fp32 by the same amount (cosine 0.99998). So it is not a port
  defect. Two bf16 effects can widen the gap: a top-6 routing near-tie, and
  the bf16 step size of the reference CPU path.
- Decode in ms per step: 42.3 eager and 13.6 with the CUDA graph at batch
  1, 43.5 and 29.5 at batch 4. At batch 4 the expert gather copies about
  0.5 GB per MoE layer.
- Both sides of the real-weight test get the same ids, because
  transformers adds no BOS here (`add_bos_token` is false).
- The multi-token-prediction layer is dropped. Tensor parallelism and the
  vision towers of Bolt and Super are not ported.
