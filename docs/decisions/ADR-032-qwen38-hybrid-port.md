# ADR-032: Qwen3.8 port (Gated DeltaNet linear attention, gated full attention, hybrid state cache)

Date: 2026-09-19
Status: Accepted

## Context

The Qwen3.8 open weights came out on 2026-08-05 as `Qwen/Qwen3.8-27B`
(55.6 GB). The checkpoint uses the `qwen3_5` architecture, and the pinned
transformers 5.14 implements it exactly. Before this port, the repo did
not load any Qwen hybrid.

The model is a 3:1 hybrid over 64 layers: 48 Gated DeltaNet (linear
attention) layers and 16 Gated Attention layers. Neither kind fits the
one-entry-per-token model of PagedAttention. So this family uses the
state-cache seam of the Kimi Linear port
([ADR-026](ADR-026-kimi-linear-kda-port.md)).

A meta-device build of the reference matched the checkpoint index: 851 text
tensors, 0 missing, 0 extra. Two facts drive the decisions:

- Gated DeltaNet has 16 key heads and 48 value heads, a 3x ratio. The
  Qwen3.5-4B/9B models use 2x.
- `q_proj` is 12288 = 24 * 256 * 2 wide, because the output gate is fused
  into it, split per head.

transformers 5.14 hardcodes the behavior of `attn_output_gate: true` and
`output_gate_type: "swish"` in the configuration, so it is a faithful oracle.

## Decision

1. Reuse the KDA delta rule for Gated DeltaNet. The per-channel decay of KDA
   generalizes the per-head scalar decay of GDN. With the scalar broadcast
   across key channels, `kda_recurrent` and `kda_chunkwise` compute the GDN
   recurrence exactly. Max abs error against the transformers reference in
   fp32 was 4.5e-08 to 8.9e-08.
2. A new `Qwen35StateCache`, not an extended `KimiStateCache`. The two share
   a lifecycle but no tensors. Qwen fuses q/k/v into one conv tail, and its
   full-attention layers use plain GQA K/V, not compressed MLA.
3. Keep the fused projections of the reference, so weights load with no
   remapping. A flat `[all queries | all gates]` split of `q_proj` gives
   0.21 max abs error, and the per-head split gives 0.0.
4. Text-only, with the multimodal rope collapsed on purpose. The load drops
   `model.visual.*` and `mtp.*`. For text-only input, the interleaved mrope
   equals plain 1D partial rope. The code uses plain partial rope, and a
   test pins the equivalence.
5. Single rank. `load_weights` raises on world_size != 1. Tensor
   parallelism is separate work.
6. Build the model in place and stream the weights in. The generic
   `load_model` peaks at about twice the half-precision weights, around
   110 GB of host RAM for the 27B. `from_checkpoint` builds in the target
   dtype on the target device, and `load_weights_streaming` copies one shard
   at a time. `StateCacheGenerator.from_pretrained` takes this route.

## Alternatives Considered

- Wait for a Qwen3.8-specific architecture. It is not needed, because the
  checkpoint is `qwen3_5`.
- Port Qwen3.5 first, before the 3.8 weights existed. The architecture was
  the main unknown, and a wrong guess meant rework.
- Generalize `KimiStateCache` to both families. Rejected per decision 2.
- Bump transformers. 5.14 already reproduces the exact tensor set.

## Consequences

- One code path covers Qwen3.5-4B/9B/27B and Qwen3.6-27B, which differ only
  in dimensions and the GDN head ratio.
- The 27B does not fit the local Apple Silicon machine. Synthetic parity
  tests cover its 3x GDN ratio. A one-GPU gate on the real weights
  (`scripts/qwen/modal_qwen38_smoke.py`, A100 80 GB, bf16) gives a prefill logit
  cosine of at least 0.99985 against transformers. Greedy output equals the
  reference token for token on all four prompts (see ADR-034 for the fused
  decode step).
- A local smoke on `Qwen/Qwen3.5-4B` covers real weights. It runs only with
  `--run-large-models`, because of its 9.3 GB download.
- `Qwen3.8-Flash-Next` (`Qwen4ExpForConditionalGeneration`, 512-expert MoE)
  is a different architecture and is not covered. The MoE variants of this
  family are not ported.
