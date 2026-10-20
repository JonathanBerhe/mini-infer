# ADR-032: Qwen3.8 port (Gated DeltaNet linear attention, gated full attention, hybrid state cache)

Date: 2026-09-19
Status: Accepted

## Context

Qwen3.8 was announced 2026-08-03 and the open weights landed 2026-08-05 as
`Qwen/Qwen3.8-27B` (apache-2.0, 55.6 GB, 1199 tensors). Despite the product
name, the checkpoint ships under the **`qwen3_5` architecture**: its
`architectures[0]` is `Qwen3_5ForConditionalGeneration` and transformers
5.14, which this repo already pins, implements it exactly. No dependency
bump is required to serve it or to parity-test it.

Before this port the repo covered only dense GQA Qwen3
(`models/qwen3.py`). The entire Qwen hybrid lineage (Qwen3-Next, 3.5, 3.6,
3.8) failed at registry lookup, so the gap was a real port rather than a
config tweak.

The architecture is a 3:1 hybrid over 64 layers: three Gated DeltaNet
(linear attention) layers for every one Gated Attention layer, 48 linear
and 16 full, given explicitly by the config's `layer_types` list. Neither
layer kind fits PagedAttention's one-entry-per-token model, which is why
this family is served through the state-cache seam that the Kimi Linear
port generalized (see [ADR-026](ADR-026-kimi-linear-kda-port.md)).

### Facts pinned from the released checkpoint

Read from the real `config.json` and `model.safetensors.index.json`, then
confirmed by instantiating the reference on a meta device and diffing
against the checkpoint index: **851 text tensors, 0 missing, 0 extra.**

- hidden 5120, 64 layers, intermediate 17408, vocab 248320, head_dim 256,
  24 attention heads / 4 KV heads, `attention_bias` false, rms_norm_eps 1e-6.
- Gated DeltaNet: 16 key heads, **48 value heads (a 3x broadcast ratio, not
  the 2x the Qwen3.5-4B/9B siblings use)**, key/value head dim 128, conv
  kernel 4. Fused `in_proj_qkv` is 2*2048 + 6144 = 10240 wide over one
  depthwise conv; `A_log` and `dt_bias` are flat `(48,)` per-head scalars.
- Full attention: `q_proj` is 12288 = 24 * 256 * 2 wide because the output
  gate is **fused into `q_proj`**, split per head. There is no separate
  gate projection in the checkpoint.
- rope: `partial_rotary_factor` 0.25, `rope_theta` 1e7, `mrope_interleaved`
  true, `mrope_section` [11, 11, 10].
- Weight layout: text under `model.language_model.*`, plus 333
  `model.visual.*` and 15 `mtp.*` tensors, and an untied `lm_head.weight`.

The config also carries `attn_output_gate: true` and
`output_gate_type: "swish"`. Neither name appears anywhere in transformers
5.14's `qwen3_5` module, meaning 5.14 hardcodes precisely the behavior they
describe. The released values match that hardcoded behavior, so 5.14 is a
faithful oracle for this checkpoint.

## Decision

1. **Reuse the KDA delta rule rather than reimplementing Gated DeltaNet.**
   KDA's per-channel decay `Diag(a_t)` strictly generalizes GDN's per-head
   scalar decay, so broadcasting the scalar across key channels makes
   `kda_recurrent` / `kda_chunkwise` compute the GDN recurrence exactly.
   Measured against transformers' `torch_recurrent_gated_delta_rule` and
   `torch_chunk_gated_delta_rule` in fp32, max abs error was 4.5e-08 to
   8.9e-08 (round-off) across square and non-square head dims, sequences
   crossing the 64-token chunk boundary, ragged padding, and a non-zero
   initial state. `blocks/gdn.py` therefore holds only what genuinely
   differs: the per-head gate formula and the SiLU gated output norm.
2. **New `Qwen35StateCache` rather than extending `KimiStateCache`.** The
   two share a lifecycle but no tensors: Qwen fuses q/k/v into one conv tail
   where Kimi has three, and Qwen's full-attention layers are plain GQA
   per-token K/V buffers where Kimi's are compressed MLA. Keeping each
   cache's indexing visible at its own call sites is the design choice
   ADR-026 already made.
3. **Keep the reference's fused projections instead of splitting them at
   load.** Both the fused `in_proj_qkv` and the fused, per-head-split
   `q_proj` are kept as the checkpoint stores them, so weights load with no
   remapping beyond the prefix strip and the conv squeeze. The per-head
   split is load-bearing: splitting `q_proj` flat into
   `[all queries | all gates]` is numerically wrong (0.21 max abs error
   against the reference, versus 0.0 for the per-head split).
4. **Text-only, with the multimodal rope collapsed on purpose.** The vision
   tower is out of scope per the repo's standing scope decision, and
   `model.visual.*` and `mtp.*` are dropped at load. For text-only input all
   three mrope position streams are identical, and the interleaved mrope
   then reduces to plain 1D partial rope exactly (`torch.equal` true on both
   cos and sin), so the implementation uses plain partial rope and pins that
   equivalence with a test.
5. **Single rank for this port.** `load_weights` raises on world_size != 1.
   Tensor parallelism would have to shard the fused conv channels and the
   per-head recurrent state, which is a separate piece of work; this follows
   the same staging the Kimi Linear port used.
6. **Build the model in place and stream the weights in.** The generic
   `load_model` builds the model in fp32 on the host first, then holds the
   whole checkpoint next to the converted model. The peak is about twice the
   half-precision weights: around 17 GB for the 4B and around 110 GB of host
   RAM for the 27B. `from_checkpoint` builds the model directly in the
   target dtype on the target device, and `load_weights_streaming` copies the
   checkpoint in one shard at a time. The peak is the model plus the
   largest shard: for the 4B, about 8.5 GB of weights plus a 5.3 GB shard. `StateCacheGenerator.from_pretrained` takes this
   route. One side effect: the rotary `inv_freq` table stays fp32, as it does
   in transformers, where `load_model` casts it to the model dtype.

## Alternatives Considered

- **Wait for a Qwen3.8-specific architecture.** Rejected on evidence: the
  checkpoint is literally `qwen3_5`, and a meta-device shape diff against
  the real index matched 851/851 tensors.
- **Port Qwen3.5 first as a head start, before 3.8 weights existed.**
  Considered at planning time and declined, because the branch decision
  (same arch vs new model_type) was the port's dominant unknown and guessing
  it wrong meant rework. Waiting cost nothing: the answer turned out to be
  "same arch", and Qwen3.5-4B now serves as the small real-weight vehicle.
- **Generalize `KimiStateCache` to cover both families.** Rejected per
  decision 2; the shared surface is the lifecycle, not the tensors.
- **Bump transformers.** Not needed. 5.14 parses the released config and
  reproduces the checkpoint's exact tensor set.

## Consequences

- The Qwen hybrid lineage is now reachable: the same code path covers
  Qwen3.5-4B/9B/27B and Qwen3.6-27B, which share the architecture and differ
  only in dims and the GDN head ratio.
- `blocks/gdn.py` is small because the expensive math was already in
  `blocks/kda.py`. The cost of the Kimi Linear port is partly recovered here.
- The 27B cannot run on the local Apple Silicon box at 55.6 GB. Its 3x
  GDN ratio is covered by synthetic parity against the reference, and by a
  one-GPU gate on the real weights (`scripts/modal_qwen38_smoke.py`, A100
  80 GB, bf16): prefill logit cosine of at least 0.99985 against
  transformers, and greedy output equal to the reference token for token on
  all four prompts. Real weights are also covered by a local-only smoke on
  `Qwen/Qwen3.5-4B` (same architecture, 2x GDN ratio). That smoke is marked
  `large_model` and runs only with `--run-large-models`, because its 9.3 GB
  download does not fit the bit-parity runner or its 10 GB cache.
- `Qwen3.8-Flash-Next` is explicitly NOT covered. It is a different
  architecture (`Qwen4ExpForConditionalGeneration`, model_type `qwen4_exp`,
  512-expert MoE) and remains a separate port.
- MoE variants of this family (the 2.4T-A95B Max tier) remain unported.
