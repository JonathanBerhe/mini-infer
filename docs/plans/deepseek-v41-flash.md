# DeepSeek-V4.1-Flash port plan

Date: 2026-10-24
Updated: 2026-10-25, Stage A done.
Status: Stage A (design study) done in
[ADR-047](../decisions/ADR-047-deepseek-v41-flash-design.md). Stage B
(text-only port) is next.

Model: `deepseek-ai/DeepSeek-V4.1-Flash` (released 2026-09-10) and its
weight refresh `deepseek-ai/DeepSeek-V4.1-Flash-1017` (released
2026-10-17), both MIT. The two repos have the same `config.json` and the
same 96,085 tensor names, so one port serves both. The tech report
(`DeepSeek_V41_Tech_Report.pdf`) and a reference implementation
(`inference/model.py`, `engram.py`, `kernel.py`, `vision.py`) ship inside
the V4.1-Flash repo.

Engine: DeepSeek released its production engine, DeepInfer
([deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer),
Apache 2.0), on 2026-10-20. It serves only this model, only offline, and
only on SM100 GPUs and Ascend 950. This plan uses it as a second
reference for serving behavior, not as code to copy.

## What V4.1-Flash is

A multimodal MoE model with 552B backbone parameters plus 196B Engram
parameters. It has 40 layers, 384 routed experts (6 active per token)
and 1 shared expert. DeepSeek reports 8B active parameters per token in
prefill and 16B in decode. The checkpoint is 510 GB, against 160 GB for
V4-Flash.

The changes against V4-Flash, in order of how much work they make for
us:

1. **CSA2 attention layout.** In V4, every compressed layer computed
   and cached its own compressed KV. In V4.1, each layer has one of
   three static modes:
   - Full: the layer computes its own compressed KV, derives indexer K
     from it, and runs the indexer. Layers 2, 8, 14 and 20
     (`kv_source_layer_ids`).
   - Reindex: the layer reuses the compressed KV and indexer K of the
     latest Full layer, but runs its own indexer query and top-k.
     Layers 24, 28, 32 and 36 (`index_source_layer_ids` minus the
     Full layers).
   - Reuse: the layer reuses the latest compressed KV and the latest
     top-k indices. It runs no indexer. All other compressed layers.

   Every layer still computes its own query and its own 128-token
   sliding-window KV. Layers 0, 1 and the 3 DSpark layers are
   sliding-window only.
2. **Causal encoder-decoder (CED).** Layers 0 to 19 form an encoder and
   layers 20 to 39 a decoder. Layer 20 projects the decoder's global KV
   from the encoder output, and layers 21 to 39 share it. Prefill
   therefore needs the full decoder only for the last few prompt
   tokens. This is where the 8B prefill figure comes from.
3. **New compression ratios.** `compress_ratios` changes from 4 and 128
   to 2 (layers 2 to 19) and 1 (layers 20 to 39). Ratio 1 means no
   pooling: the compressor is a plain projection plus norm, kept in
   BF16. The ratio-2 compressor drops the V4 overlap window and the
   position bias.
4. **Indexer changes.** Indexer K is projected from the main compressed
   latent (`attn.indexer.wk`, 512 to 128) instead of coming from a
   second compressor. The Hadamard rotation of V4 is gone. The decoder
   adds a hierarchical step. Layer 20 scores all positions and keeps
   the 2,048 best blocks of 8 positions (`candidate_topk_blocks`,
   `candidate_block_size`). The later Reindex layers search only inside
   that pool. Indexer heads drop from 64 to 32.
5. **FP4 caches.** Compressed KV is stored as E2M1 with one E4M3 scale
   per 16 channels, quantized after RoPE over the whole 512-wide
   vector. Indexer K is E2M1 with one E8M0 scale per 32. Window KV
   stays FP8, quantized over the whole vector, RoPE tail included.
   The global cache is 890 bytes per token, about a quarter of
   V4-Flash.
6. **Single-pass mHC.** Each sublayer mixes its input with the
   coefficients that the previous sublayer computed, not its own.
   The first block uses a one-hot mix, and the final head uses the
   last FFN's mix. The V4 `hc_head_*` weights are gone. The Sinkhorn
   math itself is unchanged.
7. **Engram.** Two lookup modules, at layers 1 and 14. Each token
   hashes the n-grams that end at it (sizes 2, 3 and 4, with 8 hash
   heads each). The hashes land in prime-sized buckets of one FP8
   table per module, about 384M rows of 256 values. A projection turns
   the 24 fetched rows into a key per residual copy and one shared
   value. A sigmoid gate then adds the value into the residual stream.
   Token ids first go through a normalized "compressed" vocabulary of
   99,092 entries, built from the tokenizer.
8. **DSpark in the checkpoint.** Three draft blocks under `mtp.*`:
   sliding-window attention only, MoE with 128 experts and 3 active,
   block size 5, inputs tapped from layers 37, 38 and 39, a rank-256
   Markov head and a confidence head. The reference implements only
   the draft forward pass, not the verify loop.
9. **Smaller items.**
   - FP8 weights use 32x32 blocks, not 128x128.
   - No hash-routed MoE layers.
   - A second router bias, `gate.bias_vl`, for image tokens.
   - `rms_norm_eps` is 1e-20.
   - A 32-layer ViT plus a two-layer aligner for images.
   - No Jinja chat template and no `generation_config.json`. The
     prompt format is defined by `encoding/encoding.py`, with test
     fixtures.

Configuration changes that matter for shapes:

| Field | V4-Flash | V4.1-Flash |
| :--- | :---: | :---: |
| `hidden_size` | 4096 | 5120 |
| `num_hidden_layers` | 43 | 40 |
| `n_routed_experts` | 256 | 384 |
| `moe_intermediate_size` | 2048 | 2304 |
| `q_lora_rank` | 1024 | 1280 |
| `index_n_heads` | 64 | 32 |
| `num_nextn_predict_layers` | 1 | 3 |
| FP8 `weight_block_size` | 128x128 | 32x32 |

## Why this maps to mini-infer

- The V4 port already has most of the parts:
  - the attention core: MQA over one 512-wide KV head, attention
    sinks (`models/deepseek_v4/sink.py`), the grouped low-rank output
    projection (`models/deepseek_v4/grouped_output.py`) and the sliding
    window
  - the Sinkhorn split (ADR-018), whose math is unchanged
  - the `sqrtsoftplus` router with `noaux_tc` bias and the SwiGLU clamp
  - the FP4 expert format. It is the same packed E2M1 with E8M0 scales
    per 32 that the W4A8 grouped path reads (ADR-040, ADR-041,
    ADR-043).
- The serving path carries over: per-request state cache (ADR-019),
  ragged decode, the CUDA-graph decode step (ADR-042), fused RoPE
  (ADR-044) and fused RMS norm (ADR-045).
- The DSpark machinery exists: Markov head, confidence head, proposal
  and verify loop, and multi-request scheduling
  (`src/mini_infer/engine/dspark/`, ADR-027, ADR-031). V4.1 is the first
  model that ships its own DSpark weights, so this closes the
  "V4-DSpark" item that the DSpark plan left out of scope.
- The parity pattern exists: `tests/unit/models/deepseek_v4/_v4_reference_helpers.py`
  stubs the reference's TileLang kernels with PyTorch so the reference
  runs on CPU.
- Differentiation: as of 2026-10-24, transformers (5.20.2) has no
  V4.1 class, and SGLang supports V4.1 only in a preview image. A
  readable, parity-tested port fits this project.

## What has to change

- **Layer roles.** `DeepseekV4DecoderLayer` picks CSA or HCA from the
  ratio (`CSA_COMPRESSION_RATIO = 4`,
  `src/mini_infer/models/deepseek_v4/decoder_layer.py:75`). V4.1
  needs a role per layer (sliding-window, Full, Reindex, Reuse) plus a
  source layer for each shared tensor.
- **Compressor.** `TokenLevelCompressor`
  (`src/mini_infer/models/deepseek_v4/compressor.py`) has overlap mode
  and a position bias. V4.1 needs neither, and needs a ratio-1 path.
- **Indexer.** `LightningIndexer` owns a second compressor
  (`src/mini_infer/models/deepseek_v4/lightning_indexer.py:140`). V4.1
  derives K from the main latent and adds the candidate-block mask.
- **State cache.** `StateLayerSpec`
  (`src/mini_infer/cache/state_cache.py:57`) and
  `build_state_cache_layer_specs`
  (`src/mini_infer/models/deepseek_v4/deepseek_v4.py:661`) allocate compressed KV
  and indexer state per layer. V4.1 allocates window KV in all 43
  layers but compressed KV and indexer K in only 4. Prefix sharing
  (ADR-020) and the PD transfer layout (ADR-016) follow from this.
- **Hyper-connections.** `hc_pre` computes and applies its mix in one
  call (`deepseek_v4/decoder_layer.py:496-505`). V4.1 carries the mix
  from one sublayer to the next, and `HCHeadReduction` goes away.
- **Loader.** The FP8 dequant hardcodes `block_size=(128, 128)`
  (`src/mini_infer/models/deepseek_v4/deepseek_v4.py:1393`). The configuration is nested
  (`text_config`, `vision_config`), `model_type` is `deepseek_v41`, and
  there are new tensor names: `engram.*`, `gate.bias_vl`, the
  DSpark `mtp.*` blocks, `vision.*`, `aligner.*` and `image_*`.
- **New modules.** Engram with its n-gram hash state, the V4.1 DSpark
  drafter, and (optional) the ViT.

## Hardware and memory

Checkpoint bytes by part, from the safetensors headers:

| Part | Size | Format |
| :--- | ---: | :--- |
| Routed experts | 288.8 GB | FP4, E8M0 scales |
| Engram tables | 196.6 GB | FP8 |
| Engram table scales | 6.1 GB | E8M0 |
| DSpark blocks | 7.9 GB | mixed |
| Attention | 5.2 GB | FP8 and BF16 |
| Embedding and head | 2.6 GB | BF16 |
| Shared experts | 1.4 GB | FP8 |
| Vision | 1.0 GB | BF16 |
| Total | 510.3 GB | |

2x B200 has 384 GB of HBM, so the whole checkpoint does not fit. The
non-Engram weights (about 307 GB) do fit, at about 154 GB per GPU. The
loader dequantizes the FP8 attention and shared-expert weights to BF16,
which adds about 6 GB.

The Engram tables can live in host memory. Their row ids depend only
on the token ids, so the rows can be gathered on the CPU before
layer 1 runs. One token reads 24 rows of 264 bytes per module, about
12.7 KB for both modules. A decode step at batch 16 reads about 200 KB,
and a 32K-token prefill reads about 415 MB. SGLang's preview image has
this host-table mode and reports bitwise-equal outputs.

Deployment options, in order of preference:

1. 2x B200, Engram in pinned host memory. Same GPUs as the V4-Flash
   work. Needs about 210 GB of host RAM in the Modal container.
2. 4x B200, everything on the GPUs.
3. 8x H100 (640 GB), everything on the GPUs, with little headroom.
   The FP4 experts run through the Hopper grouped path (ADR-041).
   Neither DeepInfer nor SGLang supports Hopper for this model.

KV memory is small at our scale. Stored in BF16 (no FP4), the global
cache is about 3.2 KB per token. 16 requests at 128K tokens need about
6.7 GB. The window KV adds about 5.6 MB per request in BF16.

## Out of scope (explicit)

- **Training or QAT.** We consume released weights only.
- **DeepInfer's cluster features**: wide expert parallelism, EPLB,
  data-parallel attention, encoder/prefill/decode disaggregation and
  the SSD KV store. They pay off at 64 GPUs, not at 2.
- **Single-kernel Mega MoE and Mega-mHC** in the first pass. These are
  Stage E candidates and need a profile first.
- **Encoder SWA bounded replay for a persistent KV tier.** We have no
  SSD KV tier. Decoder bounded replay is in scope (Stage E).
- **Vision** until Stage F.
- **Ascend.**

## Stages and gates

### Stage A: design study (CPU only, no code shipped), done

Done (2026-10-25): see
[ADR-047](../decisions/ADR-047-deepseek-v41-flash-design.md). All eight
unknowns have answers from the reference code, a CPU run of the reference
with PyTorch kernel stand-ins, and a rebuild of the Engram layout. Three
results change the later stages:

- The reference's own prefill and decode disagree when indexer scores tie,
  because `torch.topk` breaks ties by row width. With a lower-index tie rule
  they agree within 2.4e-6. The port and the parity harness use that rule.
- With the FP8 and FP4 round trips on, float noise crosses rounding
  boundaries, so even the reference does not match itself exactly. The
  parity contract separates quantization-off equality from quantization-on
  agreement.
- The Engram layout rebuilt from the configuration and the tokenizer
  matches the checkpoint's table sizes and compressed vocabulary exactly.

**Gate** (did not trigger): shared compressed KV needs only a new field in
`StateLayerSpec`. Prefix sharing works unchanged, and the prefill/decode
split path does not serve `StateCache` models.

Original plan text follows.

Resolve the parity-critical unknowns against the reference code, not
the report's prose, and record them in a new ADR:

1. Single-pass mHC: the one-hot mix for the first block, the mix that
   the head uses, and the DSpark tap (the mean over the 4 residual
   copies of the attention input of layers 37 to 39).
2. Shared state: which tensors a Reindex or Reuse layer reads
   (compressed KV, indexer K, top-k indices, candidate mask), and how
   the top-k offset points past the window entries in the concatenated
   KV.
3. Compressor timing: when a ratio-2 group flushes in decode, and which
   RoPE position a compressed entry takes (group j takes position
   j times the ratio).
4. Indexer numerics: FP4 query and key with E8M0 per 32, no Hadamard,
   ReLU scores weighted by `weights_proj`, the all-reduce across tensor
   parallel ranks, and the candidate rule (the partly filled newest
   block always stays in).
5. Cache quantization: the V4 port's simulated round trip covers only
   the non-RoPE KV dims. V4.1 quantizes the whole vector, RoPE tail
   included, and uses E4M3 scales per 16 for compressed KV.
6. Engram hashing: the tokenizer normalizer chain, the prime search
   (start at 16,000,000, no prime used twice), the multipliers
   (`numpy.random.default_rng(10007 * layer_id)`), int64 XOR, the pad
   id, and image spans that block n-grams.
7. Router: FP32 logits, the bias picks experts but the weights come
   from raw scores, `bias_vl` for image tokens, and the 1e-20 epsilon
   in the top-k normalization.
8. Decoder bounded replay: how it truncates the window, and how many
   tokens it keeps. DeepInfer keeps 140
   (`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`), the window plus a backtrack
   margin.

Deliverable: the ADR, with alternatives considered (a separate model
class against a V4 class with role flags, BF16 against FP4 cache
storage, host-table Engram against GPU tables).

**Gate:** if shared compressed KV cannot fit the state cache without a
rewrite of prefix sharing and PD transfer, stop. Record the decision
(for example, ship V4.1 first without prefix sharing or PD).

### Stage B: text-only port, CPU micro-config parity

Build `DeepseekV41ForCausalLM` from the V4 blocks plus the new parts.
Put the reference in `third_party/deepseek_v41_reference/` with a CPU
kernel stub in the style of `tests/unit/models/deepseek_v4/_v4_reference_helpers.py`.
The stub needs `sparse_attn`, the FP8 and FP4 round trips and
`hc_split_sinkhorn`.

Parity contract, in order:

1. Random-weight micro-config tests per block: compressor (ratio 1 and
   2), indexer with and without candidates, the three attention modes,
   single-pass mHC and Engram. The reference's `ModelArgs` defaults are
   already a small model with all three modes.
2. Whole-model logits for one prefill plus decode steps that cross
   compression-group and candidate-block boundaries. Run once with the
   simulated quantization off and once with it on.
3. The DSpark draft forward: base logits, Markov-corrected logits and
   confidence per position.

Engram tests use the real `tokenizer.json`, because the reference
asserts the compressed vocabulary size against the configuration.

**Gate:** micro-config parity within the tolerances of the existing V4
parity tests, and greedy token parity for 32 decode steps.

### Stage C: serving integration (CPU, then small GPU)

- State cache with shared sources, continuous batching and ragged
  decode.
- The CUDA-graph decode step with shared state.
- Tensor-parallel sharding for TP=2: 32 indexer heads, 8 output groups,
  experts split by rank, Engram table rows.
- Prefix sharing.
- The prompt encoder, ported from `encoding/encoding.py` and tested
  against the checkpoint's `encoding/tests/` fixtures.
- Compile every new Triton kernel for sm100 in the CPU Docker image
  before any GPU run.

**Gate:** the existing V4-Flash tests do not change. The V4.1
micro-config gives equal outputs for TP=2 (gloo, CPU) and TP=1, and for
graph decode and eager decode.

### Stage D: real checkpoint on 2x B200

1. Make sure that a Modal B200 container can get about 210 GB of host
   RAM. If it cannot, move to option 2 (4x B200).
2. Load Engram into pinned host memory. Gather rows on the CPU from the
   token ids and copy them to the GPU on a side stream, before layer 1
   needs them.
3. Generate golden token fixtures with the reference `generate.py`
   (greedy, a few prompts, 64 tokens). The reference keeps Engram on
   the GPUs, so this needs 4x B200 for one short run.
4. Run the port on 2x B200 against the fixtures, with bounded replay
   off.

Ask the user before every Modal run, with GPU type, duration and cost.
Stop the app's containers after each run.

**Gate:** greedy token parity with the reference fixtures. If a
mismatch appears, look for top-k ties first.

### Stage E: performance (profile first)

Profile decode at batch 1 and 16 and a long-prompt prefill before
choosing work. Use long, real prompts. Candidates:

- Encoder-only prefill with decoder bounded replay. The report claims
  it about halves prefill work. Measure the output change against full
  prefill.
- DSpark drafting with block 5 through the batched drafter path.
  DeepInfer's own numbers show that confidence-scheduled verification
  loses 1.5% at 16 sequences per GPU and gains 5 to 11% at 60 to 96.
  At our batch sizes, the scheduler therefore comes after the batched
  drafter.
- FP4 cache storage, if KV memory becomes a limit.
- A single-pass hyper-connection kernel, the report's Mega-mHC idea,
  as a follow-up to ADR-018. It does the residual update, the input mix
  and the coefficient projection in one read of the residual.
- The host-side decode-loop fixes from the DeepInfer study (batched
  sampling, fewer syncs between ranks, graphs per batch size). These
  apply to V4-Flash too and are tracked separately.

**Gate:** none for the port. Each optimization ships with its own ADR
and measured numbers.

### Stage F: optional

- The ViT and aligner, image inputs through the API, and `bias_vl`
  routing for image tokens.
- Reasoning-effort control and the V4.1 tool-call format in the API
  server.

## Risks

- **Reference against production.** DeepSeek serves V4.1 with decoder
  bounded replay, and the reference does not use it. Outputs differ by
  design. Parity runs must pin bounded replay off, and the API must
  name which mode it runs.
- **Top-k ties.** The indexer top-k and the candidate-block top-k can
  pick other indices on ties across widths and platforms. Use the
  lower-index tie rule that the V4 indexer already uses
  (`_topk_lower_index_on_ties`). A test that passes on Mac and fails on
  Linux points here first.
- **Uninitialized parameters.** The reference builds every parameter
  with `torch.empty`. Every new module needs an explicit init, or CI
  sees flaky NaNs across processes.
- **Engram hash exactness.** One wrong multiplier or prime silently
  rehashes the whole table. The compressed-vocabulary assert catches a
  tokenizer mismatch but not a hash bug, so Stage B compares hash ids,
  not only outputs.
- **Host memory and load time.** 203 GB of pinned host memory and a
  510 GB download make the first load slow. Cache the weights on a
  Modal volume once.
- **CPU-backend gaps.** The vLLM V4 audit found a dropped expert clamp
  and BF16 router logits in a CPU path. Keep FP32 router logits and
  both clamps in every backend.
- **Reference location.** `third_party/` is not tracked. In a worktree,
  the parity tests skip unless the reference is copied or linked there.
- **TP teardown.** A live CUDA graph with captured NCCL calls hangs
  `destroy_process_group`. Free graphs before teardown, as for V4.

## References

- Weights, report and reference code:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash and
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
  (`DeepSeek_V41_Tech_Report.pdf`, `inference/`, `encoding/`)
- Report sections: 2.2 (CED), 2.3 (CSA2), 2.4 (single-pass mHC, Engram,
  DSpark, FP4 KV), 3.2 (inference system, bounded replay)
- DeepInfer: https://github.com/deepseek-ai/DeepInfer, including
  `doc/inference_deployment_qa.md` and `examples/yoco/`
- SGLang preview: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
- In-repo: ADR-014, ADR-016, ADR-018, ADR-019, ADR-020, ADR-027,
  ADR-031, ADR-042, ADR-043, ADR-044, ADR-045,
  `docs/plans/deepseek-v4-attention.md`, `docs/plans/dspark-evaluation.md`
