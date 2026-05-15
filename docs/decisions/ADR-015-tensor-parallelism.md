# ADR-015: Tensor parallelism (Megatron-style column/row/vocab/expert sharding)

Date: 2026-05-15
Status: Accepted

## Context

Several models in mini-infer's target set don't fit on a single GPU at
inference precision:

- **DeepSeek-V4-Flash**: 158 GB on disk (mixed FP4/FP8), >220 GB at
  BF16. Exceeds a single B200's 192 GB HBM once activations + KV +
  routing overhead are accounted for.
- **DeepSeek-V4-Pro**: 862 GB on disk. Needs ~8 ranks at minimum.
- **DeepSeek-V3 (671B)**, **Kimi-K2 (1T)**, **Llama-3-70B**,
  **Mixtral-8x22B**: also single-GPU-OOM at full precision.

Up to this ADR, every `load_weights` path assumed `model.load_state_dict(...)`
on a single device. That was sufficient through Phase 2's optimizations
(continuous batching, paged attention, prefix caching, INT8 weight quant,
spec decoding, TurboQuant KV) but blocks the V4 contribution from running
on real weights and blocks Llama-3-70B / Mixtral-8x22B from running at all.

We need a way to **shard model weights and split compute across N GPUs**
that:

1. Composes with the existing block library (no fork of GQA / MLA / HCA /
   CSA / SwiGLU / MoE).
2. Keeps the 393 single-device unit tests bit-identical (the
   `world_size=1` contract).
3. Is the standard mechanism inference engines use, so the code reads
   like vLLM / SGLang / the V4 reference rather than something bespoke.

## Decision

Megatron-style tensor parallelism, ported in this order: linears,
embeddings, attention, FFN (including expert-parallel MoE), LM head,
loader.

### Sharding strategy per module type

| Module                       | Sharding                    | Communication             |
|------------------------------|-----------------------------|---------------------------|
| `nn.Embedding`               | Vocab-parallel (vocab dim)  | All-reduce after lookup   |
| Q / K / V / Q-LoRA / KV-LoRA | Column-parallel (head dim)  | None at output            |
| Output projection (`o_proj`) | Row-parallel                | All-reduce after          |
| FFN gate / up                | Column-parallel             | None at output            |
| FFN down                     | Row-parallel                | All-reduce after          |
| MoE routed experts           | Expert-parallel             | All-to-all dispatch       |
| MoE shared experts           | Replicated                  | None                      |
| LM head                      | Column-parallel (vocab dim) | All-gather before sample  |
| Norms / per-head scalars     | Replicated or sharded with heads | None / implicit       |

The standard pairing `(col-parallel Q/K/V) -> attention -> (row-parallel O)`
and `(col-parallel gate/up) -> activation -> (row-parallel down)` gives
**exactly one all-reduce per attention block and one per FFN**, with no
intervening collective.

### Module hierarchy

Five files in `src/mini_infer/distributed/`:

- `group.py`: process-group lifecycle (NCCL on CUDA, gloo on CPU).
  `get_world_size()` returns `1` and `get_rank()` returns `0` when no
  group is initialised. This single invariant is what lets every TP-aware
  module build transparently on single-device.
- `comm.py`: thin wrappers over `dist.all_reduce`, `dist.all_gather`,
  `dist.all_to_all_single`, `dist.broadcast`, `dist.barrier`. Each has a
  `world_size == 1` fast path so callers can invoke them unconditionally.
- `linear.py`: `ColumnParallelLinear` and `RowParallelLinear`. Both
  subclass `nn.Linear` so anywhere the codebase asks
  `isinstance(m, nn.Linear)` (e.g. the int8 quantizer's module walk) a
  TP-aware linear at `world_size=1` is recognised as the plain
  `nn.Linear` it is bit-equivalent to.
- `embedding.py`: `VocabParallelEmbedding`. Each rank owns a slice of the
  vocab dim; the all-reduce after lookup combines the per-rank
  contributions (each rank emits zero for tokens not in its vocab slice).
- `loader.py`: `load_state_dict_with_tp(model, state_dict, *, target_device=None)`.
  Walks `state_dict` keys, dispatches to `load_full_weight` /
  `load_full_logits` / `load_full_wo_a` (V4's grouped output projection)
  on the matched TP-aware module. Handles meta-mode (replace the
  Parameter) vs allocated-mode (copy_) so V4-Flash can construct on
  meta + slice CPU-resident weights directly to GPU without ever holding
  the full BF16 model in any single HBM.

### Expert parallelism for MoE

Each rank owns `num_routed_experts // world_size` routed experts. The
hash-routed and score-topk gates both produce **global** expert indices;
the loader and the FFN block remap to local indices using
`_remap_expert_indices_to_local_rank`. Shared experts are replicated on
every rank (they're tiny and run on every token regardless of routing).
Hash-routing tables (`tid2eid`) are replicated.

Token dispatch uses `dist.all_to_all_single` with per-rank send/receive
split sizes computed from the routed expert IDs. Correctness was the
priority for the first ship; throughput optimization of the dispatch
path is a follow-up.

### The `world_size=1` contract

Every TP-aware module produces **bit-identical output to its plain-PyTorch
counterpart at `world_size=1`**, by construction:

- `ColumnParallelLinear(world_size=1) == nn.Linear` (no slicing, no
  gather, no collective).
- `RowParallelLinear(world_size=1) == nn.Linear` (the all-reduce becomes
  a no-op, the rank-0-only bias is just the bias).
- `VocabParallelEmbedding(world_size=1) == nn.Embedding`.
- `comm.all_reduce_sum` / `all_gather_along_dim` / `all_to_all_single` /
  `broadcast` / `barrier` are no-ops at `world_size=1`.

This is the invariant that kept the existing 393 single-device unit
tests green through the entire TP migration. Verified by running
`uv run pytest tests/unit/ -q` after every commit in the
TP infrastructure series.

## Why TP over alternatives

Four parallelism strategies were considered:

- **Tensor parallelism (this ADR)**. Splits each matmul across ranks
  with one all-reduce per block. Fast intra-node collectives (NVLink),
  simple weight layout, no pipeline bubbles. Standard inference choice
  (vLLM, SGLang, TensorRT-LLM, the V4 reference).
- **Pipeline parallelism**. Different layers on different ranks. Lower
  collective volume but introduces pipeline bubbles and complicates
  scheduling. Worthwhile complement at >8 ranks; unnecessary at
  2 or 4 ranks for V4-Flash.
- **FSDP / ZeRO-3**. Designed for training: shards optimizer state +
  gradients + parameters. Inference doesn't need optimizer state or
  gradients, so ZeRO-3 collapses to TP with extra abstraction cost.
- **Sequence parallelism / context parallelism**. Shards the sequence
  dim. Complementary to TP, valuable for very long contexts (>128k
  tokens). Out of scope for this ADR; possible follow-up.

TP is the right first parallelism axis to ship: it unblocks every
target model, composes with the existing block library, and is the
mechanism the reference implementations the project benchmarks against
actually use.

## Numerical correctness

Two parity tracks:

1. **`world_size=1` bit-parity** with the existing PyTorch path on every
   TP-aware module. Enforced by construction (the world_size=1
   collectives are no-ops). Validated by every single-device unit test.

2. **`world_size=2` parity** against the same model loaded once and
   replicated across two CPU processes (gloo backend). For each
   TP-aware primitive: column-parallel + row-parallel pairings produce
   the same output (within FP32 atol=1e-4 / BF16 atol=1e-2) as the
   un-sharded baseline on the same input + the same full weight loaded
   via `load_full_weight`. Validated in
   `tests/unit/test_distributed_linear.py`,
   `tests/unit/test_distributed_embedding.py`,
   `tests/unit/test_attention_tp_parity.py`,
   `tests/unit/test_ffn_tp_parity.py`.

The Qwen2.5-7B Modal smoke on 2× H100 is the real-hardware integration
test: a two-rank TP load produced finite per-rank sliced outputs with
the expected `hidden_size / world_size` head sharding, confirming the
NCCL path + the per-rank loader path work end-to-end on production
hardware. (Token-for-token greedy parity against single-device Qwen2.5-7B
on the same prompt is the next-level validation; deferred to keep
Modal spend bounded.)

## Loader integration

The loader is the part that touches every model family, because every
`load_weights` had to be routed through `load_state_dict_with_tp` (which
falls back to plain `model.load_state_dict(...)` at `world_size=1`).

Key decision: **`load_state_dict_with_tp` dispatches per-key, not
per-module.** It walks the state_dict, finds the parent module for each
key via `model.get_submodule`, and calls the right TP loader helper.
This keeps the per-family `load_weights` paths thin: they massage HF
key names (rename rules, MoE expert index remap, ...) and then hand the
result to the central loader.

For V4-Flash's meta-device path, the loader additionally:

- Keeps the full state_dict on CPU (V4-Flash is 158 GB on disk; loading
  to a single GPU at any point would OOM).
- Passes `target_device` through to `load_full_weight`, which slices
  the rank's per-tile from CPU and moves only that slice to GPU.
- Replaces the meta Parameter with a real Parameter holding the sliced
  weight, rather than `copy_`ing into a pre-allocated tensor (the
  pre-allocation itself would OOM).

This composes with the V4-specific dequant pairing: the loader sees a
`weight` key, looks up the sibling `weight_scale` key, runs the
NVFP4 or block-FP8 dequant on CPU, then slices to the rank's tile.
Peak per-rank HBM tracks the model's sharded share + activations + KV,
never the full 158 GB.

## Consequences

**Positive**:

- Every model family in the registry is now TP-capable. Llama-3-70B,
  Mixtral-8x22B, DeepSeek-V3 (671B), Kimi-K2 (1T), V4-Flash, V4-Pro all
  have a path to running on multi-GPU.
- Single-device behaviour is unchanged. The 393 single-device unit
  tests stay bit-identical; no opt-in flag, no parallel code paths to
  maintain.
- TP-aware loaders are the foundation for the V4-Flash mixed-quant
  loader (ADR-014 update): the same `load_state_dict_with_tp` path
  handles block-FP8 + NVFP4 dequant via the per-key dispatch.
- The expert-parallel MoE path lays the groundwork for any future
  large-MoE model (V4-Pro's 256 experts, Mixtral-8x22B's 8 experts,
  Switch-style hash-routed MoEs).

**Negative**:

- Weight-only INT8 quantization (`Int8Linear`) is single-rank only
  today. Quantizing a `ColumnParallelLinear` or `RowParallelLinear`
  needs a TP-aware INT8 path (per-rank scales + per-rank GEMM); not
  built yet. The TP-aware linears subclass `nn.Linear` so the existing
  quantizer's `isinstance` check fires, but the resulting Int8Linear
  would silently lose the sharding metadata. Listed as a follow-up.
- TP-aware FlashInfer (FP8 / NVFP4 KV) hasn't been validated end-to-end.
  The KV cache layout is per-layer per-stream and shards naturally with
  the attention heads, but the FlashInfer wrapper hasn't been exercised
  on a TP run. Follow-up.
- Loader complexity bumped: per-rank shard tracking, meta-device
  awareness, dequant pairing. Concentrated in `loader.py` +
  per-family `load_weights`; doesn't bleed into the block library.
- Multi-process tests are slower than single-device tests
  (~3-5s startup per test for the gloo PG init). All gated behind a
  `requires_distributed` marker so CI's CPU-only path stays fast.

**Reversibility**: clean. `world_size=1` is the no-op path on every
TP primitive. Reverting to single-device is the trivial case the code
already takes when no PG is initialised.

## Validation

- **Unit (CPU, `world_size=1`)**: 393 existing unit tests stay green
  after every commit in the series. TP-aware linears, embedding, and
  loader produce bit-identical output to their plain-PyTorch
  counterparts.
- **Multi-process (CPU, `world_size=2`, gloo)**:
  - `tests/unit/test_distributed_linear.py`: column-parallel +
    row-parallel pairing parity.
  - `tests/unit/test_distributed_embedding.py`: vocab-parallel
    embedding parity.
  - `tests/unit/test_attention_tp_parity.py`: GQA, MLA, HCA, CSA,
    SWA parity at `world_size=2`.
  - `tests/unit/test_ffn_tp_parity.py`: SwiGLU + expert-parallel
    MoE (Mixtral-shape + hash-routed) parity at `world_size=2`.
- **Real hardware (2× H100, NCCL)**: Qwen2.5-7B two-rank TP load
  produced finite per-rank sliced outputs with the expected
  `hidden_size / world_size` head sharding
  (`scripts/modal_llama_tp_smoke.py`; chose Qwen2.5-7B over Llama-3-8B
  since Llama-3 weights require an HF gating step the project doesn't
  have access to).
- **V4-Flash load path**: dry-run against the real V4-Flash safetensors
  index matches all 34,223 model parameters (zero missing / zero
  unexpected after dequant pairing consumes 33,389 sibling `.scale`
  companions); dequant numerically verified against a real V4-Flash
  shard. Forward pass on 2× B200 is the remaining hardware gate, with
  three dtype + meta-init fixes landed since the load was first proven.

## Pointers

- TP primitives:
  - `src/mini_infer/distributed/group.py` (PG lifecycle)
  - `src/mini_infer/distributed/comm.py` (collectives)
  - `src/mini_infer/distributed/linear.py` (column/row-parallel)
  - `src/mini_infer/distributed/embedding.py` (vocab-parallel)
  - `src/mini_infer/distributed/loader.py` (per-rank loader)
- TP-aware blocks:
  - `src/mini_infer/models/blocks/gqa.py`
  - `src/mini_infer/models/blocks/mla.py`
  - `src/mini_infer/models/blocks/hca.py` / `csa.py` / `swa.py`
  - `src/mini_infer/models/blocks/v4/lightning_indexer.py`
  - `src/mini_infer/models/blocks/swiglu.py`
  - `src/mini_infer/models/blocks/mixtral_moe.py`
  - `src/mini_infer/models/blocks/hash_routed_moe_ffn.py`
  - `src/mini_infer/models/blocks/hash_routed_gate.py`
- Per-family loaders routed through TP:
  - Every `*ForCausalLM.load_weights` in `src/mini_infer/models/`
- Tests:
  - `tests/unit/test_distributed_linear.py`
  - `tests/unit/test_distributed_embedding.py`
  - `tests/unit/test_attention_tp_parity.py`
  - `tests/unit/test_ffn_tp_parity.py`
- Modal smokes:
  - `scripts/modal_llama_tp_smoke.py` (Qwen2.5-7B on 2× H100; validated)
  - `scripts/modal_v4_flash_smoke.py` (V4-Flash on 2× B200; load
    validated, forward remains the gate)

## Follow-ups

- **TP-aware INT8 weight quantization**: extend `Int8Linear` to handle
  the column-parallel / row-parallel slice shape and per-rank scales.
  Unblocks quantized large-model serving.
- **TP-aware FlashInfer FP8 / NVFP4 KV**: validate the FlashInfer
  wrapper on a TP run. KV layout shards naturally with attention heads
  but hasn't been exercised end-to-end.
- **V4-Flash forward end-to-end validation on 2× B200**: the final gate
  for the V4 contribution. Load is proven; three dtype + meta-init
  bugs found via Modal probes are statically fixed (compressor
  projection dtype, Hyper-Connections FP32 preservation, rotary
  inv_freq re-materialization). One more run is expected to close it.
- **Pipeline parallelism** for >8 ranks (V4-Pro at 862 GB).
- **Sequence parallelism / context parallelism** for very long contexts.
- **Optimized expert-parallel dispatch**: correctness shipped first; the
  `all_to_all_single` path is a candidate for fusing the dispatch +
  expert forward + return-dispatch into fewer collectives.
- **Token-for-token greedy parity** of the Qwen2.5-7B TP run against
  single-device Qwen2.5-7B on the same prompt (currently only
  shape + finiteness is verified on real hardware).
