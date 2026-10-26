# ADR-015: Tensor parallelism (Megatron-style column/row/vocab/expert sharding)

Date: 2026-05-15
Status: Accepted

## Context

Several target models do not fit on one GPU at inference precision.
DeepSeek-V4-Flash is 158 GB on disk (mixed FP4/FP8) and >220 GB at BF16.
With activations, KV and routing overhead, it exceeds the 192 GB HBM of one
B200. DeepSeek-V4-Pro is 862 GB on disk and needs about 8 ranks.
DeepSeek-V3 (671B), Kimi-K2 (1T), Llama-3-70B and Mixtral-8x22B also do
not fit on one GPU at full precision.

Until this ADR, every `load_weights` path called
`model.load_state_dict(...)` on one device. That blocked V4 on real
weights, and Llama-3-70B and Mixtral-8x22B did not run at all. The fix
must:

1. Compose with the existing block library, with no fork of GQA, MLA, HCA,
   CSA, SwiGLU or MoE.
2. Keep the 393 single-device unit tests bit-identical (the `world_size=1`
   contract).
3. Use the standard mechanism, so the code reads like vLLM, SGLang or the
   V4 reference.

## Decision

Use Megatron-style tensor parallelism (TP: each matmul split across GPU
processes, called ranks).

| Module | Sharding | Communication |
|---|---|---|
| `nn.Embedding` | Vocab-parallel | All-reduce after lookup |
| Q / K / V / Q-LoRA / KV-LoRA | Column-parallel (head dim) | None |
| Output projection (`o_proj`) | Row-parallel | All-reduce after |
| FFN gate / up | Column-parallel | None |
| FFN down | Row-parallel | All-reduce after |
| MoE routed experts | Expert-parallel | All-to-all dispatch |
| LM head | Column-parallel (vocab dim) | All-gather before sample |

A column-parallel projection followed by a row-parallel one gives exactly
one all-reduce per attention block and one per FFN.

The code is in `src/mini_infer/distributed/`. `group.py` owns the process
group (NCCL on CUDA, gloo on CPU). `comm.py` wraps the collectives.
`linear.py` has `ColumnParallelLinear` and `RowParallelLinear`, which
subclass `nn.Linear`. `embedding.py` has `VocabParallelEmbedding`, where
each rank owns a slice of the vocabulary. `loader.py` has
`load_state_dict_with_tp`.

Each rank owns `num_routed_experts // world_size` routed experts.
`_remap_expert_indices_to_local_rank` maps the global expert indices of the
gates to local ones. Shared experts and the hash-routing tables (`tid2eid`)
are replicated, because they are small. Token dispatch uses
`dist.all_to_all_single`. This first version puts correctness before
dispatch speed.

The `world_size=1` contract: with no process group, `get_world_size()`
returns `1`. Then every TP-aware module gives bit-identical output to its
plain PyTorch counterpart, and every collective is a no-op. This kept the
393 single-device unit tests green through the whole migration.

## Why TP over alternatives

- Tensor parallelism needs one all-reduce per block over fast NVLink and
  has no pipeline bubbles (idle stages). vLLM, SGLang, TensorRT-LLM and the
  V4 reference use it.
- Pipeline parallelism puts layers on different ranks. It adds bubbles
  and makes scheduling harder. It helps above 8 ranks, but V4-Flash needs
  only 2 or 4.
- FSDP / ZeRO-3 shards optimizer state, gradients and parameters for
  training. Inference has no optimizer state or gradients, so ZeRO-3
  becomes TP with extra abstraction.
- Sequence or context parallelism complements TP for contexts above 128k
  tokens. It is out of scope.

## Numerical correctness

At `world_size=1`, bit-parity holds by construction, and every
single-device unit test covers it. At `world_size=2`, two CPU processes
with gloo run each sharded primitive. It must match the unsharded baseline
within FP32 atol=1e-4 or BF16 atol=1e-2. The tests are
`tests/unit/distributed/test_distributed_linear.py`,
`tests/unit/distributed/test_distributed_embedding.py`,
`tests/unit/distributed/test_attention_tp_parity.py` (GQA, MLA, HCA, CSA, SWA) and
`tests/unit/distributed/test_ffn_tp_parity.py` (SwiGLU and expert-parallel MoE).

On 2× H100 with NCCL, `scripts/llama/modal_llama_tp_smoke.py` loads Qwen2.5-7B
with two-rank TP. The per-rank outputs are finite and have the expected
`hidden_size / world_size` head sharding. Qwen2.5-7B replaced Llama-3-8B,
because the Llama-3 weights need an HF gating step. Token-for-token greedy
parity against single-device Qwen2.5-7B is not tested yet.

For V4-Flash, a dry run against the real safetensors index matches all
34,223 model parameters, after dequant pairing consumes 33,389 sibling
`.scale` companions. The forward pass on 2× B200
(`scripts/deepseek_v4/modal_v4_flash_smoke.py`) is the remaining hardware gate. Three
dtype and meta-init bugs found there are fixed in code, not yet on
hardware: compressor projection dtype, Hyper-Connections FP32 preservation
and rotary inv_freq re-materialization.

## Loader integration

`load_state_dict_with_tp` falls back to `model.load_state_dict(...)` at
`world_size=1`. It dispatches per key, not per module. It finds the parent
module of each key with `model.get_submodule` and calls the right helper,
for example `load_full_weight`. So the per-family `load_weights` paths only
rename HF keys and remap MoE expert indices.

For the meta-device path of V4-Flash, the loader keeps the full state_dict
on CPU. It passes `target_device` to `load_full_weight`, which moves only
the slice of the rank to the GPU. It replaces the meta Parameter with a real
one, because a preallocated tensor will OOM. For a `weight` key, it finds
the sibling `weight_scale` key and runs the NVFP4 or block-FP8 dequant on
CPU before the slice. Peak HBM per rank is the sharded share plus
activations and KV, never the full 158 GB.

## Consequences

- Every model family in the registry can run with TP, from Llama-3-70B to
  V4-Pro. Single-device behavior does not change, with no opt-in flag and
  no second code path. Reverting is clean.
- The V4-Flash mixed-quant loader (ADR-014 update) uses the same per-key
  dispatch.
- `Int8Linear` (weight-only INT8) works on one rank only. The quantizer's
  `isinstance` test matches the TP linears, but the resulting Int8Linear
  silently loses the sharding metadata. A TP-aware INT8 path with per-rank
  scales is a follow-up.
- TP-aware FlashInfer (FP8 / NVFP4 KV) is not tested end to end.
- The loader is more complex, but that stays out of the block library.
- Multi-process tests need ~3-5s per test for the gloo PG init. The
  `requires_distributed` marker keeps the CPU-only CI path fast.
- Other follow-ups: pipeline parallelism for >8 ranks (V4-Pro at 862 GB),
  sequence parallelism, and fewer collectives in the expert dispatch.
