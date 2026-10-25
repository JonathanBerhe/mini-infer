# ADR-013: TurboQuant KV cache (V1 + V3 + V2a fused dequant; V2b deselected); FlashInfer FP8 + NVFP4

Date: 2026-04-29 (V1), 2026-04-30 (V3), 2026-05-02 (V2a/V2b, FlashInfer bf16/FP8/NVFP4)
Status: Accepted

This ADR covers six compressed KV cache paths:

- V1 (`kv_quant="turbo4"`): rotation plus per-channel uniform 4-bit. It
  saves ~62% of KV storage.
- V3 (`kv_quant="turbo3"`): the full TurboQuant recipe. It saves ~74%.
- V2a: a fused Triton dequant kernel for `turbo3`, with 6-7x the throughput
  of the Python-loop turbo4 path.
- V2b: a kernel that also fuses the attention. It is off by default.
- FlashInfer FP8 (`kv_quant="fp8"`, H100): FP8 e4m3fn paged storage, with
  the dequant fused into the FlashInfer kernel. It saves 50% of KV
  memory, with logit cosine 0.999985 against bf16 and 0.93x throughput at
  Qwen2.5-0.5B.
- FlashInfer NVFP4 (`kv_quant="nvfp4"`, B200): FP4-packed paged storage with
  per-16-element FP8 block scales and one FP32 global scale per (layer,
  side), passed via `kv_cache_sf`. It saves 71.9% of KV memory at 0.91x
  throughput at Qwen2.5-7B.

NVFP4 degrades token output under greedy decode. With outliers present, the
per-(layer, side) global scale pushes most K/V values toward FP4 zero. The
direction error is ~5% per layer (cosine 0.948 on a parity probe), and it
compounds across 28+ layers into incoherent text. A fix needs
outlier-aware preprocessing, for example per-channel scales or calibration,
and mini-infer does not implement it. The memory savings are real, and the
kernel path stays within the 1e-1 tolerance of FlashInfer. NVFP4 needs a
calibration step before use where token quality matters.

## Context

Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
([paper](https://arxiv.org/abs/2504.19874), ICLR 2026) compresses the KV
cache to 3 bits per value with almost no accuracy loss. It needs no training
and no data. The recipe has four parts:

1. A random orthogonal rotation of K/V before storage.
2. PolarQuant: a Cartesian-to-polar transform for the main compression.
3. A Lloyd-Max scalar quantizer (levels optimal for a Gaussian).
4. A QJL (Quantized Johnson-Lindenstrauss) residual sign bit.

V1 ships part 1 with a simpler per-block uniform 4-bit quantizer. It leaves
out parts 2 to 4, asymmetric K/V bits and a fused kernel. V1 builds the
storage and cache infrastructure that V2 (fused kernel) and V3 (full
algorithm) plug into.

## Decision

1. Random orthogonal rotation of K and V. Each layer has one matrix, shared
   across heads, from the QR decomposition of a Gaussian random matrix at
   engine startup. The shape is `(num_layers, head_dim, head_dim)` bf16,
   ~1 MB in total.
2. Per-channel asymmetric 4-bit quantization of each
   `(block_size, num_kv_heads, head_dim)` block. Each
   `(num_kv_heads, head_dim)` channel gets one `(low, scale)` pair from the
   per-channel min/max of the block at write time.
3. `BlockPool` gets a `kv_quant: str | None = None` parameter. With
   `"turbo4"`, `_compressed_storage` (int8 packed bytes), `_scales_storage`
   (bf16 scales) and `_rotation` (bf16 matrices) replace the bf16 `_storage`.
4. `PagedKVCache.append_kv_packed` and `materialize_packed_kv` dispatch on
   `pool.kv_quant`. A compressed write dequantizes, changes and requantizes
   each affected block. A compressed read calls `pool.read_compressed_block`
   per block, which unpacks, dequantizes and inverse-rotates to bf16.
5. K and V rotate on write and inverse-rotate on read. Q is not rotated. The
   result equals the bf16 path within quantization noise.
6. Opt-in via `ModelRunner.from_pretrained(..., kv_quant="turbo4")`. The
   default bf16 path stays byte-for-byte the same.
7. The FA paged path is off for compressed pools. The dispatcher in
   `cache/packed_attention.py` sends them to the materialized path, which
   can dequantize.

## Empirical findings (A10 bench)

| Model | bf16 KV pool | turbo4 KV pool | Savings |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 192.0 MiB | 72.2 MiB | 62.4% |
| Qwen2.5-7B | 896.0 MiB | 336.9 MiB | 62.4% |

4-bit theory gives ~75% savings. The per-channel `(low, scale)` pairs take
~33% of the compressed bytes, so compression is 2.7x instead of 4x.

On Qwen2.5-0.5B (24 layers, head_dim=64), turbo4 greedy output matches bf16
token for token. On Qwen2.5-7B (28 layers, head_dim=128), the output
diverges at index 2 into `" Paris. The capital of of France of"`. The
uniform 4-bit noise compounds across 28 layers until it exceeds the argmax
margin of bf16. Do not use V1 at 7B+ in production.

turbo4 runs at 0.06x to 0.18x of bf16 throughput on both models, at C = 1, 4
and 8 concurrent requests. The plan expected 0.5–0.8x. The cause is the
Python-side bit unpacking in `read_compressed_block` on every materialize.

## Alternatives considered

- Write the Triton fused kernel (V2) first. The kernel is hard: the
  `0xSero/turboquant` reference uses 3 separate kernels. V1 proves the
  algorithm in Python first and gives V2 a clean storage layout.
- 8-bit quantization with no rotation. It gives ~2x compression and loses no
  accuracy. It lost because it does not use the rotation that makes 4-bit
  and 3-bit possible.
- Calibrated per-tensor or per-token scales. They close part of the accuracy
  gap, but calibration loses the training-free, data-free property.

## Consequences

- KV memory drops ~2.7x on any model.
- Throughput is unusable at any scale. V2 fixes it. Accuracy degrades at
  7B+. V3 fixes it.
- Peak memory at attention time does not change, because compressed and
  materialized bf16 K/V exist together for a short time. V2 was to fix this
  (see the V2b update).
- `kv_quant=None` stays byte-for-byte the same, and a small revert removes
  the path.
- On M1, 16 unit tests cover the TurboQuant primitives, and 3 real-model
  integration tests pass. The existing 143-test unit suite stays green. The
  fp32 reference gives attention-logit cosine > 0.99 and first-token greedy
  parity. Tests: `tests/unit/test_turbo_quant.py`,
  `tests/unit/test_turbo_quant_integration.py`. Bench:
  `scripts/modal_packed_bench.py --config turbo`.
- Code: `src/mini_infer/cache/turbo_quant.py`,
  `src/mini_infer/cache/block_pool.py`,
  `src/mini_infer/cache/paged_kv_cache.py`.

## V3 update (2026-04-30): full algorithm shipped

`kv_quant="turbo3"` adds the left-out parts:

- V3a (PolarQuant): each K/V vector becomes a radius and a unit vector. A
  per-vector L2 norm replaces the `(low, scale)` pair.
- V3c (Lloyd-Max codebook): fixed quantizers for N(0,1), with 8 levels
  (3-bit) and 16 levels (4-bit), on the unit-vector coordinates.
- V3b (QJL residual sign bit): one bit tells if the value was above or below
  the chosen center. Dequant moves the value a quarter step that way.
- V3d (asymmetric K/V bits): K uses the 3-bit codebook plus the 1-bit QJL, 4
  bits stored. V uses the 4-bit codebook. The packed layout is the same as
  V1.

Results (`docs/benchmarks/2026-04-30-turboquant-v3.md`):

- Storage: ~74% savings on 0.5B and 7B, so ~3.7x compression against 2.7x.
- 7B: V3 wins. V1 gives the degenerate `of of France of`, and V3 gives
  `capital capital the`, which is also imperfect but more coherent.
- 0.5B: V1 wins. The 3-bit K of V3 flips the argmax at shallow depth,
  although the logit cosine stays > 0.99.
- Throughput: V3 is ~1.6x slower than V1 in the Python loop.

The turbo3 contract is "high-fidelity logits, not argmax parity": logit
cosine > 0.99 against bf16, and the argmax can flip. Both `turbo4` and
`turbo3` ship. `turbo4` is the simpler mode on small models.

## V2a update (2026-05-02): fused dequant kernel shipped

One Triton kernel per K/V side per layer replaces the per-block Python loop
in `materialize_packed_kv`. The kernel applies the V3 codec, inverse-rotates
with `tl.dot` and writes into the packed `(total_k, num_kv_heads, head_dim)`
buffer. `flash_attn_varlen_func` reads that buffer unchanged.

Qwen2.5-0.5B on an A10, real ~2000-token prompt
(`docs/benchmarks/2026-05-02-turboquant-v2a.md`):

- 6-7x the throughput of the Python-loop dequant at C ∈ {1, 4, 8}: 5.34 /
  6.10 / 6.31 t/s against 0.86 / 0.90 / 0.92 t/s for turbo4.
- 0.31-0.41x of bf16, against 0.04-0.18x in the V3 baseline.
- Storage savings stay at 73.3% on 0.5B, because the layout does not change.
- The tests in [test_turbo_kernel.py](../../tests/unit/test_turbo_kernel.py)
  compare the kernel with the Python loop, not with bf16.

Launch overhead, not arithmetic, was the bottleneck in V3. One kernel per
layer replaces hundreds of small launches.

## V2b update (2026-05-02): shipped, then deselected

V2b adds `_turbo_fused_attn_decode_kernel`. For each (request, q_head), it
dequantizes the compressed K/V tiles in registers and runs FA-2 online
softmax in the kernel. bf16 K/V never reach HBM. It handles decode only
(q_len=1 per request), and chunked prefill uses V2a. Its output matches V2a
at cosine 0.999996 on Qwen2.5-7B GQA (28 q_heads, 4 kv_heads, head_dim=128).

The bench (`docs/benchmarks/2026-05-02-turboquant-v2b.md`) tested the "fit
longer contexts" claim of the V2 follow-up:

- Qwen2.5-0.5B on an A10: 1.03x the throughput of V2a, identical peak HBM.
- Qwen2.5-7B on an A10: 0.88x the throughput (12% slower), identical peak HBM.

There are two reasons:

1. V2a calls `flash_attn_varlen_func`, a tensor-core kernel tuned over years.
   The Triton softmax of V2b keeps up at head_dim 64 but loses at 128.
2. The avoided buffer is too small to show. The materialized K/V per call is
   1-4 MB at 0.5-7B and ~2k tokens. The PyTorch caching allocator reuses the
   same bytes across layers, so peak memory does not change. Even
   Llama-405B at 32k tokens needs only ~250 MB, < 0.03% of 820 GB.

V2b stays in the code, off by default:
`_FUSED_ATTN_DISABLED_FOR_BENCH = True`. Set it to `False` for A/B
benchmarks or memory-constrained workloads. V2a is the attention path for
compressed pools.

### Production-grade alternatives

The production path for quantized KV depends on the GPU:

| Target GPU | Native KV format | Library | Notes |
|---|---|---|---|
| Ampere (A10, A100) | bf16 only | custom kernels | turbo3 + V2a |
| Hopper (H100, H200) | FP8 | FlashInfer FP8 KV / vLLM | ~50% memory vs bf16 |
| Blackwell (B200, B300, RTX 50) | FP4 (NVFP4) | FlashInfer NVFP4 KV / vLLM | ~75% memory vs bf16 |

FlashInfer NVFP4 ([v0.6.10rc1](https://github.com/flashinfer-ai/flashinfer/releases),
[NVIDIA blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/))
came out on 2026-04-30. It gives the 4-bit compression of turbo3, with the
dequant fused into attention through tensor-core FP4 support. That is the
design that V2b reached for, built by the hardware vendor. On Ampere, V2a
with FA varlen on materialized K/V is the right path.

## Follow-ups (in order of value/effort)

- V2a-for-turbo4, with the `(low, scale)` codec. The priority is low.
- Per-head rotation instead of per-layer, ~1% accuracy for a little more
  storage.
- Codebook calibration for one model. It loses the training-free property
  but can close the 7B parity gap.
- More aggressive QJL, for example a multi-bit residual. K accuracy
  improves, and storage grows.
