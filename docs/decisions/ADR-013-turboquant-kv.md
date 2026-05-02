# ADR-013: TurboQuant KV cache (V1 + V3 + V2a fused dequant; V2b deselected); FlashInfer FP8 + NVFP4

Date: 2026-04-29 (V1), 2026-04-30 (V3), 2026-05-02 (V2a/V2b, FlashInfer bf16/FP8/NVFP4)
Status: Accepted

V1 (`kv_quant="turbo4"`): rotation + per-channel uniform 4-bit. Shipped
2026-04-29. ~62% storage savings; parity holds on 0.5B, breaks at 7B.

V3 (`kv_quant="turbo3"`): full TurboQuant — rotation + polar transform +
Lloyd-Max codebook + QJL residual + asymmetric K (3-bit + QJL = 4 bits
stored) / V (4-bit Lloyd-Max). Shipped 2026-04-30. ~74% storage savings;
0.5B coherent-but-different argmax; 7B less degenerate than V1.

V2a (fused dequant Triton kernel for `turbo3`): one launch per K/V side
per layer replaces the per-block Python loop in `materialize_packed_kv`.
Shipped 2026-05-02. **6-7x throughput over the Python-loop turbo4 path**
on Qwen2.5-0.5B + A10; turbo3 throughput recovers from 0.04-0.18x of
bf16 (V3 baseline) to 0.31-0.41x of bf16. Storage layout unchanged.

V2b (fully-fused dequant + online softmax for decode): a second Triton
kernel that walks compressed K/V tiles in registers and avoids
materializing bf16 K/V in HBM. Shipped 2026-05-02 but **deselected as
the default attention path the same day**: at Qwen2.5-7B + A10, V2b is
12% slower than V2a (FA varlen on materialized K/V) and the avoided
transient buffer is too small to register on the peak-memory meter.
Kept opt-in via `_FUSED_ATTN_DISABLED_FOR_BENCH = False` (default True)
for memory-constrained edge cases.

FlashInfer FP8 KV (`kv_quant="fp8"`): FP8 e4m3fn paged storage routed
through FlashInfer's tensor-core paged-attention wrapper, which fuses
dequant into the kernel. Shipped 2026-05-02 on H100. **50% KV memory
savings**, logit cos sim 0.999985 vs bf16, 0.93x throughput at
Qwen2.5-0.5B. The production-grade 8-bit KV path on Hopper.

FlashInfer NVFP4 KV (`kv_quant="nvfp4"`): FP4-packed paged storage with
per-16-element FP8 block scales + per-(layer, side) FP32 global scale,
routed through the same FlashInfer wrapper via `kv_cache_sf`. Shipped
2026-05-02 on B200. **71.9% KV memory savings**, 0.91x throughput at
Qwen2.5-7B. **Token-level accuracy is degraded under greedy decode**:
the textbook per-(layer, side) global scale leaves bulk K/V values
quantized toward FP4-zero in the presence of outliers, and the
per-layer ~5% direction error (cos sim 0.948 on a parity probe)
compounds across 28+ layers into incoherent token output. Production
NVFP4 KV deployments need outlier-aware preprocessing (per-channel
scales, SmoothQuant-style transforms, calibration) we haven't
implemented. Integration infrastructure is correct (memory savings
real, kernel path validated within FlashInfer's own 1e-1 tolerance);
treat this as a working FP4 plumbing layer that needs a calibration
slice on top before token-quality-sensitive use.

## Context

Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
([paper](https://arxiv.org/abs/2504.19874), ICLR 2026, ~weeks old as of this
slice) is a training-free, data-oblivious algorithm that compresses the KV
cache to 3 bits per value with effectively no accuracy loss. The full recipe:

1. Random orthogonal rotation of K/V before storage.
2. PolarQuant (Cartesian → polar) for primary compression.
3. Lloyd-Max scalar quantizer applied to rotated values.
4. QJL (Quantized Johnson-Lindenstrauss) residual sign bit.

This ADR ships V1: items 1 + a simpler per-block uniform 4-bit quantizer
(no PolarQuant, no QJL, no Lloyd-Max, no fused dequant-attention kernel).
The point is to demonstrate the rotation-based pipeline end-to-end and
build the storage / cache infrastructure that V2 (fused kernel) and V3
(full algorithm) plug into.

This is a different layer of the engine than what we've quantized so far:

| What | Where it lives | What we have |
|---|---|---|
| Weight quant (W8A16, ADR-010/012) | `Int8Linear`, fused kernel | INT8 weights, ~30% memory savings, fused 2.74x at 7B decode |
| **KV cache** | `BlockPool`, paged storage | Now: 4-bit + rotation (this ADR), or bf16/fp16 (default) |

## Decision

1. **Random orthogonal rotation** of K and V, one matrix per layer
   (shared across heads), generated at engine startup via QR decomposition
   of a Gaussian random matrix. `(num_layers, head_dim, head_dim)` bf16,
   ~1 MB total for any model we care about.
2. **Per-channel asymmetric 4-bit quantization** of each
   `(block_size, num_kv_heads, head_dim)` block. One `(low, scale)` pair
   per `(num_kv_heads, head_dim)` channel, computed at write time from
   the block's per-channel min/max. ~2.7x compression vs bf16 (the
   `(low, scale)` overhead is ~50% of the compressed values' bytes — see
   "Empirical findings" below).
3. **`BlockPool` gains a `kv_quant: str | None = None` parameter**.
   When set to `"turbo4"`, the bf16 main `_storage` is replaced by:
   - `_compressed_storage`: int8 packed bytes,
     `(num_layers, 2, num_blocks, packed_bytes_per_block)`.
   - `_scales_storage`: bf16 scales,
     `(num_layers, 2, num_blocks, num_kv_heads, head_dim, 2)`.
   - `_rotation`: bf16 rotation matrices,
     `(num_layers, head_dim, head_dim)`.
4. **`PagedKVCache.append_kv_packed` and `materialize_packed_kv`
   dispatch on `pool.kv_quant`**. Compressed-mode write does
   dequant→modify→requantize on each affected block (slow but correct;
   the proper fix is a fused write kernel in V2). Compressed-mode read
   calls `pool.read_compressed_block` per block, which unpacks +
   dequantizes + inverse-rotates back to bf16.
5. **Symmetric attention path**: rotation on write, inverse-rotation
   on read. Q is NOT rotated. Math is identity vs the bf16 path within
   quantization noise (`(K @ R)^T` would be needed to attend against
   `Q @ R`; we skip the round-trip by inverse-rotating K on read).
6. **Opt-in via `ModelRunner.from_pretrained(..., kv_quant="turbo4")`**.
   The default is the existing bf16 path, byte-for-byte unchanged.
7. **FA paged path is gated off when compressed**. The dispatcher in
   `cache/packed_attention.py` checks `pool.kv_quant`; compressed pools
   route to the materialized path which knows how to dequant.

## Why this scope (V1 cuts)

The full TurboQuant recipe has four layered techniques. V1 ships only
the first plus a simpler quantizer:

| Component | V1 | Full TurboQuant |
|---|---|---|
| Random orthogonal rotation | ✓ | ✓ |
| Per-block uniform 4-bit quant | ✓ | — |
| PolarQuant (Cartesian → polar) | — | ✓ |
| QJL (residual sign bit) | — | ✓ |
| Lloyd-Max codebook | — | ✓ |
| Asymmetric 3-bit K + 2/4-bit V | — | ✓ |
| Fused dequant-attention kernel | — | ✓ |

V1 establishes the structural pieces (rotation lifecycle, compressed
storage, dequant pipeline, materialized-attention dispatcher gate). Each
deferred component is documented as a follow-up and the bench numbers
make clear which one each follow-up is buying.

## Empirical findings (A10 bench)

### Storage (~62.4% savings, both models)

| Model | bf16 KV pool | turbo4 KV pool | Savings |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 192.0 MiB | 72.2 MiB | 62.4% |
| Qwen2.5-7B | 896.0 MiB | 336.9 MiB | 62.4% |

The 62.4% figure is consistent across model sizes — the compression
ratio is a property of the per-block quant scheme, not the model.

The "missing" 13% (4-bit theory says ~75% savings) goes to the
per-block per-channel `(low, scale)` overhead: ~33% of compressed bytes
are scales, only ~67% are the actual quantized values. PolarQuant (V3)
removes the `low` parameter (polar coordinates are signed-around-zero)
and would reclaim most of this overhead, getting close to the paper's
~5x figure.

### Accuracy: 0.5B passes, 7B doesn't

On Qwen2.5-0.5B (24 layers, head_dim=64), V1's rotation + uniform
4-bit produces **full token-for-token parity** vs bf16:

```
prompt:  "The capital of France is"
bf16:    [12095, 13, 1084, 374, 279, 7772, 3283, 304]
turbo4:  [12095, 13, 1084, 374, 279, 7772, 3283, 304]   (full match)
text:    " Paris. It is the largest city in"
```

On Qwen2.5-7B (28 layers, head_dim=128), V1's accuracy **breaks down**.
First token matches, full sequence diverges at index 2 with degenerate
output:

```
prompt:  "The capital of France is"
bf16:    [12095, 13, 15920, 315, 279, 2701, 12239, 374]
         " Paris. Which of the following statements is"
turbo4:  [12095, 13, 576, 6722, 315, 315, 9625, 315]
         " Paris. The capital of of France of"
```

Per-block uniform 4-bit noise compounds across 28 layers; the
cumulative drift exceeds the bf16-baseline argmax margin and produces
the observable "of of" repetition. This is exactly the regime the full
TurboQuant recipe is calibrated for: PolarQuant + Lloyd-Max + QJL
together close ~2 bits of the residual error budget.

V1's accuracy is acceptable for ≤ 0.5B-class models for demonstration
purposes; **V1 should not be used at 7B+ in production**. V3 (full
algorithm) is the path forward.

### Throughput: catastrophic regression at any scale

V1 dequantizes to bf16 inside `materialize_packed_kv`, which calls
`read_compressed_block` per block via Python-side bit unpacking +
dequant + inverse-rotate. The overhead is severe:

| Model | C | bf16 t/s | turbo4 t/s | turbo / bf16 |
|---|---:|---:|---:|---:|
| 0.5B | 1 | 27.18 | 3.44 | **0.13x** |
| 0.5B | 4 | 50.31 | 3.80 | **0.08x** |
| 0.5B | 8 | 59.84 | 3.87 | **0.06x** |
| 7B | 1 | 16.09 | 2.87 | **0.18x** |
| 7B | 4 | 34.11 | 3.23 | **0.09x** |
| 7B | 8 | 42.77 | 3.29 | **0.08x** |

V1 is **bench-wrecking** for throughput. The plan anticipated a
0.5–0.8x regression; the actual numbers are an order of magnitude
worse, dominated by Python-side bit fiddling on every attention
materialize. This is exactly the V2 fused-kernel territory.

## Alternatives considered

- **Skip V1, jump straight to a Triton fused kernel (V2)**. Would
  deliver real performance. Rejected for V1 because the kernel is hard
  (3 separate kernels in `0xSero/turboquant`'s reference implementation;
  our fused-INT8 kernel took 3 Modal iterations to compile). Splitting
  the work into "infra + correctness first, kernel second" lets V1
  validate the algorithm in Python and gives V2 a clean storage layout
  to optimize against.
- **8-bit quantization without rotation (no TurboQuant insight)**.
  Simpler, ~2x compression, no accuracy loss at any scale. Rejected
  because it doesn't demonstrate the rotation-based technique that
  makes 4-bit/3-bit viable — that's the whole point of TurboQuant.
- **Calibration-based (per-tensor or per-token) scales**. Would close
  some of the accuracy gap. Rejected for V1 because TurboQuant's pitch
  is "training-free, data-oblivious" — calibration would lose that
  property.

## Consequences

- **Positive**:
  - Engine now has a TurboQuant-style cache compression path. ~2.7x
    persistent KV memory savings on any model.
  - Rotation infrastructure (per-layer matrices, write-rotate /
    read-inverse-rotate plumbing) is in place. V2 reuses it directly.
  - 0.5B parity holds end-to-end with this V1 (token-for-token decode),
    so the cache plumbing is correct.
  - Front-edge implementation: the paper is week-old; V1 is among the
    first public rotation-based KV-quant integrations into a continuous
    batching engine.
- **Negative**:
  - Throughput is unusable at any scale (Python-loop dequant on every
    materialize). Fixed by V2.
  - Accuracy degrades at 7B+ (per-block uniform 4-bit isn't accurate
    enough through 28+ layers). Fixed by V3.
  - Peak memory at attention time is unchanged (compressed +
    materialized bf16 coexist transiently). Fixed by V2.
  - Per-block per-channel scales overhead is ~33% of compressed bytes,
    dropping the realized compression from 4x theoretical to 2.7x
    actual. Fixed by V3 (PolarQuant removes `low` parameter).
- **Reversibility**: clean. `kv_quant=None` is unchanged byte-for-byte.
  Removing the path is a small revert.

## Validation

- **M1 (CPU/MPS, fp32/fp16)**: 16 unit tests for the standalone
  TurboQuant primitives + 3 real-model integration tests + cache
  round-trip. Existing 143-test unit suite stays green; the
  bf16-cache path is unchanged. M1 fp32 reference produces:
  cosine sim of attention logits > 0.99 vs baseline; first-token
  greedy decode parity; "Paris" smoke green.
- **CUDA (A10, bf16)**: 0.5B full-sequence parity, 7B first-token
  parity but full-sequence divergence (per above). Storage savings
  ~62.4% on both. Throughput regresses 5–15x.

## Pointers

- Implementation: `src/mini_infer/cache/turbo_quant.py` (rotation +
  quant primitives), `src/mini_infer/cache/block_pool.py` (compressed
  storage layout), `src/mini_infer/cache/paged_kv_cache.py` (compressed
  append/materialize).
- Integration: `ModelRunner.from_pretrained(kv_quant="turbo4")`.
- Unit tests: `tests/unit/test_turbo_quant.py`,
  `tests/unit/test_turbo_quant_integration.py`.
- Bench: `scripts/modal_packed_bench.py --config turbo`.
- Paper: https://arxiv.org/abs/2504.19874
- Reference repos: `0xSero/turboquant`, `tonbistudio/turboquant-pytorch`.

## V3 update (2026-04-30) — full algorithm shipped

The follow-ups V3a-d listed at the bottom of this ADR were all
implemented and shipped under `kv_quant="turbo3"`:

- **V3a (PolarQuant)**: Cartesian K/V → polar (radius + unit vector).
  Per-vector L2 norm replaces the per-channel `(low, scale)` pair. The
  unit vectors are bounded and ~Gaussian after rotation.
- **V3c (Lloyd-Max codebook)**: precomputed optimal-for-N(0,1) scalar
  quantizer with 8 levels (3-bit) and 16 levels (4-bit), hardcoded as
  module constants. Replaces uniform quant on the unit-vector coords.
- **V3b (QJL residual sign bit)**: 1-bit sign telling whether the
  pre-quant value was above or below the chosen Lloyd-Max center, used
  to nudge the dequantized value by a quarter-step.
- **V3d (asymmetric K/V bits)**: K side uses 3-bit Lloyd-Max codebook
  + 1-bit QJL = 4 bits stored. V side uses 4-bit Lloyd-Max codebook
  directly. Same packed layout as V1 (4 bits/element, two per byte).

Full bench in `docs/benchmarks/2026-04-30-turboquant-v3.md`. Headline
findings:

- **Storage**: V3 saves ~74% (vs V1's 62%) on both 0.5B and 7B. The
  per-vector radii layout removes most of V1's per-channel scale
  overhead. ~3.7x compression vs V1's 2.7x.
- **Accuracy at depth**: V3 wins at 7B. V1 produces degenerate
  `of of France of` repetition; V3 produces `capital capital the` —
  also imperfect but more coherent. The Lloyd-Max + QJL machinery
  recovers ~1 bit of effective precision that V1 loses.
- **Accuracy at small depth**: V1 wins on 0.5B. V1's full-sequence
  parity vs V3's argmax flip. V3's 3-bit K is more aggressive than
  V1's 4-bit and tips the argmax at shallow depth even though logit
  cosine sim still > 0.99.
- **Throughput**: V3 is ~1.6x slower than V1 (Lloyd-Max table lookup
  + QJL bit fiddle on the Python-loop dequant). Both unusable until
  V2 (fused kernel).

Both `kv_quant="turbo4"` and `kv_quant="turbo3"` ship; `turbo4` is
preserved for the V1-vs-V3 comparison and as the simpler choice on
small models. The shipped V3 demonstrates the full algorithm; the
remaining gap to the paper's claims is on more aggressive QJL step
tuning, per-head rotations, and calibration of the codebook to the
specific model's K/V distributions — outside the V3 scope.

## V2a update (2026-05-02) — fused dequant kernel shipped

V2 is split into two stages; V2a (this update) ships the fused dequant
half. A single Triton kernel per K/V side per layer replaces the
per-block Python loop in `materialize_packed_kv` that called
`pool.read_compressed_block(...)` twice per block per layer per step.
The kernel reads packed nibbles + per-vector radii + per-layer rotation,
applies the V3 codec (3-bit Lloyd-Max + 1-bit QJL on K, 4-bit Lloyd-Max
on V), inverse-rotates via `tl.dot` against the rotation transpose, and
writes directly into the packed `(total_k, num_kv_heads, head_dim)`
buffer. The existing `flash_attn_varlen_func` consumes that buffer
unchanged.

Headline numbers on Qwen2.5-0.5B + A10, real ~2000-token prompt
(`docs/benchmarks/2026-05-02-turboquant-v2a.md`):

- **6-7x throughput** over the same Python-loop dequant at C ∈ {1, 4, 8}
  (5.34 / 6.10 / 6.31 t/s vs turbo4's 0.86 / 0.90 / 0.92 t/s).
- **0.31-0.41x of bf16** (vs the V3 baseline's 0.04-0.18x). Different
  prompt length than the V3 doc so not a clean apples-to-apples, but the
  *same-class* turbo3-vs-turbo4 comparison above is the clean evidence.
- Storage savings unchanged (73.3% on 0.5B) — V2a doesn't touch the
  layout. Codebooks + rotation matrices are the same tensors.
- turbo3's coherent-but-different-argmax property carries through; the
  unit-test [`test_qwen_05b_turbo3_greedy_matches_python_path`](../../tests/unit/test_turbo_kernel.py)
  is the relevant correctness check (kernel vs Python loop, not vs bf16).

This confirms the V2 hypothesis: launch overhead, not arithmetic, was
the bottleneck in V3. One kernel per layer replaces hundreds of small
launches and the savings track perfectly with the number of blocks
walked per step.

V2b (fuse FA online softmax into the same kernel so K/V tiles are
dequanted in registers and never materialized) remains as the secondary
follow-up — captures the peak-memory reduction needed for "fit longer
contexts on the same GPU" but, on the throughput axis specifically, the
big win is already in.

## V2b update (2026-05-02) — shipped, then deselected

V2b shipped a second Triton kernel
(`_turbo_fused_attn_decode_kernel`) that walks compressed K/V tiles for
each (request, q_head), dequants them in registers using the V2a codec
body, and runs FA-2 online softmax in-kernel — never materializing
bf16 K/V in HBM. Decode-only contract (q_len=1 per request); chunked
prefill falls through to the V2a path. Cosine sim parity vs V2a's
attention output: 0.999996 on Qwen2.5-7B GQA (28 q_heads, 4 kv_heads,
head_dim=128).

Bench (`docs/benchmarks/2026-05-02-turboquant-v2b.md`) tested the
"fit longer contexts" thesis directly. **The result deselected V2b as
the default**:

- Qwen2.5-0.5B + A10: V2b 1.03x throughput vs V2a, peak HBM identical.
- Qwen2.5-7B + A10: V2b **0.88x throughput** (12% slower) vs V2a, peak
  HBM identical.

Two reasons:

1. **V2a uses FlashAttention; V2b reimplements it.** V2a calls
   `flash_attn_varlen_func` on the materialized bf16 K/V — a kernel
   tuned over years on tensor cores. V2b's hand-rolled Triton online
   softmax keeps up at small head_dim (64) but loses to FA at 128.
   Beating FA in our own kernel is a separate, large project.
2. **The avoided buffer is too small to register.** V2a's per-call
   materialized K/V is `total_k × num_kv_heads × head_dim × 4 bytes` —
   1-4 MB at our scales (0.5-7B at ~2k tokens). PyTorch's caching
   allocator reuses the same physical bytes across layers, so historical
   peak doesn't change. Even on Llama-405B at 32k tokens the buffer is
   ~250 MB — < 0.03% of the 820 GB working set.

V2b therefore stays in the codebase but is **off by default**:
`_FUSED_ATTN_DISABLED_FOR_BENCH = True`. Opt-in via the toggle for
A/B benchmarks or memory-pressured workloads. V2a is the recommended
attention path for compressed pools.

### Production-grade alternatives we'd reach for instead

The V2b deselection raises a fair question: is there a library that
does fused dequant + attention well enough to be worth integrating
instead of writing our own? Today's landscape:

| Library | Status | What it gives | Hardware required |
|---|---|---|---|
| **FlashAttention-3** ([github](https://github.com/Dao-AILab/flash-attention)) | Production | State-of-the-art tensor-core attention; FP8 KV | Hopper+ (H100/H200) for FP8 |
| **FlashInfer FP8 KV** ([github](https://github.com/flashinfer-ai/flashinfer)) | Production | Page-aware FA + FP8 KV powering vLLM/SGLang | Hopper+ |
| **FlashInfer NVFP4 KV** ([v0.6.10rc1](https://github.com/flashinfer-ai/flashinfer/releases), [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)) | **Production (2026-04)** | **4-bit** KV with native NVFP4 tensor-core dequant fused into attention | **Blackwell only** (B200/B300/RTX 50/DGX Spark) |
| **vLLM** with FP8/NVFP4 KV ([code](https://github.com/vllm-project/vllm)) | Production | Wraps FlashInfer's quantized-KV kernels end-to-end | Same as kernels |
| **TensorRT-LLM** | Production (NVIDIA) | INT8/FP8 KV with custom kernels | Hopper+ |
| **CUTLASS attention** ([github](https://github.com/NVIDIA/cutlass)) | Building blocks | Templates for FA-class kernels with custom epilogues | n/a (you write the kernel) |
| **`0xSero/turboquant`** | Research | Reference TurboQuant kernels (3 separate); paper-aligned | A100-targeted |

**The realistic production path for KV-quantized inference depends on
target hardware**:

| Target GPU | Native KV format | Library | Notes |
|---|---|---|---|
| Ampere (A10, A100) | bf16 only natively | n/a — custom kernels | What we run today; turbo3 + V2a is reasonable |
| Hopper (H100, H200) | **FP8** | FlashInfer FP8 KV / vLLM | ~50% memory vs bf16; production-grade |
| Blackwell (B200, B300, RTX 50) | **FP4 (NVFP4)** | **FlashInfer NVFP4 KV / vLLM** | ~75% memory vs bf16; **same 4-bit compression as turbo3 but native HW path** |

The 2026-04-30 FlashInfer NVFP4 release is the **right answer for
4-bit KV in production** as soon as you're on Blackwell: the kernel
team gets the dequant fused into attention via tensor-core FP4
support, which is exactly the architectural shape V2b reached for —
delivered with HW-vendor engineering quality.

**For mini-infer's purposes**, V2a + FA varlen on materialized K/V is
the right answer for the Ampere hardware our budget covers. V2b stays
as a kernel-engineering study — useful for showing the design space
of the problem, kept out of the default path because the production
answer is "use FlashInfer NVFP4 on Blackwell" rather than "write a
better Triton kernel on Ampere."

## Follow-ups (in order of value/effort)

- **~~V2b — Fuse attention into the dequant kernel.~~ Shipped and
  deselected (above).** No further V2b work planned.
- **FP8 / NVFP4 KV cache** (the production answer for KV-quant on
  modern hardware). Use FlashInfer FP8 KV on Hopper (H100/H200) or
  FlashInfer NVFP4 KV on Blackwell (B200+). Both are production-ready
  in vLLM as of mid-2026. NVFP4 specifically matches turbo3's 4-bit
  compression ratio with native tensor-core support — the
  architectural shape V2b tried to build, delivered by the kernel
  team. mini-infer would integrate by replacing the
  TurboQuant-specific code paths with a FlashInfer attention call
  against an FP8/FP4-quantized pool. Out of current scope (we're on
  A10), but the right next step for any production target.
- **V2a-for-turbo4.** Same kernel pattern, different codec body
  (per-channel `(low, scale)` instead of polar + Lloyd-Max + QJL). Wire
  if 7B turbo4 ever becomes a target; turbo3 is the recommended V3 mode
  so this is low priority.
- **Per-head rotation** (instead of per-layer). Slightly better
  decorrelation. ~1% accuracy at modest extra storage.
- **Calibration-based codebook tuning** for K/V distributions
  specific to a model. Loses the "training-free" property but should
  close the 7B parity gap.
- **More aggressive QJL** (multi-bit residual, fp4 or fp8 storage of
  residual offsets). Narrows the K-side accuracy at the cost of
  storage budget.
