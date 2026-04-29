# ADR-013: TurboQuant KV cache (V1 — rotation + 4-bit, materialize-on-read)

Date: 2026-04-29
Status: Accepted (with explicit V2/V3 follow-ups based on the empirical limits surfaced)

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

## Follow-ups (in order of value/effort)

- **V2 — Fused dequant-attention Triton kernel**. Reads compressed
  K/V tiles, dequants in shared mem, runs the attention math without
  materializing to bf16. The single highest-value follow-up: turns
  the unusable 0.06–0.18x throughput into something workable, AND
  unlocks "fit longer contexts on the same GPU" (peak attention
  memory drops because we don't materialize). ~1-2 Modal iterations
  of kernel debugging.
- **V3a — PolarQuant**. Replaces the per-channel `(low, scale)` pair
  with a single per-channel magnitude (polar coordinates are
  zero-centered), reclaiming ~50% of the scale overhead. Pushes
  realized compression from 2.7x toward the paper's 5x. Implementation:
  Cartesian → polar transform on write, polar → Cartesian on read.
- **V3b — QJL residual sign bit**. ~1 extra bit per dimension of
  error correction. Reduces per-block quantization error and is what
  closes the 7B accuracy gap.
- **V3c — Lloyd-Max codebook**. Optimal scalar quantizer for the
  post-rotation distribution (which is roughly Gaussian by the
  Johnson-Lindenstrauss argument). Replaces uniform quant.
- **V3d — Asymmetric 3-bit K + 2-bit (or 4-bit) V**. Reference repo
  reports values are the bottleneck and 4-bit V suffices; 2-bit V
  is the most aggressive option. Reaches the paper's 3-bit headline.
- **Per-head rotation** (instead of per-layer). Slightly better
  decorrelation at the cost of more rotation storage.
- **Calibration-based scales** for cases where training-free isn't
  required.
