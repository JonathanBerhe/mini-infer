# TurboQuant V2b (fully-fused decode attention) on A10 — V2b loses to V2a at 7B

Date: 2026-05-02
Hardware: NVIDIA A10 (Ampere, SM_86), bf16 model
Engine: mini-infer @ this slice (V2b fused dequant + online softmax kernel)
Script: `scripts/modal_packed_bench.py --config turbo_v2b`

V2b extends the V2a fused dequant kernel into a fully-fused
attention path: a single Triton program walks compressed K/V blocks
for one (request, q_head) pair, dequants tiles in registers using
the V2a codec, and runs FA-2 online softmax — no materialized bf16
K/V buffer in HBM. The original ADR-013 motivation: capture a peak
attention-memory reduction beyond V2a's throughput win, supporting
"fit longer contexts on the same GPU".

The bench tests that motivation directly. **It doesn't pan out.**

## Workload

- Real long prompt (`scripts/modal_packed_bench.py:_TECHNICAL_PASSAGE`):
  ~2000 tokens of varied technical prose.
- `max_tokens=128`, single request (`C=1`). Decode-heavy regime where
  V2b's per-step savings (no transient materialized buffer) would be
  most visible.
- `kv_quant="turbo3"` on Qwen2.5-0.5B and Qwen2.5-7B.

## Results

### Qwen2.5-0.5B

```
                                       t/s     peak HBM
V2a (materialized + FA varlen)        11.06    1084.5 MiB
V2b (fused attention)                 11.35    1084.5 MiB
V2b vs V2a: throughput 1.03x, peak HBM identical
```

V2b is 3% faster, zero measurable memory delta.

### Qwen2.5-7B

```
                                       t/s     peak HBM
V2a (materialized + FA varlen)         8.18   14885.4 MiB
V2b (fused attention)                  7.16   14885.4 MiB
V2b vs V2a: throughput 0.88x, peak HBM identical
```

**V2b is 12% slower at 7B.** Memory still identical.

## Reading the data

### Throughput: V2b loses to FlashAttention at scale

V2a's attention math runs through `flash_attn_varlen_func` — FlashAttention,
the result of years of NVIDIA + FA-team kernel-engineering on tensor cores.
V2b reimplements the same math (online softmax, V accumulation) inside a
custom Triton kernel that wraps it around the dequant logic. At small
head_dim (0.5B uses 64) my Triton softmax loop keeps up with FA's
hand-tuned version. At larger head_dim (7B uses 128):

- The rotation matrix in shared memory grows to **32 KB per program**
  on Qwen2.5-7B (`head_dim^2 * 2 bytes`), occupying ⅓ of A10's 96 KB
  SMEM per SM. Occupancy drops to one CTA per SM.
- The K/V tiles after dequant are larger (`block_size * head_dim` =
  16 × 128 = 2048 fp32 each = 8 KB), pressuring registers.
- The custom online softmax loop isn't tensor-core-friendly the way
  FA's is.

Net effect: V2b's kernel processes each K-block slower than the
equivalent (FA-on-materialized-K/V) does, and the win from skipping the
HBM round-trip doesn't make up the gap.

### Memory: the buffer V2a allocates is too small to register on the meter

V2a's per-call materialized buffer is `total_k × num_kv_heads × head_dim × 4 bytes`
(K + V combined, bf16):

| Model | head_dim × num_kv_heads | At 2128 tokens | As % of model |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 64 × 2 = 128 | 1.1 MiB | 0.10% of 1 GiB weights |
| Qwen2.5-7B  | 128 × 4 = 512 | 4.4 MiB | 0.03% of 14 GiB weights |
| Llama-70B   | 128 × 8 = 1024 | 8.7 MiB | 0.006% of 140 GiB weights |
| Llama-405B  | 128 × 16 = 2048 | 17.4 MiB | 0.002% of 810 GiB weights |

The buffer is allocated and freed per-attention-call. PyTorch's caching
allocator reuses the same physical bytes for each layer's call, so the
historical peak observed by `torch.cuda.max_memory_allocated()` only
records the largest single allocation. That allocation is the same
size in V2a and V2b's run because the same prefill path runs in both:
chunked prefill always uses V2a (multi-token Q falls outside V2b's
decode-only contract), and prefill's last chunk allocates a similarly-
sized materialized buffer.

So the meter shows zero delta. The actual savings exists, but only
during decode-time attention calls — and only on the order of MB on
even very large models. **The "fit longer contexts" headline from the
ADR-013 V2 follow-up is overstated at any model size we can reasonably
serve on a single GPU.**

### What V2b would need to be useful

To deliver a measurable win:

1. **Beat FlashAttention on the attention math itself.** V2b would need
   to be a competitive FA-3 reimplementation with TurboQuant's dequant
   folded in. This is a serious kernel-engineering project (FA-3 took
   the original team months on H100). A reasonable cap would be to
   match FA's perf with the dequant mostly hidden by load-store
   overlap, which is plausible but not what V2b ships today.
2. **Be the compute-bound path.** At very long contexts (e.g. 128k+),
   the materialized buffer grows enough that allocation pressure
   matters. We don't run that regime; if we did, the buffer would still
   be small relative to the cache itself.

Neither matches the project's current scope.

## Decision: V2a is the default; V2b is opt-in

`_FUSED_ATTN_DISABLED_FOR_BENCH` defaults to True (V2b OFF). The V2b
kernel stays in the codebase as opt-in via that toggle, and the unit
tests verify it remains correct (cosine sim > 0.999 vs V2a on Qwen
0.5B and 7B GQA shapes — the parity numbers from this slice's
`turbo_parity` Modal run).

V2a remains the recommended attention path for compressed pools:
- `materialize_packed_kv` runs the V2a fused dequant kernel
- `_packed_attention_materialized_flash` calls FA varlen on the
  materialized bf16 K/V

Combined throughput: 5.34 t/s at C=1 on Qwen2.5-0.5B with the long
prompt (2026-05-02 turbo bench), 8.18 t/s on Qwen2.5-7B (this run).

## Caveats

- A10 only. H100 has 228 KB SMEM per SM (vs A10's 96 KB), so V2b's
  rotation tile would fit more comfortably and occupancy would
  improve. But V2a uses FA which is already well-tuned on H100, so
  the gap likely persists. We didn't run on H100 — the 7B A10 result
  was decisive enough to flip the default.
- Decode-only scope. V2b never claimed to handle chunked prefill.
- The cosine-sim parity bar (> 0.999 on V2a/V2b attention output) is
  unchanged from V2a's contract; this isn't a correctness failure,
  it's a performance deselection.

## Reproduce

```
uv run modal run scripts/modal_packed_bench.py --config turbo_v2b
uv run modal run scripts/modal_packed_bench.py --config turbo_v2b \
    --model "Qwen/Qwen2.5-7B-Instruct"
```

## Pointers

- ADR: [ADR-013](../decisions/ADR-013-turboquant-kv.md).
- V2a baseline: [2026-05-02-turboquant-v2a.md](2026-05-02-turboquant-v2a.md).
- V2b kernel: [turbo_kernel.py](../../src/mini_infer/cache/turbo_kernel.py)
  (`_turbo_fused_attn_decode_kernel` and `fused_turbo_attention_decode`).
- V2b parity tests: [test_turbo_kernel.py](../../tests/unit/test_turbo_kernel.py)
  (`test_fused_attn_decode_matches_materialized_*`).
- Production-grade alternatives for KV-quantized fused attention:
  see ADR-013 V2b update.
