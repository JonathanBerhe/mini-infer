# Plan: integrate FlashInfer as the production attention backend

## Context

mini-infer's CUDA attention path today uses `flash_attn_varlen_func`
from the [`flash-attn`](https://github.com/Dao-AILab/flash-attention)
package, both for the materialized FA path (`_packed_attention_materialized_flash`)
and the paged FA path (`_packed_attention_paged_flash`). Both work, but
they're a single backend with no FP8/NVFP4 KV support.

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the kernel
library that powers vLLM and SGLang. Its appeal for mini-infer:

- **FP8 KV cache (Hopper, H100/H200)**: ~50% KV memory savings vs bf16
  with native FP8 tensor-core dequant fused into attention. Already
  in production via vLLM.
- **NVFP4 KV cache (Blackwell, B200/B300/RTX 50)**: ~75% KV memory
  savings — same compression as our `turbo3` but with a hand-tuned
  vendor kernel. Shipped in FlashInfer v0.6.10rc1 on 2026-04-30.
- **Paged FA equivalent**: drop-in replacement for `flash_attn_varlen_func`
  with the plan/run lifecycle and CSR-style page-index layout.
- **No flash-attn dependency**: FlashInfer is standalone — replaces
  rather than wraps flash-attn.

ADR-013 V2b deselected our custom V2b decode kernel after it lost to
FlashAttention by 12% at 7B. The honest production answer for
KV-quantized attention is **use the vendor library**. This plan adds
that path.

## Scope — three stages, each independently shippable

| Stage | What ships | Hardware to validate | Why |
|---|---|---|---|
| **1: FlashInfer as bf16 backend** | New `attention_backend="flashinfer"` selector. Same numerics as today. | A10 (~$0.20) | Validates the integration end-to-end with the bf16 path we already trust. Smallest change with the most plumbing. |
| **2: FP8 KV mode** | `kv_quant="fp8"` backed by FlashInfer's FP8 paged attention. ~50% KV savings. | **H100** (~$0.50/run) | The first real production win. Requires Hopper. |
| **3 (optional): NVFP4 KV mode** | `kv_quant="nvfp4"` backed by FlashInfer's NVFP4 path. ~75% KV savings. | **B200** (~$2/run) | Same compression as turbo3 with vendor kernels. Skip if Blackwell access is impractical. |

Each stage is independently revertible; if Stage 2 stalls the engine
keeps Stage 1's value.

## Stage 1: FlashInfer as a bf16 attention backend

### Files to touch

- **NEW** `src/mini_infer/cache/flashinfer_backend.py` — wrapper module
  that owns the FlashInfer `BatchPrefillWithPagedKVCacheWrapper` and
  `BatchDecodeWithPagedKVCacheWrapper` instances, manages the workspace
  buffer, and exposes one `flashinfer_attention_forward()` function
  with the same contract as `_packed_attention_paged_flash`.
- **EDIT** `src/mini_infer/cache/packed_attention.py:packed_attention_forward`
  — add a third dispatch arm gated on `cache._pool.attention_backend == "flashinfer"`,
  before the existing flash-attn paths.
- **EDIT** `src/mini_infer/cache/block_pool.py:BlockPool.__init__` — add
  `attention_backend: str = "flash_attn"` parameter (validated against
  `("flash_attn", "flashinfer")`). Stored on the pool; the dispatcher
  reads it.
- **EDIT** `src/mini_infer/engine/model_runner.py:from_pretrained` —
  forward an `attention_backend` kwarg through to `BlockPool`.
- **EDIT** `pyproject.toml` — add `flashinfer-python` (and optionally
  `flashinfer-cubin` for pre-compiled kernels) to the `[cuda]` extra,
  alongside `flash-attn`.
- **NEW** `tests/unit/test_flashinfer_backend.py` — CPU mechanics tests
  (predicate returns False, dispatcher falls through) + CUDA parity
  tests (`@pytest.mark.requires_cuda`) comparing FlashInfer output to
  flash-attn output at cosine sim > 0.999.
- **EDIT** `scripts/modal_packed_bench.py` — add `--config flashinfer`
  for the same workload as `--config turbo` but with the new backend,
  for A/B comparison.

### KV layout mapping

FlashInfer accepts paged K/V in `(num_pages, page_size, num_kv_heads, head_dim)`
NHD layout — **the same layout mini-infer's `BlockPool._storage` uses
for the bf16 path**. Per-layer slice via
`pool._storage[layer_idx, 0]` and `[layer_idx, 1]` for K and V; pass
both as `k_cache` and `v_cache` to the wrapper.

CSR-style page indexing:
- `kv_page_indices`: int32, `(total_pages_across_batch,)` flat
  concatenation of each request's block IDs. Build from
  `cache._block_ids` once per layer.
- `kv_page_indptr`: int32, `(B+1,)` cumulative count of
  pages-per-request.
- `kv_last_page_len`: int32, `(B,)` how many tokens are valid in each
  request's last page (`seq_len % page_size`, or `page_size` if exact).

These three tensors map 1:1 to information already available in
`cache._block_ids` and `cache._num_tokens`. Build a small helper on
`PagedKVCache` (`flashinfer_paged_index(device)`) that returns the
triple and reuses across layers in a step.

### Lifecycle (`plan` then `run`)

FlashInfer wrappers split work into:
- `plan()`: takes shapes + index tensors, computes layout/scheduling
  metadata, allocates per-call buffers off the persistent workspace.
  Cheap; called once per step (not per layer) since shapes are
  consistent across layers.
- `run(q, k_cache, v_cache, ...)`: executes attention. Called once
  per layer.

mini-infer's per-step flow becomes:
1. After all `append_kv_packed` calls for the step, build the index
   triple from cache state.
2. Call `prefill_wrapper.plan(...)` once per step (or
   `decode_wrapper.plan(...)` for pure decode batches; pick by checking
   `q.shape[0] == B`).
3. Per layer: `output = wrapper.run(q_layer, k_cache_layer, v_cache_layer)`.

The engine's existing per-layer loop in
`engine/attention_patches/qwen2.py` already calls
`packed_attention_forward(q, cache, layer_idx, cu_seqlens_q)` once per
layer, so the per-layer `run()` slots in directly. The per-step
`plan()` call needs a small new hook — either at the top of the model
forward or lazily on first attention call of the step.

### Workspace allocation

FlashInfer needs a persistent `uint8` workspace buffer (recommended
128 MiB). Allocate once on the `BlockPool` at construction time when
`attention_backend == "flashinfer"`. Same buffer reused for prefill
and decode wrappers (separate wrappers share the buffer).

### Tests

CPU (`tests/unit/test_flashinfer_backend.py`):
- Predicate `supports_flashinfer_backend(device)` returns False on CPU.
- BlockPool with `attention_backend="flashinfer"` allocates the
  workspace; with `"flash_attn"` doesn't.

CUDA (`@pytest.mark.requires_cuda`):
- Build a populated bf16 BlockPool, run a forward via FlashInfer and
  via flash-attn, assert cosine sim > 0.999 on attention output.
- Cover decode-only (q_len=1 per request), prefill (q_len > 1), mixed
  batches, GQA shapes (Qwen2.5-0.5B `14×2`, Qwen2.5-7B `28×4`).

### Modal validation (~$0.20)

Run `scripts/modal_packed_bench.py --config turbo` first with
`--attention-backend flash_attn` (current default), then with
`--attention-backend flashinfer`. Same workload, different backend.
Acceptance: cosine sim > 0.999 on greedy logits between the two; tok/s
within ±5% (FlashInfer is generally on-par with FA on Ampere; the win
shows up in Stages 2-3).

## Stage 2: FP8 KV mode

### What changes

- **NEW** `kv_quant="fp8"` accepted by `BlockPool`. Storage layout: a
  single fp8 tensor of shape `(num_layers, 2, num_blocks, page_size,
  num_kv_heads, head_dim)` dtype `torch.float8_e4m3fn`, plus per-head
  scales `(num_layers, 2, num_kv_heads)` bf16 (FlashInfer's per-head
  scale convention).
- **NEW** Append-side quantization: when `kv_quant="fp8"`, `append_kv`
  computes per-head abs-max over the new tokens, divides by 448
  (`fp8_e4m3` representable max), stores the int8 bytes (`fp8_e4m3` bit
  pattern) plus updated scales. Dequant on the FlashInfer side is
  fused into attention; mini-infer does no dequant.
- **EDIT** `packed_attention_forward` dispatch — `kv_quant="fp8"`
  forces `attention_backend="flashinfer"` (FlashInfer is the only path
  that handles FP8 KV; flash-attn doesn't).
- **EDIT** `materialize_packed_kv` — for `kv_quant="fp8"`, dequantize
  per-block to bf16 in the CPU/MPS fallback path. Keep the existing
  Python fallback for testability.

### Validation strategy

Modal A10 doesn't support FP8 in tensor cores. FP8 must be tested on
**H100 or H200**. Allocate ~$0.50 budget for an H100 run.

- Parity vs bf16 baseline: cosine sim on first-token logits > 0.99 on
  Qwen2.5-7B-Instruct `kv_quant="fp8"` vs bf16 (FP8 is lossy but the
  loss is small).
- Memory: KV pool size measurement. Expected savings: 50% vs bf16.
- Throughput: same workload as the turbo bench. FP8 KV should match or
  beat bf16 (less memory bandwidth per attention call).

### Acceptance

- Greedy parity: first-token argmax matches bf16 on `kv_quant="fp8"`.
- Storage: 50% reduction.
- Throughput: ≥ bf16 on Qwen2.5-7B.

## Stage 3 (optional): NVFP4 KV mode

Same shape as Stage 2 but using FlashInfer's NVFP4 path
(`nvfp4_kv_quantize` for write, attention reads NVFP4 directly). Same
50% → 75% storage savings, but **Blackwell-only** (B200, B300, RTX 50).

Modal Blackwell access is more expensive (~$3-5/run). Skip if the
project's hardware budget can't accommodate this. The plan terminates
at Stage 2 unless a Blackwell run is approved.

## Cross-cutting concerns

### Dependency: `flashinfer-python`

Install via `pyproject.toml`'s `[cuda]` extra:

```toml
cuda = [
    "flash-attn>=2.8",          # kept for the existing path
    "flashinfer-python>=0.6.10",  # NEW
]
```

FlashInfer ships pre-built wheels for SM 7.5+. Optional
`flashinfer-cubin` adds pre-compiled kernels (avoids JIT on first call).

### Backend selector default

Default `attention_backend="flash_attn"` to preserve existing
behavior. Engine users opt into FlashInfer via
`ModelRunner.from_pretrained(..., attention_backend="flashinfer")`.

When `kv_quant in ("fp8", "nvfp4")`, force `attention_backend="flashinfer"`
(only path that handles those modes); raise if user requests an
incompatible combo.

### Fallback & A/B toggles

Mirror the existing `_FUSED_DISABLED_FOR_BENCH` pattern:
- `_FLASHINFER_DISABLED_FOR_BENCH = False` — set True to force the
  flash-attn path even when FlashInfer is selected. Lets the bench A/B
  the two on the same model load.

### Risks

1. **FlashInfer's `plan()` overhead per step**: cheap but non-zero.
   Profile to ensure it's not a per-token-latency regression on small
   batches.
2. **JIT compile latency on first call**: mitigate by installing
   `flashinfer-cubin` in the Modal image. Falls back to JIT otherwise
   (~10s on first call, then cached).
3. **FP8 numerical drift across deep models**: 7B is the test bar; if
   logit cosine sim drops below 0.99, escalate (per-token scales,
   different scale strategy, etc.).

### Per-stage shipping

- **Stage 1 commit**: `flashinfer_backend.py` + dispatcher edit + tests
  + bench config + ADR-014 amendment + `pyproject.toml` dep add.
- **Stage 2 commit**: `kv_quant="fp8"` storage + append-side quant +
  dispatcher gate + tests + H100 bench doc + ADR-014 amendment.
- **Stage 3 commit (if pursued)**: `kv_quant="nvfp4"` + Blackwell bench
  + ADR-014 amendment.

## Critical files

- `src/mini_infer/cache/flashinfer_backend.py` (NEW)
- `src/mini_infer/cache/packed_attention.py` (dispatcher + workspace
  initialization at engine startup)
- `src/mini_infer/cache/block_pool.py` (`attention_backend` param,
  workspace alloc, FP8 storage in Stage 2)
- `src/mini_infer/cache/paged_kv_cache.py` (`flashinfer_paged_index`
  helper for CSR index triple)
- `src/mini_infer/engine/model_runner.py` (forward `attention_backend`
  through `from_pretrained`)
- `src/mini_infer/engine/attention_patches/qwen2.py` (per-step `plan()`
  hook, if not lazy)
- `tests/unit/test_flashinfer_backend.py` (NEW, both stages)
- `scripts/modal_packed_bench.py` (new `flashinfer` config, FP8 bench)
- `pyproject.toml` (`flashinfer-python` dep)
- `docs/decisions/ADR-014-flashinfer-backend.md` (NEW, captures the
  decision and per-stage status)

## Verification

```bash
# Local (M1)
uv run ruff format .
uv run ruff check .
uv run mypy src/
uv run pytest tests/unit/ -v        # Stage 1 + 2 CPU mechanics tests
uv run pytest tests/golden/ -v      # No regression

# Modal A10 (Stage 1)
modal run scripts/modal_packed_bench.py --config flashinfer
# Compare logits cosine sim and tok/s vs --config turbo

# Modal H100 (Stage 2)
MINI_INFER_BENCH_GPU=H100 modal run scripts/modal_packed_bench.py --config flashinfer_fp8
# Compare KV storage, parity, throughput vs bf16 + flash-attn baseline
```

## What this plan deliberately does NOT cover

- Replacing speculative decoding's draft-model attention path.
  FlashInfer doesn't expose anything special for spec-dec; the
  draft-model uses the same `BatchDecodeWithPagedKVCacheWrapper` and
  the spec-dec orchestration in `scheduler/` is unchanged.
- TurboQuant (`kv_quant in {"turbo3", "turbo4"}`) integration with
  FlashInfer. The TurboQuant codec doesn't map onto FlashInfer's
  primitives — those modes stay on the V2a custom kernel path.
- Replacing the radix prefix cache. mini-infer's `PrefixCache` is
  separate from FlashInfer's KV-cache layout and stays as-is.
