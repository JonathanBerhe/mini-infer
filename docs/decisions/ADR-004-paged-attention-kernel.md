# ADR-004: Triton paged attention kernel for the decode path

Date: 2026-04-25
Status: Accepted

## Context

Slice 2.1 (ADR-003) put a block-based KV cache in place but kept materialization in `PagedKVCache.update()` so HF's stock attention path could read the K/V history as a contiguous tensor. The materialization is correct but defeats the point of paged storage: it allocates a fresh contiguous tensor every layer per decode step. The plumbing only pays off if some compute layer reads K/V directly out of blocks. That layer is the attention compute itself, and it doesn't exist as part of HF's standard SDPA path — it has to be ours.

This slice writes a Triton kernel that does paged decode-step attention without materialization, validates it numerically, and integrates it via per-architecture monkey-patches.

## Decision

1. **Decode-only kernel, prefill stays on materialization.** Prefill is one forward per request and not the hot path; the algorithmic complexity of a paged prefill kernel (causal masking across query positions and blocks) is not worth the engineering vs. payoff at our model scale.
2. **Triton kernel, hand-rolled.** Online-softmax decode kernel adapted for block storage. Compile-time constants for `HEAD_DIM` and `BLOCK_SIZE`; runtime kernel arg for `seq_len`. ~80 lines of Triton.
3. **PyTorch reference path always available.** `paged_attention_decode_torch` mirrors the kernel's contract using `repeat_interleave` for GQA and standard tensor ops. It runs anywhere (MPS / CPU / CUDA) and is the dispatcher's fallback when the kernel isn't usable.
4. **Per-architecture integration via monkey-patch + registry.** `attention_patches/__init__.py` exposes a `REGISTRY` mapping a substring of the model class name to its patch function. `engine/attention_patch.py` is a thin dispatcher that picks the right patch (or logs and skips, falling back to materialization). Slice 2.2 ships the Qwen2 patch in `attention_patches/qwen2.py`; future model families add their own files + registry entries.
5. **Single source of truth for "device supports the kernel".** `cache/paged_attention.supports_paged_kernel(device)` is the only place we check whether to use the kernel path. Replaces the `device == "cuda"` strings that had crept into model_runner, the patch, the smoke script, and tests.
6. **A/B switch: `use_paged_kernel: bool = True`** on `ModelRunner.from_pretrained`. Default True (ship the kernel); benchmarks pass False to compare against the materialization path on the same hardware.

## Alternatives considered

- **Wrap vLLM's kernel as a dependency.** Would have shipped faster and with higher confidence, but doesn't demonstrate the kernel-engineering chops the project is supposed to show, and adds a heavy dep.
- **Generic kernel that handles arbitrary `head_dim`/`block_size` at runtime.** Reasonable but pushes complexity into the kernel. Compile-time constants are simpler and Triton's JIT autotune means new shapes are cheap to compile when they show up. We can generalize later.
- **Patch HF's `Cache` rather than the attention layer.** Too late in the call chain; by the time `Cache.update()` returns, HF's attention already wants a contiguous K/V tensor. The patch must live at the attention layer to bypass the SDPA call entirely.
- **Skip the kernel and accept materialization as good-enough.** Phase 2 is meant to demonstrate production-realistic patterns. Materialization undermines the whole point of the block manager. Worth doing the kernel even if the speedup at our model scale is modest (it is — see numbers below).
- **Subclass HF model classes instead of monkey-patching.** Cleaner architecturally but invasive (we'd own a copy of the Qwen2 model code) and more work to keep in sync with transformers updates. Monkey-patching is the standard pattern (vLLM and SGLang both do variants of this).

## Consequences

- **Positive**:
  - End-to-end paged attention now actually pages — no contiguous materialization on the decode hot path on CUDA.
  - The architecture-registry pattern means adding Llama / Mistral / Gemma support is ~30 lines + an entry, not a refactor.
  - The PyTorch reference path keeps the engine fully runnable on M1 / CPU; the kernel is a CUDA optimization, not a CUDA requirement.
  - Slice 2.3 (continuous batching) inherits a working paged path; nothing in Phase 2 has to revisit this.
- **Negative**:
  - Speedup at our model scale (0.5B, A10) is modest: 3–8% median across seq_lens 16–1024. The reason is that decode latency is dominated by linear-projection and MLP compute, not attention. Documented in `docs/benchmarks/2026-04-25-paged-decode.md`.
  - Two paths to maintain (kernel + reference). Tests cover both.
  - The patch is per-architecture and tied to transformers' internal forward signature. A future transformers major version could break it; mitigation is the smoke + golden tests catching regressions.
- **Reversibility**:
  - Setting `use_paged_kernel=False` on `from_pretrained` cleanly disables the patch and falls back to materialization with no other changes needed. The Slice 2.1 fallback stays as the always-correct path.
  - Removing the kernel module entirely would break only the patched fast path; the engine would still work via materialization.

## Validation summary

- **Smoke A (Modal A10):** patched Qwen2.5-0.5B generates `" Paris. It is the largest city in"` — token-for-token identical to the CPU/fp32 reference.
- **Pool accounting:** `num_free_blocks` returns to 1024 after generation; `try/finally` in scheduler verified.
- **Local tests:** 48 passed + 1 `requires_cuda` skipped on M1; PyTorch reference matches SDPA-on-materialized-K/V across block-aligned, partial-last-block, GQA-grouped (14/2), and seq_len=1 cases.
- **Benchmark (Modal A10):** see `docs/benchmarks/2026-04-25-paged-decode.md`.

## Pointers

- Kernel + reference: `src/mini_infer/cache/paged_attention.py`
- Cache append + block-table accessor: `src/mini_infer/cache/paged_kv_cache.py`
- Dispatcher + registry: `src/mini_infer/engine/attention_patch.py`, `attention_patches/__init__.py`
- Qwen2 integration: `src/mini_infer/engine/attention_patches/qwen2.py`
- Tests: `tests/unit/test_paged_attention.py` (+ shared helpers in `tests/unit/_kernel_test_utils.py`)
- Benchmark harness: `tests/benchmarks/bench_decode_attention.py`
- Modal entrypoints: `scripts/modal_paged_smoke.py`, `scripts/modal_bench_paged.py`
