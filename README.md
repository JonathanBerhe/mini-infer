# mini-infer

An open-source LLM inference engine built from scratch in Python and Triton, implementing the techniques that production engines (vLLM, SGLang, TensorRT-LLM) are known for: continuous batching, PagedAttention with a Triton decode kernel, and an OpenAI-compatible FastAPI server with SSE streaming.

## What's working today

* **Continuous batching scheduler** (`ContinuousScheduler`) on a dedicated engine thread with FIFO admission, per-request handles, and backpressure. One forward pass per step over all in-flight decoding requests.
* **Chunked prefill + packed-varlen forward**: long prompts advance one chunk per step alongside in-flight decoders, eliminating head-of-line blocking. Single `model.forward(...)` per step via FlashAttention's varlen API on CUDA, PyTorch reference elsewhere.
* **PagedAttention** with a fixed-size block pool, batch-aware `PagedKVCache`, and per-architecture monkey-patch registry (`Qwen2` shipping; the engine itself is model-agnostic via the HF `Cache` interface).
* **Prefix caching**: chained-hash, block-granular, refcounted LRU. Repeat or shared-prefix prompts skip prefill on the cached prefix; opt-in via `prefix_cache=True`. Verified token-for-token against the no-cache path.
* **Weight-only INT8 quantization (W8A16)**: symmetric per-output-channel scales applied at load time; opt-in via `quant="int8"`. Drops model-weight HBM by ~30% on Qwen2.5-0.5B with cosine-sim > 0.99 on logits and first-token greedy parity preserved.
* **Triton decode kernel** (single-request and batched variants) with online-softmax accumulation, validated against a PyTorch reference within cosine similarity > 0.99.
* **OpenAI-compatible HTTP API** (`/v1/completions`) with both non-streaming responses and SSE streaming.
* **Sampler** with greedy, temperature, top-k, top-p; pure-logic unit tests.
* **CI** running `ruff`, `mypy --strict`, and the full unit + golden test suite (CPU-only path) on every push.

Throughput on Qwen2.5-0.5B-Instruct, NVIDIA A10, bf16:

| Concurrency | Tokens/sec | Multiplier vs C=1 |
|---:|---:|---:|
| 1 | 45.7 | 1.00x |
| 2 | 79.5 | 1.74x |
| 4 | 127.7 | 2.79x |
| 8 | 187.5 | 4.10x |

Full report: [docs/benchmarks/2026-04-27-continuous-batching.md](docs/benchmarks/2026-04-27-continuous-batching.md).

Prefix caching on a 15.9k-token shared system prompt + 8 unique short user questions, A10, bf16:

| Concurrency | cache OFF | cache ON | Throughput speedup |
|---:|---:|---:|---:|
| 1 | 2.57 tok/s | 34.1 tok/s | **13.27x** |
| 4 | 2.87 tok/s | 74.3 tok/s | **25.88x** |
| 8 | 2.91 tok/s | 94.4 tok/s | **32.45x** |

Warm-cache TTFT: ~74ms vs ~11.7s (cold), a 158x reduction once the system prompt has been served once. Full report: [docs/benchmarks/2026-04-28-prefix-caching.md](docs/benchmarks/2026-04-28-prefix-caching.md).

Weight-only INT8 (W8A16) on Qwen2.5-0.5B, A10: model-weight HBM drops from 1142 MiB (fp16) to 794 MiB (**−30.5%**) with cosine similarity > 0.99 on logits and first-token greedy parity preserved. Throughput is ~neutral (no FLOP savings; a fused dequant kernel is the path to compute speedup). Full report: [docs/benchmarks/2026-04-28-int8-weight-quant.md](docs/benchmarks/2026-04-28-int8-weight-quant.md).

## Quickstart

```bash
# Install (Python 3.11+; uv as the package manager)
uv sync

# Run the unit tests (CPU only, no model download)
uv run pytest tests/unit/ -v

# Run the API server (downloads Qwen2.5-0.5B-Instruct on first start)
uv run python -m mini_infer.api.server

# In another shell, hit the API
curl -N http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":8,"stream":true}'
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`; override with `MINI_INFER_MODEL=...`.

Device selection is automatic: CUDA if available, else MPS on Apple Silicon, else CPU. The Triton paged kernel runs on CUDA; other devices fall back to a numerically-equivalent PyTorch reference.

## Architecture

```
HTTP request
   │
   ▼
FastAPI (api/server.py) ──► ContinuousScheduler.submit(Request)
                                    │
                              engine thread
                                    │
                ┌───────────────────┼─────────────────────┐
                ▼                   ▼                     ▼
          admit waiting       prefill (B=1)         batched decode
                                    │                     │
                                    └────► PagedKVCache ◄─┘
                                                │
                                                ▼
                                         BlockPool (shared)
                                                │
                                                ▼
                                Triton kernel  /  PyTorch reference
                                  (CUDA)           (CPU/MPS)
```

* **`engine/`** loads the model, owns prefill and decode, applies the per-architecture attention patch.
* **`scheduler/`** owns the engine thread, admission control, the running batch, and the shared `PagedKVCache`.
* **`cache/`** holds the block pool, the batch-aware cache, and the paged attention kernel + reference.
* **`api/`** is FastAPI with OpenAI-compatible endpoints and SSE streaming.

## Design decisions

Each non-trivial choice has an Architecture Decision Record under [docs/decisions/](docs/decisions/):

* [ADR-001](docs/decisions/ADR-001-uv-as-package-manager.md): `uv` as package manager
* [ADR-002](docs/decisions/ADR-002-starter-model.md): Qwen2.5-0.5B-Instruct as the starter model
* [ADR-003](docs/decisions/ADR-003-paged-kv-cache.md): block-based KV cache layout
* [ADR-004](docs/decisions/ADR-004-paged-attention-kernel.md): Triton paged attention kernel
* [ADR-005](docs/decisions/ADR-005-continuous-batching-integration.md): batch-aware cache and the continuous-batching forward pass
* [ADR-006](docs/decisions/ADR-006-chunked-prefill.md): chunked prefill (two-forward design)
* [ADR-007](docs/decisions/ADR-007-packed-forward-integration.md): packed-varlen forward, one `model.forward` per step
* [ADR-008](docs/decisions/ADR-008-paged-fa-varlen.md): paged FlashAttention varlen, kept as a tunable
* [ADR-009](docs/decisions/ADR-009-prefix-caching.md): prefix caching (chained-hash, block-granular, refcounted LRU)
* [ADR-010](docs/decisions/ADR-010-int8-weight-quant.md): weight-only INT8 quantization (W8A16)

## Tests

Three layers, each with a clear contract:

* **Unit** (`tests/unit/`): CPU-only, fast, run on every CI push. Covers the block pool, paged cache, paged attention reference, scheduler, sampler, tokenizer, API schemas.
* **Golden** (`tests/golden/`): CPU/fp32. Token-for-token comparison against Hugging Face's reference output for a fixed set of prompts. Catches numerical drift before it ships.
* **Modal smokes + benchmarks** (`scripts/`): real CUDA validation on Modal A10. Runs on demand only.

Markers: `requires_model` (skipped in CI; needs an HF model download) and `requires_cuda` (skipped on non-CUDA hardware).

```bash
uv run pytest tests/unit/ -q                       # CI-equivalent
uv run pytest tests/golden/ -q                     # token-for-token vs HF
uv run pytest tests/unit/ -m "not requires_model"  # skip model-dependent tests
```

## What's NOT in here

By design:

* No multimodal support (text-only).
* No fine-tuning or training pipeline (inference only).
* No reimplementation of FlashAttention or other heavily-optimized kernels; we use `torch.nn.functional.scaled_dot_product_attention` for prefill and our own Triton kernel for paged decode.

## License

(Pending.)
