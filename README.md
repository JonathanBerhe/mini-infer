# mini-infer

An open-source LLM inference engine built from scratch in Python and Triton, implementing the techniques that production engines (vLLM, SGLang, TensorRT-LLM) are known for: continuous batching, PagedAttention with a Triton decode kernel, a multi-model registry of owned `nn.Module` implementations, and an OpenAI-compatible FastAPI server with SSE streaming. Six model families register today: Qwen2, Qwen3, Llama, Mistral, Gemma 3, Mixtral.

## What's working today

* **Continuous batching scheduler** (`ContinuousScheduler`) on a dedicated engine thread with FIFO admission, per-request handles, and backpressure. One forward pass per step over all in-flight decoding requests.
* **Chunked prefill + packed-varlen forward**: long prompts advance one chunk per step alongside in-flight decoders, eliminating head-of-line blocking. Single `model.forward(...)` per step via FlashAttention's varlen API on CUDA, PyTorch reference elsewhere.
* **PagedAttention** with a fixed-size block pool and a batch-aware `PagedKVCache`. Supports per-layer heterogeneous KV shape — different `(num_kv_heads, head_dim)` per layer (Stage C1 of the multi-model plan), the foundation for Gemma 4 31B's mixed sliding/global head dims, DeepSeek MLA's latent KV, and DeepSeek-V4's CSA + HCA hybrid attention.
* **Multi-model framework**: `ModelRegistry` looks up an HF `config.architectures[0]` string and dispatches to an owned `nn.Module` (Llama-style or family-specific). Six families register today: `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `LlamaForCausalLM` (covers Llama 2/3/4 + SmolLM2 + TinyLlama + the Llama-shape Nemotron variants), `MistralForCausalLM`, `Gemma3ForCausalLM`, `MixtralForCausalLM`. Adding a new model is one ~80-line file composing the shared block library — RMSNorm + RoPE + GQA + SwiGLU plus the family-specific extensions (sliding-window attention, dual RoPE, partial RoPE, per-head Q/K norm, sandwich norm, top-k MoE FFN). HF safetensors weights load identity-rename via `model.load_state_dict(...)`. No HF runtime monkey-patching.
* **Prefix caching**: chained-hash, block-granular, refcounted LRU. Repeat or shared-prefix prompts skip prefill on the cached prefix; opt-in via `prefix_cache=True`. Verified token-for-token against the no-cache path.
* **Weight-only INT8 quantization (W8A16)**: symmetric per-output-channel scales applied at load time; opt-in via `quant="int8"`. Drops model-weight HBM by ~30% on Qwen2.5-0.5B with cosine-sim > 0.99 on logits and first-token greedy parity preserved. Forward dispatches to a fused Triton W8A16 GEMM kernel on CUDA — keeps weights in INT8 in HBM and dequants tile-by-tile in registers, skipping the bf16-weight HBM round-trip the naive path pays.
* **Speculative decoding** (vanilla two-model, greedy V1): small draft model proposes K tokens, large target verifies them in one forward, accept-reject emits target's argmax sequence. `PagedKVCache.truncate_to` rolls back on rejections. 1.14x decode throughput on Qwen2.5-7B target + 0.5B draft on A10 at bf16; the regime is constrained by the modest target/draft size ratio (the same implementation scales to the published 1.5–2x range at 70B+ on Hopper).
* **TurboQuant KV cache** (V1 + V3 full + fused dequant kernel): rotation + per-block 4-bit (V1, `kv_quant="turbo4"`) or rotation + polar transform + Lloyd-Max codebook + QJL residual sign + asymmetric K(3-bit + QJL)/V(4-bit) (V3, `kv_quant="turbo3"`). V3 saves ~74% KV memory (vs V1's 62%) and produces less-degenerate output at 7B; V1 still preserves full argmax parity at 0.5B. A Triton kernel collapses turbo3's per-block dequant into a single launch per layer, lifting decode throughput from 0.04–0.18x of bf16 (Python loop) to 0.31–0.41x on Qwen2.5-0.5B + A10. Per-storage-class throughput vs the same-class Python loop is 6.2–6.9x.
* **FlashInfer attention backend** with three KV storage modes selectable via `attention_backend="flashinfer"`:
   * **bf16** (default with the FlashInfer backend): same paged storage as flash-attn, routed through FlashInfer's tensor-core wrapper. Token-for-token parity with the flash-attn path.
   * **FP8** (`kv_quant="fp8"`): FP8 e4m3fn paged storage with FlashInfer fusing dequant into the attention kernel. **50% KV memory savings vs bf16** on Hopper (H100/H200); logit cosine sim 0.999985, 0.93x throughput on Qwen2.5-0.5B at 2k context. The production-grade 8-bit KV path on Hopper.
   * **NVFP4** (`kv_quant="nvfp4"`, Blackwell only): FP4-packed paged storage + per-16-element FP8 block scales, dequant fused into the kernel via `kv_cache_sf`. **71.9% KV memory savings vs bf16** on B200; 0.91x throughput on Qwen2.5-7B. Token-level accuracy is degraded under greedy decode (per-(layer,side) global scale leaves bulk K/V values quantized toward FP4-zero in the presence of outliers); production NVFP4 deployments need outlier-aware preprocessing we haven't built. Integration infrastructure is correct (kernel path validated within FlashInfer's own 1e-1 tolerance); treat this as working FP4 plumbing pending a calibration layer.
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

Fused W8A16 Triton kernel on Qwen2.5-7B, A10, bf16: same kernel that loses 0.74–0.99x to cuBLAS at 0.5B wins **2.74x at decode (C=1)** and 1.47–1.65x at higher concurrency on 7B. The regime flip is exactly where the math predicts: at 7B the bf16-weight HBM round-trip in the naive int8 path becomes the dominant cost per forward, and that's what the fused kernel skips. Full report: [docs/benchmarks/2026-04-29-fused-int8-kernel.md](docs/benchmarks/2026-04-29-fused-int8-kernel.md).

Speculative decoding on Qwen2.5-7B target + Qwen2.5-0.5B draft, bf16, K=4: aggregate **1.14x on A10**, **1.00x on H100** vs target-alone greedy on a 3-prompt workload. Mean acceptance 2.3–3.4 / K=4 (58–85%). Below the published 1.5–2x range; the H100 result is the more interesting finding — faster baseline decode on Hopper *narrows* the win at this target size rather than widening it (1.29x on the longest prompt, 0.83x on the shorter ones). Full report: [docs/benchmarks/2026-04-29-speculative-decoding.md](docs/benchmarks/2026-04-29-speculative-decoding.md).

FlashInfer FP8 KV on Qwen2.5-0.5B + H100 (2k-token prefill, single request): KV pool 192 MiB → 96 MiB (**−50.0%**), logit cos sim 0.999985 vs bf16, 0.93x throughput. Full report: [docs/benchmarks/2026-05-02-flashinfer-fp8-kv.md](docs/benchmarks/2026-05-02-flashinfer-fp8-kv.md).

FlashInfer NVFP4 KV on Qwen2.5-7B + B200 (2k-token prefill, single request): KV pool 1792 MiB → 504 MiB (**−71.9%**), 0.91x throughput vs bf16. Token-level accuracy degraded under greedy decode (FP4 precision tradeoff; production deployments need outlier-aware calibration). Full report: [docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md](docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md).

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

## How to use mini-infer

### Programmatic API

The engine entry point is `mini_infer.engine.model_runner.ModelRunner.from_pretrained`. It accepts knobs for all the optional features:

```python
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    # Default: bf16 KV / flash-attn varlen on CUDA, PyTorch reference elsewhere.
    quant=None,                  # "int8" enables W8A16 weight-only quantization
    kv_quant=None,               # "turbo4" | "turbo3" | "fp8" | "nvfp4"
    attention_backend="flash_attn",  # "flashinfer" required for fp8/nvfp4
    block_size=16,
    num_blocks=512,
)

scheduler = ContinuousScheduler(runner)
scheduler.start()
try:
    result = scheduler.run(Request(
        prompt="The capital of France is",
        sampling_params=SamplingParams(temperature=0.0),  # greedy
        max_tokens=32,
    ))
    print(result.text)
finally:
    scheduler.stop()
```

### Supported model families

Six HF architecture keys register today; pass any HF model id whose config matches and `ModelRunner.from_pretrained(...)` will route through the right owned class.

| HF architecture | Examples | Family-specific primitives |
|---|---|---|
| `Qwen2ForCausalLM` | Qwen2.5-0.5B / 7B / 32B-Instruct | Biased Q/K/V projections |
| `Qwen3ForCausalLM` | Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B | Per-head Q/K norm, tied embeddings |
| `LlamaForCausalLM` | Llama 2 / 3 / 4, SmolLM2, TinyLlama, Llama-Nemotron variants | Standard Llama-shape baseline |
| `MistralForCausalLM` | Mistral-7B-Instruct-v0.1/0.2/0.3, Mistral Small / Large | Same as Llama (registered separately for the HF arch key) |
| `Gemma3ForCausalLM` | Gemma 3 1B / 4B (text-only) | Sliding-window + global alternating attention, dual RoPE, sandwich norms, GemmaRMSNorm (`(1+w)*x`), GeGLU (`gelu_pytorch_tanh`), embed scaling, Q/K norm |
| `MixtralForCausalLM` | Mixtral-8x7B / 8x22B-Instruct | Top-k sparse MoE FFN (8 experts, top-2) |

Validated path: each family has at least one CPU/MPS smoke test that loads an ungated checkpoint and produces "Paris" for `"The capital of France is"`. Mixtral 8x7B (47B params) is too large for M1 fp16 — the MoE block was bit-validated against HF's `MixtralSparseMoeBlock` on synthetic input instead.

Adding a model family is a one-file change in `src/mini_infer/models/<family>.py`: declare a config + a class that composes the shared blocks, decorate with `@register_model`, and add the import to `_register_builtin_models()` in `src/mini_infer/models/__init__.py`. Examples to copy:

* Pure Llama-shape (no biases, no quirks): [src/mini_infer/models/mistral.py](src/mini_infer/models/mistral.py) — 30 lines.
* Llama-shape + Q/K norm + tied embeddings: [src/mini_infer/models/qwen3.py](src/mini_infer/models/qwen3.py) — ~140 lines.
* Heavier per-family deltas (sandwich norm, GemmaRMSNorm, dual RoPE): [src/mini_infer/models/gemma3.py](src/mini_infer/models/gemma3.py) — ~170 lines.

### Picking a KV-cache mode

| `kv_quant` | `attention_backend` | KV memory vs bf16 | Best on | Notes |
|---|---|---|---|---|
| `None` (default) | `"flash_attn"` | 100% (baseline) | All CUDA | Token-for-token parity; the reference path. |
| `None` | `"flashinfer"` | 100% | All CUDA | bf16 KV via FlashInfer; same parity, different kernel. Useful as a stepping stone before enabling FP8/NVFP4. |
| `"turbo4"` | `"flash_attn"` | 38% (−62%) | A10/A100 | Argmax parity at 0.5B; 7B drifts. Production-grade kernel not built. |
| `"turbo3"` | `"flash_attn"` | 26% (−74%) | A10/A100 | Coherent-but-different output at 0.5B; less degenerate at 7B. Fused Triton kernel keeps decode at 0.31–0.41x of bf16. |
| `"fp8"` | `"flashinfer"` | 50% (−50%) | **H100/H200** | Logit cos sim 0.999985 vs bf16; 0.93x throughput. The production answer for 8-bit KV on Hopper. |
| `"nvfp4"` | `"flashinfer"` | 28.1% (−71.9%) | **B200** | 4-bit KV on Blackwell. 0.91x throughput. Token-level accuracy degraded under greedy decode without outlier-aware calibration; ship infrastructure only, not production-quality outputs (see [bench doc](docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md)). |

Quick examples for each non-default mode:

```python
# Weight-only INT8 (any CUDA GPU)
runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quant="int8")

# TurboQuant V3 (KV compression to ~26% of bf16, A10-grade)
runner = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", kv_quant="turbo3")

# FP8 KV via FlashInfer (Hopper)
runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    kv_quant="fp8",
    attention_backend="flashinfer",
)

# NVFP4 KV via FlashInfer (Blackwell only)
runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    kv_quant="nvfp4",
    attention_backend="flashinfer",
)
```

### Speculative decoding

Use `SpeculativeRunner` when you have a target + draft pair (same vocab family, e.g. Qwen2.5-7B target + Qwen2.5-0.5B draft):

```python
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.speculative import SpeculativeRunner

target = ModelRunner.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
draft = ModelRunner.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
runner = SpeculativeRunner(target=target, draft=draft, K=4)
token_ids, stats = runner.run_greedy(prompt="Explain recursion.", max_tokens=128)
print(target.tokenizer.decode(token_ids))
print(f"acceptance: {stats.mean_acceptance_per_iter:.2f}/{runner.K}")
```

### HTTP server

The `/v1/completions` endpoint mirrors the OpenAI Completions API (subset). The server entry point is `mini_infer.api.server`; today only `MINI_INFER_MODEL` is wired as an env var. For non-default KV quantization or attention backends, build the runner programmatically and pass it to `make_app(...)` in your own startup script — the server module is small enough to copy-and-modify.

```bash
# Default: bf16 KV / flash-attn, model from MINI_INFER_MODEL or Qwen2.5-0.5B-Instruct
uv run python -m mini_infer.api.server

# Override the model only
MINI_INFER_MODEL="Qwen/Qwen2.5-7B-Instruct" uv run python -m mini_infer.api.server
```

### Modal benchmarks

Each major feature ships with a Modal bench script that reproduces the published numbers. Examples:

```bash
# Continuous-batching throughput on A10
modal run scripts/modal_packed_bench.py --config throughput

# Prefix caching speedup on a long shared prompt
modal run scripts/modal_packed_bench.py --config prefix

# FP8 KV on H100
MINI_INFER_BENCH_GPU=H100 modal run scripts/modal_packed_bench.py \
    --config flashinfer_fp8 --model "Qwen/Qwen2.5-0.5B-Instruct"

# NVFP4 KV on B200 (HF token required for 7B+ model downloads)
HF_TOKEN=$(hf auth token) MINI_INFER_BENCH_GPU=B200 \
    modal run scripts/modal_packed_bench.py \
    --config flashinfer_nvfp4 --model "Qwen/Qwen2.5-7B-Instruct" --num-blocks 2048
```

See `scripts/modal_packed_bench.py --help` for the full list of configs.

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

* **`engine/`** loads the model via `mini_infer.models.load_model(...)`, owns prefill and decode.
* **`models/`** is the multi-model registry: per-family `nn.Module` implementations + a shared block library (RMSNorm, RoPE, GQA, SwiGLU, MoE FFN, GemmaDecoderLayer, ...). HF safetensors weights load directly into our hierarchies with no runtime patching.
* **`scheduler/`** owns the engine thread, admission control, the running batch, and the shared `PagedKVCache`.
* **`cache/`** holds the block pool, the batch-aware cache, and the paged attention kernel + reference. The block pool supports per-layer KV shape (Gemma 4 / MLA / V4 prerequisite).
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
* [ADR-011](docs/decisions/ADR-011-speculative-decoding.md): speculative decoding (vanilla two-model, greedy, single-request)
* [ADR-012](docs/decisions/ADR-012-fused-int8-kernel.md): fused W8A16 Triton kernel for `Int8Linear`
* [ADR-013](docs/decisions/ADR-013-turboquant-kv.md): TurboQuant KV cache (V1 + V3 + V2a fused dequant; V2b deselected); FlashInfer FP8 + NVFP4 KV

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

* No multimodal support (text-only). Gemma 3's vision/audio towers, Gemma 4's image/audio inputs, and Phi-3/4-Vision are explicitly skipped — the text decoder is what we implement.
* No fine-tuning or training pipeline (inference only).
* No reimplementation of FlashAttention or other vendor kernels for attention math itself. We use `flash-attn` 2.8+ for varlen attention on CUDA, FlashInfer for FP8/NVFP4 paged paths, and a PyTorch SDPA reference on CPU/MPS. We do own custom Triton kernels for INT8 GEMM (`Int8Linear`) and TurboQuant KV dequant.

Future model support (planned, see [docs/plans/multi-model-support.md](docs/plans/multi-model-support.md)):

* **Gemma 4 31B** — heterogeneous-KV cache is in place; needs `attention_k_eq_v` (V = K on global layers) and `v_norm` to ship.
* **DeepSeek-V2 / V3 + Kimi-K2** — Multi-head Latent Attention (MLA). Generalizes the per-layer-shape primitive into a per-layer storage descriptor (latent KV + RoPE-K).
* **DeepSeek-V4** — hybrid Compressed Sparse Attention + Heavily Compressed Attention. See [docs/plans/deepseek-v4-attention.md](docs/plans/deepseek-v4-attention.md).
* **State-space hybrids** (Mamba, Nemotron-H, Jamba) — different cache abstraction entirely; deliberately out of scope.

## License

(Pending.)
