# mini-infer

A from-scratch LLM inference engine for newly-published architectures, bit-parity validated against the upstream reference implementation. The niche neither vLLM nor SGLang fills: a research-paper-readable engine that ports new architectures fast and matches the reference behaviour exactly. Ten model families register today, from Qwen2 through complete from-scratch ports of DeepSeek-V4 and GLM-5.2.

Positioning, non-goals, and roadmap: [docs/plans/roadmap-2026.md](docs/plans/roadmap-2026.md).

## What works

* **Models**. Ten HF architecture families: Qwen2, Qwen3, Llama (2/3/4 + SmolLM2 + TinyLlama + Nemotron variants), Mistral, Gemma 3, Gemma 4, Mixtral, DeepSeek-V2/V3/Kimi-K2 (MLA), DeepSeek-V4 (hybrid CSA/HCA), and GLM-5.2 (MLA + DeepSeek Sparse Attention + IndexShare). Adding one is a single file. See [model families](#model-families).
* **Serving**. Continuous-batching scheduler on a dedicated engine thread, chunked prefill, one packed-varlen `model.forward` per step (FlashAttention varlen on CUDA, PyTorch reference elsewhere), PagedAttention with per-layer named KV streams, prefix caching, OpenAI-compatible HTTP API with SSE streaming.
* **Quantization & KV**. Weight-only INT8 W8A16 (fused Triton GEMM), TurboQuant KV (V1 4-bit + V3 3-bit with a fused dequant kernel), and FlashInfer paged KV in FP8 (Hopper) and NVFP4 (Blackwell).
* **Scale**. Megatron-style tensor parallelism (column/row/vocab-parallel linears + expert-parallel MoE, no-op at `world_size=1`), prefill/decode disaggregation, and two-model speculative decoding.

Every non-trivial choice is an ADR in [docs/decisions/](docs/decisions/) (ADR-001 through ADR-021). Paper-section to code walkthroughs live in [docs/architectures/](docs/architectures/).

## Benchmarks

Continuous batching, Qwen2.5-0.5B-Instruct, A10, bf16:

| Concurrency | Tokens/sec | vs C=1 |
|---:|---:|---:|
| 1 | 45.7 | 1.00x |
| 2 | 79.5 | 1.74x |
| 4 | 127.7 | 2.79x |
| 8 | 187.5 | 4.10x |

Other validated results (full reports in [docs/benchmarks/](docs/benchmarks/)):

* **Prefix caching**: up to **32.45x** throughput on a 15.9k-token shared prompt; warm TTFT ~74ms vs ~11.7s cold (158x). [report](docs/benchmarks/2026-04-28-prefix-caching.md)
* **Fused INT8 kernel**: **2.74x** decode at C=1 on Qwen2.5-7B (the regime where the bf16-weight HBM round-trip dominates). [report](docs/benchmarks/2026-04-29-fused-int8-kernel.md)
* **FP8 KV** (H100): **-50%** KV memory, logit cosine sim 0.999985, 0.93x throughput. [report](docs/benchmarks/2026-05-02-flashinfer-fp8-kv.md)
* **NVFP4 KV** (B200): **-71.9%** KV memory, 0.91x throughput (accuracy needs outlier-aware calibration; infrastructure only). [report](docs/benchmarks/2026-05-02-flashinfer-nvfp4-kv.md)
* **TurboQuant V3 KV**: **-74%** KV memory. [report](docs/benchmarks/2026-04-30-turboquant-v3.md)
* **Speculative decoding**: 1.14x on A10 (7B target + 0.5B draft, K=4). [report](docs/benchmarks/2026-04-29-speculative-decoding.md)
* **INT8 weights**: **-30.5%** weight HBM, cosine sim > 0.99, first-token greedy parity. [report](docs/benchmarks/2026-04-28-int8-weight-quant.md)

## Quickstart

```bash
# Install (Python 3.11+, uv as the package manager)
uv sync

# Unit tests (CPU only, no model download)
uv run pytest tests/unit/ -v

# API server (downloads Qwen2.5-0.5B-Instruct on first start)
uv run python -m mini_infer.api.server

# In another shell, hit the API
curl -N http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":8,"stream":true}'

# Demo: DeepSeek-V4 hybrid CSA/HCA backbone (synthetic weights, no download)
uv run python scripts/demo_deepseek_v4_hybrid.py
```

Default model is `Qwen/Qwen2.5-0.5B-Instruct` (override with `MINI_INFER_MODEL`). Device is automatic: CUDA, else MPS, else CPU. The Triton paged kernel runs on CUDA; other devices use a numerically-equivalent PyTorch reference.

## Model families

`ModelRegistry` maps an HF `config.architectures[0]` string to an owned `nn.Module`. HF safetensors load directly (`load_state_dict`, or `load_state_dict_with_tp` for multi-GPU); no runtime monkey-patching.

| HF architecture | Examples | Family-specific primitives |
|---|---|---|
| `Qwen2ForCausalLM` | Qwen2.5-0.5B / 7B / 32B | Biased Q/K/V projections |
| `Qwen3ForCausalLM` | Qwen3 0.6B-32B | Per-head Q/K norm, tied embeddings ([doc](docs/architectures/qwen3.md)) |
| `LlamaForCausalLM` | Llama 2/3/4, SmolLM2, TinyLlama, Nemotron | Llama-shape baseline ([doc](docs/architectures/llama.md)) |
| `MistralForCausalLM` | Mistral 7B / Small / Large | Llama-shape (separate HF key) ([doc](docs/architectures/mistral.md)) |
| `Gemma3ForCausalLM` | Gemma 3 1B / 4B | Sliding + global attention, dual RoPE, sandwich norms, GeGLU ([doc](docs/architectures/gemma-3.md)) |
| `Gemma4ForConditionalGeneration` | Gemma 4 31B-it (text-only) | Heterogeneous per-layer KV, `k_eq_v`, dual RoPE with different head_dim, logit softcap, SDPA backend override ([doc](docs/architectures/gemma-4.md)) |
| `MixtralForCausalLM` | Mixtral 8x7B / 8x22B | Top-k sparse MoE (8 experts, top-2) ([doc](docs/architectures/mixtral.md)) |
| `DeepseekV2ForCausalLM` | V2-Lite / V2 / V3 / Kimi-K2 | Multi-head Latent Attention (~7x smaller cache), low-rank Q, interleaved RoPE, dense + MoE FFN ([doc](docs/architectures/deepseek-v2-mla.md)) |
| `DeepseekV4ForCausalLM` | V4-Pro / V4-Flash | Every published V4 primitive: three-mode attention (SWA/CSA/HCA), Lightning Indexer, token compressor, attention sink, grouped output, hash-routed MoE, Hyper-Connections, YaRN ([doc](docs/architectures/deepseek-v4.md), [ADR-014](docs/decisions/ADR-014-deepseek-v4-hybrid-attention.md)) |
| `GlmMoeDsaForCausalLM` | GLM-5.2 (753B FP8) | DeepSeek-V3.2-style MLA + DeepSeek Sparse Attention (Lightning Indexer top-k) + GLM IndexShare + noaux_tc sigmoid MoE; block-FP8 per-expert + FP8-resident loader ([ADR-021](docs/decisions/ADR-021-glm-moe-dsa-port.md)) |

Adding a family is one file in `src/mini_infer/models/<family>.py`: declare a config + a class composing the shared blocks (RMSNorm, RoPE, GQA, SwiGLU, MoE FFN, ...), decorate with `@register_model`, add one import. Copy the closest example, from [mistral.py](src/mini_infer/models/mistral.py) (Llama-shape, 30 lines) up to [deepseek_v4.py](src/mini_infer/models/deepseek_v4.py) (hybrid attention + Hyper-Connections, ~1.6k lines across the family).

**Validation.** Each loadable family has a CPU/MPS smoke that loads an ungated checkpoint and emits "Paris" for "The capital of France is". Models past M1's memory budget (Mixtral, Gemma 4, DeepSeek-V2-Lite, V4, GLM-5.2) are validated on Modal and/or bit-parity-checked against the HF block on synthetic input. **DeepSeek-V4**: every primitive is bit-parity vs the [V4-Pro reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) (cosine 1.0000, max abs diff < 1e-9); the V4-Flash loader matches all 34,223 params of the real safetensors index, with a full multi-GPU forward as the remaining gate. **GLM-5.2**: bit-parity vs the HF `glm_moe_dsa` reference on CPU, a single L4 (fp32-exact + bf16), and 2x L4 under NCCL TP; the FP8-resident loader fits the 753B checkpoint on a single 8xH200 node, with the funded full-weight run pending.

## Usage

### Programmatic

```python
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request

runner = ModelRunner.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quant=None,                       # "int8" enables W8A16 weight-only quant
    kv_quant=None,                    # "turbo4" | "turbo3" | "fp8" | "nvfp4"
    attention_backend="flash_attn",   # "flashinfer" required for fp8/nvfp4
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

### KV-cache modes

| `kv_quant` | `attention_backend` | KV vs bf16 | Best on | Notes |
|---|---|---|---|---|
| `None` (default) | `flash_attn` | 100% | All CUDA | Token-for-token parity; the reference path |
| `None` | `flashinfer` | 100% | All CUDA | bf16 KV via FlashInfer; same parity, different kernel |
| `turbo4` | `flash_attn` | 38% | A10/A100 | Argmax parity at 0.5B; 7B drifts |
| `turbo3` | `flash_attn` | 26% | A10/A100 | Coherent at 0.5B; fused kernel keeps decode at 0.31-0.41x of bf16 |
| `fp8` | `flashinfer` | 50% | **H100/H200** | cos sim 0.999985, 0.93x throughput; the 8-bit KV answer on Hopper |
| `nvfp4` | `flashinfer` | 28% | **B200** | 4-bit KV; 0.91x throughput; needs outlier-aware calibration (infra only) |

### HTTP server

`/v1/completions` mirrors the OpenAI Completions API (subset), non-streaming or SSE. Env vars: `MINI_INFER_MODEL`, `MINI_INFER_USE_PD=1` (back the API with the PD pipeline), `MINI_INFER_PD_MODE=serial|parallel`. For non-default quant/backends, build the runner programmatically and pass it to `make_app(...)`.

```bash
uv run python -m mini_infer.api.server
MINI_INFER_MODEL="Qwen/Qwen2.5-7B-Instruct" uv run python -m mini_infer.api.server
```

### Other entry points

* **Speculative decoding**: `SpeculativeRunner(target=..., draft=..., K=4).run_greedy(...)` for a target + draft pair (same vocab family).
* **Modal benches**: `modal run scripts/modal_packed_bench.py --config <throughput|prefix|flashinfer_fp8|flashinfer_nvfp4|...>` reproduces the published numbers. See `--help` for the full list.

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
                            Backend kernels       /  PyTorch reference
                   (CUDA/Triton, TPU/Pallas, Trainium/NKI)  (CPU/MPS)
```

* **`engine/`** loads the model and owns prefill and decode.
* **`models/`** is the registry: per-family `nn.Module`s + a shared block library (RMSNorm, RoPE, GQA, SwiGLU, MoE FFN, ...).
* **`scheduler/`** owns the engine thread, admission control, the running batch, and the shared `PagedKVCache`.
* **`cache/`** holds the block pool, the batch-aware cache, and the paged attention kernel + reference. Per-layer named `StreamSpec`s handle heterogeneous KV (Gemma 4) and MLA's `kv_latent` + `k_rope`; V4 adds a per-request `StateCache`.
* **`api/`** is FastAPI with OpenAI-compatible endpoints and SSE streaming.

## Tests

* **Unit** (`tests/unit/`): CPU-only, fast, on every CI push. Block pool, cache, attention reference, scheduler, sampler, API schemas.
* **Golden** (`tests/golden/`): CPU/fp32 token-for-token vs Hugging Face, catches numerical drift.
* **Modal smokes + benches** (`scripts/`): real CUDA validation, on demand.

A separate weekly workflow (`.github/workflows/bitparity.yml`) runs the `requires_model` tests against pinned HF revisions (`tests/_pinned_models.toml`) so upstream changes don't silently affect us.

```bash
uv run pytest tests/unit/ -q                       # CI-equivalent (no model loads)
uv run pytest tests/golden/ -q                     # token-for-token vs HF
uv run pytest -m requires_model -v                 # bit-parity subset (downloads models)
```

## What's NOT in here

By design:

* **No multimodal** (text-only). Gemma 3/4 vision/audio towers and Phi-Vision are skipped; we implement the text decoder.
* **No training or fine-tuning** (inference only).
* **No reimplemented attention kernels beyond what a vendor library lacks.** flash-attn for varlen on CUDA, FlashInfer for FP8/NVFP4 paged paths, PyTorch SDPA on CPU/MPS, XLA/Pallas primitives on TPU. We own hand-written kernels (Triton on GPU, Pallas on TPU, NKI on Trainium) only where the math isn't in a vendor library: INT8 GEMM, TurboQuant KV dequant, the Hyper-Connections Sinkhorn port, and paged attention on TPU.

Planned work (DeepSeek-V3/Kimi-K2 large-checkpoint runs, V4-Flash multi-GPU forward, the funded GLM-5.2 753B run) is tracked in [docs/plans/roadmap-2026.md](docs/plans/roadmap-2026.md).

## License

(Pending.)
