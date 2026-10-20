# mini-infer

mini-infer is an LLM inference engine, written from scratch. It ports new model architectures soon after they are published. Each port matches the upstream reference implementation bit for bit.

## Features

mini-infer has these features:

* Thirteen model families, from Qwen2 and Gemma to DeepSeek-V4, GLM-5.2, MiniMax-M3, Inkling, and Kimi Linear. See [Supported models](#supported-models).
* A continuous-batching scheduler. New requests join the running batch at each step.
* A paged KV cache (the attention memory, stored in fixed-size blocks), with prefix caching and chunked prefill.
* An HTTP server that is compatible with the OpenAI Completions API, with SSE streaming.
* INT8 weight quantization, and KV-cache quantization in FP8, NVFP4, and TurboQuant (3-bit and 4-bit).
* Tensor parallelism across GPUs, and separate workers for prefill and decode.
* Speculative decoding with a draft model, and DSpark (a drafter that proposes a block of tokens at a time). DSpark also works with many requests in flight.
* Custom kernels for NVIDIA GPUs (Triton), Google TPUs (Pallas), and AWS Trainium (NKI). Other devices use a PyTorch reference path.

Benchmark results are in [docs/benchmarks/](docs/benchmarks/). Design decisions are in [docs/decisions/](docs/decisions/), one ADR (architecture decision record) per decision.

## Quickstart

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run the unit tests. They run on the CPU and do not download a model.

```bash
uv run pytest tests/unit/ -q
```

Start the API server. The first start downloads `Qwen/Qwen2.5-0.5B-Instruct`.

```bash
uv run python -m mini_infer.api.server
```

In a second shell, send a request to the server:

```bash
curl -N http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":8,"stream":true}'
```

The engine selects the device automatically: CUDA, then MPS, then CPU.

## Supported models

The engine selects the model class from `config.architectures[0]` in the Hugging Face configuration. It loads Hugging Face safetensors directly.

| Hugging Face architecture | Examples | Notes |
|---|---|---|
| `Qwen2ForCausalLM` | Qwen2.5 0.5B to 32B | |
| `Qwen3ForCausalLM` | Qwen3 0.6B to 32B | [doc](docs/architectures/qwen3.md) |
| `Qwen3_5ForConditionalGeneration` | Qwen3.8-27B, Qwen3.5, Qwen3.6 (text only) | [doc](docs/architectures/qwen3-5.md) |
| `LlamaForCausalLM` | Llama 2/3/4, SmolLM2, TinyLlama, Nemotron | [doc](docs/architectures/llama.md) |
| `MistralForCausalLM` | Mistral 7B, Small, Large | [doc](docs/architectures/mistral.md) |
| `Gemma3ForCausalLM` | Gemma 3 1B, 4B | [doc](docs/architectures/gemma-3.md) |
| `Gemma4ForConditionalGeneration` | Gemma 4 31B (text only) | [doc](docs/architectures/gemma-4.md) |
| `MixtralForCausalLM` | Mixtral 8x7B, 8x22B | [doc](docs/architectures/mixtral.md) |
| `DeepseekV2ForCausalLM` | DeepSeek-V2, V3, Kimi-K2 | [doc](docs/architectures/deepseek-v2-mla.md) |
| `DeepseekV4ForCausalLM` | DeepSeek-V4-Pro, V4-Flash | [doc](docs/architectures/deepseek-v4.md) |
| `GlmMoeDsaForCausalLM` | GLM-5.2, GLM-5.3 | [ADR-021](docs/decisions/ADR-021-glm-moe-dsa-port.md) |
| `MiniMaxM3SparseForConditionalGeneration` | MiniMax-M3 (text only) | [doc](docs/architectures/minimax-m3.md) |
| `InklingForConditionalGeneration` | Inkling (text only) | [doc](docs/architectures/inkling.md) |
| `KimiLinearForCausalLM` | Kimi Linear 48B | [ADR-026](docs/decisions/ADR-026-kimi-linear-kda-port.md) |
| `NemotronHForCausalLM` | Nemotron 3.5 Lightning 30B-A3B, Nemotron 3 Nano 4B (text only) | [doc](docs/architectures/nemotron-h.md) |

To add a model family, add one file in `src/mini_infer/models/`. Build the model from the shared blocks (RMSNorm, RoPE, attention, MoE) and decorate the class with `@register_model`. [mistral.py](src/mini_infer/models/mistral.py) is the shortest example.

Tests compare each port with the Hugging Face reference. Small models load a real checkpoint. Models too large for a laptop run on Modal GPUs, or each block runs against the reference block on synthetic input.

## Usage

### Python

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

| `kv_quant` | `attention_backend` | KV size vs bf16 | Hardware |
|---|---|---|---|
| `None` (default) | `flash_attn` or `flashinfer` | 100% | All CUDA GPUs |
| `turbo4` | `flash_attn` | 38% | A10, A100 |
| `turbo3` | `flash_attn` | 26% | A10, A100 |
| `fp8` | `flashinfer` | 50% | H100, H200 |
| `nvfp4` | `flashinfer` | 28% | B200 |

### HTTP server

The `/v1/completions` endpoint supports a subset of the OpenAI Completions API. You set the server configuration with environment variables:

* `MINI_INFER_MODEL`: the Hugging Face model ID. The default is `Qwen/Qwen2.5-0.5B-Instruct`.
* `MINI_INFER_USE_PD=1`: use separate prefill and decode workers.
* `MINI_INFER_PD_MODE`: `serial` or `parallel`.

```bash
MINI_INFER_MODEL="Qwen/Qwen2.5-7B-Instruct" uv run python -m mini_infer.api.server
```

For quantization or a different attention backend, build the runner in Python and pass it to `make_app(...)`.

## Project layout

* `src/mini_infer/models/`: the model families and the shared blocks.
* `src/mini_infer/engine/`: model loading, prefill, decode, sampling, and speculative decoding.
* `src/mini_infer/scheduler/`: the engine thread, request admission, and the running batch.
* `src/mini_infer/cache/`: the block pool, the paged KV cache, and the paged attention kernels.
* `src/mini_infer/api/`: the FastAPI server.
* `src/mini_infer/backends/`: the TPU kernels.

## Tests

```bash
uv run pytest tests/unit/ -q
uv run pytest tests/golden/ -q
uv run pytest -m requires_model -v
```

The unit tests run on the CPU on each push. The golden tests compare the output token by token with Hugging Face. The `requires_model` tests download models and compare them bit for bit. A weekly workflow runs them against pinned Hugging Face revisions. Tests marked `large_model` download several GB, so they run only when you add `--run-large-models`.

## Out of scope

* Images, audio, and video. mini-infer runs the text decoder only.
* Training and fine-tuning.
* Kernels that a vendor library already supplies. mini-infer uses flash-attn, FlashInfer, PyTorch SDPA, and XLA where they exist.

## License

MIT. See [LICENSE](LICENSE).
