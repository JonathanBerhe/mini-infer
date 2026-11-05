# Time to first token and tokens per second of each model

Date: 2026-11-05

This report measures every supported model that fits one node, on one
NVIDIA H100 and one B200, with one workload. The
[README](../../README.md#performance) shows a summary of these numbers.

## Setup

- Code: `main` at commit `054e120`.
- GPUs: one NVIDIA H100 80GB HBM3 (Modal `H100!`, which allows no H200 in
  its place) and one NVIDIA B200. DeepSeek-V4-Flash runs on two B200s.
- Image: CUDA 13.2, `torch==2.14.1+cu132`, `triton==3.8.0`,
  `flashinfer-python==0.6.10`, `transformers==5.14.0`.
- Script: `scripts/bench/modal_model_metrics.py`. Each model runs in its own
  container, through the scheduler that the API server uses for it.

## Workload

- Each prompt is a window of 1,024 tokens from
  `scripts/data/technical_passage.md`. Each window starts at another offset,
  so no two prompts share a prefix, and the prefix cache never hits.
- Each request asks for up to 128 output tokens, with greedy sampling and
  bf16 weights. A request can stop earlier at an end-of-text token. The mean
  output length was 120 to 128 tokens in every case.
- TTFT (time to first token) is the median over 5 requests, sent one at a
  time, from the submit to the first output token.
- Decode speed is the output speed of one request: the median over the same
  5 requests of (output tokens - 1) divided by the time from the first token
  to the last token.
- Throughput is the total output speed of 16 requests that start together:
  all output tokens divided by the time from the first submit to the last
  token. The 16 prompts are prefilled in this time too.
- Warm-up requests run first, so the timed requests include no kernel
  compile.

Each model runs on the fastest path that the code has for it. A paged model
uses FlashInfer attention with the decode and prefill CUDA graphs, if the
model allows them. A model with its own state cache uses its decode CUDA
graph, if it has one. The Path column names what ran.

## Results

TTFT and the decode speed are for one request. Throughput and "TTFT at 16"
are for 16 requests at once.

| Model | GPU | Path | TTFT (ms) | Decode (tok/s) | Throughput (tok/s) | TTFT at 16 (ms) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | H100 | Paged, FlashInfer, CUDA graphs | 40 | 155 | 1,365 | 415 |
| Qwen2.5-7B | B200 | Paged, FlashInfer, CUDA graphs | 29 | 297 | 2,738 | 233 |
| Qwen3-8B | H100 | Paged, FlashInfer, CUDA graphs | 44 | 143 | 1,230 | 460 |
| Qwen3-8B | B200 | Paged, FlashInfer, CUDA graphs | 33 | 244 | 2,357 | 244 |
| Llama 3.1 8B Instruct | H100 | Paged, FlashInfer, CUDA graphs | 48 | 147 | 1,313 | 435 |
| Llama 3.1 8B Instruct | B200 | Paged, FlashInfer, CUDA graphs | 34 | 276 | 2,668 | 230 |
| Mistral 7B Instruct v0.3 | H100 | Paged, FlashInfer, CUDA graphs | 46 | 158 | 1,358 | 411 |
| Mistral 7B Instruct v0.3 | B200 | Paged, FlashInfer, CUDA graphs | 33 | 287 | 2,682 | 221 |
| Gemma 3 1B | H100 | Paged, FlashInfer, no graph | 155 | 40 | 542 | 193 |
| Gemma 3 1B | B200 | Paged, FlashInfer, no graph | 77 | 84 | 1,088 | 111 |
| Gemma 4 31B | H100 | Out of memory | - | - | - | - |
| Gemma 4 31B | B200 | Paged, torch attention, no graph | 284 | 28 | 67 | 2,591 |
| Mixtral 8x7B Instruct | B200 | Paged, FlashInfer, CUDA graphs | 127 | 99 | 547 | 488 |
| DeepSeek-V2-Lite-Chat | H100 | Paged, torch attention, no graph | 590 | 25 | 66 | 6,407 |
| DeepSeek-V2-Lite-Chat | B200 | Paged, torch attention, no graph | 416 | 36 | 86 | 4,658 |
| Qwen3.8-27B | H100 | Own cache, decode graph | 217 | 33 | 271 | 3,473 |
| Qwen3.8-27B | B200 | Own cache, decode graph | 144 | 50 | 410 | 2,288 |
| Nemotron 3.5 Lightning 30B-A3B | H100 | Own cache, decode graph | 92 | 66 | 457 | 1,398 |
| Nemotron 3.5 Lightning 30B-A3B | B200 | Own cache, decode graph | 61 | 87 | 673 | 940 |
| Kimi Linear 48B-A3B Instruct | B200 | Own cache, no graph | 3,375 | 2.1 | 14 | 48,890 |
| DeepSeek-V4-Flash | 2x B200 | Tensor parallel, own cache, decode graph | 226 | 29.5 | 240 | 3,505 |

The checkpoints are `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B`,
`unsloth/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
`unsloth/gemma-3-1b-it`, `google/gemma-4-31B-it`,
`mistralai/Mixtral-8x7B-Instruct-v0.1`, `deepseek-ai/DeepSeek-V2-Lite-Chat`,
`Qwen/Qwen3.8-27B`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
`moonshotai/Kimi-Linear-48B-A3B-Instruct` and
`deepseek-ai/DeepSeek-V4-Flash`.

## Why some paths are slow

- Gemma 3 has sliding-window layers. The paged CUDA graphs plan one
  attention call per step, so they need full-attention layers only. Gemma 3
  therefore decodes eagerly, and the host sets its speed.
- Gemma 4 31B has a head size of 512, above the FlashInfer limit, so it uses
  the torch attention and no graph. On the H100 its 62 GB of weights and the
  working memory of the torch attention do not fit in 80 GB.
- DeepSeek-V2 uses MLA, which needs the torch attention, so it has no graph.
- Kimi Linear runs its KDA layers in PyTorch and its experts one at a time,
  with no CUDA graph.
- Mixtral 8x7B (93 GB) and Kimi Linear 48B (98 GB) do not fit one 80 GB
  H100 in bf16, so they ran on the B200 only.

## Models not measured

These models need more GPUs than one node of this setup, or a loader that
does not exist yet:

| Model | Size | Why not measured |
|---|---|---|
| Kimi K3, Kimi K3.1 Pro | 2.8T | Needs a streaming multi-GPU loader. No real-weight run yet |
| Inkling | 975B | Needs a streaming multi-GPU loader |
| GLM-5.2, GLM-5.3 | 753B | Needs eight H200s and a streaming loader |
| MiniMax-M3 | 428B | Needs four H200s. It ran once in July ([report](2026-07-03-minimax-m3-428b-gate.md)) |
| DeepSeek-V4-Pro | Larger than V4-Flash | Needs more than two GPUs |
| DeepSeek-V3, Kimi-K2 | 671B, 1T | Need a multi-GPU node |

## Reproduce

Run from the repository root:

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token) uv run modal run scripts/bench/modal_model_metrics.py --out metrics.jsonl
```

`--models` takes a comma-separated subset of the keys in `MODELS`, and
`--gpus` takes `h100`, `b200` or both.
