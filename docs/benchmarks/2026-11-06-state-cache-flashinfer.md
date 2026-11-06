# FlashInfer attention for the state-cache models on one H100 and one B200

Date: 2026-11-06
Hardware: one NVIDIA H100 80GB HBM3 (compute capability 9.0, 132 SMs) and
one NVIDIA B200 (compute capability 10.0, 148 SMs), on Modal
Software on both GPUs: CUDA 13.2, torch 2.14.1+cu132, Triton 3.8.0,
FlashInfer 0.6.10, no flash-attn
Models: `Qwen/Qwen3.8-27B` at `1d4bf0f` and
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at `a9904d2`, in bf16
Workload: a prompt of real prose (War and Peace), then greedy decode from
the decode CUDA graph: 3 warm-up steps and 20 timed steps, every row a copy
of one prefilled request, one host sync per step for sampling
Engine: mini-infer at `93b0096` (branch `feat/state-cache-flashinfer-attention`)
Script: `scripts/bench/modal_state_cache_attention_ab.py`, one run per GPU,
both runs at the same time

## Question

Qwen3.8 and Nemotron-H keep the attention keys of each request in a buffer
of `max_seq_len` positions. The PyTorch decode step reads the whole buffer
and masks the empty positions. How much faster is a step that reads only
the filled pages through FlashInfer
([ADR-059](../decisions/ADR-059-state-cache-flashinfer-attention.md))? And
which FlashInfer backend is faster for a one-token step?

## Result

The script runs three arms on one model load: the PyTorch path, FlashInfer
with its automatic backend, and FlashInfer with FA2. The tables give
milliseconds per decode step. The gain is the PyTorch time divided by the
FA2 time, which is what ships.

One H100. The automatic backend is FlashAttention 3 here:

| Model | Prompt | Buffer | Batch | PyTorch | FlashInfer FA3 | FlashInfer FA2 | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 2,048 | 2,048 | 1 | 26.38 | 26.01 | 25.04 | 1.05x |
| Qwen3.8-27B | 2,048 | 2,048 | 4 | 27.57 | 27.05 | 26.19 | 1.05x |
| Qwen3.8-27B | 2,048 | 65,536 | 1 | 77.39 | 26.01 | 25.06 | 3.09x |
| Qwen3.8-27B | 32,768 | 32,768 | 1 | 51.15 | 42.01 | 25.63 | 2.00x |
| Qwen3.8-27B | 32,768 | 32,768 | 4 | 53.35 | 43.58 | 28.95 | 1.84x |
| Nemotron-H 30B | 2,048 | 2,048 | 1 | 9.36 | 9.45 | 9.27 | 1.01x |
| Nemotron-H 30B | 2,048 | 2,048 | 4 | 10.28 | 10.46 | 10.26 | 1.00x |
| Nemotron-H 30B | 2,048 | 65,536 | 1 | 16.87 | 9.43 | 9.28 | 1.82x |
| Nemotron-H 30B | 32,768 | 32,768 | 1 | 12.97 | 11.82 | 9.32 | 1.39x |
| Nemotron-H 30B | 32,768 | 32,768 | 4 | 13.93 | 12.92 | 10.52 | 1.32x |

One B200. The automatic backend is FA2 here, so both FlashInfer arms ran
the same kernels, and the table gives the FA2 arm:

| Model | Prompt | Buffer | Batch | PyTorch | FlashInfer FA2 | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 2,048 | 2,048 | 1 | 16.65 | 14.88 | 1.12x |
| Qwen3.8-27B | 2,048 | 2,048 | 4 | 17.68 | 15.90 | 1.11x |
| Qwen3.8-27B | 2,048 | 65,536 | 1 | 79.01 | 14.91 | 5.30x |
| Qwen3.8-27B | 32,768 | 32,768 | 1 | 46.83 | 15.24 | 3.07x |
| Qwen3.8-27B | 32,768 | 32,768 | 4 | 48.34 | 17.11 | 2.83x |
| Nemotron-H 30B | 2,048 | 2,048 | 1 | 7.76 | 7.68 | 1.01x |
| Nemotron-H 30B | 2,048 | 2,048 | 4 | 8.62 | 8.57 | 1.01x |
| Nemotron-H 30B | 2,048 | 65,536 | 1 | 15.73 | 7.68 | 2.05x |
| Nemotron-H 30B | 32,768 | 32,768 | 1 | 11.60 | 7.70 | 1.51x |
| Nemotron-H 30B | 32,768 | 32,768 | 4 | 12.55 | 8.74 | 1.44x |

Prefill of a 32,768-token prompt in chunks of 4,096 tokens, PyTorch
against FlashInfer:

| Model | H100 | B200 |
|---|---|---|
| Qwen3.8-27B | 7.25 s to 6.69 s (1.08x) | 4.76 s to 4.70 s (1.01x) |
| Nemotron-H 30B | 1.91 s to 1.79 s (1.07x) | 1.32 s to 1.27 s (1.04x) |

Accuracy, FlashInfer against PyTorch, after a 4,096-token prompt:

- The cosine of the last-position logits is 0.99962 (Qwen3.8, H100),
  0.99966 (Qwen3.8, B200), 0.99994 (Nemotron-H, H100) and 0.99992
  (Nemotron-H, B200). The top-1 token is the same in each case, and the
  same for both FlashInfer backends.
- Of 16 greedy tokens, 11 (Qwen3.8, H100), 3 (Qwen3.8, B200), 8 and 8
  (Nemotron-H) are equal. The two paths sum in another order in bf16, and
  the text splits after the first token that differs.

Before the models load, the script runs 52 GPU tests on each GPU. They
include the FlashInfer kernel against the masked SDPA path and a CUDA-graph
step against an eager step, on tiny models with head size 64. All passed.

## What the numbers show

1. On the PyTorch path, a short request in a long buffer pays for the whole
   buffer. A 2,048-token Qwen3.8 step takes 77.39 ms in a 65,536-position
   buffer and 26.38 ms in a 2,048-position buffer (H100). A server sizes the
   buffer for its longest prompt, so every request pays this. With FlashInfer
   the step costs the same in both buffers.
2. At 32,768 tokens, FlashAttention 3 on the H100 gives each query tile one
   CTA and does not split the row, so few SMs read the keys: 42.01 ms. FA2
   splits the row: 25.63 ms, near the 2,048-token step. On the B200 the
   32,768-token step (15.24 ms) is within 3% of the 2,048-token step.
3. For a short request in a short buffer the gain is 1.00x to 1.12x, because
   the weight reads set the step time.
4. Prefill gains little, because the Gated DeltaNet and Mamba-2 layers set
   the prefill time.

## Reproduce

```bash
MINI_INFER_TEST_GPU='H100!' uv run modal run scripts/bench/modal_state_cache_attention_ab.py
MINI_INFER_TEST_GPU=B200 uv run modal run scripts/bench/modal_state_cache_attention_ab.py
```

The weights and the prose come from the `mini-infer-hf-cache` volume, which
`scripts/qwen/modal_qwen38_smoke.py`, `scripts/nemotron_h/modal_nemotron_h_smoke.py`
and `scripts/qwen/modal_qwen38_prefill_ab.py` fill.
