# Llama 3.1 8B against transformers on an H100

Date: 2026-11-05

This report compares the engine with transformers on Llama 3.1 8B. It
measures the llama3 RoPE rescale and the bf16 cast of the RoPE table.

## Setup

- Script: `scripts/llama/modal_llama_hf_parity.py`.
- GPU: one H100 80GB HBM3 (requested as `H100!`).
- Stack: the main stack (CUDA 13.2, torch 2.14.1+cu132, Triton 3.8.0,
  FlashInfer 0.6.10, transformers 5.14.0).
- Model: `unsloth/Meta-Llama-3.1-8B-Instruct` in bf16. It holds the same
  weights as the gated meta-llama repo.

The script loads the model in transformers (SDPA attention) and in the
engine (`ModelRunner` with FlashInfer, no CUDA graph). Then it puts four
RoPE tables, one at a time, into the engine model. The short prompt is "The
capital of France is" with BOS (6 tokens), with 8 greedy tokens. The long
prompt is 10240 tokens of repo documents. The script compares the logits of
its last 256 positions. All of them sit past position 8192, the original
context of Llama 3.1.

## Short prompt

Transformers and all four tables give " a city of romance, art, fashion".
The H200 gate of 2026-11-05 gave the same text. So this text is the greedy
output of the model without a chat template, and the engine is correct here.
The "Paris" check of `scripts/smoke/modal_smoke.py` does not fit this model.

## Long prompt

| RoPE table | Top-1 agreement | Mean logit gap | Max logit gap | Mean KL |
|---|---|---|---|---|
| llama3 rescale, fp32 (the table of transformers) | 97.3% | 0.032 | 1.75 | 0.00079 |
| llama3 rescale, bf16 (the table that `load_model` leaves) | 98.0% | 0.056 | 5.05 | 0.00202 |
| plain, bf16 (the port before the fix) | 87.1% | 0.329 | 10.57 | 0.05411 |
| plain, fp32 | 86.7% | 0.321 | 10.15 | 0.04946 |

The mean KL is the KL divergence from the transformers distribution to the
engine distribution, per position.

The llama3 rescale is the large effect. It cuts the mean KL by a factor of
27 (0.054 to 0.0020) and the top-1 misses from 12.9% to 2.0%.

The bf16 cast of `inv_freq` is the second effect. With the fp32 table, the
mean KL is 2.6 times smaller and the max logit gap is 2.9 times smaller. At
256 positions, the top-1 agreement does not separate the two tables.

The gap that stays with the fp32 llama3 table probably comes from bf16
arithmetic in different kernels: SDPA against FlashInfer, and two sets of
GEMM kernels.
