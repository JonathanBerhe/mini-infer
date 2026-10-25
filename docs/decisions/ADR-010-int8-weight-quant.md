# ADR-010: Weight-only INT8 quantization (W8A16)

Date: 2026-04-28
Status: Accepted

## Context

After ADR-009 the engine has continuous batching, chunked prefill, paged
attention and prefix caching. Quantization is the next step. The change is
local (the engine replaces modules at load time), and smaller weights leave
more memory for the KV cache.

The target is weight-only INT8 (W8A16): INT8 weights with `bf16` or `fp16`
activations. bitsandbytes `Linear8bitLt` and vLLM use the same
method. We write it from scratch so that the integration is explicit.

## Decision

1. Symmetric scales per output channel. For `W: (out, in)`,
   `s_i = max(|W[i, :]|) / 127` and
   `W_q = round(W / s).clamp(-127, 127).to(int8)`. Weights are centered on
   zero, so they need no zero point.
2. `src/mini_infer/quant/int8.py` holds `quantize_per_channel(weight)`,
   `dequantize_per_channel(q, s, dtype)`, `Int8Linear(nn.Module)` (a
   drop-in replacement for `nn.Linear`) and
   `quantize_model_to_int8(model, skip_modules)`.
3. A naive forward: `w_dq = weight.to(x.dtype) * scales.to(x.dtype).unsqueeze(1)`,
   then `F.linear(x, w_dq, bias.to(x.dtype))`. It saves no FLOPs. The only
   gain is less HBM read traffic for the weights.
4. `ModelRunner.from_pretrained(quant="int8")` replaces the modules after
   load. The `quant` parameter defaults to `None`, and then the fp path
   does not change.
5. `lm_head` is skipped by default, and `quant_lm_head=True` includes it.
   On Qwen2.5-0.5B it is the largest Linear (151,936 × 896, 28% of all
   Linear params). Its output is the logits, and softmax amplifies the
   quantization noise.
6. The int8 weight and the scales are buffers, not Parameters, because they
   have no autograd path. The bias stays an `nn.Parameter`, so state-dict
   names match `nn.Linear`.

## Alternatives considered

- W8A8 (deferred). INT8 GEMM has 2x the bf16 throughput on Ampere and
  Hopper. But it needs calibrated activation scales and outlier handling
  such as SmoothQuant.
- W4A16 with GPTQ or AWQ (deferred). It saves 75% of the weight memory, but
  it needs calibration and a custom dequant kernel.
- bitsandbytes `Linear8bitLt`. Its fused CUTLASS kernel is faster than our
  naive path at small batches. Rejected, because the load-time module walk
  is part of what this work shows.
- Per-group scales (group_size = 128). The quality gain is small for the
  extra code.
- Float8 (E4M3 or E5M2). The A10 reference GPU has no fp8 GEMM.

## Consequences

Per-channel INT8 usually keeps cosine similarity above 0.999 between
quantized and fp logits on Qwen-class models. The tests require a cosine
above 0.99 on Qwen2.5-0.5B-Instruct:

- `tests/unit/test_int8_model_integration.py::test_quantized_logits_close_to_fp_reference`
  compares the last-position logits.
- `tests/unit/test_int8_model_integration.py::test_quantized_model_decodes_paris`
  makes sure that the quantized model answers "Paris".
- `tests/stress/test_int8_load.py::test_int8_first_tokens_match_fp_reference`
  compares the first greedy token over several prompts.

We claim first-token parity only. Quantization noise flips an argmax after
some tokens.

Qwen2.5-0.5B has 169 `nn.Linear` modules with ~494M params. The default
skip keeps the 136M-param `lm_head` in fp16:

| Mode                  | Linear weight bytes | Savings vs fp16 |
|-----------------------|---------------------|-----------------|
| fp16 baseline         | 988 MB              | -               |
| int8 (skip lm_head)   | ~630 MB             | ~36%            |
| int8 (incl. lm_head)  | ~494 MB             | ~50%            |

`tests/unit/test_int8_model_integration.py::test_quantized_weight_storage_smaller_than_fp`
asserts ≥30% savings with the default skip. On an M1 CPU, 15 unit tests,
5 real-model tests and 1 stress test pass.

On CUDA (A10, Qwen2.5-0.5B), model-weight HBM drops 30.5% with the default
skip. Throughput is within ~6% of fp16 at C=1 and neutral at C=4 and C=8
(`docs/benchmarks/2026-04-28-int8-weight-quant.md`). The dequant adds work
on top of the same matmul. A fused dequant-matmul kernel is the path to a
real speedup.

Qwen2.5 has `tie_word_embeddings=True`. `quant_lm_head=True` adds an int8
copy of `lm_head` but cannot free the fp weight, because `embed_tokens`
still holds it. So on Qwen2.5-0.5B it saves less memory (−19.1%) than the
default skip (−30.5%). The flag stays for models with untied embeddings.

Loading in fp and then quantizing needs ~2x weight memory for a moment.
Larger models need streaming quantization, one shard at a time.
Follow-ups: a fused dequant-matmul Triton kernel, quantization that
handles tied embeddings, W8A8 and W4A16.
