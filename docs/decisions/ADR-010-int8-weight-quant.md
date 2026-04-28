# ADR-010: Weight-only INT8 quantization (W8A16)

Date: 2026-04-28
Status: Accepted

## Context

After ADR-009 the engine ships continuous batching, chunked prefill, paged
attention, and prefix caching. The remaining Phase 2 ROADMAP items are
quantization and tensor parallelism. Quantization is the cheaper next slice:
the integration is local (load-time module replacement), validation budget
is small, and the memory savings stack with prefix caching (smaller weights
leave more room for K/V cache).

The implementation target is **weight-only INT8 (W8A16)**: every quantizable
`nn.Linear` weight matrix is quantized symmetrically per output channel;
activations stay in `bf16` / `fp16`. This is the same family as
HuggingFace's `load_in_8bit`, bitsandbytes' `Linear8bitLt`, and the W8A16
path in vLLM's quantized GEMM kernels — implemented from scratch here so the
integration is explicit.

## Decision

1. **Symmetric, per-output-channel scales.** For `W: (out, in)`,
   `s_i = max(|W[i, :]|) / 127`; `W_q = round(W / s).clamp(-127, 127).to(int8)`.
   Per-channel keeps per-row dynamic range tight without the complexity of
   per-group scales. Symmetric is fine for weights (zero-centered
   distributions); asymmetric would only be needed for activation
   quantization.
2. **Standalone primitives** (`src/mini_infer/quant/int8.py`):
   - `quantize_per_channel(weight)` / `dequantize_per_channel(q, s, dtype)`
     — pure tensor ops.
   - `Int8Linear(nn.Module)` — drop-in replacement for `nn.Linear` with int8
     `weight` buffer + per-channel `scales` buffer + float `bias`.
   - `quantize_model_to_int8(model, skip_modules)` — walks the model and
     replaces each `nn.Linear` (not in `skip_modules`) with `Int8Linear`.
3. **Naive dequant-then-bf16-matmul forward.**
   `w_dq = weight.to(x.dtype) * scales.to(x.dtype).unsqueeze(1)` followed by
   `F.linear(x, w_dq, bias.to(x.dtype))`. The dequant is `O(out * in)`
   regardless of batch; for batch=1 decode it adds a fixed cost on top of
   the matmul, for prefill it amortizes across the matmul. No FLOP savings;
   the win is HBM read traffic on the weight matrices.
4. **`ModelRunner.from_pretrained(quant="int8")`** triggers the replacement
   after `AutoModelForCausalLM.from_pretrained`. Default `skip_modules =
   {"lm_head"}`; flip via `quant_lm_head=True`.
5. **`lm_head` skipped by default.** It's the largest single Linear on
   Qwen2.5-0.5B (151,936 × 896, 28% of all Linear params) and its output IS
   the logits — quantization noise here is amplified through softmax. The
   industry default is to skip it; ours follows.
6. **Buffers, not Parameters, for the int8 weight + scales.** They have no
   autograd path and we don't want the optimizer to find them. The bias
   stays as an `nn.Parameter` (with `requires_grad=False`) for state-dict
   naming compatibility with `nn.Linear`.

## Why W8A16 and not W8A8 / W4A16

Three options were considered:

- **W8A16 (this slice).** Quantize weights only. No calibration. Drop-in
  replacement. Numerical impact small (cosine sim > 0.99 on typical models).
  Memory savings ~50% on Linear weights. No FLOP savings.
- **W8A8 (deferred).** Quantize weights and activations. Requires per-tensor
  or per-token activation scales calibrated on a representative dataset, plus
  outlier handling (SmoothQuant) since LLM activation distributions have
  heavy tails. Real FLOP savings on Tensor Cores (INT8 GEMM is 2x bf16
  throughput on Ampere/Hopper). Significant additional implementation
  surface.
- **W4A16 (GPTQ / AWQ, deferred).** 4-bit weights. Bigger memory savings
  (75%) but requires calibration (Hessian-free for AWQ, Hessian-based for
  GPTQ) and a custom kernel for dequant-on-the-fly. ~2x more implementation
  surface than this slice.

W8A16 is the right scope for one slice: the mechanism is simple and the
integration is clean. W8A8 and W4A16 are explicit follow-ups when
benchmarks justify the additional complexity.

## Numerical correctness

Per-channel symmetric INT8 typically preserves cosine similarity > 0.999 on
quantized vs fp logits for Qwen-class models. We require > 0.99 in tests as
a generous floor.

Verified on Qwen2.5-0.5B-Instruct:

- `tests/unit/test_int8_model_integration.py::test_quantized_logits_close_to_fp_reference`
  — cosine similarity of last-position logits between fp and int8 paths.
- `tests/unit/test_int8_model_integration.py::test_quantized_model_decodes_paris`
  — quantized model produces "Paris" for the canonical prompt.
- `tests/stress/test_int8_load.py::test_int8_first_tokens_match_fp_reference`
  — greedy-decode first-token parity holds across multiple prompts.

The first-token parity claim is the strictest result that holds robustly:
quantization noise eventually flips an argmax decision, so we don't claim
multi-token greedy parity.

## Memory math (Qwen2.5-0.5B, fp16 baseline)

The model has 169 `nn.Linear` modules totaling ~494M params. Default skip
(`{"lm_head"}`) leaves the 136M-param `lm_head` in fp16; the remaining
358M-param Linears are quantized.

| Mode                  | Linear weight bytes | Savings vs fp16 |
|-----------------------|---------------------|-----------------|
| fp16 baseline         | 988 MB              | -               |
| int8 (skip lm_head)   | ~630 MB             | ~36%            |
| int8 (incl. lm_head)  | ~494 MB             | ~50%            |

Verified by `tests/unit/test_int8_model_integration.py::test_quantized_weight_storage_smaller_than_fp`,
which asserts ≥30% savings in default-skip mode.

## Alternatives considered

- **Use bitsandbytes `Linear8bitLt`.** Plug-and-play. Rejected for this
  project: the integration story (load-time module walk + custom Linear
  class) is part of what we're demonstrating. The tradeoff is no fused
  kernel; bitsandbytes ships a CUTLASS-backed dequant-fused matmul that
  outperforms our naive path on small batches.
- **Per-group scales** (group_size = 128 within `in_features`). Modest
  quality improvement; not worth the implementation surface at this size.
- **Asymmetric / zero-point quantization.** Necessary for activation quant
  (heavy-tailed distributions); unnecessary for weights.
- **Quantization-aware fine-tuning.** Out of scope for an inference engine.
- **Float8 (E4M3 / E5M2).** Better numerical fidelity than INT8 on Hopper;
  requires fp8 GEMM hardware which our reference A10 lacks.

## Consequences

- **Positive**:
  - 36% reduction in Linear weight memory at default settings; 50% with
    `quant_lm_head=True`. Frees HBM for K/V cache or larger models.
  - The `Int8Linear` is a clean ~150 LOC module that's easy to test in
    isolation and serves as the foundation for a future fused-kernel or
    W4A16 variant.
  - `ModelRunner.from_pretrained(quant="int8")` is a single-flag opt-in.
- **Negative**:
  - Throughput is ~neutral or slightly negative on small-batch decode
    (dequant overhead on the same matmul). Honest reporting; a fused kernel
    is the path to actual throughput speedup.
  - Loading fp first then quantizing peaks weight memory at ~2x for a
    moment. Acceptable at 0.5B scale; for larger models, streaming
    quantization (load-quantize-discard per shard) would be needed.
  - State-dict serialization of int8 buffers works in PyTorch 2.4+, but we
    don't depend on it (we always quantize on load). Documented as
    out-of-scope.
- **Reversibility**: removing the path is a clean revert. The `quant`
  parameter on `ModelRunner.from_pretrained` defaults to `None`; the
  fp-only path is byte-for-byte unchanged.

## Validation

- **M1 (CPU, fp32)**: 15 unit tests for the standalone module + 5 real-model
  integration tests + 1 stress test pass. Existing 106-test suite stays
  green; the quant-disabled path is unchanged.
- **Real-model parity (Qwen2.5-0.5B, M1)**: cosine sim > 0.99 on logits;
  first-token parity on greedy decode across multiple prompts.
- **CUDA (A10, Qwen2.5-0.5B)**: model-weight HBM **−30.5%** at default
  (skip `lm_head`); throughput within ~6% of fp16 at C=1, neutral at C=4
  and C=8. Numbers in `docs/benchmarks/2026-04-28-int8-weight-quant.md`.

### Weight-tying finding (`quant_lm_head=True`)

The benchmark surfaced a real interaction: Qwen2.5 has
`tie_word_embeddings=True`, so `embed_tokens.weight` and `lm_head.weight`
share storage. Replacing `lm_head` with `Int8Linear` allocates a new int8
buffer but does **not** free the original fp weight (still held by
`embed_tokens`). Net effect on Qwen2.5-0.5B: `quant_lm_head=True` saves
**less** memory (−19.1%) than the default skip (−30.5%).

Default behaviour (skip `lm_head`) is unaffected and remains the right
choice. The flag is preserved for models with untied embeddings; using it
on a tied-embedding model is documented as a no-op or slight regression.
A proper fix would treat `embed_tokens` and `lm_head` as a tied pair —
either untie before quantizing, or replace `embed_tokens` with a
quantized embedding too. Tracked as a follow-up.

## Pointers

- Standalone module: `src/mini_infer/quant/int8.py`.
- Model integration: `src/mini_infer/engine/model_runner.py::from_pretrained`
  (the `quant=` and `quant_lm_head=` flags).
- Unit tests: `tests/unit/test_int8_quant.py`.
- Real-model tests: `tests/unit/test_int8_model_integration.py`.
- Stress: `tests/stress/test_int8_load.py`.
- Earlier ADRs: ADR-005..009 for the engine architecture this slots into.

## Follow-ups

- **Tied-embedding-aware quantization.** Replace `embed_tokens` and
  `lm_head` together (or untie first) so `quant_lm_head=True` actually
  reduces memory on tied-embedding models like Qwen2.
- **Fused dequant-matmul Triton kernel.** Would eliminate the per-call
  dequant overhead and produce a real throughput win. Phase 3b stretch;
  worthwhile only after benchmarks confirm dequant is a meaningful slice
  of step time.
- **W8A8** (full INT8 path with calibrated activations). Real FLOP savings
  on Tensor Cores. Requires SmoothQuant-style outlier handling.
- **W4A16** (GPTQ / AWQ). Bigger memory savings; requires calibration and
  a custom dequant kernel.
- **Streaming load-quantize.** For models too large to load fp before
  quantizing, we'd need to quantize each shard as it arrives. Not needed
  at our model sizes.
