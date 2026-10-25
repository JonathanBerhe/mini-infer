# ADR-002: Starter model is Qwen2.5-0.5B-Instruct

Date: 2026-04-25
Status: Accepted

## Context

Phase 1 needs a real LLM to test the engine, from the tokenizer to the
golden tests. It must load in seconds, run on an M1 Pro, not hide bugs,
and have a license that suits an open-source repository.

## Decision

Use `Qwen/Qwen2.5-0.5B-Instruct`. It has 494M parameters (about 1GB on
disk) and an Apache 2.0 license with no usage gates. It is a standard
decoder-only transformer with GQA, RoPE and SwiGLU, like the larger target
models. Instruction tuning makes manual smoke tests readable.

## Alternatives Considered

- TinyLlama-1.1B-Chat. Its common checkpoint has no GQA, and its tuning is
  weaker.
- microsoft/Phi-3.5-mini-instruct (3.8B). It is too heavy on an M1 Pro.
- meta-llama/Llama-3.2-1B-Instruct. Its license requires acceptance.
- google/gemma-2-2b-it. It doubles the memory, and its license requires
  acceptance.

## Consequences

- `transformers` supports the model natively.
- At 0.5B, the output is good for sanity tests, not for a public
  benchmark.
- Pick again if a later technique needs a missing feature, for example
  MoE. Also pick again for a stronger model in the Phase 4 comparison with
  vLLM and SGLang. The same holds for a better small model with a
  permissive license.
