# ADR-002: Starter model is Qwen2.5-0.5B-Instruct

Date: 2026-04-25
Status: Accepted

## Context

Phase 1 needs a real LLM to load and exercise the engine plumbing (tokenizer, model loading, sampling, KV cache, decode loop, golden tests). The choice constrains:

- How much memory we need for development (M1 Pro unified memory; cloud CUDA later).
- Tokenizer behavior, special tokens, and chat template surface area.
- What architectural assumptions our prefill / decode / KV-cache code can make (GQA, RoPE, layer count, head dim).
- License posture for an open-source repo.

The model itself is not the point of this project; the inference engine is. We want something that loads in seconds, runs at a usable speed on M1 Pro for development, and isn't so trivial that it hides bugs.

## Decision

Use `Qwen/Qwen2.5-0.5B-Instruct` as the Phase 1 starter model.

- 494M parameters, ~1GB on disk in safetensors.
- Apache 2.0 license, no usage gates.
- Standard decoder-only transformer with GQA, RoPE, SwiGLU. Representative of the architecture family used by the production-scale models we want to support later.
- Instruction-tuned, so generations look sensible at small scales (helps with manual smoke testing).

## Alternatives Considered

- **TinyLlama-1.1B-Chat**: similar size class, but slightly older architecture (no GQA in the most common checkpoint), and the instruction tuning is weaker than Qwen2.5.
- **microsoft/Phi-3.5-mini-instruct (3.8B)**: stronger model, but 3.8B params is heavier on M1 Pro and overkill for plumbing tests. Reserve for later phases if we want a more capable demo.
- **meta-llama/Llama-3.2-1B-Instruct**: good architecture and quality, but the Llama license requires acceptance and is awkward for a fully open repo.
- **google/gemma-2-2b-it**: solid model, but the 2B size doubles our memory footprint vs. Qwen2.5-0.5B for no Phase 1 benefit; license also requires acceptance.

## Consequences

- **Positive**: small, fast to load and iterate on; permissive license; modern architecture exercises the code paths (GQA, RoPE) we'll lean on in Phase 2/3; Hugging Face hosts it and `transformers` supports it natively.
- **Negative**: at 0.5B, output quality is fine for "the capital of France is..." style sanity checks but not impressive in a public benchmark. We're explicitly accepting this for Phase 1; benchmarks come in Phase 4 and may use a larger model.
- **Re-pick triggers**: re-evaluate the choice if (a) a Phase 2/3 technique requires architectural features Qwen2.5-0.5B lacks (e.g. a Mixture-of-Experts pathway, multi-token prediction head); (b) Phase 4's public benchmark needs a more impressive model to compare credibly with vLLM / SGLang; (c) a clearly better permissively-licensed small model lands.
