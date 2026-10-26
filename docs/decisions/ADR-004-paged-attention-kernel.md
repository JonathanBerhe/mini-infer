# ADR-004: Triton paged attention kernel for the decode path

Date: 2026-04-25
Status: Accepted

## Context

Slice 2.1 (ADR-003) stored the KV cache in blocks, but
`PagedKVCache.update()` still copied the blocks into one contiguous tensor
for the stock HF attention. This copy (materialization) runs for every layer
at every decode step, so paging gave no gain. Only an attention layer that
reads K/V from the blocks removes it, and HF SDPA has none.

## Decision

1. Decode only. Prefill runs once per request and keeps materialization,
   because a paged prefill kernel with causal masks is not worth it at our
   scale.
2. A Triton kernel of about 80 lines: online softmax (one pass over the keys)
   on block storage. `HEAD_DIM` and `BLOCK_SIZE` are compile-time constants,
   and `seq_len` is a runtime argument.
3. A PyTorch reference, `paged_attention_decode_torch`, with the same
   contract. It runs on any device and is the fallback.
4. Per-architecture monkey-patches. `REGISTRY` in
   `attention_patches/__init__.py` maps a model class name substring to a
   patch. `engine/attention_patch.py` applies it, or falls back to
   materialization. Slice 2.2 ships `attention_patches/qwen2.py`.
5. `cache/paged_attention.supports_paged_kernel(device)` is the only place
   that decides on the kernel. It replaces scattered `device == "cuda"`
   strings.
6. `use_paged_kernel: bool = True` on `ModelRunner.from_pretrained` is an A/B
   switch. Benchmarks pass False to compare with materialization.

## Alternatives considered

- Use the vLLM kernel as a dependency. It adds a heavy dependency and does
  not show our own kernel work.
- Runtime `head_dim` and `block_size`. Compile-time constants keep the
  kernel simpler.
- Patch the HF `Cache`. When `Cache.update()` returns, HF attention already
  wants contiguous K/V.
- Keep materialization. It defeats the purpose of the block manager.
- Subclass the HF model classes. We then own a copy of the Qwen2 code and
  must sync it with transformers.

## Consequences

- On CUDA, decode reads K/V from the blocks with no copy.
- A new family (Llama, Mistral, Gemma) needs about 30 lines and a registry
  entry.
- The kernel is a CUDA speedup, not a CUDA requirement.
- The gain on 0.5B and A10 is small: 3–8% median across seq_lens 16–1024.
  Projections and the MLP dominate decode time, not attention
  (`docs/benchmarks/2026-04-25-paged-decode.md`).
- Two paths need upkeep. A new transformers major version can break the
  patch, and the smoke and golden tests catch that.
- `use_paged_kernel=False` restores the Slice 2.1 path, which stays correct.
- On Modal A10, the patched Qwen2.5-0.5B generates
  `" Paris. It is the largest city in"`, token-for-token equal to the
  CPU/fp32 reference. `num_free_blocks` returns to 1024 afterwards.
- On M1, 48 tests pass and 1 `requires_cuda` test skips. In
  `tests/unit/cache/test_paged_attention.py`, the reference matches SDPA for
  block-aligned, partial last block, GQA (14/2) and seq_len=1 cases.
