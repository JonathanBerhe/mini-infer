# ADR-055: Kimi K3.1 Pro port (Gated DSA on the K3 stack)

Date: 2026-11-04
Status: Accepted

## Context

Moonshot released Kimi K3.1 Pro (`moonshotai/Kimi-K3.1-Pro`) on 2026-11-04.
The announcement called it a post-trained K3 with the same architecture, and
ADR-049 planned to load it through the K3 path. The release is different.
Its 24 MLA layers are Gated DeepSeek Sparse Attention (DSA) layers. Before
the attention, an indexer picks for each query the 2048 earlier tokens that
the attention can read.

The checkpoint ships a configuration file but no modeling code, so no
reference exists to run. The configuration adds `text_config.dsa_config`,
and the weights add five indexer tensors per DSA layer. The open pull
requests that add K3.1 to vLLM and SGLang are the only implementations.

## Decision

1. K3.1 is the K3 stack with one more configuration switch, not a new
   family. `KimiLinearConfig.from_hf` reads `dsa_config`, and each MLA layer
   gets a `KimiDsaIndexer` (`kimi_linear/dsa_indexer.py`) when the switch is
   on. `KimiK31ForConditionalGeneration` is the K3 wrapper with a
   configuration that requires `dsa_config`.
2. The indexer math is the DeepSeek-V3.2 indexer with the K3.1 values from
   SGLang's K3.1 support (sgl-project/sglang#48415). The query reads the
   output of `q_a_layernorm`. The key is a LayerNorm in fp32 with eps
   `rms_norm_eps`. The head gate is rounded to the model dtype. There is no
   RoPE and no Hadamard rotation. The scores run in fp32.
3. `from_hf` rejects every value that this port does not implement:
   `index_topk_freq` other than 1, a non-zero `index_rope_dim`, a key norm
   other than LayerNorm, and DSA layers that differ from the MLA layers.
   SGLang rejects the same values. With a frequency of 1, vLLM's IndexShare
   rule never skips a layer, so `index_skip_topk_offset` has no effect.
4. The layer masks the dense attention scores with the selection. When a
   request holds at most `index_topk` tokens, the selection keeps every key
   and the layer skips the scoring. The state cache keeps one indexer key
   per token next to the compressed KV.
5. With no reference, the tests pin three facts. The indexer matches its
   formula. The layer matches a loop over the kept keys. K3.1 equals K3 bit
   for bit while every key stays. The tests accept either side of a tie at
   the top-k boundary, because engines break ties in different ways.

## Consequences

- `scripts/kimi_linear/check_kimi_k3_checkpoint.py` passes on the released
  K3.1 Pro checkpoint, from its shard headers. All 249876 text tensors land
  on a parameter of the same shape, the 120 BF16 indexer tensors included.
  The script now treats a subclass of the K3 wrapper as K3.
- The selection can differ from the engines' selection at a tie, because
  they quantize the indexer to FP8. The first GPU run must compare the
  output with vLLM or SGLang on the same prompts.
- The dense mask saves no compute. A kernel that reads only the kept keys
  is a follow-up.
- No real-weight run exists. The 2.8T checkpoint still needs a streaming
  loader and a multi-GPU script (ADR-049).
