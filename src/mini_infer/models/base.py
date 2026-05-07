"""Contract every owned causal LM honors.

Tighter than HF's `PreTrainedModel.forward`: we pass exactly what the
scheduler builds (packed input_ids, position_ids, paged cache,
varlen boundaries) and return the raw logits tensor — no
`CausalLMOutputWithPast` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import torch
from torch import nn

from mini_infer.cache.block_pool import LayerAttentionSpec, StreamSpec

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class ModelConfigLike(Protocol):
    """Marker for per-model config dataclasses; each declares `from_hf` itself."""


@dataclass(frozen=True)
class KVCacheDims:
    """Dimensions the `BlockPool` needs to size storage for an architecture.

    Every owned model exposes these through `BaseCausalLM.kv_cache_dims` so
    the model runner can build the pool without re-loading HF's config.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int


class BaseCausalLM(nn.Module):
    """Common contract for owned causal LMs.

    Subclasses declare:
      - `HF_ARCHITECTURE`: the string in HF `config.architectures[0]`
        used for registry lookup (e.g. `"Qwen2ForCausalLM"`).
      - `Config`: a per-model dataclass with `from_hf(hf_config)`.
      - `load_weights(model, hf_state_dict)`: in-place state_dict copy
        with any name remapping.
      - `kv_cache_dims`: the architectural dims for the block pool.
      - `expected_missing_state_keys()` (optional): aliases of tied
        weights or other parameters that `load_state_dict` will report
        as missing but that aren't a real load failure.
    """

    HF_ARCHITECTURE: ClassVar[str] = ""
    Config: ClassVar[type]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Returns packed logits `(1, total_q, vocab_size)`."""
        raise NotImplementedError

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        raise NotImplementedError

    def per_layer_attention(self) -> list[LayerAttentionSpec]:
        """Per-layer attention pattern for the block pool.

        Default: every layer is full causal — matches Qwen2 / Llama. Models
        with sliding-window layers (Gemma 3+, Mistral SWA, ...) override to
        return a list of `"full"` and `("sliding", window_size)` entries.
        """
        return ["full"] * self.kv_cache_dims.num_layers

    def per_layer_kv_shape(self) -> list[tuple[int, int]] | None:
        """Per-layer (num_kv_heads, head_dim) for the block pool.

        Default: None — homogeneous, the pool synthesizes the list from
        `kv_cache_dims`. Models with per-layer-type heterogeneous attention
        (Gemma 4 31B, MLA, V4) override to return an explicit list.
        """
        return None

    def per_layer_streams(self) -> list[list[StreamSpec]] | None:
        """Per-layer storage descriptor for the block pool.

        Default: None — the pool synthesizes the legacy `["k", "v"]` pair
        per layer (with shape from `per_layer_kv_shape()` or
        `kv_cache_dims`). Models that need a non-K/V stream layout —
        DeepSeek-V2/V3 MLA stores `compressed_kv` (1xkv_lora_rank) plus
        `k_rope` (1xqk_rope_head_dim) per layer rather than per-head K/V
        — override to return the explicit per-layer descriptor.
        """
        return None

    def expected_missing_state_keys(self) -> set[str]:
        """Names present in the module hierarchy but expected to be absent
        from the HF state_dict (e.g. aliases of tied weights).

        `load_weights` should subtract these from `load_state_dict`'s
        reported `missing` before declaring a load failure. Default: none.
        """
        return set()

    def required_attention_backend(self) -> str | None:
        """Name of an attention backend the model REQUIRES, overriding user choice.

        Returned when no installed kernel handles the model's attention
        shape — e.g. Gemma 4 31B has `head_dim=512` on full layers, which
        FlashAttention 2 and FlashInfer's prefill kernel both reject.
        `ModelRunner.from_pretrained` honors this (with a log line so the
        override is visible) and the matching value gets threaded into
        `BlockPool(attention_backend=...)`. Default: `None` — model is
        compatible with any user-selected backend. Mirrors vLLM's
        `verify_and_update_config` pattern (`gemma4` forces TRITON_ATTN
        when `max_head_dim > 256`).
        """
        return None

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError
