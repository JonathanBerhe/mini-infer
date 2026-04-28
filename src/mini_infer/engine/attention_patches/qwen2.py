"""Qwen2 family attention patch for the packed varlen forward path."""

from typing import Any

import torch

from mini_infer.cache.packed_attention import packed_attention_forward
from mini_infer.cache.paged_kv_cache import PagedKVCache


def patch_qwen2(model: Any) -> None:
    """Replace each Qwen2Attention.forward with our packed-attention wrapper."""
    # transformers Qwen2ForCausalLM has model.model.layers; each layer has self_attn.
    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward
        attn.forward = _make_packed_forward(attn, original_forward)


def _make_packed_forward(attn_module: Any, original_forward: Any) -> Any:
    """Build the patched forward closure for varlen packed attention.

    When the caller passes `cu_seqlens_q` in kwargs and the cache is a
    `PagedKVCache`, we run the unified packed path: Q/K/V projections + RoPE,
    write new K/V to the right per-request slots via `append_kv_packed`,
    materialize the full per-request K/V history into packed form, and
    dispatch `packed_attention_forward` (FlashAttention varlen on CUDA, the
    PyTorch reference elsewhere). One forward per scheduler step, regardless
    of whether the in-flight requests are prefilling or decoding.

    Without `cu_seqlens_q`, we fall through to HF's stock attention so non-
    scheduler entry points (golden tests via the older `prefill`/`decode`
    style) keep working unchanged.
    """
    # Imported here so transformers internals aren't pulled at module import time.
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        if cu_seqlens_q is not None and isinstance(past_key_values, PagedKVCache):
            return _packed_attention_path(
                attn_module,
                apply_rotary_pos_emb,
                hidden_states,
                position_embeddings,
                past_key_values,
                cu_seqlens_q,
                kwargs,
            )

        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    return patched_forward


def _packed_attention_path(
    attn_module: Any,
    apply_rotary_pos_emb: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: PagedKVCache,
    cu_seqlens_q: torch.Tensor,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    """Compute Q/K/V, append new K/V to cache, run packed varlen attention.

    The dispatcher inside `packed_attention_forward` handles the K/V read:
    paged-aware FlashAttention on CUDA reads directly from `BlockPool` storage
    via `block_table` (no per-layer gather); the PyTorch reference path
    materializes per-request K/V internally for the SDPA loop.
    """
    del kwargs  # No longer needed; dispatcher computes max_seqlen_q internally.
    # hidden_states is (1, total_q, hidden). The "batch" dim is always 1 here —
    # all per-request boundaries live in cu_seqlens_q.
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)
    query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # Shapes: query_states (1, num_q_heads, total_q, head_dim);
    #         key/value_states (1, num_kv_heads, total_q, head_dim).

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Reshape to packed (total_q, num_*_heads, head_dim) for both the cache
    # append and the varlen attention call.
    new_keys_packed = key_states.transpose(1, 2).squeeze(0).contiguous()
    new_values_packed = value_states.transpose(1, 2).squeeze(0).contiguous()
    queries_packed = query_states.transpose(1, 2).squeeze(0).contiguous()

    past_key_values.append_kv_packed(
        new_keys_packed, new_values_packed, cu_seqlens_q, attn_module.layer_idx
    )

    attn_packed = packed_attention_forward(
        queries_packed, past_key_values, attn_module.layer_idx, cu_seqlens_q
    )
    # attn_packed: (total_q, num_q_heads, head_dim).

    # Reshape back to (1, total_q, num_q_heads * head_dim) for o_proj.
    attn_output = attn_packed.unsqueeze(0).reshape(*input_shape, -1).contiguous()
    attn_output = attn_module.o_proj(attn_output)
    return attn_output, None
