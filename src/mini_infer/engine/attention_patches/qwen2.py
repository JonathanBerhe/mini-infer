"""Qwen2 family attention patch for the paged decode kernel."""

from typing import Any

import torch

from mini_infer.cache.paged_attention import (
    paged_attention_decode_batched,
    supports_paged_kernel,
)
from mini_infer.cache.paged_kv_cache import PagedKVCache


def patch_qwen2(model: Any) -> None:
    """Replace each Qwen2Attention.forward with our paged-decode-aware wrapper."""
    # transformers Qwen2ForCausalLM has model.model.layers; each layer has self_attn.
    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward
        attn.forward = _make_paged_forward(attn, original_forward)


def _make_paged_forward(attn_module: Any, original_forward: Any) -> Any:
    """Build the patched forward closure that owns the decode fast path.

    Mirrors transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward up to
    Q/K/V projections and RoPE; then either calls the batched paged kernel
    (decode, q_len=1, B>=1) or delegates to original_forward (prefill,
    q_len > 1) which uses materialization through `update()`.
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
        q_len = hidden_states.shape[1]

        # Decode fast path: single-token query per request, populated paged cache,
        # kernel-capable device. The device check is defensive — the runner only
        # installs this patch on kernel-capable devices today, but the explicit
        # gate documents intent and survives any future change to that policy.
        if (
            q_len == 1
            and isinstance(past_key_values, PagedKVCache)
            and past_key_values.batch_size > 0
            and all(n > 0 for n in past_key_values.seq_lens_list())
            and supports_paged_kernel(hidden_states.device)
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)
            query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Append the new K/V to blocks; do NOT materialize.
            past_key_values.append_kv(key_states, value_states, attn_module.layer_idx)

            # Slice this layer's K/V pool: (num_blocks, block_size, num_kv_heads, head_dim).
            pool_storage = past_key_values._pool.storage
            k_pool_layer = pool_storage[attn_module.layer_idx, 0]
            v_pool_layer = pool_storage[attn_module.layer_idx, 1]
            block_tables = past_key_values.block_tables_per_request_tensor(hidden_states.device)
            seq_lens = past_key_values.seq_lens_list()

            # query_states is (B, num_q_heads, q_len=1, head_dim); kernel wants
            # (B, num_q_heads, head_dim).
            q_decode = query_states.squeeze(2)
            attn_decode = paged_attention_decode_batched(
                q_decode, k_pool_layer, v_pool_layer, block_tables, seq_lens
            )
            # Restore q_len dim so the standard reshape below produces
            # (B, q_len, num_q_heads * head_dim).
            attn_output = attn_decode.unsqueeze(1)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, None

        # Prefill or empty-cache: defer to the original HF attention path.
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    return patched_forward
