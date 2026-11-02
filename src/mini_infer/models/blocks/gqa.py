"""Grouped-Query Attention with native packed-varlen path.

This block replaces the HF-attention monkey-patch we used to apply at
runtime. The forward computes Q/K/V projections, applies RoPE,
appends new K/V to the paged cache, dispatches the packed varlen
attention kernel, and applies the output projection. One layer
forward = one call to `packed_attention_forward`.

`o_proj` aligns with the HF Llama/Qwen2 name. The q, k and v projections
live in one `qkv_proj` (`QKVParallelLinear`), so a step runs one GEMM for
the three; `merge_qkv_weights` concatenates the HF `q_proj`, `k_proj` and
`v_proj` tensors into it at load time.

Tensor parallelism
------------------
The Q/K/V projection is column-parallel along the head axis (each rank
holds a contiguous slice of heads of each of q, k and v). The output
projection is row-parallel along its input. This is the standard
Megatron pairing for self-attention: one all-reduce per block, none in
between. The softmax / SDPA kernel itself operates per-head and so is
unchanged under TP. At `world_size=1` both wrappers reduce to plain
`nn.Linear`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_infer.cache.packed_attention import packed_attention_forward
from mini_infer.distributed.linear import QKVParallelLinear, RowParallelLinear
from mini_infer.models.blocks.rope import apply_rotary_pos_emb
from mini_infer.models.blocks.rope_write_kernel import (
    rope_and_write_kv,
    supports_rope_write_kernel,
)

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_bias: bool,
        layer_idx: int,
        query_scale: float | None = None,
        q_norm: nn.Module | None = None,
        k_norm: nn.Module | None = None,
        attention_k_eq_v: bool = False,
        v_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.attention_k_eq_v = attention_k_eq_v
        # One column-parallel layer holds q, k and v, so a step runs one GEMM
        # for the three. Each rank gets `num_*_heads // world_size` heads of
        # each; the layer checks that both head counts divide `world_size`.
        # Gemma 4 full layers (`attention_k_eq_v=True`) reuse the k projection
        # as V, so their layer has no v rows and the loader skips `v_proj`.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=qkv_bias,
            with_v=not attention_k_eq_v,
        )
        # Kept apart from the layer: a quantizer can swap the layer for one
        # that only knows the merged output.
        self._qkv_sizes = self.qkv_proj.split_sizes
        # Row-parallel output projection: input dim is the sharded
        # head-merged activation, output is the replicated hidden state.
        # Triggers exactly one all-reduce per attention block.
        self.o_proj = RowParallelLinear(num_q_heads * head_dim, hidden_size, bias=False)
        # Optional per-head norm on Q and K AFTER projection and BEFORE RoPE.
        # Gemma 3+ uses GemmaRMSNorm(head_dim) here; the names `q_norm` and
        # `k_norm` align with HF's parameter naming so weight loading is
        # identity rename. None = pass-through (Qwen2, Llama).
        self.q_norm = q_norm
        self.k_norm = k_norm
        # Gemma 4 also normalizes V with an unscaled RMSNorm AFTER capturing
        # V from k_proj output (or v_proj output). None = pass-through.
        self.v_norm = v_norm
        # `query_scale` overrides the default `1/sqrt(head_dim)` softmax
        # scale. Gemma 3 uses `1/sqrt(query_pre_attn_scalar)`. Gemma 4 sets
        # this to 1.0 (q_norm/k_norm absorb the magnitude). `None` keeps the
        # default (Qwen2, Llama).
        self._query_scale = query_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (1, total_q, hidden). The leading "1" is the engine's
        # packed-batch convention; per-request boundaries live in cu_seqlens_q.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # (1, total_q, num_*_heads, head_dim): the packed layout, head dim last.
        # On CUDA the three are views into the one projection output.
        q, k, v = self._project(hidden_states)
        query_states = q.view(hidden_shape)
        key_states = k.view(hidden_shape)
        # Gemma 4 full layers: V reuses the post-projection K tensor BEFORE
        # k_norm and BEFORE RoPE. Capturing here keeps `value_states`
        # pointing at the un-normalized k output even after the
        # `key_states = self.k_norm(...)` reassignment below (Python rebinds
        # the name; the original tensor stays alive via `value_states`).
        value_states = key_states if v is None else v.view(hidden_shape)

        # Per-head Q/K norm (Gemma 3+). Acts on the last dim (head_dim). The
        # fused norm kernel wants rows it can view as a 2-D matrix, which the
        # views into the merged output are not, so they become contiguous first.
        if self.q_norm is not None:
            query_states = self.q_norm(query_states.contiguous())
        if self.k_norm is not None:
            key_states = self.k_norm(key_states.contiguous())

        # v_norm is applied AFTER V is captured (so for k_eq_v layers, it
        # operates on the un-k_normed, un-RoPE'd K tensor). Gemma 4's v_norm
        # uses `with_scale=False` so it's a pure RMS rescale.
        if self.v_norm is not None:
            value_states = self.v_norm(value_states)

        cos, sin = position_embeddings
        queries_packed = self._rope_and_append(
            query_states, key_states, value_states, cos, sin, past_key_values, cu_seqlens_q
        )

        attn_packed = packed_attention_forward(
            queries_packed,
            past_key_values,
            self.layer_idx,
            cu_seqlens_q,
            softmax_scale=self._query_scale,
        )
        # attn_packed: (total_q, num_q_heads, head_dim).

        attn_output = attn_packed.unsqueeze(0).reshape(*input_shape, -1).contiguous()
        out: torch.Tensor = self.o_proj(attn_output)
        return out

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """`(q, k, v)` of the packed hidden states; `v` is None when k doubles as v."""
        proj = self.qkv_proj
        if isinstance(proj, QKVParallelLinear):
            return proj.project(hidden_states)
        # A replacement layer (the int8 quantizer swaps in `Int8Linear`) gives
        # the merged output; split it the way the merged layer would.
        parts = torch.split(proj(hidden_states), list(self._qkv_sizes), dim=-1)
        return parts[0], parts[1], parts[2] if len(parts) == 3 else None

    def _rope_and_append(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate q and k, append the new k and v to the cache, return packed q.

        The inputs are `(1, total_q, heads, head_dim)`. On CUDA with an
        uncompressed pool one Triton kernel (`rope_write_kernel.py`) rotates q
        and k and stores k and v straight into the pool pages, with the
        rounding of the PyTorch path. Other devices rotate with
        `apply_rotary_pos_emb` and append through `append_kv_packed`.
        """
        pool = past_key_values._pool
        if pool.kv_quant is None:
            q_packed = query_states.squeeze(0)
            k_packed = key_states.squeeze(0)
            v_packed = value_states.squeeze(0)
            cos_rows, sin_rows = cos.squeeze(0), sin.squeeze(0)
            if supports_rope_write_kernel(q_packed, k_packed, v_packed, cos_rows, sin_rows):
                k_pool, v_pool = pool.storage_for_layer(self.layer_idx)
                rotated: list[torch.Tensor] = []

                def write(block_ids: torch.Tensor, slots: torch.Tensor) -> None:
                    rotated.append(
                        rope_and_write_kv(
                            q_packed,
                            k_packed,
                            v_packed,
                            cos_rows,
                            sin_rows,
                            k_pool,
                            v_pool,
                            block_ids,
                            slots,
                        )
                    )

                past_key_values.append_kv_with(cu_seqlens_q, self.layer_idx, write)
                return rotated[0]

        # The PyTorch path: `apply_rotary_pos_emb` wants the head dim at index 1.
        query_states, key_states = apply_rotary_pos_emb(
            query_states.transpose(1, 2), key_states.transpose(1, 2), cos, sin
        )
        new_keys_packed = key_states.transpose(1, 2).squeeze(0).contiguous()
        new_values_packed = value_states.squeeze(0).contiguous()
        queries_packed = query_states.transpose(1, 2).squeeze(0).contiguous()
        past_key_values.append_kv_packed(
            new_keys_packed, new_values_packed, cu_seqlens_q, self.layer_idx
        )
        return queries_packed


def merge_qkv_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Turn the `q_proj`, `k_proj` and `v_proj` entries of an HF state dict into
    the `qkv_proj` entry of every `GroupedQueryAttention` in `model`, in place.

    For each block the weights, and the biases when present, are concatenated in
    the order of `QKVParallelLinear`: q, k, then v for a block with a v projection.
    A block whose entries are not all present is left alone, so the loader
    reports what is missing. Entries already named `qkv_proj` stay as they are.
    """
    for name, module in model.named_modules():
        if not isinstance(module, GroupedQueryAttention):
            continue
        prefix = f"{name}." if name else ""
        projections = ["q_proj", "k_proj"]
        if not module.attention_k_eq_v:
            projections.append("v_proj")
        for suffix in ("weight", "bias"):
            keys = [f"{prefix}{projection}.{suffix}" for projection in projections]
            if all(key in state_dict for key in keys):
                merged = torch.cat([state_dict.pop(key) for key in keys])
                state_dict[f"{prefix}qkv_proj.{suffix}"] = merged
