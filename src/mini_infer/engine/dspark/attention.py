"""DSpark drafter attention: injected target context + bidirectional block self-attention.

Every draft layer's keys/values come from TWO sources concatenated together:
projected target-model hidden states (`target_hidden_states`, the injected
context) and this layer's own projection of the draft block's mask-token
embeddings (`hidden_states`). Queries only come from the draft block, so
`q_len <= k_len` always (`k_len = ctx_len + q_len`).

No attention mask is built here. At single-request inference there is always
exactly one active block (never multiple anchors packed into one forward, the
way DeepSpec's training path does), so DeepSpec's own mask reduces to
"attend to everything": every context key is, by construction, from an
already-committed/verified token (never a future one), and the training
mask's same-block-only restriction on draft positions is trivially satisfied
when there is only one block. DeepSpec's own inference call
(`forward_dspark_draft_block`) passes `attention_mask=None` for exactly this
reason. Batched multi-request inference (deferred, see ADR-027 Stage D) is
where multiple blocks would get packed into one call and a real mask would be
needed again.

RoPE needs its own rotation helper because Q and K have different lengths
here: Q is rotated with the LAST `q_len` positions of the shared cos/sin
table (the draft block's own positions), K is rotated with the FULL table
(context positions for its head, draft positions for its tail). This is
`deepspec/modeling/dspark/qwen3/modeling.py`'s local `apply_rotary_pos_emb`
override, not the plain one every other owned model uses.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import rotate_half


def apply_dspark_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for the drafter's asymmetric Q/K lengths (`q_len < k_len = ctx_len + q_len`)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.shape[-2]
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DSparkAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        layer_idx: int,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={num_attention_heads} not a multiple of "
                f"num_key_value_heads={num_key_value_heads}"
            )
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: DSparkDraftCache | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden_states.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_dspark_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(self.layer_idx, k, v)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        out: torch.Tensor = self.o_proj(attn_output)
        return out
