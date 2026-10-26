"""Building blocks that two or more model families share.

A block that only one family uses lives in that family's package instead
(for example `mini_infer.models.deepseek_v4.csa`).
"""

from mini_infer.cache.block_pool import StreamSpec
from mini_infer.models.blocks.gemma_rmsnorm import GemmaRMSNorm
from mini_infer.models.blocks.gqa import GroupedQueryAttention
from mini_infer.models.blocks.mla import MLAAttention
from mini_infer.models.blocks.moe.glm_moe_gate import GlmMoeFFN, GlmNoAuxTcGate
from mini_infer.models.blocks.moe.mixtral_moe import MixtralExpert, MoEFFN
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from mini_infer.models.blocks.swiglu import SwiGLU
from mini_infer.models.blocks.transformer_block import TransformerBlock

__all__ = [
    "GemmaRMSNorm",
    "GlmMoeFFN",
    "GlmNoAuxTcGate",
    "GroupedQueryAttention",
    "MLAAttention",
    "MixtralExpert",
    "MoEFFN",
    "RMSNorm",
    "RotaryEmbedding",
    "StreamSpec",
    "SwiGLU",
    "TransformerBlock",
    "apply_rotary_pos_emb",
    "rotate_half",
]
