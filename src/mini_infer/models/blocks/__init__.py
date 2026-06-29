"""Canonical building blocks shared across owned model families."""

from mini_infer.cache.block_pool import StreamSpec
from mini_infer.models.blocks.csa import CSAAttention
from mini_infer.models.blocks.deepseek_v4_decoder_layer import DeepseekV4DecoderLayer
from mini_infer.models.blocks.geglu import GeGLU
from mini_infer.models.blocks.gemma4_decoder_layer import Gemma4DecoderLayer
from mini_infer.models.blocks.gemma_decoder_layer import GemmaDecoderLayer
from mini_infer.models.blocks.gemma_rmsnorm import GemmaRMSNorm
from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN, GlmNoAuxTcGate
from mini_infer.models.blocks.gqa import GroupedQueryAttention
from mini_infer.models.blocks.hash_routed_gate import HashRoutedGate
from mini_infer.models.blocks.hash_routed_moe_ffn import HashRoutedMoEFFN
from mini_infer.models.blocks.hca import HCAAttention
from mini_infer.models.blocks.hyper_connections import HyperConnections, hc_split_sinkhorn
from mini_infer.models.blocks.mixtral_decoder_layer import MixtralDecoderLayer
from mini_infer.models.blocks.mixtral_moe import MixtralExpert, MoEFFN
from mini_infer.models.blocks.mla import MLAAttention
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from mini_infer.models.blocks.swiglu import SwiGLU
from mini_infer.models.blocks.transformer_block import TransformerBlock
from mini_infer.models.blocks.v4 import (
    AttentionSink,
    GroupedOutputProjection,
    LightningIndexer,
    TokenLevelCompressor,
)

__all__ = [
    "AttentionSink",
    "CSAAttention",
    "DeepseekV4DecoderLayer",
    "GeGLU",
    "Gemma4DecoderLayer",
    "GemmaDecoderLayer",
    "GemmaRMSNorm",
    "GlmDsaIndexer",
    "GlmMoeFFN",
    "GlmNoAuxTcGate",
    "GroupedOutputProjection",
    "GroupedQueryAttention",
    "HCAAttention",
    "HashRoutedGate",
    "HashRoutedMoEFFN",
    "HyperConnections",
    "LightningIndexer",
    "MLAAttention",
    "MixtralDecoderLayer",
    "MixtralExpert",
    "MoEFFN",
    "RMSNorm",
    "RotaryEmbedding",
    "StreamSpec",
    "SwiGLU",
    "TokenLevelCompressor",
    "TransformerBlock",
    "apply_rotary_pos_emb",
    "hc_split_sinkhorn",
    "rotate_half",
]
