"""MiniMax-M3 family: owned text-model implementation (MSA, arXiv 2606.13392).

Text-only port (M3 is multimodal; vision is out of scope). Architecturally this
is a standard GQA transformer with per-head Gemma QK-norm and partial RoPE, an
MoE that reuses GLM's `noaux_tc` sigmoid gate, and MiniMax Sparse Attention:

  - **MSA**: on the sparse layers a `MiniMaxM3Indexer` scores 128-token KV blocks
    and selects the top-k (+ the local block) per query; the selection becomes a
    per-request additive mask (`build_block_mask`) that REPLACES the causal mask
    in the `torch` attention path. Full KV is kept (standard `PagedKVCache`), only
    the attended blocks are chosen; see `minimax_m3_indexer.py` and the
    `block_mask` path in `cache/packed_attention.py`.
  - **MoE**: `GlmMoeFFN` (DeepSeek-V3-style sigmoid gate + aux-loss-free selection
    bias + shared expert) with `swigluoai` experts.
  - **swigluoai**: the clamped GLU used by the dense MLP and every expert.

Layers `[0, first_dense_layers)` are dense full-attention + dense SwiGLU; the rest
are MSA + MoE. RoPE is NeoX/non-interleaved on the first `rotary_dim` dims
(`apply_rotary_pos_emb_partial`). Multi-token-prediction (MTP) draft weights are
dropped at load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import StreamSpec
from mini_infer.cache.packed_attention import packed_attention_torch
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import RotaryEmbedding, SwiGLU
from mini_infer.models.blocks.activations import swigluoai
from mini_infer.models.blocks.gemma_rmsnorm import GemmaRMSNorm
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN
from mini_infer.models.blocks.minimax_m3_indexer import MiniMaxM3Indexer
from mini_infer.models.blocks.rope import apply_rotary_pos_emb

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class MiniMaxM3Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    dense_intermediate_size: int  # dense-layer FFN width
    moe_intermediate_size: int  # per-expert width
    shared_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    tie_word_embeddings: bool
    # MSA indexer.
    index_block_size: int
    index_topk_blocks: int
    index_n_heads: int
    index_head_dim: int
    index_local_blocks: int
    # Layers [0, first_dense_layers) are dense (full attn + dense MLP); the rest
    # are MSA + MoE.
    first_dense_layers: int
    expert_dtype: str = "bf16"

    @classmethod
    def from_hf(cls, hf_config: Any) -> MiniMaxM3Config:
        text = getattr(hf_config, "text_config", hf_config) or hf_config
        rope_params = getattr(text, "rope_parameters", None) or {}
        rope_theta = float(
            rope_params.get("rope_theta") or getattr(text, "rope_theta", None) or 5_000_000.0
        )
        head_dim = int(
            getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
        )
        rotary_dim = int(
            getattr(text, "rotary_dim", None)
            or round(getattr(text, "partial_rotary_factor", 0.5) * head_dim)
        )
        # Dense/sparse split: HF encodes it via first_k_dense_replace or the
        # sparse/moe frequency arrays; layer i is dense iff i < first_dense.
        first_dense = getattr(text, "first_k_dense_replace", None)
        if first_dense is None:
            mlp_types = getattr(text, "mlp_layer_types", None)
            first_dense = sum(1 for t in mlp_types if t == "dense") if mlp_types else 0
        return cls(
            vocab_size=text.vocab_size,
            hidden_size=text.hidden_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=head_dim,
            dense_intermediate_size=int(
                getattr(text, "dense_intermediate_size", None) or text.intermediate_size
            ),
            moe_intermediate_size=int(text.intermediate_size),
            shared_intermediate_size=int(
                getattr(text, "shared_intermediate_size", None) or text.intermediate_size
            ),
            n_routed_experts=int(getattr(text, "num_local_experts", None) or text.n_routed_experts),
            num_experts_per_tok=int(text.num_experts_per_tok),
            n_shared_experts=int(getattr(text, "n_shared_experts", 1) or 0),
            n_group=int(getattr(text, "n_group", 1)),
            topk_group=int(getattr(text, "topk_group", 1)),
            routed_scaling_factor=float(getattr(text, "routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
            rms_norm_eps=float(getattr(text, "rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            rotary_dim=rotary_dim,
            tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
            index_block_size=int(getattr(text, "index_block_size", None) or 128),
            index_topk_blocks=int(getattr(text, "index_topk_blocks", None) or 16),
            index_n_heads=int(getattr(text, "index_n_heads", None) or text.num_key_value_heads),
            index_head_dim=int(getattr(text, "index_head_dim", None) or 128),
            index_local_blocks=int(getattr(text, "index_local_blocks", 1)),
            first_dense_layers=int(first_dense),
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_dense_layers

    def is_sparse_attn_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_dense_layers


class _MiniMaxM3Attention(nn.Module):
    """GQA with per-head Gemma QK-norm + partial RoPE; MSA block mask on sparse layers."""

    def __init__(self, cfg: MiniMaxM3Config, layer_idx: int, *, sparse: bool) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.layer_idx = layer_idx
        self.softmax_scale = cfg.head_dim**-0.5
        nqh, nkvh, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = ColumnParallelLinear(cfg.hidden_size, nqh * d, bias=False)
        self.k_proj = ColumnParallelLinear(cfg.hidden_size, nkvh * d, bias=False)
        self.v_proj = ColumnParallelLinear(cfg.hidden_size, nkvh * d, bias=False)
        self.o_proj = RowParallelLinear(nqh * d, cfg.hidden_size, bias=False)
        # Per-head Gemma (1+w) RMSNorm on q and k, pre-RoPE (v un-normed).
        self.q_norm = GemmaRMSNorm(d, eps=cfg.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(d, eps=cfg.rms_norm_eps)
        # Indexer is replicated (small: 4 heads); the block selection is global
        # per query, so no TP sharding of the selection is needed.
        self.indexer: MiniMaxM3Indexer | None = (
            MiniMaxM3Indexer(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.index_n_heads,
                head_dim=cfg.index_head_dim,
                block_size=cfg.index_block_size,
                topk_blocks=cfg.index_topk_blocks,
                local_blocks=cfg.index_local_blocks,
                rms_norm_eps=cfg.rms_norm_eps,
            )
            if sparse
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # Per-head QK-norm BEFORE transpose/RoPE (v un-normed), matching HF.
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        keys_packed = k.transpose(1, 2).squeeze(0).contiguous()
        values_packed = v.transpose(1, 2).squeeze(0).contiguous()
        queries_packed = q.transpose(1, 2).squeeze(0).contiguous()
        # Named-stream API (not the legacy K/V path): the index_k stream on sparse
        # layers makes the pool layout non-legacy, so k/v are named streams too.
        # append "k" on layer 0 triggers per-step block allocation.
        past_key_values.append_stream_packed(keys_packed, cu_seqlens_q, self.layer_idx, "k")
        past_key_values.append_stream_packed(values_packed, cu_seqlens_q, self.layer_idx, "v")

        block_mask: list[torch.Tensor] | None = None
        if self.indexer is not None:
            block_mask = self.indexer.forward_cached(
                hidden_states,
                cos,
                sin,
                past_key_values,
                cu_seqlens_q,
                self.layer_idx,
                dtype=queries_packed.dtype,
            )

        keys_full, cu_seqlens_k, _ = past_key_values.materialize_packed_stream(self.layer_idx, "k")
        values_full, _, _ = past_key_values.materialize_packed_stream(self.layer_idx, "v")
        attn_packed = packed_attention_torch(
            queries_packed,
            keys_full,
            values_full,
            cu_seqlens_q,
            cu_seqlens_k,
            self.softmax_scale,
            block_mask=block_mask,
        )
        attn_output = attn_packed.unsqueeze(0).reshape(*input_shape, -1).contiguous()
        out: torch.Tensor = self.o_proj(attn_output)
        return out


class _MiniMaxM3DecoderLayer(nn.Module):
    """Pre-norm block: (dense full-attn | MSA) + (dense SwiGLU | MoE), both swigluoai."""

    def __init__(self, cfg: MiniMaxM3Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = _MiniMaxM3Attention(
            cfg, layer_idx, sparse=cfg.is_sparse_attn_layer(layer_idx)
        )
        if cfg.is_moe_layer(layer_idx):
            self.mlp: nn.Module = GlmMoeFFN(
                hidden_size=cfg.hidden_size,
                moe_intermediate_size=cfg.moe_intermediate_size,
                n_routed_experts=cfg.n_routed_experts,
                top_k=cfg.num_experts_per_tok,
                n_shared_experts=cfg.n_shared_experts,
                n_group=cfg.n_group,
                topk_group=cfg.topk_group,
                norm_topk_prob=cfg.norm_topk_prob,
                routed_scaling_factor=cfg.routed_scaling_factor,
                expert_dtype=cfg.expert_dtype,
                activation=swigluoai,
            )
        else:
            self.mlp = SwiGLU(cfg.hidden_size, cfg.dense_intermediate_size, activation=swigluoai)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q)
        hidden_states = residual + x
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        out: torch.Tensor = residual + x
        return out


class _MiniMaxM3InnerModel(nn.Module):
    """Embedding + N decoder layers + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: MiniMaxM3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [_MiniMaxM3DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


# FFN weights arrive fused (HF in-memory `state_dict()`): the dense MLP and the
# shared expert stack gate+up into one `gate_up_proj` (out dim 2*I; gate = first
# half, up = second, matching swigluoai's `chunk(2, -1)`), and the routed experts
# stack all E into 3D `experts.gate_up_proj` (E, 2*I, H) / `experts.down_proj`
# (E, H, I). Split into our SwiGLU (`gate_proj/up_proj/down_proj`) and MixtralExpert
# (`w1`=gate, `w3`=up, `w2`=down) params. A separate-projection checkpoint passes
# through (dense) or renames (shared / per-expert).
_DENSE_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.gate_up_proj\.weight$")
_EXPERT_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.gate_up_proj$")
_EXPERT_DOWN_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.down_proj$")
_PER_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)
_PER_EXPERT_W = {"gate": "w1", "up": "w3", "down": "w2"}
_SHARED_GATE_UP_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.shared_experts\.gate_up_proj\.weight$"
)
_SHARED_EXPERT_RENAME: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.mlp\.shared_experts\.gate_proj\.weight$"), r".mlp.shared_experts.w1.weight"),
    (re.compile(r"\.mlp\.shared_experts\.down_proj\.weight$"), r".mlp.shared_experts.w2.weight"),
    (re.compile(r"\.mlp\.shared_experts\.up_proj\.weight$"), r".mlp.shared_experts.w3.weight"),
]
# Indexer projections: self_attn.index_{q,k}_{proj,norm} -> self_attn.indexer.{q,k}_{proj,norm}.
_INDEXER_RE = re.compile(r"\.self_attn\.index_(q|k)_(proj|norm)\.")
# Drop: vision / MTP / multimodal.
_DROP_RE = re.compile(r"(^|\.)(vision_tower|multi_modal_projector|patch_merge_mlp|mtp)\.")


@register_model
class MiniMaxM3ForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "MiniMaxM3SparseForConditionalGeneration"
    Config: ClassVar[type] = MiniMaxM3Config

    def __init__(self, cfg: MiniMaxM3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _MiniMaxM3InnerModel(cfg)
        # Full RoPE over the whole head_dim (theta 5e6). The transformers M3
        # reference uses a full head_dim rope table under rope_type="default"
        # (inv_freq has head_dim/2 non-zero freqs); the config's rotary_dim /
        # partial_rotary_factor are not wired into it, so this matches HF exactly.
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta)
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, gather_output=True
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.model.layers:
            x = layer(x, (cos, sin), past_key_values, cu_seqlens_q)
        x = self.model.norm(x)
        logits: torch.Tensor = self.lm_head(x)
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
        )

    def per_layer_streams(self) -> list[list[StreamSpec]]:
        # Standard K/V per layer; sparse layers add a single-head index-K stream
        # (RoPE'd index keys) so decode scores against the full history. index_k
        # first so a sparse layer's append triggers block allocation before K/V.
        streams: list[list[StreamSpec]] = []
        for layer_idx in range(self.cfg.num_hidden_layers):
            layer = [
                StreamSpec("k", self.cfg.num_key_value_heads, self.cfg.head_dim),
                StreamSpec("v", self.cfg.num_key_value_heads, self.cfg.head_dim),
            ]
            if self.cfg.is_sparse_attn_layer(layer_idx):
                layer.insert(0, StreamSpec("index_k", 1, self.cfg.index_head_dim))
            streams.append(layer)
        return streams

    def required_attention_backend(self) -> str | None:
        # The MSA per-query block mask is applied in the materialized SDPA
        # reference (packed_attention_torch); no flash/FlashInfer kernel takes it.
        return "torch"

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, MiniMaxM3ForCausalLM):
            raise TypeError(
                f"MiniMaxM3ForCausalLM.load_weights expects a MiniMaxM3ForCausalLM, "
                f"got {type(model).__name__}"
            )
        from mini_infer.distributed.group import get_rank, get_world_size
        from mini_infer.distributed.linear import _split_size

        cfg = model.cfg
        moe_inter = cfg.moe_intermediate_size
        per_rank = _split_size(cfg.n_routed_experts, get_world_size(), "n_routed_experts")
        local_start = get_rank() * per_rank
        local_end = local_start + per_rank

        def _keep(global_j: int) -> bool:
            return local_start <= global_j < local_end  # expert-parallel: this rank's slice

        remapped: dict[str, torch.Tensor] = {}
        for raw_key, tensor in hf_state_dict.items():
            # Strip the language_model. prefix; drop vision / MTP / multimodal.
            key = raw_key
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            if _DROP_RE.search(raw_key) or _DROP_RE.search(key):
                continue

            dense = _DENSE_GATE_UP_RE.match(key)
            if dense is not None:  # dense MLP fused gate+up -> gate_proj / up_proj
                half = tensor.shape[0] // 2
                remapped[f"{dense.group(1)}.mlp.gate_proj.weight"] = tensor[:half]
                remapped[f"{dense.group(1)}.mlp.up_proj.weight"] = tensor[half:]
                continue
            gate_up = _EXPERT_GATE_UP_RE.match(key)
            if gate_up is not None:  # stacked 3D (E, 2*I, H): per expert -> w1 / w3
                prefix = gate_up.group(1)
                for global_j in range(tensor.shape[0]):
                    if not _keep(global_j):
                        continue
                    local_j = global_j - local_start
                    remapped[f"{prefix}.mlp.experts.{local_j}.w1.weight"] = tensor[
                        global_j, :moe_inter
                    ]
                    remapped[f"{prefix}.mlp.experts.{local_j}.w3.weight"] = tensor[
                        global_j, moe_inter:
                    ]
                continue
            down = _EXPERT_DOWN_RE.match(key)
            if down is not None:  # stacked 3D (E, H, I): per expert -> w2
                prefix = down.group(1)
                for global_j in range(tensor.shape[0]):
                    if not _keep(global_j):
                        continue
                    remapped[f"{prefix}.mlp.experts.{global_j - local_start}.w2.weight"] = tensor[
                        global_j
                    ]
                continue
            per_expert = _PER_EXPERT_RE.match(key)
            if per_expert is not None:  # separate per-expert projections
                prefix, global_j = per_expert.group(1), int(per_expert.group(2))
                if not _keep(global_j):
                    continue
                w_name = _PER_EXPERT_W[per_expert.group(3)]
                remapped[f"{prefix}.mlp.experts.{global_j - local_start}.{w_name}.weight"] = tensor
                continue
            shared_gate_up = _SHARED_GATE_UP_RE.match(key)
            if shared_gate_up is not None:  # shared expert fused gate+up -> w1 / w3
                half = tensor.shape[0] // 2
                remapped[f"{shared_gate_up.group(1)}.mlp.shared_experts.w1.weight"] = tensor[:half]
                remapped[f"{shared_gate_up.group(1)}.mlp.shared_experts.w3.weight"] = tensor[half:]
                continue

            new_key = key
            if _INDEXER_RE.search(new_key):
                new_key = _INDEXER_RE.sub(
                    lambda m: f".self_attn.indexer.{m.group(1)}_{m.group(2)}.", new_key
                )
            for pattern, replacement in _SHARED_EXPERT_RENAME:
                if pattern.search(new_key):
                    new_key = pattern.sub(replacement, new_key)
                    break
            remapped[new_key] = tensor

        from mini_infer.distributed.loader import load_state_dict_with_tp

        missing, unexpected = load_state_dict_with_tp(model, remapped)
        whitelist = model.expected_missing_state_keys()
        missing = {m for m in missing if m not in whitelist}
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for MiniMaxM3ForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
