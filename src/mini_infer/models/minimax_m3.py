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
are MSA + MoE. RoPE is NeoX/non-interleaved and PARTIAL: width-`rotary_dim`
tables (64 of 128 real) rotate the leading dims and pass the tail, matching
HF's slice-to-cos-width apply under the deployment config's
`partial_rotary_factor`. Multi-token-prediction (MTP) draft weights are
dropped at load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import StreamSpec
from mini_infer.cache.msa_paged_attention import msa_paged_decode
from mini_infer.cache.packed_attention import packed_attention_torch
from mini_infer.cache.paged_attention import paged_attention_decode_batched
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import RotaryEmbedding, SwiGLU
from mini_infer.models.blocks.activations import swigluoai
from mini_infer.models.blocks.gemma_rmsnorm import GemmaRMSNorm
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN
from mini_infer.models.blocks.minimax_m3_indexer import MiniMaxM3Indexer
from mini_infer.models.blocks.rope import apply_rotary_pos_emb_partial

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
        # Partial RoPE width. HF's rope init reads `partial_rotary_factor` out
        # of the (standardized) rope_parameters; the deployment config ships it
        # as a flat field (0.5) alongside `rotary_dim` (64). Follow HF's
        # precedence: rope_parameters first, then the flat fields, then full.
        partial_factor = rope_params.get("partial_rotary_factor") or getattr(
            text, "partial_rotary_factor", None
        )
        if partial_factor:
            rotary_dim = int(head_dim * float(partial_factor))
        else:
            rotary_dim = int(getattr(text, "rotary_dim", None) or head_dim)
        # Dense/sparse split: HF encodes it via first_k_dense_replace or the
        # sparse/moe frequency arrays; layer i is dense iff i < first_dense.
        first_dense = getattr(text, "first_k_dense_replace", None)
        if first_dense is None:
            mlp_types = getattr(text, "mlp_layer_types", None)
            first_dense = sum(1 for t in mlp_types if t == "dense") if mlp_types else 0
        # FP8-quantized checkpoint (e.g. the pre-quantized staging of the 854 GB
        # bf16 release): keep routed experts e4m3-resident so the model fits.
        quant = getattr(hf_config, "quantization_config", None) or getattr(
            text, "quantization_config", None
        )
        quant_method = (quant or {}).get("quant_method") if isinstance(quant, dict) else None
        expert_dtype = "fp8" if quant_method == "fp8" else "bf16"
        return cls(
            expert_dtype=expert_dtype,
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
        # Opt-in paged decode kernels (off by default; the end-to-end A/B gates
        # default-on). When on, pure decode steps read K/V directly from the
        # pool blocks: sparse layers read ONLY the indexer-selected blocks
        # (`msa_paged_decode`), dense layers walk the full history
        # (`paged_attention_decode_batched`). Prefill and mixed batches always
        # take the materialized torch path. Single-rank only for now.
        self.use_decode_kernel = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        paged_ctx: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # Per-head QK-norm BEFORE transpose/RoPE (v un-normed), matching HF.
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # Partial RoPE: cos/sin are width `rotary_dim`; the helper rotates the
        # leading dims and passes the tail through, matching HF's slice-to-cos
        # width apply. (rotary_dim == head_dim degenerates to full rope.)
        q, k = apply_rotary_pos_emb_partial(q, k, cos, sin)

        keys_packed = k.transpose(1, 2).squeeze(0).contiguous()
        values_packed = v.transpose(1, 2).squeeze(0).contiguous()
        queries_packed = q.transpose(1, 2).squeeze(0).contiguous()
        # Named-stream API (not the legacy K/V path): the index_k stream on sparse
        # layers makes the pool layout non-legacy, so k/v are named streams too.
        # append "k" on layer 0 triggers per-step block allocation.
        past_key_values.append_stream_packed(keys_packed, cu_seqlens_q, self.layer_idx, "k")
        past_key_values.append_stream_packed(values_packed, cu_seqlens_q, self.layer_idx, "v")

        if self.use_decode_kernel and self._decode_kernel_applicable(past_key_values, cu_seqlens_q):
            attn_packed = self._forward_decode_paged(
                hidden_states, cos, sin, past_key_values, cu_seqlens_q, queries_packed, paged_ctx
            )
        else:
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

            keys_full, cu_seqlens_k, _ = past_key_values.materialize_packed_stream(
                self.layer_idx, "k"
            )
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

    def _decode_kernel_applicable(
        self, past_key_values: PagedKVCache, cu_seqlens_q: torch.Tensor
    ) -> bool:
        """Paged decode path applies to pure decode steps only (every request
        contributes exactly one token), single rank, with the index block a
        multiple of the pool block. Everything else takes the torch path."""
        from mini_infer.distributed.group import get_world_size

        if get_world_size() != 1:
            return False
        if not bool((cu_seqlens_q[1:] - cu_seqlens_q[:-1] == 1).all()):
            return False
        if self.indexer is not None:
            pool_block_size = past_key_values.pool_storage_for_stream(self.layer_idx, "k").shape[1]
            if self.indexer.block_size % pool_block_size != 0:
                return False
        return True

    def _forward_decode_paged(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        queries_packed: torch.Tensor,
        paged_ctx: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Decode step via paged reads, no K/V materialization.

        Sparse layers: `select_cached` picks the blocks (the exact routine the
        oracle's mask path uses), then `msa_paged_decode` attends over only the
        selected blocks. Dense layers: the standard paged decode kernel over
        the full history. `queries_packed` is `(B, num_heads_local, head_dim)`
        because a pure decode step has one token per request. `paged_ctx` is
        the per-step `(block_tables, seq_lens)` hoisted by the model forward
        (they are layer-invariant; rebuilding them per layer is pure host
        overhead on the decode hot path).
        """
        device = queries_packed.device
        if paged_ctx is not None and "block_tables" in paged_ctx:
            block_tables = paged_ctx["block_tables"]
            seq_lens = paged_ctx["seq_lens"]
        else:
            # This runs after this layer's k/v appends, so counts include the
            # step's token and any block allocated for it (layer 0 allocates).
            block_tables = past_key_values.block_tables_per_request_tensor(device)
            seq_lens = past_key_values.seq_lens_list()
            if paged_ctx is not None:
                paged_ctx["block_tables"] = block_tables
                paged_ctx["seq_lens"] = seq_lens
        k_pool = past_key_values.pool_storage_for_stream(self.layer_idx, "k")
        v_pool = past_key_values.pool_storage_for_stream(self.layer_idx, "v")
        if self.indexer is None:
            return paged_attention_decode_batched(
                queries_packed, k_pool, v_pool, block_tables, seq_lens
            )
        selections = self.indexer.select_cached(
            hidden_states, cos, sin, past_key_values, cu_seqlens_q, self.layer_idx
        )
        selected = [ids[0] for ids, _ in selections]  # decode: one query row each
        return msa_paged_decode(
            queries_packed,
            k_pool,
            v_pool,
            block_tables,
            seq_lens,
            selected,
            index_block_size=self.indexer.block_size,
            scale=self.softmax_scale,
        )


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
        paged_ctx: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q, paged_ctx)
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


# FFN weights arrive in two layouts. HF in-memory `state_dict()`: the dense MLP
# and the shared expert stack gate+up into one `gate_up_proj` (out dim 2*I; gate
# = first half, up = second, matching swigluoai's `chunk(2, -1)`), and the routed
# experts stack all E into 3D `experts.gate_up_proj` (E, 2*I, H) /
# `experts.down_proj` (E, H, I). The on-disk 428B checkpoint instead ships the
# MoE under `block_sparse_moe.` with per-expert `experts.E.{w1,w3,w2}` (already
# split; w1=gate, w3=up, w2=down) and the router bias at the block level
# (`block_sparse_moe.e_score_correction_bias`). Both map onto our SwiGLU
# (`gate_proj/up_proj/down_proj`) and MixtralExpert (`w1/w3/w2`) params.
_DENSE_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.gate_up_proj\.weight$")
_EXPERT_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.gate_up_proj$")
_EXPERT_DOWN_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.down_proj$")
_PER_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)
_PER_EXPERT_W = {"gate": "w1", "up": "w3", "down": "w2"}
# Disk layout: per-expert weights already in our w1/w3/w2 names; only the
# expert-parallel global->local index remap is needed.
_PER_EXPERT_W_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
# Block-FP8 per-expert scale (the pre-quantized staged checkpoint): pairs with
# the e4m3 weight of the same name minus the `_scale_inv` suffix.
_PER_EXPERT_W_SCALE_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(w1|w2|w3)\.weight_scale_inv$"
)
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
        # Partial RoPE (theta 5e6): a width-`rotary_dim` table (64 of 128 for
        # the real model). HF standardizes the deployment config's flat
        # `partial_rotary_factor` into rope_parameters and builds inv_freq of
        # length rotary_dim/2; its apply then rotates only the first
        # rotary_dim head dims. A harness config that omits the factor makes
        # HF degenerate to full rope, which is why parity configs must carry
        # it (the real checkpoint is trained partial).
        self.rotary_emb = RotaryEmbedding(cfg.rotary_dim, base=cfg.rope_theta)
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
        # Shared per-step paged-decode context: block tables + seq lens are
        # layer-invariant AFTER layer 0's append (which does the step's block
        # allocation), so layer 0's kernel path populates this dict once and
        # the remaining layers reuse it instead of rebuilding per layer.
        paged_ctx: dict[str, Any] | None = None
        first_attn = self.model.layers[0].self_attn
        if isinstance(first_attn, _MiniMaxM3Attention) and first_attn.use_decode_kernel:
            paged_ctx = {}
        for layer in self.model.layers:
            x = layer(x, (cos, sin), past_key_values, cu_seqlens_q, paged_ctx)
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

    def set_decode_kernel(self, enabled: bool) -> None:
        """Toggle the paged decode path on every attention layer.

        Off by default (the materialized torch oracle). On: pure decode steps
        read K/V straight from the pool blocks; sparse layers attend over only
        the indexer-selected blocks (`msa_paged_decode`). The end-to-end A/B
        gates whether this becomes default-on.
        """
        for module in self.modules():
            if isinstance(module, _MiniMaxM3Attention):
                module.use_decode_kernel = enabled

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
        from mini_infer.distributed.loader import load_state_dict_with_tp

        remapped = _remap_m3_state(model.cfg, hf_state_dict)
        missing, unexpected = load_state_dict_with_tp(model, remapped)
        whitelist = model.expected_missing_state_keys()
        missing = {m for m in missing if m not in whitelist}
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for MiniMaxM3ForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

    @staticmethod
    def load_weights_streaming(
        model: BaseCausalLM,
        name_or_path: str,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Load a safetensors checkpoint shard by shard (peak host RAM ~ one shard).

        The 854 GB checkpoint cannot be materialized as one in-RAM state_dict;
        each shard is remapped and copied into the (TP/EP-sliced) module params,
        then freed. FP8 weight/scale pairs must be co-located in one shard (the
        staging script writes them that way). Coverage is verified at the end,
        same contract as `load_weights`.
        """
        if not isinstance(model, MiniMaxM3ForCausalLM):
            raise TypeError(
                f"MiniMaxM3ForCausalLM.load_weights_streaming expects a "
                f"MiniMaxM3ForCausalLM, got {type(model).__name__}"
            )
        from mini_infer.distributed.loader import load_state_dict_with_tp
        from mini_infer.models.loader import iter_safetensors_shards

        model_keys = set(model.state_dict().keys())
        consumed: set[str] = set()
        for shard in iter_safetensors_shards(name_or_path, device="cpu", dtype=dtype):
            remapped = _remap_m3_state(model.cfg, shard)
            if not remapped:
                continue
            missing, unexpected = load_state_dict_with_tp(model, remapped, target_device=device)
            if unexpected:
                raise ValueError(f"streaming load hit unexpected keys: {sorted(unexpected)[:8]}")
            consumed |= model_keys - missing
        leftover = model_keys - consumed - model.expected_missing_state_keys()
        if leftover:
            raise ValueError(
                f"streaming load left {len(leftover)} model keys unloaded, e.g. "
                f"{sorted(leftover)[:8]}"
            )


def _remap_m3_state(
    cfg: MiniMaxM3Config, hf_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """One shard's (or the whole checkpoint's) HF tensors -> our module keys.

    Handles both weight layouts (HF in-memory fused forms and the on-disk
    `block_sparse_moe.` per-expert form), expert-parallel rank filtering, and
    block-FP8: with `expert_dtype="fp8"` routed-expert e4m3 weights and their
    `weight_scale_inv` scales route onto the `Fp8Expert` buffers; otherwise
    (and for any non-expert FP8 tensor) they dequantize to the model dtype.
    """
    from mini_infer.distributed.group import get_rank, get_world_size
    from mini_infer.distributed.linear import _split_size
    from mini_infer.quant.nvfp4 import dequantize_block_fp8_to_bf16_partial

    moe_inter = cfg.moe_intermediate_size
    per_rank = _split_size(cfg.n_routed_experts, get_world_size(), "n_routed_experts")
    local_start = get_rank() * per_rank
    local_end = local_start + per_rank
    keep_fp8 = cfg.expert_dtype == "fp8"

    def _keep(global_j: int) -> bool:
        return local_start <= global_j < local_end  # expert-parallel: this rank's slice

    # Normalize keys first (prefix strip, drops, block_sparse_moe -> mlp) so the
    # FP8 weight/scale pairing below can look keys up by normalized name.
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in hf_state_dict.items():
        key = raw_key
        if key.startswith("language_model."):
            key = key[len("language_model.") :]
        if _DROP_RE.search(raw_key) or _DROP_RE.search(key):
            continue
        # Disk layout: the MoE block lives under `block_sparse_moe.` with the
        # router bias at the block level; normalize to the in-memory `mlp.`
        # names so one mapping path serves both layouts.
        if ".block_sparse_moe." in key:
            key = key.replace(".block_sparse_moe.", ".mlp.")
            key = key.replace(".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
        normalized[key] = tensor

    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in normalized.items():
        if key.endswith(".weight_scale_inv"):
            scale_m = _PER_EXPERT_W_SCALE_RE.match(key)
            if keep_fp8 and scale_m is not None:
                global_j = int(scale_m.group(2))
                if _keep(global_j):
                    local_j = global_j - local_start
                    remapped[
                        f"{scale_m.group(1)}.mlp.experts.{local_j}.{scale_m.group(3)}_scale"
                    ] = tensor
            continue  # otherwise consumed by the paired weight's dequant below

        scale = normalized.get(key + "_scale_inv") if key.endswith(".weight") else None
        if scale is not None and tensor.dtype == torch.float8_e4m3fn:
            fp8_expert = _PER_EXPERT_W_RE.match(key)
            if keep_fp8 and fp8_expert is not None:
                global_j = int(fp8_expert.group(2))
                if _keep(global_j):
                    local_j = global_j - local_start
                    # e4m3-resident -> Fp8Expert buffer `w{n}` (no `.weight`).
                    remapped[
                        f"{fp8_expert.group(1)}.mlp.experts.{local_j}.{fp8_expert.group(3)}"
                    ] = tensor
                continue
            tensor = dequantize_block_fp8_to_bf16_partial(tensor, scale)

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
                remapped[f"{prefix}.mlp.experts.{local_j}.w1.weight"] = tensor[global_j, :moe_inter]
                remapped[f"{prefix}.mlp.experts.{local_j}.w3.weight"] = tensor[global_j, moe_inter:]
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
        per_expert_w = _PER_EXPERT_W_RE.match(key)
        if per_expert_w is not None:  # disk layout: our names, EP remap only
            prefix, global_j = per_expert_w.group(1), int(per_expert_w.group(2))
            if not _keep(global_j):
                continue
            local_j = global_j - local_start
            remapped[f"{prefix}.mlp.experts.{local_j}.{per_expert_w.group(3)}.weight"] = tensor
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

    return remapped
