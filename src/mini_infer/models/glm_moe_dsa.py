"""GLM-MoE-DSA family: owned model implementation (text-only).

Targets `GlmMoeDsaForCausalLM` (z.ai GLM-5.2). Architecturally this is
DeepSeek-V3.2 plus GLM's IndexShare, so it reuses most of the DeepSeek-V2
machinery with three deltas:

  - **DeepSeek Sparse Attention (DSA)**: the `MLAAttention` carries a
    `GlmDsaIndexer` that scores raw tokens and selects `index_topk` keys per
    query; the rest are masked to `-inf`. See `glm_dsa_indexer.py` and the
    DSA path in `mla_attention.py`.
  - **IndexShare**: the indexer runs only on `"full"` layers; the following
    `"shared"` layers reuse that selection (`index_topk_freq` layers share one
    indexer pass). The per-layer top-k threads through the decoder stack.
  - **`noaux_tc` MoE**: DeepSeek-V3-style sigmoid gate with an aux-loss-free
    selection bias and grouped top-k (`GlmMoeFFN`), vs Mixtral's softmax gate.

GLM also uses **non-interleaved (NeoX) RoPE** for both the main attention and
the indexer (`MLAAttention(use_interleaved_rope=False)`), unlike DeepSeek-V2/V3.
Layers `< first_k_dense_replace` are dense `SwiGLU`; the rest are MoE. The MTP
draft layer (`num_nextn_predict_layers`) is a speculative-decoding accelerator
and is not modeled (its checkpoint keys are dropped at load).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import StreamSpec
from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import MLAAttention, RMSNorm, RotaryEmbedding, SwiGLU
from mini_infer.models.blocks.glm_dsa_indexer import GlmDsaIndexer
from mini_infer.models.blocks.glm_moe_gate import GlmMoeFFN

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class GlmMoeDsaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    tie_word_embeddings: bool
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    # Per-layer markers: "dense"/"sparse" FFN and "full"/"shared" indexer.
    mlp_layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]

    @classmethod
    def from_hf(cls, hf_config: Any) -> GlmMoeDsaConfig:
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = float(
            rope_params.get("rope_theta") or getattr(hf_config, "rope_theta", None) or 10000.0
        )
        num_layers = hf_config.num_hidden_layers
        # HF's config always populates these (defaults derived in __post_init__).
        mlp_types = tuple(getattr(hf_config, "mlp_layer_types", None) or ["sparse"] * num_layers)
        idx_types = tuple(getattr(hf_config, "indexer_types", None) or ["full"] * num_layers)
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hf_config.num_attention_heads,
            kv_lora_rank=hf_config.kv_lora_rank,
            q_lora_rank=hf_config.q_lora_rank,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            n_routed_experts=hf_config.n_routed_experts,
            n_shared_experts=getattr(hf_config, "n_shared_experts", 1) or 0,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            n_group=int(getattr(hf_config, "n_group", 1)),
            topk_group=int(getattr(hf_config, "topk_group", 1)),
            routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=rope_theta,
            attention_bias=bool(getattr(hf_config, "attention_bias", False)),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            index_topk=hf_config.index_topk,
            index_head_dim=hf_config.index_head_dim,
            index_n_heads=hf_config.index_n_heads,
            mlp_layer_types=mlp_types,
            indexer_types=idx_types,
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    def indexer_is_shared(self, layer_idx: int) -> bool:
        return self.indexer_types[layer_idx] == "shared"


class _GlmMoeDsaDecoderLayer(nn.Module):
    """One GLM-MoE-DSA block: MLA+DSA attention + dense SwiGLU or MoE FFN.

    Standard pre/post-norm shape. The attention's DSA indexer runs on `"full"`
    indexer layers; `"shared"` layers reuse the previous layer's selection
    (IndexShare). `forward` returns the hidden state plus the top-k indices to
    thread to the next layer (or `None` when the next layer recomputes).
    """

    def __init__(self, cfg: GlmMoeDsaConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        indexer = GlmDsaIndexer(
            hidden_size=cfg.hidden_size,
            q_lora_rank=cfg.q_lora_rank,
            num_heads=cfg.index_n_heads,
            head_dim=cfg.index_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            index_topk=cfg.index_topk,
        )
        self.self_attn = MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            q_lora_rank=cfg.q_lora_rank,
            rms_norm_eps=cfg.rms_norm_eps,
            attention_bias=cfg.attention_bias,
            layer_idx=layer_idx,
            use_interleaved_rope=False,  # GLM uses NeoX/Llama RoPE
            indexer=indexer,
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
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
            )
        else:
            self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        # IndexShare flags (mirror HF GlmMoeDsaAttention.skip_topk/next_skip_topk).
        self.skip_topk = cfg.indexer_is_shared(layer_idx)
        self.next_skip_topk = (
            cfg.indexer_is_shared(layer_idx + 1) if layer_idx + 1 < cfg.num_hidden_layers else False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
        prev_topk: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        # DSA selection: a "shared" layer reuses the prior "full" layer's
        # top-k; otherwise compute it here (matches HF's `not skip or prev is None`).
        if self.skip_topk and prev_topk is not None:
            topk = prev_topk
        else:
            topk = self.self_attn.compute_dsa_topk(
                x, position_embeddings, past_key_values, cu_seqlens_q
            )
        x = self.self_attn(x, position_embeddings, past_key_values, cu_seqlens_q, dsa_topk=topk)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        out: torch.Tensor = residual + x
        # Thread the selection forward only while the next layer shares it.
        return out, (topk if self.next_skip_topk else None)


class _GlmMoeDsaInnerModel(nn.Module):
    """Embedding + N decoder layers + final norm. HF parameter prefix is `model.`."""

    def __init__(self, cfg: GlmMoeDsaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [_GlmMoeDsaDecoderLayer(cfg, layer_idx) for layer_idx in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


# Shared-expert rename: HF `mlp.shared_experts.{gate,up,down}_proj` → our
# single MixtralExpert `w1/w3/w2`. (Dense `mlp.{gate,up,down}_proj` already
# match SwiGLU's names, so they pass through unchanged.)
_SHARED_EXPERT_RENAME: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.mlp\.shared_experts\.gate_proj\.weight$"), r".mlp.shared_experts.w1.weight"),
    (re.compile(r"\.mlp\.shared_experts\.down_proj\.weight$"), r".mlp.shared_experts.w2.weight"),
    (re.compile(r"\.mlp\.shared_experts\.up_proj\.weight$"), r".mlp.shared_experts.w3.weight"),
]
# Routed experts appear in two layouts. HF's in-memory `GlmMoeDsaNaiveMoe` stacks
# them as 3D `experts.gate_up_proj` / `down_proj` (what a synthetic `state_dict()`
# yields); the published checkpoint stores them PER-EXPERT as
# `experts.{j}.{gate,up,down}_proj.weight` (confirmed via the safetensors index).
# `load_weights` handles both. w-name map: gate->w1, up->w3, down->w2.
_EXPERT_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.gate_up_proj$")
_EXPERT_DOWN_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.down_proj$")
_PER_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)
_PER_EXPERT_W = {"gate": "w1", "up": "w3", "down": "w2"}


def _dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight to BF16, handling partial blocks.

    The published checkpoint quantizes 2-D weights in `block x block` tiles with
    a per-tile scale; some weights (e.g. `kv_a_proj_with_mqa`, 576 rows) are not a
    multiple of `block`, so the scale grid is `ceil(M/block) x ceil(N/block)` with
    partial last tiles. Expand the scale by `block` along each axis, then crop to
    the weight shape. (mini_infer.quant.dequantize_block_fp8_to_bf16 assumes exact
    divisibility, so this ceil-aware variant lives here.)
    """
    rows, cols = weight.shape
    expanded = scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return (weight.float() * expanded[:rows, :cols]).to(torch.bfloat16)


@register_model
class GlmMoeDsaForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "GlmMoeDsaForCausalLM"
    Config: ClassVar[type] = GlmMoeDsaConfig

    def __init__(self, cfg: GlmMoeDsaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _GlmMoeDsaInnerModel(cfg)
        # GLM-5.2 uses plain RoPE (theta 8e6, no YaRN) on the qk_rope slice.
        self.rotary_emb = RotaryEmbedding(cfg.qk_rope_head_dim, base=cfg.rope_theta)
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
        # IndexShare: thread the DSA selection through the decoder stack.
        prev_topk: list[torch.Tensor] | None = None
        for layer in self.model.layers:
            x, prev_topk = layer(x, (cos, sin), past_key_values, cu_seqlens_q, prev_topk)
        x = self.model.norm(x)
        logits: torch.Tensor = self.lm_head(x)
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=1,
            head_dim=self.cfg.kv_lora_rank,
        )

    def per_layer_streams(self) -> list[list[StreamSpec]]:
        streams: list[list[StreamSpec]] = []
        for layer_idx in range(self.cfg.num_hidden_layers):
            layer = [
                StreamSpec("kv_latent", 1, self.cfg.kv_lora_rank),
                StreamSpec("k_rope", 1, self.cfg.qk_rope_head_dim),
            ]
            # "full" indexer layers cache their DSA selection keys so decode
            # steps score against the full history. index_k goes FIRST so layer
            # 0 (always "full") triggers block allocation on its append, before
            # the main KV streams write into the slot.
            if not self.cfg.indexer_is_shared(layer_idx):
                layer.insert(0, StreamSpec("index_k", 1, self.cfg.index_head_dim))
            streams.append(layer)
        return streams

    def required_attention_backend(self) -> str | None:
        # The DSA per-query sparse mask is applied in the materialized SDPA
        # reference (`mla_packed_attention_forward`); no flash/FlashInfer
        # kernel supports it, so force the torch path.
        return "torch"

    def expected_missing_state_keys(self) -> set[str]:
        if self.cfg.tie_word_embeddings:
            return {"lm_head.weight"}
        return set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, GlmMoeDsaForCausalLM):
            raise TypeError(
                f"GlmMoeDsaForCausalLM.load_weights expects a GlmMoeDsaForCausalLM, "
                f"got {type(model).__name__}"
            )
        cfg = model.cfg
        moe_inter = cfg.moe_intermediate_size
        # Expert-parallel sharding: each rank materializes only its contiguous
        # slice of routed experts, at LOCAL indices. HF ships all experts stacked
        # at global indices, so map global -> local and drop off-rank experts.
        # At world_size=1 this is the identity (per_rank == n_routed_experts).
        from mini_infer.distributed.group import get_rank, get_world_size
        from mini_infer.distributed.linear import _split_size

        per_rank = _split_size(cfg.n_routed_experts, get_world_size(), "n_routed_experts")
        local_start = get_rank() * per_rank
        local_end = local_start + per_rank

        # Pass 1: block-FP8 dequant. The published checkpoint ships e4m3 weights
        # paired with a `.weight_scale_inv` per-block scale; dequantize to BF16
        # and drop the scale companion. Synthetic / HF in-memory state_dicts are
        # BF16 with no scales, so this is a pass-through for them.
        dequantized: dict[str, torch.Tensor] = {}
        for key, tensor in hf_state_dict.items():
            if key.endswith(".weight_scale_inv"):
                continue  # consumed by its paired weight
            scale = hf_state_dict.get(key + "_scale_inv") if key.endswith(".weight") else None
            if scale is not None and tensor.dtype == torch.float8_e4m3fn:
                dequantized[key] = _dequant_block_fp8(tensor, scale)
            else:
                dequantized[key] = tensor

        # Pass 2: drop the MTP draft layer (model.layers.{num_hidden_layers}.* and
        # any mtp.*), rename/expand routed experts to this rank's local w1/w2/w3,
        # and rename shared experts. Dense `mlp.{gate,up,down}_proj` and the router
        # `mlp.gate.weight` pass through (names already match).
        mtp_layer_re = re.compile(rf"^model\.layers\.{cfg.num_hidden_layers}\.")
        remapped: dict[str, torch.Tensor] = {}
        for key, tensor in dequantized.items():
            if key.startswith("mtp.") or mtp_layer_re.match(key):
                continue
            gate_up = _EXPERT_GATE_UP_RE.match(key)
            if gate_up is not None:
                # Stacked routed experts (HF in-memory): (num_experts, 2*moe_inter,
                # hidden). Split each into w1 (gate) / w3 (up); keep this rank's slice.
                prefix = gate_up.group(1)
                for global_j in range(tensor.shape[0]):
                    if not (local_start <= global_j < local_end):
                        continue  # off-rank expert (expert-parallel)
                    local_j = global_j - local_start
                    expert = tensor[global_j]
                    remapped[f"{prefix}.mlp.experts.{local_j}.w1.weight"] = expert[:moe_inter]
                    remapped[f"{prefix}.mlp.experts.{local_j}.w3.weight"] = expert[moe_inter:]
                continue
            down = _EXPERT_DOWN_RE.match(key)
            if down is not None:
                # Stacked routed experts down_proj: (num_experts, hidden, moe_inter).
                prefix = down.group(1)
                for global_j in range(tensor.shape[0]):
                    if not (local_start <= global_j < local_end):
                        continue  # off-rank expert (expert-parallel)
                    local_j = global_j - local_start
                    remapped[f"{prefix}.mlp.experts.{local_j}.w2.weight"] = tensor[global_j]
                continue
            per_expert = _PER_EXPERT_RE.match(key)
            if per_expert is not None:
                # Per-expert checkpoint layout: experts.{j}.{gate,up,down}_proj.weight.
                prefix = per_expert.group(1)
                global_j = int(per_expert.group(2))
                if not (local_start <= global_j < local_end):
                    continue  # off-rank expert (expert-parallel)
                local_j = global_j - local_start
                w_name = _PER_EXPERT_W[per_expert.group(3)]
                remapped[f"{prefix}.mlp.experts.{local_j}.{w_name}.weight"] = tensor
                continue
            new_key = key
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
                f"weight load mismatch for GlmMoeDsaForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
