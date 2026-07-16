"""Inkling family (thinkingmachines/inkling): owned text-model implementation.

Text-only port of Thinking Machines' Inkling (975B/41B-active MoE; the
276B/12B Inkling-Small preview shares the architecture). Vision and audio
towers are out of scope by design; the parity reference is transformers
5.14's `InklingForCausalLM`.

Architecturally this is a no-RoPE hybrid-attention MoE decoder:

  - **Hybrid attention, 5:1**: `layer_types` interleaves `hybrid_sliding`
    (window 512, its own head config: 64q/16kv on the released checkpoints)
    with `hybrid` (global, 64q/8kv). Per-head RMS QK-norm, and 1/d (not
    1/sqrt(d)) logit scaling because q and k are unit-RMS per head.
  - **Relative position bias instead of RoPE**: `r_proj` emits per-token,
    per-head features (d_rel=16); `InklingRelativeLogits` mixes a trained
    bank of bias-vs-distance profiles into an additive attention bias,
    zero beyond `rel_extent` (the window on sliding layers, 1024 on global
    layers). Global layers additionally scale q and the bias by a
    log-length factor `tau = 1 + alpha*log(max(1, (pos+1)/n_floor))`.
  - **Short convolutions (SConv)**: four depthwise causal convs (kernel 4)
    per layer (on the K and V projections and on the attention/MLP branch
    outputs), each with the residual folded inside, computed in fp32.
  - **MoE with a shared-expert sink**: sigmoid router, aux-loss-free
    selection bias, and expert weights normalized JOINTLY over the top-k
    routed logits and the 2 shared experts' logits (`blocks/inkling_moe.py`).
    The first `dense_mlp_idx` layers are dense SwiGLU (times a learned
    global scale).
  - **muP unembedding**: hidden states are divided by
    `logits_mup_width_multiplier` before the (untied) lm_head, and logits
    are sliced to `unpadded_vocab_size`.

Serving-side, the SConvs make decode stateful: a step needs the previous
`kernel_size - 1` PRE-conv inputs of each conv. Rather than a per-request
rolling state (which would break under `truncate_to` and prefix-cache
rollback), the pre-conv inputs are stored as per-token streams in the
`PagedKVCache` (`conv_k`/`conv_v`/`conv_attn`/`conv_mlp`) next to the
post-conv `k`/`v`, and each step gathers the tail it needs. That costs
extra pool memory (~2*hidden + 2*kv_dim per token per layer); a rolling
conv-state buffer is a known follow-up, benchmark-gated like every
optimization here.

The relative bias + window + causality are folded into a per-request
additive mask consumed by `packed_attention_torch`'s `block_mask` path
(the MiniMax-M3 MSA precedent); no flash/FlashInfer kernel takes a
per-head additive bias, so the model pins the `torch` attention backend,
exactly as HF disables flash-attn for this family. Multi-token-prediction
(MTP) draft weights are dropped at load. Single-rank only for now (the
SConv channel dim would shard with TP's KV split; deferred).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from mini_infer.cache.block_pool import LayerAttentionSpec, StreamSpec
from mini_infer.cache.packed_attention import packed_attention_torch
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks.inkling_moe import InklingDenseMLP, InklingMoE
from mini_infer.models.blocks.inkling_rel_bias import InklingRelativeLogits
from mini_infer.models.blocks.inkling_sconv import InklingShortConv
from mini_infer.models.blocks.rmsnorm import RMSNorm

if TYPE_CHECKING:
    from mini_infer.cache.paged_kv_cache import PagedKVCache


@dataclass
class InklingConfig:
    vocab_size: int
    unpadded_vocab_size: int | None
    hidden_size: int
    num_hidden_layers: int
    # Global ("hybrid") layers' attention shape.
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    # Sliding ("hybrid_sliding") layers' attention shape.
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    # Relative position bias.
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int | None
    log_scaling_alpha: float
    # Per-layer patterns ("hybrid_sliding"/"hybrid", "dense"/"sparse").
    layer_types: list[str]
    mlp_layer_types: list[str]
    # FFN widths.
    intermediate_size: int  # dense-layer SwiGLU width
    moe_intermediate_size: int  # per-expert width
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    route_scale: float
    conv_kernel_size: int
    rms_norm_eps: float
    logits_mup_width_multiplier: float

    @classmethod
    def from_hf(cls, hf_config: Any) -> InklingConfig:
        text = getattr(hf_config, "text_config", hf_config) or hf_config
        num_layers = int(text.num_hidden_layers)
        # HF's InklingTextConfig materializes both per-layer patterns in
        # __post_init__; replicate its defaults for configs that omit them
        # (5 sliding : 1 global; all-sparse MLP).
        layer_types = list(getattr(text, "layer_types", None) or [])
        if not layer_types:
            local_ids = getattr(text, "local_layer_ids", None)
            local = (
                set(local_ids)
                if local_ids is not None
                else {i for i in range(num_layers) if (i + 1) % 6}
            )
            layer_types = ["hybrid_sliding" if i in local else "hybrid" for i in range(num_layers)]
        mlp_layer_types = list(getattr(text, "mlp_layer_types", None) or [])
        if not mlp_layer_types:
            dense_mlp_idx = int(getattr(text, "dense_mlp_idx", 0) or 0)
            mlp_layer_types = [
                "dense" if i < dense_mlp_idx else "sparse" for i in range(num_layers)
            ]
        head_dim = int(
            getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
        )
        # Deployment configs ship `dense_intermediate_size` for the dense
        # layers next to the MoE's `intermediate_size`; HF folds the former
        # into `intermediate_size` at config build. Prefer the explicit field.
        dense_inter = int(getattr(text, "dense_intermediate_size", None) or text.intermediate_size)
        n_floor = getattr(text, "log_scaling_n_floor", None)
        return cls(
            vocab_size=int(text.vocab_size),
            unpadded_vocab_size=(
                int(text.unpadded_vocab_size)
                if getattr(text, "unpadded_vocab_size", None) is not None
                else None
            ),
            hidden_size=int(text.hidden_size),
            num_hidden_layers=num_layers,
            num_attention_heads=int(text.num_attention_heads),
            num_key_value_heads=int(text.num_key_value_heads),
            head_dim=head_dim,
            swa_num_attention_heads=int(
                getattr(text, "swa_num_attention_heads", None) or text.num_attention_heads
            ),
            swa_num_key_value_heads=int(
                getattr(text, "swa_num_key_value_heads", None) or text.num_key_value_heads
            ),
            swa_head_dim=int(getattr(text, "swa_head_dim", None) or head_dim),
            sliding_window_size=int(text.sliding_window_size),
            d_rel=int(text.d_rel),
            rel_extent=int(text.rel_extent),
            log_scaling_n_floor=int(n_floor) if n_floor is not None else None,
            log_scaling_alpha=float(getattr(text, "log_scaling_alpha", 0.1)),
            layer_types=layer_types,
            mlp_layer_types=mlp_layer_types,
            intermediate_size=dense_inter,
            moe_intermediate_size=int(text.moe_intermediate_size),
            n_routed_experts=int(text.n_routed_experts),
            num_experts_per_tok=int(text.num_experts_per_tok),
            n_shared_experts=int(text.n_shared_experts),
            route_scale=float(getattr(text, "route_scale", 8.0)),
            conv_kernel_size=int(
                getattr(text, "conv_kernel_size", None)
                or getattr(text, "sconv_kernel_size", None)
                or 4
            ),
            rms_norm_eps=float(getattr(text, "rms_norm_eps", 1e-6)),
            logits_mup_width_multiplier=float(getattr(text, "logits_mup_width_multiplier", 1.0)),
        )

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "hybrid_sliding"

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    def layer_heads(self, layer_idx: int) -> tuple[int, int, int]:
        """(num_q_heads, num_kv_heads, head_dim) for one layer's type."""
        if self.is_sliding(layer_idx):
            return self.swa_num_attention_heads, self.swa_num_key_value_heads, self.swa_head_dim
        return self.num_attention_heads, self.num_key_value_heads, self.head_dim


def _sconv_cached(
    module: InklingShortConv,
    x_packed: torch.Tensor,
    cache: PagedKVCache,
    cu_seqlens_q: torch.Tensor,
    layer_idx: int,
    stream_name: str,
) -> torch.Tensor:
    """Run one SConv over packed varlen tokens with cached left context.

    Appends this step's PRE-conv inputs `(total_q, channels)` to the named
    stream, then convolves each request's new tokens against the last
    `kernel_size - 1` cached inputs. The current tokens come from
    `x_packed` directly (not read back from the pool) so a low-precision
    pool only rounds the tail, mirroring HF's fp32-strict conv as closely
    as the storage dtype allows.
    """
    total_q, channels = x_packed.shape
    spec = cache._pool.stream_spec(layer_idx, stream_name)
    cache.append_stream_packed(
        x_packed.view(total_q, spec.num_kv_heads, spec.head_dim).to(cache._pool.dtype),
        cu_seqlens_q,
        layer_idx,
        stream_name,
    )
    history, cu_seqlens_k, _ = cache.materialize_packed_stream(layer_idx, stream_name)
    out = torch.empty_like(x_packed)
    tail_len = module.kernel_size - 1
    for batch_idx in range(cu_seqlens_q.shape[0] - 1):
        q_start, q_end = int(cu_seqlens_q[batch_idx]), int(cu_seqlens_q[batch_idx + 1])
        if q_end == q_start:
            continue
        k_start, k_end = int(cu_seqlens_k[batch_idx]), int(cu_seqlens_k[batch_idx + 1])
        new_tokens = q_end - q_start
        # History BEFORE this step's tokens (which were just appended).
        hist_end = k_end - new_tokens
        tail_start = max(k_start, hist_end - tail_len)
        tail = history[tail_start:hist_end].reshape(-1, channels)
        out[q_start:q_end] = module(x_packed[q_start:q_end], tail)
    return out


class _InklingAttention(nn.Module):
    """Hybrid GQA: SConv'd K/V, per-head QK RMSNorm, relative bias, 1/d scale."""

    def __init__(self, cfg: InklingConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = cfg.is_sliding(layer_idx)
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.layer_heads(layer_idx)
        self.sliding_window = cfg.sliding_window_size if self.is_sliding else None
        self.d_rel = cfg.d_rel
        rel_extent = cfg.sliding_window_size if self.is_sliding else cfg.rel_extent
        # q/k are RMS-normalized per head, hence 1/d rather than 1/sqrt(d).
        self.softmax_scale = 1.0 / self.head_dim
        self.log_scaling_n_floor = None if self.is_sliding else cfg.log_scaling_n_floor
        self.log_scaling_alpha = cfg.log_scaling_alpha

        hidden = cfg.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.r_proj = nn.Linear(hidden, self.num_heads * cfg.d_rel, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.k_sconv = InklingShortConv(kv_dim, cfg.conv_kernel_size)
        self.v_sconv = InklingShortConv(kv_dim, cfg.conv_kernel_size)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rel_logits_proj = InklingRelativeLogits(cfg.d_rel, rel_extent)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        total_q = input_shape[-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1])  # (total_q, hidden)
        positions = position_ids.reshape(-1)

        q = self.q_norm(self.q_proj(x).view(total_q, self.num_heads, self.head_dim))
        # K/V run through their SConvs BEFORE caching; the cache stores what
        # attention consumes (post-conv, and for K post-norm), while the
        # conv_k/conv_v streams keep the pre-conv inputs decode needs.
        pre_k = self.k_proj(x)
        pre_v = self.v_proj(x)
        k = _sconv_cached(
            self.k_sconv, pre_k, past_key_values, cu_seqlens_q, self.layer_idx, "conv_k"
        )
        v = _sconv_cached(
            self.v_sconv, pre_v, past_key_values, cu_seqlens_q, self.layer_idx, "conv_v"
        )
        k = self.k_norm(k.view(total_q, self.num_kv_heads, self.head_dim))
        v = v.view(total_q, self.num_kv_heads, self.head_dim)
        past_key_values.append_stream_packed(k, cu_seqlens_q, self.layer_idx, "k")
        past_key_values.append_stream_packed(v, cu_seqlens_q, self.layer_idx, "v")

        rel_states = self.r_proj(x).view(total_q, self.num_heads, self.d_rel)

        # Global layers scale q (post-norm) and the bias by the log-length
        # factor, in fp32 like the reference.
        tau: torch.Tensor | None = None
        if self.log_scaling_n_floor is not None:
            effective_n = (positions + 1).float()
            tau = 1.0 + self.log_scaling_alpha * torch.log(
                (effective_n / self.log_scaling_n_floor).clamp(min=1.0)
            )
            q = (q.float() * tau[:, None, None]).to(q.dtype)

        keys_full, cu_seqlens_k, _ = past_key_values.materialize_packed_stream(self.layer_idx, "k")
        values_full, _, _ = past_key_values.materialize_packed_stream(self.layer_idx, "v")

        # Per-request additive mask: relative bias + causality (+ window),
        # consumed by the torch backend's block_mask path (which REPLACES
        # its built-in causal fill).
        block_mask: list[torch.Tensor] = []
        for batch_idx in range(cu_seqlens_q.shape[0] - 1):
            q_start, q_end = int(cu_seqlens_q[batch_idx]), int(cu_seqlens_q[batch_idx + 1])
            k_start, k_end = int(cu_seqlens_k[batch_idx]), int(cu_seqlens_k[batch_idx + 1])
            k_len = k_end - k_start
            q_positions = positions[q_start:q_end]
            k_positions = torch.arange(k_len, device=x.device)
            bias = self.rel_logits_proj(rel_states[q_start:q_end], q_positions, k_positions)
            bias = bias.float()
            if tau is not None:
                bias = bias * tau[q_start:q_end, None, None]
            distance = q_positions[:, None] - k_positions[None, :]
            invalid = distance < 0  # causal
            if self.sliding_window is not None:
                invalid = invalid | (distance > self.sliding_window - 1)
            block_mask.append(bias.masked_fill(invalid[:, None, :], -float("inf")))

        attn_packed = packed_attention_torch(
            q,
            keys_full,
            values_full,
            cu_seqlens_q,
            cu_seqlens_k,
            self.softmax_scale,
            block_mask=block_mask,
        )
        out: torch.Tensor = self.o_proj(attn_packed.reshape(total_q, -1))
        return out.view(*input_shape, -1)


class _InklingDecoderLayer(nn.Module):
    """Pre-norm block with SConvs on both branch outputs (inside the residual)."""

    def __init__(self, cfg: InklingConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = _InklingAttention(cfg, layer_idx)
        self.attn_sconv = InklingShortConv(cfg.hidden_size, cfg.conv_kernel_size)
        self.mlp_sconv = InklingShortConv(cfg.hidden_size, cfg.conv_kernel_size)
        if cfg.is_moe_layer(layer_idx):
            self.mlp: nn.Module = InklingMoE(
                hidden_size=cfg.hidden_size,
                moe_intermediate_size=cfg.moe_intermediate_size,
                n_routed_experts=cfg.n_routed_experts,
                n_shared_experts=cfg.n_shared_experts,
                top_k=cfg.num_experts_per_tok,
                route_scale=cfg.route_scale,
            )
        else:
            self.mlp = InklingDenseMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, position_ids, past_key_values, cu_seqlens_q)
        x = _sconv_cached(
            self.attn_sconv,
            x.reshape(-1, input_shape[-1]),
            past_key_values,
            cu_seqlens_q,
            self.layer_idx,
            "conv_attn",
        ).view(input_shape)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        x = _sconv_cached(
            self.mlp_sconv,
            x.reshape(-1, input_shape[-1]),
            past_key_values,
            cu_seqlens_q,
            self.layer_idx,
            "conv_mlp",
        ).view(input_shape)
        out: torch.Tensor = residual + x
        return out


class _InklingInnerModel(nn.Module):
    """Embedding (+ embed_norm) + N decoder layers + final norm. HF prefix `model.`."""

    def __init__(self, cfg: InklingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList(
            [_InklingDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


# Stacked-3D FFN tensors in HF's state_dict, split to our per-expert modules.
_EXPERT_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.gate_up_proj$")
_EXPERT_DOWN_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.down_proj$")
_SHARED_STACKED_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.shared_experts\.(gate|up|down)_proj$")
_SHARED_W = {"gate": "w1", "up": "w3", "down": "w2"}
# HF wraps each SConv weight in an nn.Conv1d: (C, 1, K) -> our (C, K).
_SCONV_RE = re.compile(r"^(.*\.(?:k|v|attn|mlp)_sconv)\.conv1d\.weight$")
# Drop: vision / audio towers and MTP draft layers.
_DROP_RE = re.compile(r"(^|\.)(vision_tower|audio_tower|mtp)\.")


@register_model
class InklingForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "InklingForConditionalGeneration"
    Config: ClassVar[type] = InklingConfig

    def __init__(self, cfg: InklingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _InklingInnerModel(cfg)
        # `embed` and `unembed` are separate tensors in the checkpoints, never tied.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PagedKVCache,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.embed_norm(self.model.embed_tokens(input_ids))
        for layer in self.model.layers:
            x = layer(x, position_ids, past_key_values, cu_seqlens_q)
        x = self.model.norm(x)
        # muP unembedding: divide by the width multiplier, then slice the
        # padded head rows off the logits.
        x = x / self.cfg.logits_mup_width_multiplier
        logits: torch.Tensor = self.lm_head(x)
        unpadded = self.cfg.unpadded_vocab_size
        if unpadded is not None and unpadded < logits.shape[-1]:
            logits = logits[..., :unpadded]
        return logits

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
        )

    def per_layer_attention(self) -> list[LayerAttentionSpec]:
        return [
            ("sliding", self.cfg.sliding_window_size) if self.cfg.is_sliding(i) else "full"
            for i in range(self.cfg.num_hidden_layers)
        ]

    def per_layer_kv_shape(self) -> list[tuple[int, int]]:
        return [
            (self.cfg.layer_heads(i)[1], self.cfg.layer_heads(i)[2])
            for i in range(self.cfg.num_hidden_layers)
        ]

    def per_layer_streams(self) -> list[list[StreamSpec]]:
        # Stream order == append order in the forward: conv_k first (its
        # layer-0 append triggers the step's block allocation), conv_mlp
        # last (its last-layer append triggers prefix-cache publish).
        streams: list[list[StreamSpec]] = []
        for layer_idx in range(self.cfg.num_hidden_layers):
            _, num_kv_heads, head_dim = self.cfg.layer_heads(layer_idx)
            streams.append(
                [
                    StreamSpec("conv_k", num_kv_heads, head_dim),
                    StreamSpec("conv_v", num_kv_heads, head_dim),
                    StreamSpec("k", num_kv_heads, head_dim),
                    StreamSpec("v", num_kv_heads, head_dim),
                    StreamSpec("conv_attn", 1, self.cfg.hidden_size),
                    StreamSpec("conv_mlp", 1, self.cfg.hidden_size),
                ]
            )
        return streams

    def required_attention_backend(self) -> str | None:
        # The relative-bias additive mask only exists in the materialized
        # SDPA reference (packed_attention_torch's block_mask path); HF
        # likewise ships this family with flash-attn disabled.
        return "torch"

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, InklingForCausalLM):
            raise TypeError(
                f"InklingForCausalLM.load_weights expects an InklingForCausalLM, "
                f"got {type(model).__name__}"
            )
        from mini_infer.distributed.group import get_world_size

        if get_world_size() != 1:
            raise NotImplementedError(
                "InklingForCausalLM is single-rank only for now: the SConv "
                "channel dim would shard with TP's KV split and the shared-"
                "expert sink normalization spans ranks. TP/EP is a follow-up."
            )
        remapped = _remap_inkling_state(hf_state_dict)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"weight load mismatch for InklingForCausalLM: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )


def _remap_inkling_state(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF tensors -> our module keys.

    Handles both the text-only `InklingForCausalLM` layout (`model.*`) and
    the multimodal `InklingForConditionalGeneration` one
    (`model.language_model.*` + dropped vision/audio towers). Stacked-3D
    expert tensors split into per-expert `MixtralExpert` weights; SConv
    weights lose HF's Conv1d singleton dim.
    """
    remapped: dict[str, torch.Tensor] = {}
    for raw_key, tensor in hf_state_dict.items():
        key = raw_key
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
        if _DROP_RE.search(key):
            continue

        gate_up = _EXPERT_GATE_UP_RE.match(key)
        if gate_up is not None:  # (E, 2*I, H): per expert -> w1 / w3
            prefix = gate_up.group(1)
            inter = tensor.shape[1] // 2
            for j in range(tensor.shape[0]):
                remapped[f"{prefix}.mlp.experts.{j}.w1.weight"] = tensor[j, :inter]
                remapped[f"{prefix}.mlp.experts.{j}.w3.weight"] = tensor[j, inter:]
            continue
        down = _EXPERT_DOWN_RE.match(key)
        if down is not None:  # (E, H, I): per expert -> w2
            for j in range(tensor.shape[0]):
                remapped[f"{down.group(1)}.mlp.experts.{j}.w2.weight"] = tensor[j]
            continue
        shared = _SHARED_STACKED_RE.match(key)
        if shared is not None:  # (S, I, H) / (S, H, I): per shared expert
            w_name = _SHARED_W[shared.group(2)]
            for j in range(tensor.shape[0]):
                remapped[f"{shared.group(1)}.mlp.shared_experts.{j}.{w_name}.weight"] = tensor[j]
            continue
        sconv = _SCONV_RE.match(key)
        if sconv is not None:  # (C, 1, K) -> (C, K)
            remapped[f"{sconv.group(1)}.weight"] = tensor.squeeze(1)
            continue

        remapped[key] = tensor
    return remapped
