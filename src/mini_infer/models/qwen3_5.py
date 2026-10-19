"""Qwen hybrid text model (Qwen3.5, Qwen3.6, Qwen3.8) on the state-cache path.

Qwen3.8 ships under the `qwen3_5` architecture: `Qwen/Qwen3.8-27B` declares
`architectures: ["Qwen3_5ForConditionalGeneration"]`, and the same class
covers Qwen3.5-4B/9B/27B and Qwen3.6-27B. They differ only in dims and in
the Gated DeltaNet head ratio.

The decoder is a 3:1 hybrid given explicitly by the config's `layer_types`
list: three Gated DeltaNet (linear attention) layers for every one Gated
Attention (softmax) layer. For the 27B that is 48 linear and 16 full over
64 layers.

  - **Gated DeltaNet** layers carry a fixed-size recurrent matrix state plus
    one fused short-convolution tail, not per-token KV. The recurrence
    itself is `blocks/kda.py` (see `blocks/gdn.py` for why that reuse is
    exact); this file supplies the projections around it. The 27B broadcasts
    16 key heads up to 48 value heads, a 3x ratio where the 4B and 9B use 2x,
    so the ratio is always read from the config and never assumed.
  - **Gated Attention** layers are ordinary grouped-query attention with two
    twists: the output gate is FUSED into `q_proj` (which is therefore
    `num_attention_heads * head_dim * 2` wide and split per head, not as
    `[all queries | all gates]`), and rotary embedding is partial, covering
    only the leading `head_dim * partial_rotary_factor` dimensions.

Every RMSNorm in the decoder is the `(1 + weight)` form with zero-stored
weights, which is `GemmaRMSNorm`. The one exception is the norm inside the
Gated DeltaNet output gate, which is a plain ones-init affine and lives in
`blocks/gdn.py`.

Serving: `USES_STATE_CACHE = True` routes this family to the per-request
`Qwen35StateCache` and the state-cache scheduler, the same seam the Kimi
Linear port generalized.

Text only. The released checkpoints bundle a vision tower and a
multi-token-prediction head; both are dropped at load. The text backbone's
rotary embedding is nominally a three-component mrope, but for text-only
input all three position streams are identical and the interleaved mrope
then reduces to plain 1D partial rope exactly, so that is what this
implementation computes. `tests/unit/test_qwen3_5_parity.py` pins the
equivalence against the reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional

from mini_infer.cache.qwen3_5_state_cache import (
    Qwen35StateCache,
    QwenAttentionLayerState,
    QwenAttentionStateSpec,
    QwenGdnLayerState,
    QwenGdnStateSpec,
    QwenStateLayerSpec,
)
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks import GemmaRMSNorm, RotaryEmbedding, SwiGLU
from mini_infer.models.blocks.gdn import broadcast_gate, gated_rmsnorm_silu, gdn_gate
from mini_infer.models.blocks.kda import (
    causal_conv1d_prefill,
    causal_conv1d_step,
    kda_chunkwise,
    kda_recurrent,
    l2norm,
)
from mini_infer.models.blocks.rope import apply_rotary_pos_emb_partial

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# Tensors the released checkpoints ship that this text-only port does not use:
# the vision tower and the multi-token-prediction head.
_DROP_RE = re.compile(r"^(model\.visual\.|mtp\.)")
# The multimodal checkpoint nests the text backbone one level deeper than a
# text-only one would.
_LANGUAGE_MODEL_PREFIX = "model.language_model."


@dataclass
class Qwen35Config:
    """Text-backbone configuration for the Qwen hybrid families."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: list[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    max_position_embeddings: int

    @property
    def rotary_dim(self) -> int:
        """Leading slice of each head that actually rotates."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_channels(self) -> int:
        """Width of the fused q/k/v convolution."""
        return 2 * self.linear_key_dim + self.linear_value_dim

    @property
    def linear_head_ratio(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    @classmethod
    def from_hf(cls, hf_config: Any) -> Qwen35Config:
        """Build from a HuggingFace config, nested `text_config` or flat.

        The released checkpoints are multimodal, so the text fields sit under
        `text_config`. A raw `SimpleNamespace` (what the loader falls back to
        when it avoids `trust_remote_code`) can carry that as a plain dict,
        so both forms are accepted.
        """
        text = getattr(hf_config, "text_config", None) or hf_config
        if isinstance(text, dict):
            text = _DictNamespace(text)

        def get(name: str, default: Any = _MISSING) -> Any:
            value = getattr(text, name, _MISSING)
            if value is _MISSING:
                if default is _MISSING:
                    raise ValueError(f"Qwen3.5 config is missing required field {name!r}")
                return default
            return value

        # These are the assumptions the math below hardcodes. transformers
        # 5.14 hardcodes them too (neither flag appears anywhere in its
        # qwen3_5 module), so a checkpoint that set them differently would be
        # silently mis-served rather than loudly rejected.
        if get("attention_bias", False):
            raise ValueError("Qwen3.5 port assumes attention_bias=False")
        if not get("attn_output_gate", True):
            raise ValueError("Qwen3.5 port assumes attn_output_gate=True (gate fused into q_proj)")
        gate_type = get("output_gate_type", "swish")
        if gate_type != "swish":
            raise ValueError(f"Qwen3.5 port assumes output_gate_type='swish', got {gate_type!r}")

        rope = get("rope_parameters", None) or {}
        if not isinstance(rope, dict):
            rope = dict(getattr(rope, "__dict__", {}))
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"Qwen3.5 port assumes rope_type='default', got {rope_type!r}")
        rope_theta = float(rope.get("rope_theta", get("rope_theta", 10000.0)))
        partial = float(rope.get("partial_rotary_factor", get("partial_rotary_factor", 0.25)))

        num_layers = int(get("num_hidden_layers"))
        layer_types = get("layer_types", None)
        if layer_types is None:
            interval = int(get("full_attention_interval", 4))
            layer_types = [
                FULL_ATTENTION if (i + 1) % interval == 0 else LINEAR_ATTENTION
                for i in range(num_layers)
            ]
        layer_types = [str(t) for t in layer_types]
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but num_hidden_layers is {num_layers}"
            )
        unknown = sorted(set(layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION})
        if unknown:
            raise ValueError(f"unsupported layer_types {unknown}")

        num_k_heads = int(get("linear_num_key_heads"))
        num_v_heads = int(get("linear_num_value_heads"))
        if num_k_heads <= 0 or num_v_heads % num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({num_v_heads}) must be a positive multiple of "
                f"linear_num_key_heads ({num_k_heads})"
            )

        hidden_size = int(get("hidden_size"))
        return cls(
            vocab_size=int(get("vocab_size")),
            hidden_size=hidden_size,
            intermediate_size=int(get("intermediate_size")),
            num_hidden_layers=num_layers,
            num_attention_heads=int(get("num_attention_heads")),
            num_key_value_heads=int(get("num_key_value_heads")),
            head_dim=int(get("head_dim", hidden_size // int(get("num_attention_heads")))),
            layer_types=layer_types,
            linear_num_key_heads=num_k_heads,
            linear_num_value_heads=num_v_heads,
            linear_key_head_dim=int(get("linear_key_head_dim")),
            linear_value_head_dim=int(get("linear_value_head_dim")),
            linear_conv_kernel_dim=int(get("linear_conv_kernel_dim", 4)),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            partial_rotary_factor=partial,
            tie_word_embeddings=bool(get("tie_word_embeddings", False)),
            max_position_embeddings=int(get("max_position_embeddings", 32768)),
        )


_MISSING = object()


class _DictNamespace:
    """Attribute view over a dict, for raw configs that nest `text_config` as one."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def build_qwen35_state_cache_specs(
    cfg: Qwen35Config, *, max_seq_len: int
) -> list[QwenStateLayerSpec]:
    """One cache spec per decoder layer, keyed off `cfg.layer_types`."""
    specs: list[QwenStateLayerSpec] = []
    for kind in cfg.layer_types:
        if kind == LINEAR_ATTENTION:
            specs.append(
                QwenGdnStateSpec(
                    num_v_heads=cfg.linear_num_value_heads,
                    head_k_dim=cfg.linear_key_head_dim,
                    head_v_dim=cfg.linear_value_head_dim,
                    conv_channels=cfg.linear_conv_channels,
                    conv_kernel_size=cfg.linear_conv_kernel_dim,
                )
            )
        else:
            specs.append(
                QwenAttentionStateSpec(
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    max_seq_len=max_seq_len,
                )
            )
    return specs


class _DepthwiseConv1d(nn.Module):
    """Holder for the fused depthwise conv taps.

    The checkpoint stores `(channels, 1, kernel)`; the taps are squeezed to
    `(channels, kernel)` at load so this stays the shape `blocks/kda.py`'s
    convolution helpers expect.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size))


class _GdnOutputNorm(nn.Module):
    """Ones-init affine for the Gated DeltaNet output gate.

    Separate from the decoder's `GemmaRMSNorm` because this one is a plain
    RMSNorm affine, not the `(1 + weight)` form.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _QwenGatedDeltaNet(nn.Module):
    """Linear-attention layer: fused q/k/v projection, short conv, delta rule.

    Attribute names mirror the checkpoint so weights load with no remapping
    beyond the prefix strip and the conv squeeze.
    """

    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, cfg.linear_conv_channels, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, cfg.linear_value_dim, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, cfg.linear_num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, cfg.linear_num_value_heads, bias=False)
        self.conv1d = _DepthwiseConv1d(cfg.linear_conv_channels, cfg.linear_conv_kernel_dim)
        self.A_log = nn.Parameter(torch.zeros(cfg.linear_num_value_heads))
        self.dt_bias = nn.Parameter(torch.zeros(cfg.linear_num_value_heads))
        self.norm = _GdnOutputNorm(cfg.linear_value_head_dim)
        self.out_proj = nn.Linear(cfg.linear_value_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_state: QwenGdnLayerState | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        del row_starts  # A linear-attention layer is position-free.
        cfg = self.cfg
        batch, seq_len, _ = hidden.shape

        # z, beta and the decay gate read the PRE-convolution hidden state;
        # only the fused q/k/v path is convolved.
        z = self.in_proj_z(hidden)
        beta = self.in_proj_b(hidden).sigmoid().float()
        gate = gdn_gate(self.in_proj_a(hidden), self.A_log, self.dt_bias)

        mixed = self.in_proj_qkv(hidden)
        conv_state = layer_state.conv if layer_state is not None else None
        if layer_state is not None and seq_len == 1:
            conved, new_conv = causal_conv1d_step(mixed[:, 0], self.conv1d.weight, layer_state.conv)
            conved = conved.unsqueeze(1)
        else:
            conved, new_conv = causal_conv1d_prefill(mixed, self.conv1d.weight, conv_state)
        if layer_state is not None:
            # Write in place, as the Kimi KDA layer does. Rebinding would turn the
            # cache tensor into an inference tensor that the scheduler's row copy
            # then cannot update outside `torch.inference_mode()`.
            layer_state.conv.copy_(new_conv)

        key_dim = cfg.linear_key_dim
        query, key, value = torch.split(conved, [key_dim, key_dim, cfg.linear_value_dim], dim=-1)
        query = query.reshape(batch, seq_len, -1, cfg.linear_key_head_dim)
        key = key.reshape(batch, seq_len, -1, cfg.linear_key_head_dim)
        value = value.reshape(batch, seq_len, -1, cfg.linear_value_head_dim)

        ratio = cfg.linear_head_ratio
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        # The reference normalizes and scales inside its kernel; we keep it at
        # the call site so the shared recurrence stays the bare delta rule.
        query = l2norm(query) * (cfg.linear_key_head_dim**-0.5)
        key = l2norm(key)
        gate_channels = broadcast_gate(gate, cfg.linear_key_head_dim)

        initial_state = layer_state.recurrent_state if layer_state is not None else None
        if layer_state is not None and seq_len == 1:
            core, final_state = kda_recurrent(
                query, key, value, gate_channels, beta, initial_state=initial_state
            )
        else:
            core, final_state = kda_chunkwise(
                query, key, value, gate_channels, beta, initial_state=initial_state
            )
        if layer_state is not None:
            layer_state.recurrent_state.copy_(final_state)

        core = core.reshape(-1, cfg.linear_value_head_dim)
        gated = gated_rmsnorm_silu(
            core, z.reshape(-1, cfg.linear_value_head_dim), self.norm.weight, cfg.rms_norm_eps
        )
        out: torch.Tensor = self.out_proj(gated.reshape(batch, seq_len, -1))
        return out


class _QwenGatedAttention(nn.Module):
    """Grouped-query attention with a fused output gate and partial rotary.

    `q_proj` is twice the usual width: each head's slice holds its query and
    its gate side by side, so the split is per head. Splitting the flat
    projection into `[all queries | all gates]` produces a plausible-looking
    but numerically wrong result.
    """

    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(
            cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.rotary_dim, base=cfg.rope_theta)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_state: QwenAttentionLayerState | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch, seq_len, _ = hidden.shape
        heads, kv_heads, head_dim = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )

        query, gate = torch.chunk(
            self.q_proj(hidden).view(batch, seq_len, -1, head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch, seq_len, -1)
        query = self.q_norm(query.reshape(batch, seq_len, heads, head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden).view(batch, seq_len, kv_heads, head_dim)).transpose(
            1, 2
        )
        value = self.v_proj(hidden).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)

        # Absolute positions per row, so a chunked-prefill continuation or a
        # ragged decode step positions each request correctly.
        offsets = torch.arange(seq_len, device=hidden.device, dtype=torch.long)
        positions = row_starts.view(batch, 1) + offsets.view(1, seq_len)
        cos, sin = self.rotary(hidden, positions)
        query, key = apply_rotary_pos_emb_partial(query, key, cos, sin)

        if layer_state is None:
            keys, values = key, value
            key_positions = positions
        else:
            rows = torch.arange(batch, device=hidden.device).view(batch, 1)
            layer_state.k[rows, positions] = key.transpose(1, 2).to(layer_state.k.dtype)
            layer_state.v[rows, positions] = value.transpose(1, 2).to(layer_state.v.dtype)
            total = int(positions.max().item()) + 1
            keys = layer_state.k[:, :total].transpose(1, 2).to(query.dtype)
            values = layer_state.v[:, :total].transpose(1, 2).to(query.dtype)
            key_positions = (
                torch.arange(total, device=hidden.device).view(1, total).expand(batch, total)
            )

        # A key is visible to a query when it is at or before the query's
        # absolute position. With a cache this also masks the buffer slots
        # this request has not written yet.
        allowed = key_positions.view(batch, 1, 1, -1) <= positions.view(batch, 1, seq_len, 1)
        keys = keys.repeat_interleave(heads // kv_heads, dim=1)
        values = values.repeat_interleave(heads // kv_heads, dim=1)
        attn = functional.scaled_dot_product_attention(
            query, keys, values, attn_mask=allowed, scale=head_dim**-0.5
        )
        attn = attn.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)
        out: torch.Tensor = self.o_proj(attn * torch.sigmoid(gate))
        return out


class _QwenDecoderLayer(nn.Module):
    """One decoder layer; `layer_kind` picks which mixer it holds."""

    def __init__(self, cfg: Qwen35Config, layer_kind: str) -> None:
        super().__init__()
        self.layer_kind = layer_kind
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if layer_kind == LINEAR_ATTENTION:
            self.linear_attn = _QwenGatedDeltaNet(cfg)
        else:
            self.self_attn = _QwenGatedAttention(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_state: QwenGdnLayerState | QwenAttentionLayerState | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        mixer = self.linear_attn if self.layer_kind == LINEAR_ATTENTION else self.self_attn
        hidden = hidden + mixer(self.input_layernorm(hidden), layer_state, row_starts)
        out: torch.Tensor = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return out


class _QwenInnerModel(nn.Module):
    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(_QwenDecoderLayer(cfg, kind) for kind in cfg.layer_types)
        self.norm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@register_model
class Qwen35ForConditionalGeneration(BaseCausalLM):
    """Qwen hybrid text backbone (Qwen3.5 / Qwen3.6 / Qwen3.8)."""

    HF_ARCHITECTURE: ClassVar[str] = "Qwen3_5ForConditionalGeneration"
    Config: ClassVar[type] = Qwen35Config
    USES_STATE_CACHE: ClassVar[bool] = True

    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _QwenInnerModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _run(
        self,
        input_ids: torch.Tensor,
        state_cache: Qwen35StateCache | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        for layer_idx, layer_module in enumerate(self.model.layers):
            layer_state = state_cache.layer(layer_idx) if state_cache is not None else None
            hidden = layer_module(hidden, layer_state, row_starts)
        hidden = self.model.norm(hidden)
        logits: torch.Tensor = self.lm_head(hidden)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standalone stateless prefill: `(B, T)` -> `(B, T, vocab)` logits.

        `past_key_values` and `cu_seqlens_q` keep the registry signature but
        are unused; this family decodes through `forward_decode_with_cache`,
        not PagedKVCache. `position_ids`, when given, supplies each row's
        starting absolute position.
        """
        del past_key_values, cu_seqlens_q
        batch = input_ids.shape[0]
        if position_ids is None:
            row_starts = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        else:
            row_starts = position_ids.reshape(batch, -1)[:, 0].to(torch.long)
        return self._run(input_ids, None, row_starts)

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        """Reported size only; the family uses `Qwen35StateCache`.

        Only the full-attention layers hold per-token entries, so the
        reported per-token cost counts those heads.
        """
        return KVCacheDims(
            num_layers=sum(1 for k in self.cfg.layer_types if k == FULL_ATTENTION),
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
        )

    def build_state_cache(
        self,
        *,
        max_seq_len: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Qwen35StateCache:
        return Qwen35StateCache(
            build_qwen35_state_cache_specs(self.cfg, max_seq_len=max_seq_len),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        state_cache: Qwen35StateCache,
    ) -> torch.Tensor:
        """Cache-aware prefill of `(B, T)`, CONTINUING from `state_cache.start_pos`.

        At `start_pos == 0` this is a fresh prefill; at `start_pos > 0` it is
        a chunked-prefill continuation, where the recurrent state and conv
        tails carry the left context and the attention buffers append at the
        offset. The caller advances `state_cache.start_pos` by `T` afterwards.
        """
        self._check_cache(state_cache)
        batch, seq_len = input_ids.shape
        state_cache.check_capacity(state_cache.start_pos + seq_len)
        row_starts = torch.full(
            (batch,), state_cache.start_pos, dtype=torch.long, device=input_ids.device
        )
        return self._run(input_ids, state_cache, row_starts)

    def forward_decode_with_cache(
        self,
        input_id: torch.Tensor,
        *,
        start_pos: int,
        state_cache: Qwen35StateCache,
    ) -> torch.Tensor:
        """One decode step at a uniform position: `(B, 1)` -> `(B, 1, vocab)`."""
        if input_id.shape[-1] != 1:
            raise ValueError(
                f"forward_decode_with_cache expects shape (B, 1), got {tuple(input_id.shape)}"
            )
        self._check_cache(state_cache)
        state_cache.check_capacity(start_pos + 1)
        batch = input_id.shape[0]
        positions = torch.full((batch,), start_pos, dtype=torch.long, device=input_id.device)
        return self._run(input_id, state_cache, positions)

    def forward_decode_with_cache_ragged(
        self,
        input_id: torch.Tensor,
        *,
        positions: torch.Tensor,
        state_cache: Qwen35StateCache,
    ) -> torch.Tensor:
        """One ragged decode step: B requests, each at its own `positions[b]`.

        The Gated DeltaNet layers are position-free so for them this is the
        batched step; only the full-attention layers consult `positions`, for
        per-row buffer offsets, rotary angles and causal lengths.
        """
        if input_id.shape[-1] != 1:
            raise ValueError(
                f"forward_decode_with_cache_ragged expects (B, 1), got {tuple(input_id.shape)}"
            )
        self._check_cache(state_cache)
        batch = input_id.shape[0]
        if positions.shape != (batch,):
            raise ValueError(f"positions shape {tuple(positions.shape)} != (B={batch},)")
        state_cache.check_capacity(int(positions.max().item()) + 1)
        return self._run(input_id, state_cache, positions.to(torch.long))

    def _check_cache(self, state_cache: Qwen35StateCache) -> None:
        if state_cache.num_layers != self.cfg.num_hidden_layers:
            raise ValueError(
                f"state_cache has {state_cache.num_layers} layers, "
                f"model has {self.cfg.num_hidden_layers}"
            )

    def expected_missing_state_keys(self) -> set[str]:
        return {"lm_head.weight"} if self.cfg.tie_word_embeddings else set()

    @classmethod
    def from_checkpoint(
        cls,
        name_or_path: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Qwen35ForConditionalGeneration:
        """Build the model in `dtype` on `device`, then stream the weights in.

        The generic `load_model` builds the model in fp32 on the host and then
        holds the whole checkpoint next to it, about 4x the bf16 weights. For
        the 4B that is more than a 16 GB machine has. Here the model is built
        directly in `dtype` on `device` and the checkpoint is copied in one
        shard at a time, so peak memory is the model plus one shard.
        `StateCacheGenerator.from_pretrained` calls this method.
        """
        from transformers import AutoConfig

        cfg = Qwen35Config.from_hf(AutoConfig.from_pretrained(name_or_path))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = cls(cfg)
        finally:
            torch.set_default_dtype(previous_dtype)
        model.eval()
        cls.load_weights_streaming(model, name_or_path)
        return model

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, Qwen35ForConditionalGeneration):
            raise TypeError(
                f"Qwen35ForConditionalGeneration.load_weights expects a "
                f"Qwen35ForConditionalGeneration, got {type(model).__name__}"
            )
        _require_single_rank()
        remapped = _remap_qwen35_state(hf_state_dict)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        missing_set = set(missing) - model.expected_missing_state_keys()
        if missing_set or unexpected:
            raise ValueError(
                f"weight load mismatch for Qwen35ForConditionalGeneration: "
                f"missing={sorted(missing_set)}, unexpected={sorted(unexpected)}"
            )

    @staticmethod
    def load_weights_streaming(model: BaseCausalLM, name_or_path: str) -> None:
        """Copy a safetensors checkpoint into `model` one shard at a time.

        Same key handling and coverage check as `load_weights`. Each shard is
        read on the host in the model's dtype, copied into the existing
        parameters (which keep their device), and released before the next
        shard loads.
        """
        if not isinstance(model, Qwen35ForConditionalGeneration):
            raise TypeError(
                f"Qwen35ForConditionalGeneration.load_weights_streaming expects a "
                f"Qwen35ForConditionalGeneration, got {type(model).__name__}"
            )
        _require_single_rank()
        from mini_infer.models.loader import iter_safetensors_shards

        dtype = next(model.parameters()).dtype
        model_keys = set(model.state_dict().keys())
        consumed: set[str] = set()
        for shard in iter_safetensors_shards(name_or_path, device="cpu", dtype=dtype):
            remapped = _remap_qwen35_state(shard)
            # Drop this loop's references before the next shard loads, or two
            # shards are alive at once.
            del shard
            if remapped:
                missing, unexpected = model.load_state_dict(remapped, strict=False)
                if unexpected:
                    raise ValueError(
                        f"weight load mismatch for Qwen35ForConditionalGeneration: "
                        f"unexpected={sorted(unexpected)}"
                    )
                consumed |= model_keys - set(missing)
            del remapped
        leftover = model_keys - consumed - model.expected_missing_state_keys()
        if leftover:
            raise ValueError(
                f"weight load mismatch for Qwen35ForConditionalGeneration: "
                f"missing={sorted(leftover)}"
            )


def _require_single_rank() -> None:
    from mini_infer.distributed.group import get_world_size

    if get_world_size() != 1:
        raise NotImplementedError(
            "Qwen35ForConditionalGeneration is single-rank only for now: TP would "
            "shard the fused Gated DeltaNet conv channels and the per-head matrix "
            "state. A follow-up, like Kimi Linear."
        )


def _remap_qwen35_state(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint keys to module keys.

    Three adjustments: drop the vision tower and the multi-token-prediction
    head, lift the text backbone out of `model.language_model.` (the
    multimodal wrapper's nesting), and squeeze the depthwise conv taps from
    `(channels, 1, kernel)` to `(channels, kernel)`.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        if _DROP_RE.match(key):
            continue
        name = key
        if name.startswith(_LANGUAGE_MODEL_PREFIX):
            name = "model." + name[len(_LANGUAGE_MODEL_PREFIX) :]
        if name.endswith("conv1d.weight") and tensor.dim() == 3:
            tensor = tensor.squeeze(1)
        remapped[name] = tensor
    return remapped


__all__ = [
    "Qwen35Config",
    "Qwen35ForConditionalGeneration",
    "build_qwen35_state_cache_specs",
]
