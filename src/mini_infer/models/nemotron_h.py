"""NVIDIA Nemotron-H hybrid (Nemotron 3.5 Lightning and family) on the state-cache path.

Every decoder layer is one RMSNorm and one mixer with a residual add. The
mixer kind comes from the config's `layers_block_type` list:

  - **Mamba-2** (`mamba`): a selective state-space layer with a fixed-size
    state per head and a short conv over its x, B and C channels. See
    `blocks/mamba2.py`.
  - **Attention** (`attention`): grouped-query attention with no positional
    encoding. Order comes from the Mamba layers.
  - **MoE** (`moe`): DeepSeek-V3-style sigmoid router with an aux-free
    selection bias (`GlmNoAuxTcGate`), routed experts that are NON-gated
    MLPs with a squared-ReLU activation, and one always-on shared expert.
    An optional latent projection wraps the routed experts.
  - **Dense MLP** (`mlp`): one squared-ReLU MLP.

Nemotron 3.5 Lightning 30B-A3B has 52 layers: 23 Mamba-2, 23 MoE and 6
attention. The checkpoint stores the backbone under `backbone.`, routed
experts one tensor per expert, the conv taps as `(channels, 1, kernel)`, and
one multi-token-prediction layer under `mtp.`, which this port drops.

Serving: `USES_STATE_CACHE = True` routes the family to
`NemotronHStateCache` and the state-cache scheduler. The decode step has no
host sync and no data-dependent shape, so it declares
`SUPPORTS_DECODE_CUDA_GRAPH`.

Numerics follow the `mamba_ssm` kernels the model is served with: the SSM
runs in fp32 and the step size is not clamped (`time_step_limit` defaults to
`(0, inf)`). The transformers CPU fallback also clamps the step size at
`time_step_min`, which only matters for steps below 1e-3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional

from mini_infer.cache.nemotron_h_state_cache import (
    NemotronAttentionLayerState,
    NemotronAttentionStateSpec,
    NemotronHStateCache,
    NemotronMambaLayerState,
    NemotronMambaStateSpec,
    NemotronStateLayerSpec,
)
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks.glm_moe_gate import GlmNoAuxTcGate
from mini_infer.models.blocks.grouped_experts import grouped_expert_mlp, supports_grouped_experts
from mini_infer.models.blocks.mamba2 import (
    gated_group_rmsnorm,
    mamba_conv_prefill,
    mamba_conv_step,
    ssd_chunked,
    ssd_recurrent,
)
from mini_infer.models.blocks.rmsnorm import RMSNorm

MAMBA = "mamba"
ATTENTION = "attention"
MOE = "moe"
MLP = "mlp"

# The config spells layer kinds three ways: the on-disk names, the names
# transformers remaps them to, and the one-letter `hybrid_override_pattern`.
_LAYER_KIND = {
    "mamba": MAMBA,
    "linear_attention": MAMBA,
    "M": MAMBA,
    "attention": ATTENTION,
    "full_attention": ATTENTION,
    "*": ATTENTION,
    "moe": MOE,
    "E": MOE,
    "mlp": MLP,
    "-": MLP,
}

_DROP_RE = re.compile(r"^mtp\.")
_EXPERT_RE = re.compile(
    r"^(backbone\.layers\.\d+\.mixer)\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)

# Without grouped matmuls (CPU, MPS, pre-Hopper GPUs): up to this many
# (token, expert) pairs, the MoE gathers the chosen experts' weights instead
# of looping over experts. The gather has no host sync, so a decode step stays
# capturable; the loop is cheaper for long prefills.
_GATHER_EXPERT_LIMIT = 64

# Host memory the streaming loader holds at once.
_STREAM_BATCH_BYTES = 1 << 30


@dataclass
class NemotronHConfig:
    """Configuration of the Nemotron-H text backbone."""

    vocab_size: int
    hidden_size: int
    layer_types: list[str]
    norm_eps: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    # Attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    # Mamba-2
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    n_groups: int
    conv_kernel: int
    use_conv_bias: bool
    use_bias: bool
    chunk_size: int
    time_step_limit: tuple[float, float]
    # MLP and MoE
    intermediate_size: int
    mlp_bias: bool
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    moe_latent_size: int | None

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    @property
    def mamba_intermediate_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_channels(self) -> int:
        """Width of the conv over the concatenated x, B and C channels."""
        return self.mamba_intermediate_size + 2 * self.n_groups * self.ssm_state_size

    @classmethod
    def from_hf(cls, hf_config: Any) -> NemotronHConfig:
        """Build from a transformers config or a raw `config.json` namespace."""

        def get(name: str, default: Any = None, *, required: bool = False) -> Any:
            value = getattr(hf_config, name, None)
            if value is None:
                if required:
                    raise ValueError(f"Nemotron-H config is missing required field {name!r}")
                return default
            return value

        raw_types = get("layers_block_type") or get("layer_types")
        if raw_types is None:
            pattern = get("hybrid_override_pattern", required=True)
            raw_types = list(pattern)
        unknown = sorted({str(kind) for kind in raw_types} - set(_LAYER_KIND))
        if unknown:
            raise ValueError(f"unsupported Nemotron-H layer types {unknown}")
        layer_types = [_LAYER_KIND[str(kind)] for kind in raw_types]

        # The math below hardcodes these; reject a checkpoint that differs.
        if get("mamba_hidden_act", "silu") != "silu":
            raise ValueError("Nemotron-H port assumes mamba_hidden_act='silu'")
        if get("mlp_hidden_act", "relu2") != "relu2":
            raise ValueError("Nemotron-H port assumes mlp_hidden_act='relu2'")
        if get("sliding_window") is not None:
            raise ValueError("Nemotron-H port does not support sliding_window")

        heads = int(get("mamba_num_heads", required=True))
        groups = int(get("n_groups", 1))
        if heads % groups:
            raise ValueError(f"mamba_num_heads ({heads}) must be a multiple of n_groups ({groups})")
        num_attention_heads = int(get("num_attention_heads", required=True))
        num_kv_heads = int(get("num_key_value_heads", num_attention_heads))
        if num_attention_heads % num_kv_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        hidden_size = int(get("hidden_size", required=True))
        limit = get("time_step_limit", (0.0, float("inf")))
        latent = get("moe_latent_size")
        return cls(
            vocab_size=int(get("vocab_size", required=True)),
            hidden_size=hidden_size,
            layer_types=layer_types,
            norm_eps=float(get("layer_norm_epsilon", get("norm_eps", 1e-5))),
            tie_word_embeddings=bool(get("tie_word_embeddings", False)),
            max_position_embeddings=int(get("max_position_embeddings", 4096)),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=int(get("head_dim", hidden_size // num_attention_heads)),
            attention_bias=bool(get("attention_bias", False)),
            mamba_num_heads=heads,
            mamba_head_dim=int(get("mamba_head_dim", required=True)),
            ssm_state_size=int(get("ssm_state_size", required=True)),
            n_groups=groups,
            conv_kernel=int(get("conv_kernel", 4)),
            use_conv_bias=bool(get("use_conv_bias", True)),
            use_bias=bool(get("use_bias", False)),
            chunk_size=int(get("chunk_size", 128)),
            time_step_limit=(float(limit[0]), float(limit[1])),
            intermediate_size=int(get("intermediate_size", 0)),
            mlp_bias=bool(get("mlp_bias", False)),
            n_routed_experts=int(get("n_routed_experts", 0)),
            num_experts_per_tok=int(get("num_experts_per_tok", 0)),
            moe_intermediate_size=int(get("moe_intermediate_size", 0)),
            moe_shared_expert_intermediate_size=int(get("moe_shared_expert_intermediate_size", 0)),
            n_group=int(get("n_group", 1)),
            topk_group=int(get("topk_group", 1)),
            norm_topk_prob=bool(get("norm_topk_prob", True)),
            routed_scaling_factor=float(get("routed_scaling_factor", 1.0)),
            moe_latent_size=None if latent is None else int(latent),
        )


def build_nemotron_h_state_cache_specs(
    cfg: NemotronHConfig, *, max_seq_len: int
) -> list[NemotronStateLayerSpec]:
    """One cache spec per layer; MoE and MLP layers get None."""
    specs: list[NemotronStateLayerSpec] = []
    for kind in cfg.layer_types:
        if kind == MAMBA:
            specs.append(
                NemotronMambaStateSpec(
                    num_heads=cfg.mamba_num_heads,
                    head_dim=cfg.mamba_head_dim,
                    state_size=cfg.ssm_state_size,
                    conv_channels=cfg.mamba_conv_channels,
                    conv_kernel_size=cfg.conv_kernel,
                )
            )
        elif kind == ATTENTION:
            specs.append(
                NemotronAttentionStateSpec(
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    max_seq_len=max_seq_len,
                )
            )
        else:
            specs.append(None)
    return specs


class _DepthwiseConv(nn.Module):
    """Conv taps `(channels, kernel)` and bias.

    The checkpoint stores the taps as `(channels, 1, kernel)`; the loader squeezes them.
    """

    def __init__(self, channels: int, kernel_size: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None


class _NormWeight(nn.Module):
    """Holder for the Mamba-2 gated norm's affine weight (ones-init)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _NemotronMamba2Mixer(nn.Module):
    """Mamba-2 mixer: in_proj, short conv, SSM, gated group norm, out_proj."""

    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        inner = cfg.mamba_intermediate_size
        heads = cfg.mamba_num_heads
        self.in_proj = nn.Linear(
            cfg.hidden_size, inner + cfg.mamba_conv_channels + heads, bias=cfg.use_bias
        )
        self.conv1d = _DepthwiseConv(
            cfg.mamba_conv_channels, cfg.conv_kernel, bias=cfg.use_conv_bias
        )
        self.dt_bias = nn.Parameter(torch.ones(heads))
        self.A_log = nn.Parameter(torch.arange(1, heads + 1, dtype=torch.get_default_dtype()).log())
        self.D = nn.Parameter(torch.ones(heads))
        self.norm = _NormWeight(inner)
        self.out_proj = nn.Linear(inner, cfg.hidden_size, bias=cfg.use_bias)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_state: NemotronMambaLayerState | None,
        row_starts: torch.Tensor,
        kv_len: int | None,
    ) -> torch.Tensor:
        del row_starts, kv_len  # The SSM is position-free.
        cfg = self.cfg
        batch, seq_len, _ = hidden.shape
        inner, heads, head_dim = (
            cfg.mamba_intermediate_size,
            cfg.mamba_num_heads,
            cfg.mamba_head_dim,
        )
        group_width = cfg.n_groups * cfg.ssm_state_size

        gate, xbc, dt_raw = torch.split(
            self.in_proj(hidden), [inner, cfg.mamba_conv_channels, heads], dim=-1
        )
        decode = layer_state is not None and seq_len == 1
        if layer_state is not None and decode:
            conved, new_conv = mamba_conv_step(
                xbc[:, 0], self.conv1d.weight, self.conv1d.bias, layer_state.conv
            )
            conved = conved.unsqueeze(1)
        else:
            conved, new_conv = mamba_conv_prefill(
                xbc,
                self.conv1d.weight,
                self.conv1d.bias,
                layer_state.conv if layer_state is not None else None,
            )
        if layer_state is not None:
            # In place: the cache tensors must stay the ones a CUDA graph captured.
            layer_state.conv.copy_(new_conv)

        x, b, c = torch.split(conved, [inner, group_width, group_width], dim=-1)
        x = x.reshape(batch, seq_len, heads, head_dim).float()
        repeat = heads // cfg.n_groups
        b = b.reshape(batch, seq_len, cfg.n_groups, -1).float().repeat_interleave(repeat, dim=2)
        c = c.reshape(batch, seq_len, cfg.n_groups, -1).float().repeat_interleave(repeat, dim=2)
        dt = functional.softplus(dt_raw.float() + self.dt_bias.float())
        low, high = cfg.time_step_limit
        if low > 0.0 or high != float("inf"):
            dt = dt.clamp(min=low, max=high)
        a = -torch.exp(self.A_log.float())

        initial = layer_state.ssm if layer_state is not None else None
        if decode:
            y, final_state = ssd_recurrent(x, dt, a, b, c, initial_state=initial)
        else:
            y, final_state = ssd_chunked(
                x, dt, a, b, c, chunk_size=cfg.chunk_size, initial_state=initial
            )
        if layer_state is not None:
            layer_state.ssm.copy_(final_state)

        y = y + x * self.D.float()[:, None]
        y = y.reshape(batch, seq_len, inner)
        normed = gated_group_rmsnorm(
            y, gate, self.norm.weight, group_size=inner // cfg.n_groups, eps=cfg.norm_eps
        )
        out: torch.Tensor = self.out_proj(normed.to(hidden.dtype))
        return out


class _NemotronAttention(nn.Module):
    """Grouped-query attention with no positional encoding."""

    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        heads, kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, heads * head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_heads * head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_heads * head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(heads * head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_state: NemotronAttentionLayerState | None,
        row_starts: torch.Tensor,
        kv_len: int | None,
    ) -> torch.Tensor:
        """`kv_len` is the cache length after a cached span of 2+ tokens, known on the host."""
        cfg = self.cfg
        batch, seq_len, _ = hidden.shape
        heads, kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        query = self.q_proj(hidden).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, seq_len, kv_heads, head_dim)
        value = self.v_proj(hidden).view(batch, seq_len, kv_heads, head_dim)

        offsets = torch.arange(seq_len, device=hidden.device, dtype=torch.long)
        positions = row_starts.view(batch, 1) + offsets.view(1, seq_len)
        if layer_state is None:
            keys, values = key.transpose(1, 2), value.transpose(1, 2)
            key_positions = positions
        else:
            rows = torch.arange(batch, device=hidden.device).view(batch, 1)
            layer_state.k[rows, positions] = key.to(layer_state.k.dtype)
            layer_state.v[rows, positions] = value.to(layer_state.v.dtype)
            if seq_len == 1:
                return self._project(self._decode_attention(query, layer_state, positions))
            if kv_len is None:
                raise ValueError("a cached span longer than one token needs kv_len")
            keys = layer_state.k[:, :kv_len].transpose(1, 2).to(query.dtype)
            values = layer_state.v[:, :kv_len].transpose(1, 2).to(query.dtype)
            key_positions = (
                torch.arange(kv_len, device=hidden.device).view(1, kv_len).expand(batch, kv_len)
            )

        allowed = key_positions.view(batch, 1, 1, -1) <= positions.view(batch, 1, seq_len, 1)
        group = heads // kv_heads
        attn = functional.scaled_dot_product_attention(
            query,
            keys.repeat_interleave(group, dim=1),
            values.repeat_interleave(group, dim=1),
            attn_mask=allowed,
            scale=head_dim**-0.5,
        )
        return self._project(attn.transpose(1, 2))

    def _decode_attention(
        self,
        query: torch.Tensor,
        layer_state: NemotronAttentionLayerState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """One token per row over the whole fixed-length buffer, masked on the device.

        No host sync and no data-dependent shape, so a CUDA graph can capture
        it. The query heads of each KV head are stacked on the query axis.
        Returns `(batch, 1, heads, head_dim)`.
        """
        cfg = self.cfg
        batch = query.shape[0]
        kv_heads, head_dim = cfg.num_key_value_heads, cfg.head_dim
        group = cfg.num_attention_heads // kv_heads
        buffer_len = layer_state.k.shape[1]
        slots = torch.arange(buffer_len, device=query.device)
        allowed = (slots.view(1, buffer_len) <= positions.view(batch, 1)).view(
            batch, 1, 1, buffer_len
        )
        attn = functional.scaled_dot_product_attention(
            query.reshape(batch, kv_heads, group, head_dim),
            layer_state.k.transpose(1, 2).to(query.dtype),
            layer_state.v.transpose(1, 2).to(query.dtype),
            attn_mask=allowed,
            scale=head_dim**-0.5,
        )
        return attn.reshape(batch, cfg.num_attention_heads, 1, head_dim).transpose(1, 2)

    def _project(self, attn: torch.Tensor) -> torch.Tensor:
        batch, seq_len = attn.shape[:2]
        out: torch.Tensor = self.o_proj(attn.reshape(batch, seq_len, -1))
        return out


def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    return functional.relu(x).square()


class _ReluSquaredMLP(nn.Module):
    """`down(relu(up(x)) ** 2)`: the non-gated MLP of the dense layers and the shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, bias: bool) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(functional.relu(self.up_proj(x)).square())
        return out


class _NemotronMoE(nn.Module):
    """Sigmoid-routed MoE with squared-ReLU experts and one shared expert.

    Routed experts are stored stacked, `experts_up` `(experts, intermediate,
    in)` and `experts_down` `(experts, in, intermediate)`; the loader copies
    the checkpoint's per-expert tensors into their slots. As in the
    reference, the weighted expert outputs accumulate in fp32. On Hopper and
    newer GPUs the stacked tensors feed `grouped_expert_mlp` directly; elsewhere
    the MoE gathers or loops, see `_GATHER_EXPERT_LIMIT`.
    """

    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = GlmNoAuxTcGate(
            hidden_size=cfg.hidden_size,
            n_routed_experts=cfg.n_routed_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            norm_topk_prob=cfg.norm_topk_prob,
            routed_scaling_factor=cfg.routed_scaling_factor,
        )
        expert_in = cfg.moe_latent_size or cfg.hidden_size
        self.experts_up = nn.Parameter(
            torch.empty(cfg.n_routed_experts, cfg.moe_intermediate_size, expert_in)
        )
        self.experts_down = nn.Parameter(
            torch.empty(cfg.n_routed_experts, expert_in, cfg.moe_intermediate_size)
        )
        # A checkpoint overwrites these; a fresh model must still be finite.
        nn.init.normal_(self.experts_up, std=0.02)
        nn.init.normal_(self.experts_down, std=0.02)
        self.shared_experts = _ReluSquaredMLP(
            cfg.hidden_size, cfg.moe_shared_expert_intermediate_size, bias=cfg.mlp_bias
        )
        self.fc1_latent_proj: nn.Linear | None = None
        self.fc2_latent_proj: nn.Linear | None = None
        if cfg.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                cfg.hidden_size, cfg.moe_latent_size, bias=cfg.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                cfg.moe_latent_size, cfg.hidden_size, bias=cfg.mlp_bias
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        indices, weights = self.gate(flat)
        routed_in = self.fc1_latent_proj(flat) if self.fc1_latent_proj is not None else flat
        if supports_grouped_experts(flat.device):
            routed = grouped_expert_mlp(
                routed_in,
                indices,
                weights,
                self.experts_up,
                self.experts_down,
                _relu_squared,
                num_experts=self.cfg.n_routed_experts,
                accumulate_dtype=torch.float32,
            )
        elif flat.shape[0] * self.cfg.num_experts_per_tok <= _GATHER_EXPERT_LIMIT:
            routed = self._gathered(routed_in, indices, weights)
        else:
            routed = self._looped(routed_in, indices, weights)
        if self.fc2_latent_proj is not None:
            routed = self.fc2_latent_proj(routed)
        out: torch.Tensor = (routed + self.shared_experts(flat)).view(shape)
        return out

    def _gathered(
        self, flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Gather each token's chosen experts; no host sync."""
        up = functional.relu(torch.einsum("td,tkid->tki", flat, self.experts_up[indices])).square()
        down = torch.einsum("tki,tkdi->tkd", up, self.experts_down[indices])
        return (down.float() * weights[..., None]).sum(dim=1).to(flat.dtype)

    def _looped(
        self, flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Run each expert once on its tokens; host syncs, cheaper for many tokens."""
        out = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        mask = functional.one_hot(indices, num_classes=self.cfg.n_routed_experts).permute(2, 1, 0)
        hit = torch.nonzero(mask.sum(dim=(1, 2)) > 0).squeeze(-1).tolist()
        for expert in hit:
            slot, token = torch.where(mask[expert])
            up = functional.relu(functional.linear(flat[token], self.experts_up[expert])).square()
            down = functional.linear(up, self.experts_down[expert])
            out.index_add_(0, token, (down * weights[token, slot, None]).to(out.dtype))
        return out.to(flat.dtype)


class _NemotronBlock(nn.Module):
    """One layer: `hidden + mixer(norm(hidden))`."""

    def __init__(self, cfg: NemotronHConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mixer: nn.Module
        if kind == MAMBA:
            self.mixer = _NemotronMamba2Mixer(cfg)
        elif kind == ATTENTION:
            self.mixer = _NemotronAttention(cfg)
        elif kind == MOE:
            self.mixer = _NemotronMoE(cfg)
        else:
            self.mixer = _ReluSquaredMLP(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_state: NemotronMambaLayerState | NemotronAttentionLayerState | None,
        row_starts: torch.Tensor,
        kv_len: int | None,
    ) -> torch.Tensor:
        normed = self.norm(hidden)
        if self.kind in (MAMBA, ATTENTION):
            mixed = self.mixer(normed, layer_state, row_starts, kv_len)
        else:
            mixed = self.mixer(normed)
        out: torch.Tensor = hidden + mixed
        return out


class _NemotronBackbone(nn.Module):
    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(_NemotronBlock(cfg, kind) for kind in cfg.layer_types)
        self.norm_f = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)


@register_model
class NemotronHForCausalLM(BaseCausalLM):
    """Nemotron-H hybrid text model (Nemotron 3.5 Lightning and family)."""

    HF_ARCHITECTURE: ClassVar[str] = "NemotronHForCausalLM"
    Config: ClassVar[type] = NemotronHConfig
    USES_STATE_CACHE: ClassVar[bool] = True
    # `decode_step_static` is safe to capture in a CUDA graph.
    SUPPORTS_DECODE_CUDA_GRAPH: ClassVar[bool] = True

    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _NemotronBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight

    def _run(
        self,
        input_ids: torch.Tensor,
        state_cache: NemotronHStateCache | None,
        row_starts: torch.Tensor,
        kv_len: int | None = None,
    ) -> torch.Tensor:
        """`kv_len`: cache length after this span, for cached spans of 2+ tokens."""
        hidden = self.backbone.embeddings(input_ids)
        for layer_idx, block in enumerate(self.backbone.layers):
            layer_state = state_cache.layer(layer_idx) if state_cache is not None else None
            hidden = block(hidden, layer_state, row_starts, kv_len)
        hidden = self.backbone.norm_f(hidden)
        logits: torch.Tensor = self.lm_head(hidden)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standalone stateless prefill: `(B, T)` -> `(B, T, vocab)` logits."""
        del past_key_values, cu_seqlens_q
        batch = input_ids.shape[0]
        if position_ids is None:
            row_starts = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        else:
            row_starts = position_ids.reshape(batch, -1)[:, 0].to(torch.long)
        return self._run(input_ids, None, row_starts)

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        """Reported size only: the attention layers are the per-token part."""
        return KVCacheDims(
            num_layers=sum(1 for kind in self.cfg.layer_types if kind == ATTENTION),
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
    ) -> NemotronHStateCache:
        return NemotronHStateCache(
            build_nemotron_h_state_cache_specs(self.cfg, max_seq_len=max_seq_len),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward_prefill_with_cache(
        self, input_ids: torch.Tensor, *, state_cache: NemotronHStateCache
    ) -> torch.Tensor:
        """Cache-aware prefill of `(B, T)`, continuing from `state_cache.start_pos`.

        The caller advances `state_cache.start_pos` by `T` afterwards.
        """
        self._check_cache(state_cache)
        batch, seq_len = input_ids.shape
        state_cache.check_capacity(state_cache.start_pos + seq_len)
        row_starts = torch.full(
            (batch,), state_cache.start_pos, dtype=torch.long, device=input_ids.device
        )
        return self._run(input_ids, state_cache, row_starts, state_cache.start_pos + seq_len)

    def forward_decode_with_cache(
        self, input_id: torch.Tensor, *, start_pos: int, state_cache: NemotronHStateCache
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
        state_cache: NemotronHStateCache,
    ) -> torch.Tensor:
        """One ragged decode step: B requests, each at its own `positions[b]`."""
        if input_id.shape[-1] != 1:
            raise ValueError(
                f"forward_decode_with_cache_ragged expects (B, 1), got {tuple(input_id.shape)}"
            )
        self._check_cache(state_cache)
        batch = input_id.shape[0]
        if positions.shape != (batch,):
            raise ValueError(f"positions shape {tuple(positions.shape)} != (B={batch},)")
        state_cache.check_capacity(int(positions.max().item()) + 1)
        return self.decode_step_static(input_id, positions.to(torch.long), state_cache)

    def decode_step_static(
        self, input_id: torch.Tensor, positions: torch.Tensor, state_cache: NemotronHStateCache
    ) -> torch.Tensor:
        """The ragged decode step without host-side checks, for `DecodeCudaGraph`.

        `(B, 1)` token ids and `(B,)` int64 positions on the model's device.
        No host sync, no data-dependent shape, and only in-place cache updates.
        """
        return self._run(input_id, state_cache, positions)

    def _check_cache(self, state_cache: NemotronHStateCache) -> None:
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
    ) -> NemotronHForCausalLM:
        """Build in `dtype` on `device`, then stream the weights in one shard at a time."""
        from transformers import AutoConfig

        cfg = NemotronHConfig.from_hf(AutoConfig.from_pretrained(name_or_path))
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
        if not isinstance(model, NemotronHForCausalLM):
            raise TypeError(
                f"NemotronHForCausalLM.load_weights expects a NemotronHForCausalLM, "
                f"got {type(model).__name__}"
            )
        _require_single_rank()
        tracker = _LoadTracker(model)
        tracker.load(hf_state_dict)
        tracker.finish()

    @staticmethod
    def load_weights_streaming(model: BaseCausalLM, name_or_path: str) -> None:
        """Copy a safetensors checkpoint into `model` about 1 GiB at a time.

        Every tensor loads on its own, so the loader may split a file into
        batches. Peak host memory then stays near one batch even for a
        single-file checkpoint such as the 8 GB Nemotron 3 Nano 4B.
        """
        if not isinstance(model, NemotronHForCausalLM):
            raise TypeError(
                f"NemotronHForCausalLM.load_weights_streaming expects a NemotronHForCausalLM, "
                f"got {type(model).__name__}"
            )
        _require_single_rank()
        from mini_infer.models.loader import iter_safetensors_shards

        tracker = _LoadTracker(model)
        dtype = next(model.parameters()).dtype
        for shard in iter_safetensors_shards(
            name_or_path, device="cpu", dtype=dtype, max_bytes=_STREAM_BATCH_BYTES
        ):
            tracker.load(shard)
            # Drop this loop's reference before the next shard loads.
            del shard
        tracker.finish()


class _LoadTracker:
    """Copies checkpoint tensors into a model and checks that every weight arrived.

    Plain tensors go through `load_state_dict`; routed experts are copied
    into their slot of the stacked expert parameters.
    """

    def __init__(self, model: NemotronHForCausalLM) -> None:
        self.model = model
        self.model_keys = set(model.state_dict().keys())
        self.expert_params = {
            name: param for name, param in model.named_parameters() if ".experts_" in name
        }
        self.consumed: set[str] = set()
        self.experts_loaded: dict[str, set[int]] = {name: set() for name in self.expert_params}

    def load(self, hf_state: dict[str, torch.Tensor]) -> None:
        plain: dict[str, torch.Tensor] = {}
        for key, tensor in hf_state.items():
            if _DROP_RE.match(key):
                continue
            match = _EXPERT_RE.match(key)
            if match is not None:
                suffix = "experts_up" if match.group(3) == "up_proj" else "experts_down"
                name = f"{match.group(1)}.{suffix}"
                param = self.expert_params.get(name)
                if param is None:
                    raise ValueError(f"weight load mismatch: unexpected expert tensor {key!r}")
                expert = int(match.group(2))
                with torch.no_grad():
                    param[expert].copy_(tensor)
                self.experts_loaded[name].add(expert)
                continue
            name = "backbone.embeddings.weight" if key == "backbone.embedding.weight" else key
            if name.endswith("mixer.conv1d.weight") and tensor.dim() == 3:
                tensor = tensor.squeeze(1)
            plain[name] = tensor
        if not plain:
            return
        missing, unexpected = self.model.load_state_dict(plain, strict=False)
        if unexpected:
            raise ValueError(f"weight load mismatch: unexpected={sorted(unexpected)[:8]}")
        self.consumed |= self.model_keys - set(missing)

    def finish(self) -> None:
        leftover = (
            self.model_keys
            - self.consumed
            - set(self.expert_params)
            - self.model.expected_missing_state_keys()
        )
        incomplete = [
            name
            for name, param in self.expert_params.items()
            if len(self.experts_loaded[name]) != param.shape[0]
        ]
        if leftover or incomplete:
            raise ValueError(
                f"weight load mismatch for NemotronHForCausalLM: missing={sorted(leftover)[:8]}, "
                f"incomplete experts={incomplete[:4]}"
            )


def _require_single_rank() -> None:
    from mini_infer.distributed.group import get_world_size

    if get_world_size() != 1:
        raise NotImplementedError(
            "NemotronHForCausalLM is single-rank only for now: TP would shard the "
            "Mamba heads, the conv channels and the experts."
        )


__all__ = [
    "NemotronHConfig",
    "NemotronHForCausalLM",
    "build_nemotron_h_state_cache_specs",
]
