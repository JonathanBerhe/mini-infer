"""Kimi Linear family (moonshotai/Kimi-Linear-48B-A3B, Kimi K3 text stack): owned implementation.

From-scratch port of Moonshot's hybrid linear-attention architecture
(arXiv:2510.26692), the attention foundation of Kimi K3. The parity
references are the checkpoints' own remote code (trust_remote_code, vendored
by `scripts/kimi_linear/clone_kimi_linear_reference.py` and
`scripts/kimi_linear/clone_kimi_k3_reference.py`) driving the FLA `kda` kernels;
`mini_infer.models.blocks.kda` lists the pinned semantics.

Architecturally, a decoder that interleaves two attention kinds at ~3:1 (per
the config's 1-indexed `kda_layers` / `full_attn_layers` lists):

  - **KDA layers** (`blocks/kda.py`): q/k/v projections through short causal
    convs (kernel 4, SiLU), L2-normalized q/k, a per-channel decay gate
    `-exp(A_log) * softplus(low_rank(x) + dt_bias)`, the gated delta rule
    over a `(heads, 128, 128)` matrix state, then a sigmoid-gated RMSNorm and
    output projection. Decode state is the matrix + three conv tails,
    constant per request.
  - **MLA layers**: DeepSeek-V3-shaped latent attention with **NoPE**: no
    rotary embedding exists anywhere in this family; the `qk_rope_head_dim`
    split of q and the shared `kv_a` output is used UNROTATED (all position
    information comes from the KDA layers' decay and convolutions). Cache is
    the compressed `kv_a_proj_with_mqa` output (`512 + 64` per token),
    re-decompressed on read like `blocks/mla.py`.

The FFN is dense on layer 0 (`first_k_dense_replace = 1`) and a sigmoid-routed
MoE elsewhere, with an aux-loss-free `e_score_correction_bias`.

Kimi K3 (`moonshotai/Kimi-K3`, 2.8T, 93 layers) is the same stack with six
config-gated additions, each pinned from its reference `modeling_kimi_linear.py`:

  - `hidden_act = "situ"`: SiTU-GLU in the dense MLP, the experts and the
    shared experts (`blocks/activations.situ`);
  - `q_lora_rank`: low-rank Q (`q_a_proj` -> RMSNorm -> `q_b_proj`) and
    `mla_use_output_gate`: a sigmoid gate `g_proj(x)` on the MLA output
    before `o_proj`;
  - `linear_attn_config.use_full_rank_gate` (one `g_proj` instead of the
    low-rank `g_a/g_b` output-gate pair) and `gate_lower_bound` (the bounded
    decay gate `lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))`);
  - `routed_expert_hidden_size`: latent MoE, one shared down projection
    before dispatch, experts in the latent width, RMSNorm + shared up
    projection after (the gate and the shared experts see the full width);
  - `attn_res_block_size`: AttnRes, attention over residual-stream snapshots
    before every sublayer and before the final norm (`attn_res.py`).

Two conventions differ between the two reference files and are selected by
`KimiLinearConfig.moe_weights_from_biased_scores`: the Kimi Linear file adds
the correction bias to the sigmoid scores IN PLACE, so the gathered expert
weights are the BIASED scores; the K3 file keeps the bias out of the weights
(the DeepSeek-V3 / GLM convention) and masks dropped groups with `-inf`
instead of `0.0`. `from_hf` keys the choice on the K3-only config fields.

Serving: `USES_STATE_CACHE = True` routes the family to the per-request
`KimiStateCache` and the `StateCacheContinuousScheduler`, via the same
`forward_prefill_with_cache` / `forward_decode_with_cache[_ragged]` contract
as DeepSeek-V4. Because the whole family is position-free (NoPE + linear
attention), `forward_prefill_with_cache` also supports CONTINUATION from
`state_cache.start_pos > 0`, i.e. chunked prefill, which V4's state path
does not. AttnRes mixes per token, so it adds no decode state. Single-rank
only for now (TP would shard the KDA conv channels and the per-head state;
a follow-up like Inkling). Cross-request prefix sharing (`StatePrefixCache`)
is V4-only; the KDA state-snapshot equivalent is a follow-up.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional

from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks.activations import GateUpActivation, situ_activation, swiglu
from mini_infer.models.blocks.kda import (
    causal_conv1d_prefill,
    gated_rmsnorm,
    kda_chunkwise,
    kda_gate,
    kda_lower_bound_gate,
    kda_recurrent,
    l2norm,
)
from mini_infer.models.blocks.moe.mixtral_moe import MixtralExpert
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.swiglu import SwiGLU
from mini_infer.models.kimi_linear.attn_res import AttnResQueryNorm, attn_res_mix
from mini_infer.models.kimi_linear.state_cache import (
    KimiKdaLayerState,
    KimiKdaStateSpec,
    KimiMlaLayerState,
    KimiMlaStateSpec,
    KimiStateCache,
    KimiStateLayerSpec,
)
from mini_infer.quant.mxfp4 import dequantize_mxfp4_to_bf16

# The reference builds `q_a_layernorm` / `kv_a_layernorm` as `KimiRMSNorm(rank)`
# with the class default eps, NOT `config.rms_norm_eps` (1e-5 on both checkpoints).
_LATENT_NORM_EPS = 1e-6


@dataclass
class KimiLinearConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    # MLA (full-attention) layers' shape. `mla_use_nope` is always True on
    # this family; `from_hf` enforces it.
    num_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    # KDA (linear-attention) layers' shape, from `linear_attn_config`.
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel_size: int
    # 1-indexed layer numbers, straight from the HF config (`is_kda_layer`
    # converts; keeping the raw lists makes config dumps grep-able).
    kda_layers: list[int]
    full_attn_layers: list[int]
    # FFN.
    intermediate_size: int  # dense-layer width
    moe_intermediate_size: int  # per-expert width
    num_experts: int | None
    num_experts_per_token: int
    num_shared_experts: int
    routed_scaling_factor: float
    moe_renormalize: bool
    moe_router_activation_func: str  # "sigmoid" | "softmax"
    num_expert_group: int
    topk_group: int
    first_k_dense_replace: int
    moe_layer_freq: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    # ---- Kimi K3 delta (all off for Kimi Linear 48B) ----
    hidden_act: str = "silu"  # "silu" | "situ"
    situ_beta: float = 1.0
    situ_linear_beta: float | None = None
    q_lora_rank: int | None = None
    mla_use_output_gate: bool = False
    kda_use_full_rank_gate: bool = False
    kda_gate_lower_bound: float | None = None
    routed_expert_hidden_size: int | None = None
    latent_moe_use_norm: bool = False
    attn_res_block_size: int | None = None
    # Kimi Linear's reference gathers the BIASED sigmoid scores as expert
    # weights (in-place `+=` through a view); K3's gathers the unbiased ones.
    moe_weights_from_biased_scores: bool = True
    # `quantization_config` declares compressed-tensors `mxfp4-pack-quantized`
    # (K3's routed experts); only then does the loader dequantize MXFP4 pairs.
    mxfp4_weights: bool = False

    @classmethod
    def from_hf(cls, hf_config: Any) -> KimiLinearConfig:
        linear_attn = getattr(hf_config, "linear_attn_config", None)
        if not linear_attn:
            raise ValueError(
                "KimiLinearConfig requires `linear_attn_config` (the hybrid "
                "KDA/MLA layout); a config without it is not this family"
            )
        if not getattr(hf_config, "mla_use_nope", False):
            raise ValueError(
                "Kimi Linear MLA layers are NoPE (mla_use_nope=True); a RoPE'd "
                "variant would need different math"
            )
        num_layers = int(hf_config.num_hidden_layers)
        kda_layers = [int(i) for i in linear_attn["kda_layers"]]
        full_layers = [int(i) for i in linear_attn["full_attn_layers"]]
        # The two 1-indexed lists must partition 1..num_layers; a typo here
        # would silently build the wrong hybrid, so fail loudly.
        if sorted(kda_layers + full_layers) != list(range(1, num_layers + 1)):
            raise ValueError(
                f"kda_layers + full_attn_layers must partition 1..{num_layers}, "
                f"got kda={kda_layers} full={full_layers}"
            )
        activation_func = str(getattr(hf_config, "moe_router_activation_func", "sigmoid"))
        if activation_func not in ("sigmoid", "softmax"):
            raise ValueError(f"unsupported moe_router_activation_func {activation_func!r}")
        hidden_act = str(getattr(hf_config, "hidden_act", "silu") or "silu")
        if hidden_act not in ("silu", "situ"):
            raise ValueError(f"unsupported hidden_act {hidden_act!r} (expected silu or situ)")
        num_experts = getattr(hf_config, "num_experts", None)
        # Check the K3 knobs here, not in the first forward: a bad value
        # must fail before the checkpoint loads. The float checks are written
        # so that NaN fails them too.
        situ_linear_beta = getattr(hf_config, "activation_situ_linear_beta", None)
        if situ_linear_beta is not None and not (
            math.isfinite(float(situ_linear_beta)) and float(situ_linear_beta) > 0
        ):
            raise ValueError(
                f"activation_situ_linear_beta must be positive or None, got {situ_linear_beta}"
            )
        q_lora_rank = getattr(hf_config, "q_lora_rank", None)
        if q_lora_rank is not None and int(q_lora_rank) <= 0:
            raise ValueError(f"q_lora_rank must be positive or None, got {q_lora_rank}")
        gate_lower_bound = linear_attn.get("gate_lower_bound", None)
        if gate_lower_bound is not None and not (
            math.isfinite(float(gate_lower_bound)) and float(gate_lower_bound) < 0
        ):
            raise ValueError(
                f"linear_attn_config.gate_lower_bound must be finite and negative, "
                f"got {gate_lower_bound}"
            )
        attn_res_block_size = getattr(hf_config, "attn_res_block_size", None)
        if attn_res_block_size is not None and int(attn_res_block_size) <= 0:
            raise ValueError(f"attn_res_block_size must be positive, got {attn_res_block_size}")
        routed_expert_hidden_size = getattr(hf_config, "routed_expert_hidden_size", None)
        if routed_expert_hidden_size is not None and int(routed_expert_hidden_size) <= 0:
            raise ValueError(
                f"routed_expert_hidden_size must be positive or None, "
                f"got {routed_expert_hidden_size}"
            )
        return cls(
            vocab_size=int(hf_config.vocab_size),
            hidden_size=int(hf_config.hidden_size),
            num_hidden_layers=num_layers,
            num_attention_heads=int(hf_config.num_attention_heads),
            kv_lora_rank=int(hf_config.kv_lora_rank),
            qk_nope_head_dim=int(hf_config.qk_nope_head_dim),
            qk_rope_head_dim=int(hf_config.qk_rope_head_dim),
            v_head_dim=int(hf_config.v_head_dim),
            kda_num_heads=int(linear_attn["num_heads"]),
            kda_head_dim=int(linear_attn["head_dim"]),
            kda_conv_kernel_size=int(linear_attn["short_conv_kernel_size"]),
            kda_layers=kda_layers,
            full_attn_layers=full_layers,
            intermediate_size=int(hf_config.intermediate_size),
            moe_intermediate_size=int(getattr(hf_config, "moe_intermediate_size", 0) or 0),
            num_experts=int(num_experts) if num_experts is not None else None,
            num_experts_per_token=int(getattr(hf_config, "num_experts_per_token", 0) or 0),
            num_shared_experts=int(getattr(hf_config, "num_shared_experts", 0) or 0),
            routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
            moe_renormalize=bool(getattr(hf_config, "moe_renormalize", True)),
            moe_router_activation_func=activation_func,
            num_expert_group=int(getattr(hf_config, "num_expert_group", 1)),
            topk_group=int(getattr(hf_config, "topk_group", 1)),
            first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 0)),
            moe_layer_freq=int(getattr(hf_config, "moe_layer_freq", 1)),
            rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            hidden_act=hidden_act,
            situ_beta=float(getattr(hf_config, "activation_situ_beta", None) or 1.0),
            situ_linear_beta=float(situ_linear_beta) if situ_linear_beta is not None else None,
            q_lora_rank=int(q_lora_rank) if q_lora_rank is not None else None,
            mla_use_output_gate=bool(getattr(hf_config, "mla_use_output_gate", False)),
            kda_use_full_rank_gate=bool(linear_attn.get("use_full_rank_gate", False)),
            kda_gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
            routed_expert_hidden_size=(
                int(routed_expert_hidden_size) if routed_expert_hidden_size is not None else None
            ),
            latent_moe_use_norm=bool(getattr(hf_config, "latent_moe_use_norm", False)),
            attn_res_block_size=(
                int(attn_res_block_size) if attn_res_block_size is not None else None
            ),
            moe_weights_from_biased_scores=not _ships_k3_modeling_code(hf_config, linear_attn),
            mxfp4_weights=_declares_mxfp4_weights(hf_config),
        )

    def is_kda_layer(self, layer_idx: int) -> bool:
        """0-indexed layer -> KDA? (the HF lists are 1-indexed, mirrored here)."""
        return (layer_idx + 1) in self.kda_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts is not None
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    def gate_up_activation(self) -> GateUpActivation:
        """The FFN combiner for `hidden_act` (dense MLP, experts, shared experts)."""
        if self.hidden_act == "situ":
            return situ_activation(self.situ_beta, self.situ_linear_beta)
        return swiglu


def _ships_k3_modeling_code(hf_config: Any, linear_attn: dict[str, Any]) -> bool:
    """Does this config belong to the K3-generation `modeling_kimi_linear.py`?

    The two reference files differ in router convention (see the module
    docstring). The K3 file's config class is the only one that knows these
    fields, so any of them being set marks a K3-generation checkpoint.
    """
    return (
        getattr(hf_config, "attn_res_block_size", None) is not None
        or getattr(hf_config, "routed_expert_hidden_size", None) is not None
        or getattr(hf_config, "activation_situ_beta", None) is not None
        or bool(getattr(hf_config, "mla_use_output_gate", False))
        or "use_full_rank_gate" in linear_attn
        or "gate_lower_bound" in linear_attn
    )


def _declares_mxfp4_weights(hf_config: Any) -> bool:
    """Does `quantization_config` declare compressed-tensors MXFP4 weights?

    No `quantization_config` means a BF16 checkpoint. Any other declared
    format fails here, before the checkpoint loads, because the loader can
    only dequantize `mxfp4-pack-quantized` pairs.
    """
    quant = getattr(hf_config, "quantization_config", None)
    if not quant:
        return False
    if isinstance(quant, dict):
        method, fmt = quant.get("quant_method"), quant.get("format")
    else:
        method, fmt = getattr(quant, "quant_method", None), getattr(quant, "format", None)
    if method == "compressed-tensors" and fmt == _MXFP4_FORMAT:
        return True
    raise ValueError(
        f"unsupported quantization_config (quant_method={method!r}, format={fmt!r}); "
        f"Kimi Linear loads BF16 or compressed-tensors {_MXFP4_FORMAT!r} checkpoints"
    )


def build_kimi_state_cache_specs(
    cfg: KimiLinearConfig, *, max_seq_len: int
) -> list[KimiStateLayerSpec]:
    """Per-layer `KimiStateCache` specs: constant-size KDA state or a dense
    `max_seq_len`-capacity compressed-KV buffer for the MLA layers."""
    specs: list[KimiStateLayerSpec] = []
    for layer_idx in range(cfg.num_hidden_layers):
        if cfg.is_kda_layer(layer_idx):
            specs.append(
                KimiKdaStateSpec(
                    num_heads=cfg.kda_num_heads,
                    head_dim=cfg.kda_head_dim,
                    conv_channels=cfg.kda_num_heads * cfg.kda_head_dim,
                    conv_kernel_size=cfg.kda_conv_kernel_size,
                )
            )
        else:
            specs.append(
                KimiMlaStateSpec(
                    kv_width=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                    max_seq_len=max_seq_len,
                )
            )
    return specs


class _KimiConv1d(nn.Module):
    """Depthwise causal conv taps, stored `(channels, kernel)`.

    HF wraps these in FLA's `ShortConvolution` (an `nn.Conv1d`, weight
    `(C, 1, W)`); the load remap squeezes the singleton. The actual conv +
    SiLU math lives in `blocks/kda.py` so prefill and decode share one
    implementation.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        # nn.Conv1d's default (kaiming-uniform at depthwise fan_in = kernel),
        # so a freshly-constructed model is finite and non-degenerate.
        nn.init.uniform_(self.weight, -(kernel_size**-0.5), kernel_size**-0.5)


class _KimiDeltaAttention(nn.Module):
    """One KDA layer: convs -> gate/beta -> delta rule -> gated RMSNorm -> o_proj."""

    def __init__(self, cfg: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.kda_num_heads
        self.head_dim = cfg.kda_head_dim
        projection_size = self.num_heads * self.head_dim

        hidden = cfg.hidden_size
        self.q_proj = nn.Linear(hidden, projection_size, bias=False)
        self.k_proj = nn.Linear(hidden, projection_size, bias=False)
        self.v_proj = nn.Linear(hidden, projection_size, bias=False)
        self.q_conv1d = _KimiConv1d(projection_size, cfg.kda_conv_kernel_size)
        self.k_conv1d = _KimiConv1d(projection_size, cfg.kda_conv_kernel_size)
        self.v_conv1d = _KimiConv1d(projection_size, cfg.kda_conv_kernel_size)
        # Per-head decay rate, stored flat `(H,)` (K3's layout; Kimi Linear
        # ships `(1, 1, H, 1)` and the load remap flattens it). Init matches
        # the reference's log-uniform(1, 16) so a fresh model is finite.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        # The reference leaves dt_bias uninitialized (torch.empty); zeros keep
        # a fresh model finite, and checkpoints always overwrite it.
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        # Low-rank decay-gate feature projection and the delta-rule step size.
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        # Output gate: low-rank `g_a/g_b` (Kimi Linear) or one full-rank
        # `g_proj` (K3, `use_full_rank_gate`).
        self.use_full_rank_gate = cfg.kda_use_full_rank_gate
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        # Decay gate form: softplus (Kimi Linear) or bounded sigmoid (K3).
        self.gate_lower_bound = cfg.kda_gate_lower_bound
        # FusedRMSNormGated equivalent: per-head RMSNorm weight, shared
        # across heads, sigmoid gate applied in fp32 (blocks/kda.gated_rmsnorm).
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.o_norm_eps = cfg.rms_norm_eps
        self.o_proj = nn.Linear(projection_size, hidden, bias=False)

    def forward(self, x: torch.Tensor, layer_state: KimiKdaLayerState | None) -> torch.Tensor:
        """Runs any span length; positions are irrelevant (the state carries
        all sequence information). With `layer_state`, conv tails and the
        recurrent state are read as the left context and written back
        updated, which serves prefill, chunked prefill, and decode alike.
        """
        batch, seq_len, _ = x.shape
        conv_q = layer_state.conv_q if layer_state is not None else None
        conv_k = layer_state.conv_k if layer_state is not None else None
        conv_v = layer_state.conv_v if layer_state is not None else None
        q_raw, new_conv_q = causal_conv1d_prefill(self.q_proj(x), self.q_conv1d.weight, conv_q)
        k_raw, new_conv_k = causal_conv1d_prefill(self.k_proj(x), self.k_conv1d.weight, conv_k)
        v_raw, new_conv_v = causal_conv1d_prefill(self.v_proj(x), self.v_conv1d.weight, conv_v)

        gate_features = self.f_b_proj(self.f_a_proj(x))
        if self.gate_lower_bound is None:
            decay = kda_gate(gate_features, self.A_log, self.dt_bias, self.head_dim)
        else:
            decay = kda_lower_bound_gate(
                gate_features, self.A_log, self.dt_bias, self.head_dim, self.gate_lower_bound
            )
        beta = self.b_proj(x).float().sigmoid()

        # L2-normalize q/k and scale q BEFORE the recurrence (the reference
        # kernels run with `use_qk_l2norm_in_kernel=True` and scale K^-0.5).
        q = l2norm(q_raw.view(batch, seq_len, self.num_heads, self.head_dim))
        q = q * self.head_dim**-0.5
        k = l2norm(k_raw.view(batch, seq_len, self.num_heads, self.head_dim))
        v = v_raw.view(batch, seq_len, self.num_heads, self.head_dim)

        initial = layer_state.recurrent_state if layer_state is not None else None
        # Recurrent form for short spans, chunked beyond. Both are the same
        # math (pinned by test_kda_block), as are the reference's two kernels.
        if seq_len <= 64:
            out, final_state = kda_recurrent(q, k, v, decay, beta, initial_state=initial)
        else:
            out, final_state = kda_chunkwise(q, k, v, decay, beta, initial_state=initial)

        if layer_state is not None:
            layer_state.recurrent_state.copy_(final_state)
            layer_state.conv_q.copy_(new_conv_q)
            layer_state.conv_k.copy_(new_conv_k)
            layer_state.conv_v.copy_(new_conv_v)

        gate = self.g_proj(x) if self.use_full_rank_gate else self.g_b_proj(self.g_a_proj(x))
        gate = gate.view(batch, seq_len, self.num_heads, self.head_dim)
        out = gated_rmsnorm(out, gate, self.o_norm_weight, self.o_norm_eps)
        result: torch.Tensor = self.o_proj(out.reshape(batch, seq_len, -1))
        return result


class _KimiMlaAttention(nn.Module):
    """One NoPE MLA layer over the compressed-KV state buffer.

    Same decompression math as `blocks/mla.py` (cache the shared
    `kv_a_proj_with_mqa` output, rebuild per-head K/V on read) but NO rotary
    embedding anywhere, and reads/writes a dense per-request buffer instead
    of PagedKVCache streams. K3 adds low-rank Q (`q_lora_rank`) and a sigmoid
    output gate (`mla_use_output_gate`).
    """

    def __init__(self, cfg: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.softmax_scale = self.qk_head_dim**-0.5

        hidden = cfg.hidden_size
        self.q_lora_rank = cfg.q_lora_rank
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(hidden, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=_LATENT_NORM_EPS)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=_LATENT_NORM_EPS)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.use_output_gate = cfg.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_state: KimiMlaLayerState | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
        x: `(batch, seq_len, hidden)` new tokens.
        layer_state: compressed-KV buffer, or None for a stateless
            full-sequence pass (parity path).
        row_starts: `(batch,)` global position of `x[:, 0]` per row.
            Uniform for (chunked) prefill; per-row for ragged decode.
        """
        batch, seq_len, _ = x.shape
        if self.q_lora_rank is None:
            q_states = self.q_proj(x)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        queries = q_states.view(batch, seq_len, self.num_heads, self.qk_head_dim)

        kv_new = self.kv_a_proj_with_mqa(x)  # (batch, seq_len, kv_lora + rope)
        if layer_state is None:
            if int(row_starts.max()) != 0:
                raise ValueError("stateless MLA forward requires row_starts == 0")
            kv_hist = kv_new
        else:
            capacity = layer_state.kv.shape[1]
            if int(row_starts.max()) + seq_len > capacity:
                raise ValueError(
                    f"MLA state buffer overflow: writing {seq_len} tokens at "
                    f"position {int(row_starts.max())} exceeds capacity {capacity}"
                )
            if seq_len == 1:
                rows = torch.arange(batch, device=x.device)
                layer_state.kv[rows, row_starts] = kv_new[:, 0].to(layer_state.kv.dtype)
            else:
                # Multi-token spans only occur in (chunked) prefill, where
                # every row sits at the same offset.
                start = int(row_starts[0])
                if not bool((row_starts == start).all()):
                    raise ValueError("multi-token MLA spans require uniform row_starts")
                layer_state.kv[:, start : start + seq_len] = kv_new.to(layer_state.kv.dtype)
            lengths = row_starts + seq_len
            kv_hist = layer_state.kv[:, : int(lengths.max())].to(x.dtype)

        kv_latent, k_rot = torch.split(kv_hist, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        decompressed = self.kv_b_proj(self.kv_a_layernorm(kv_latent)).view(
            batch, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, values = torch.split(decompressed, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # NoPE: the rope-dim split concatenates UNROTATED, broadcast to heads.
        keys = torch.cat([k_nope, k_rot.unsqueeze(2).expand(-1, -1, self.num_heads, -1)], dim=-1)

        # (batch, heads, q, k) scores; causal against each row's own length.
        scores = torch.einsum("bqhd,bkhd->bhqk", queries.float(), keys.float())
        scores = scores * self.softmax_scale
        total_k = keys.shape[1]
        key_positions = torch.arange(total_k, device=x.device)
        query_positions = row_starts.view(batch, 1, 1, 1) + torch.arange(
            seq_len, device=x.device
        ).view(1, 1, seq_len, 1)
        valid = key_positions.view(1, 1, 1, total_k) <= query_positions
        scores = scores.masked_fill(~valid, float("-inf"))
        weights = scores.softmax(dim=-1).to(x.dtype)
        attn = torch.einsum("bhqk,bkhd->bqhd", weights, values).reshape(batch, seq_len, -1)
        if self.use_output_gate:
            attn = attn * torch.sigmoid(self.g_proj(x))
        result: torch.Tensor = self.o_proj(attn)
        return result


class _KimiMoeGate(nn.Module):
    """Kimi router: sigmoid (or softmax) scores + aux-loss-free correction bias.

    Faithful to each reference `KimiMoEGate`. The bias steers SELECTION in
    both; `weights_from_biased_scores` picks which scores become the expert
    weights (Kimi Linear: the biased ones, via its in-place `+=` through a
    view; K3: the unbiased ones, the DeepSeek-V3 / GLM convention) and how
    dropped groups are masked before the top-k (`0.0` vs `-inf`).
    """

    def __init__(self, cfg: KimiLinearConfig) -> None:
        super().__init__()
        assert cfg.num_experts is not None
        self.top_k = cfg.num_experts_per_token
        self.num_experts = cfg.num_experts
        self.num_expert_group = cfg.num_expert_group
        self.topk_group = cfg.topk_group
        self.moe_renormalize = cfg.moe_renormalize
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.activation_func = cfg.moe_router_activation_func
        self.weights_from_biased_scores = cfg.moe_weights_from_biased_scores
        self.weight = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Reference ships this uninitialized (torch.empty); zeros keep a
        # fresh model finite, checkpoints overwrite.
        self.e_score_correction_bias = nn.Parameter(torch.zeros(cfg.num_experts))

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route `(tokens, hidden)`; returns `(top_k_indices, top_k_weights)`,
        weights fp32, renormalized, scaled."""
        logits = functional.linear(x_flat.float(), self.weight.float())
        scores = logits.sigmoid() if self.activation_func == "sigmoid" else logits.softmax(dim=1)
        choice = scores + self.e_score_correction_bias.float()

        num_tokens = x_flat.shape[0]
        per_group = self.num_experts // self.num_expert_group
        group_scores = (
            choice.view(num_tokens, self.num_expert_group, per_group).topk(2, dim=-1)[0].sum(-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.num_expert_group, per_group)
            .reshape(num_tokens, -1)
        )
        dropped = 0.0 if self.weights_from_biased_scores else float("-inf")
        masked_choice = choice.masked_fill(~score_mask.bool(), dropped)
        topk_indices = torch.topk(masked_choice, k=self.top_k, dim=-1, sorted=False)[1]
        weight_source = choice if self.weights_from_biased_scores else scores
        topk_weights = weight_source.gather(1, topk_indices)
        if self.top_k > 1 and self.moe_renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_indices, topk_weights * self.routed_scaling_factor


class _KimiMoe(nn.Module):
    """Sparse FFN: `_KimiMoeGate` + routed `MixtralExpert`s + shared FFN.

    Expert tensors keep the checkpoint's names (`experts.N.w1/w2/w3`,
    `shared_experts.gate/up/down_proj`) so loading is a direct copy. Routed
    contributions accumulate in fp32, matching the reference's fp32 combine.

    With `routed_expert_hidden_size` (K3's latent MoE) the routed path runs
    in a narrower latent width: `routed_expert_down_proj` once before
    dispatch, experts at the latent width, an optional RMSNorm on the
    combined output, `routed_expert_up_proj` back to the hidden width. The
    gate and the shared experts always see the full-width input.
    """

    def __init__(self, cfg: KimiLinearConfig) -> None:
        super().__init__()
        assert cfg.num_experts is not None
        self.num_experts = cfg.num_experts
        activation = cfg.gate_up_activation()
        self.latent_size = cfg.routed_expert_hidden_size
        expert_width = self.latent_size if self.latent_size is not None else cfg.hidden_size
        self.experts = nn.ModuleList(
            [
                MixtralExpert(expert_width, cfg.moe_intermediate_size, activation=activation)
                for _ in range(cfg.num_experts)
            ]
        )
        self.gate = _KimiMoeGate(cfg)
        self.shared_experts: SwiGLU | None = (
            SwiGLU(
                cfg.hidden_size,
                cfg.moe_intermediate_size * cfg.num_shared_experts,
                activation=activation,
            )
            if cfg.num_shared_experts > 0
            else None
        )
        if self.latent_size is not None:
            self.routed_expert_down_proj = nn.Linear(cfg.hidden_size, self.latent_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.latent_size, cfg.hidden_size, bias=False)
            if cfg.latent_moe_use_norm:
                self.routed_expert_norm = RMSNorm(self.latent_size, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        flat = x.view(-1, x.shape[-1])
        topk_indices, topk_weights = self.gate(flat)
        routed_in = self.routed_expert_down_proj(flat) if self.latent_size is not None else flat

        routed = torch.zeros(routed_in.shape, dtype=torch.float32, device=flat.device)
        expert_mask = functional.one_hot(topk_indices, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            weights = topk_weights[token_idx, top_k_pos, None]
            routed.index_add_(
                0, token_idx, self.experts[expert_idx](routed_in[token_idx]).float() * weights
            )
        out = routed.to(x.dtype)
        if self.latent_size is not None:
            if hasattr(self, "routed_expert_norm"):
                out = self.routed_expert_norm(out)
            out = self.routed_expert_up_proj(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(input_shape)


class _KimiDecoderLayer(nn.Module):
    """Pre-norm block: (KDA | NoPE-MLA) attention, then (dense | MoE) FFN.

    With `attn_res_block_size` (K3) the residual stream is replaced by
    AttnRes (`attn_res.py`): the layer receives the running sum and the
    block snapshots, picks each sublayer's input by attention over them,
    snapshots the running sum every `attn_res_block_size` layers, and
    returns the updated pair.
    """

    def __init__(self, cfg: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = cfg.is_kda_layer(layer_idx)
        self.rms_norm_eps = cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn: nn.Module = (
            _KimiDeltaAttention(cfg, layer_idx)
            if self.is_kda
            else _KimiMlaAttention(cfg, layer_idx)
        )
        # HF names the sparse FFN `block_sparse_moe` and the dense one `mlp`.
        if cfg.is_moe_layer(layer_idx):
            self.block_sparse_moe = _KimiMoe(cfg)
        else:
            self.mlp = SwiGLU(
                cfg.hidden_size, cfg.intermediate_size, activation=cfg.gate_up_activation()
            )
        self.attn_res_block_size = cfg.attn_res_block_size
        if self.attn_res_block_size is not None:
            self.self_attention_res_norm = AttnResQueryNorm(cfg.hidden_size)
            self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False)
            self.mlp_res_norm = AttnResQueryNorm(cfg.hidden_size)
            self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False)

    def _attention(
        self,
        x: torch.Tensor,
        layer_state: KimiKdaLayerState | KimiMlaLayerState | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_kda:
            assert layer_state is None or isinstance(layer_state, KimiKdaLayerState)
            out: torch.Tensor = self.self_attn(x, layer_state)
            return out
        assert layer_state is None or isinstance(layer_state, KimiMlaLayerState)
        out = self.self_attn(x, layer_state, row_starts)
        return out

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = (
            self.block_sparse_moe(x) if hasattr(self, "block_sparse_moe") else self.mlp(x)
        )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_state: KimiKdaLayerState | KimiMlaLayerState | None,
        row_starts: torch.Tensor,
        block_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns `(hidden_states, block_residual)`; the second is None
        without AttnRes and `(batch, seq_len, num_blocks, hidden)` with it."""
        if self.attn_res_block_size is None:
            x = self._attention(self.input_layernorm(hidden_states), layer_state, row_starts)
            hidden_states = hidden_states + x
            x = self._ffn(self.post_attention_layernorm(hidden_states))
            return hidden_states + x, block_residual

        assert block_residual is not None
        prefix_sum: torch.Tensor | None = hidden_states
        attn_input = hidden_states
        if block_residual.shape[-2] > 0:
            attn_input = attn_res_mix(
                hidden_states,
                block_residual,
                self.self_attention_res_proj.weight,
                self.self_attention_res_norm.weight,
                self.rms_norm_eps,
            )
        if self.layer_idx % self.attn_res_block_size == 0:
            # Snapshot the running sum as a new block and restart it.
            block_residual = torch.cat([block_residual, hidden_states.unsqueeze(-2)], dim=-2)
            prefix_sum = None

        attn_out = self._attention(self.input_layernorm(attn_input), layer_state, row_starts)
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        ffn_input = attn_res_mix(
            prefix_sum,
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.rms_norm_eps,
        )
        ffn_out = self._ffn(self.post_attention_layernorm(ffn_input))
        return prefix_sum + ffn_out, block_residual


class _KimiInnerModel(nn.Module):
    """Embedding + N decoder layers + final norm. HF prefix `model.`."""

    def __init__(self, cfg: KimiLinearConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [_KimiDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.attn_res_block_size is not None:
            self.output_attn_res_norm = AttnResQueryNorm(cfg.hidden_size)
            self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False)


# HF wraps each KDA conv in FLA's ShortConvolution (nn.Conv1d): (C, 1, W) -> (C, W).
_KDA_CONV_RE = re.compile(r"^(.*\.(?:q|k|v)_conv1d)\.weight$")
# FusedRMSNormGated's weight -> our flat parameter name.
_O_NORM_RE = re.compile(r"^(.*)\.o_norm\.weight$")
# compressed-tensors `mxfp4-pack-quantized` pair (K3's routed experts).
_MXFP4_FORMAT = "mxfp4-pack-quantized"
_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"


@register_model
class KimiLinearForCausalLM(BaseCausalLM):
    HF_ARCHITECTURE: ClassVar[str] = "KimiLinearForCausalLM"
    Config: ClassVar[type] = KimiLinearConfig
    # Decodes via the per-request KimiStateCache (KDA matrix state + dense
    # MLA buffers), not the shared PagedKVCache.
    USES_STATE_CACHE: ClassVar[bool] = True

    def __init__(self, cfg: KimiLinearConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _KimiInnerModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # ---- shared layer walk ----

    def _run(
        self,
        input_ids: torch.Tensor,
        state_cache: KimiStateCache | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        block_residual: torch.Tensor | None = None
        if self.cfg.attn_res_block_size is not None:
            block_residual = hidden.new_zeros(*hidden.shape[:-1], 0, hidden.shape[-1])
        for layer_idx, layer_module in enumerate(self.model.layers):
            layer_state = state_cache.layer(layer_idx) if state_cache is not None else None
            hidden, block_residual = layer_module(hidden, layer_state, row_starts, block_residual)
        if block_residual is not None:
            hidden = attn_res_mix(
                hidden,
                block_residual,
                self.model.output_attn_res_proj.weight,
                self.model.output_attn_res_norm.weight,
                self.cfg.rms_norm_eps,
            )
        hidden = self.model.norm(hidden)
        logits: torch.Tensor = self.lm_head(hidden)
        return logits

    # ---- BaseCausalLM contract ----

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standalone stateless prefill: `(B, T)` -> `(B, T, vocab)` logits.

        The `position_ids` / `past_key_values` / `cu_seqlens_q` parameters
        keep the registry signature but are ignored; the family is NoPE
        (positions carry no information) and decodes through
        `forward_decode_with_cache`, not PagedKVCache.
        """
        batch = input_ids.shape[0]
        row_starts = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        return self._run(input_ids, None, row_starts)

    @property
    def kv_cache_dims(self) -> KVCacheDims:
        """Reported size only; the family uses KimiStateCache, not PagedKVCache.
        The MLA layers' per-token cost is one shared `kv_lora_rank +
        qk_rope_head_dim` entry (the KDA layers are constant-size)."""
        return KVCacheDims(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=1,
            head_dim=self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
        )

    # ---- StateCache serving contract (mirrors DeepseekV4ForCausalLM) ----

    def build_state_cache(
        self,
        *,
        max_seq_len: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> KimiStateCache:
        return KimiStateCache(
            build_kimi_state_cache_specs(self.cfg, max_seq_len=max_seq_len),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        state_cache: KimiStateCache,
    ) -> torch.Tensor:
        """Cache-aware prefill of `(B, T)`, CONTINUING from `state_cache.start_pos`.

        At `start_pos == 0` this is a fresh prefill; at `start_pos > 0` it is
        a chunked-prefill continuation (the KDA state and conv tails carry
        the left context, the MLA buffers append at the offset). The caller
        advances `state_cache.start_pos` by `T` afterwards.
        """
        self._check_cache(state_cache)
        batch = input_ids.shape[0]
        row_starts = torch.full(
            (batch,), state_cache.start_pos, dtype=torch.long, device=input_ids.device
        )
        return self._run(input_ids, state_cache, row_starts)

    def forward_decode_with_cache(
        self,
        input_id: torch.Tensor,
        *,
        start_pos: int,
        state_cache: KimiStateCache,
    ) -> torch.Tensor:
        """One decode step at a uniform position: `(B, 1)` -> `(B, 1, vocab)`.
        The caller advances `state_cache.start_pos` afterwards."""
        if input_id.shape[-1] != 1:
            raise ValueError(
                f"forward_decode_with_cache expects shape (B, 1), got {tuple(input_id.shape)}"
            )
        self._check_cache(state_cache)
        batch = input_id.shape[0]
        positions = torch.full((batch,), start_pos, dtype=torch.long, device=input_id.device)
        return self._run(input_id, state_cache, positions)

    def forward_decode_with_cache_ragged(
        self,
        input_id: torch.Tensor,
        *,
        positions: torch.Tensor,
        state_cache: KimiStateCache,
    ) -> torch.Tensor:
        """One ragged decode step: B requests, each at its own `positions[b]`.

        KDA layers are position-free so the ragged step is the batched step;
        only the MLA layers consult `positions` (per-row buffer offsets and
        causal lengths). Returns `(B, 1, vocab)` logits.
        """
        if input_id.shape[-1] != 1:
            raise ValueError(
                f"forward_decode_with_cache_ragged expects (B, 1), got {tuple(input_id.shape)}"
            )
        self._check_cache(state_cache)
        batch = input_id.shape[0]
        if positions.shape != (batch,):
            raise ValueError(f"positions shape {tuple(positions.shape)} != (B={batch},)")
        return self._run(input_id, state_cache, positions.to(torch.long))

    def _check_cache(self, state_cache: KimiStateCache) -> None:
        if state_cache.num_layers != self.cfg.num_hidden_layers:
            raise ValueError(
                f"state_cache has {state_cache.num_layers} layers, "
                f"model has {self.cfg.num_hidden_layers}"
            )

    def expected_missing_state_keys(self) -> set[str]:
        return {"lm_head.weight"} if self.cfg.tie_word_embeddings else set()

    @staticmethod
    def load_weights(model: BaseCausalLM, hf_state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(model, KimiLinearForCausalLM):
            raise TypeError(
                f"KimiLinearForCausalLM.load_weights expects a KimiLinearForCausalLM, "
                f"got {type(model).__name__}"
            )
        from mini_infer.distributed.group import get_world_size

        if get_world_size() != 1:
            raise NotImplementedError(
                "KimiLinearForCausalLM is single-rank only for now: TP would "
                "shard the KDA conv channels and the per-head matrix state. "
                "A follow-up, like Inkling."
            )
        remapped = _remap_kimi_state(
            hf_state_dict,
            mxfp4_weights=model.cfg.mxfp4_weights,
            kda_num_heads=model.cfg.kda_num_heads,
        )
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        missing_set = set(missing) - model.expected_missing_state_keys()
        if missing_set or unexpected:
            raise ValueError(
                f"weight load mismatch for {type(model).__name__}: "
                f"missing={sorted(missing_set)}, unexpected={sorted(unexpected)}"
            )


def _remap_kimi_state(
    hf_state_dict: dict[str, torch.Tensor], *, mxfp4_weights: bool, kda_num_heads: int
) -> dict[str, torch.Tensor]:
    """HF tensors -> our module keys.

    The module tree deliberately mirrors the reference names, so almost
    everything copies straight through. Four exceptions: the KDA conv
    weights lose HF's Conv1d singleton dim, `o_norm.weight` maps to our flat
    `o_norm_weight` parameter, `A_log` becomes `(H,)` (see `_flat_a_log`),
    and, when the config declares
    `mxfp4-pack-quantized` (`mxfp4_weights`), every `weight_packed` /
    `weight_scale` pair (K3's routed experts) becomes its dequantized BF16
    `weight`. A packed tensor without its scale, or the reverse, is a
    corrupt shard and fails loudly. Without the declaration the pair keys
    pass through and fail the load as unexpected keys.
    """
    remapped: dict[str, torch.Tensor] = {}
    packed: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        if mxfp4_weights and key.endswith(_PACKED_SUFFIX):
            packed[key[: -len(_PACKED_SUFFIX)]] = tensor
            continue
        if mxfp4_weights and key.endswith(_SCALE_SUFFIX):
            scales[key[: -len(_SCALE_SUFFIX)]] = tensor
            continue
        conv = _KDA_CONV_RE.match(key)
        if conv is not None:  # (C, 1, W) -> (C, W)
            remapped[f"{conv.group(1)}.weight"] = tensor.squeeze(1)
            continue
        o_norm = _O_NORM_RE.match(key)
        if o_norm is not None:
            remapped[f"{o_norm.group(1)}.o_norm_weight"] = tensor
            continue
        if key.endswith(".A_log"):
            remapped[key] = _flat_a_log(key, tensor, kda_num_heads)
            continue
        remapped[key] = tensor
    no_scale = sorted(set(packed) - set(scales))
    no_packed = sorted(set(scales) - set(packed))
    if no_scale or no_packed:
        raise ValueError(
            "MXFP4 weight_packed / weight_scale pairs do not match: "
            f"{len(no_scale)} weight_packed without a weight_scale {no_scale[:5]}, "
            f"{len(no_packed)} weight_scale without a weight_packed {no_packed[:5]}"
        )
    for module_name, packed_weight in packed.items():
        remapped[f"{module_name}.weight"] = dequantize_mxfp4_to_bf16(
            packed_weight, scales[module_name]
        )
    return remapped


def _flat_a_log(key: str, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """A checkpoint `A_log` -> our `(H,)` parameter.

    Kimi Linear ships `(1, 1, H, 1)`. The released K3 checkpoint ships a
    flat vector padded with zeros from its 96 heads to 128 entries, while the
    reference parameter has `num_heads` entries, so the padding carries no
    meaning: drop it, and fail if it is not all zero. A meta tensor (the
    shape-only checkpoint check) has no values to test. A vector shorter
    than `num_heads` passes through and fails the load's shape check.
    """
    flat = tensor.reshape(-1)
    if flat.numel() <= num_heads:
        return flat
    padding = flat[num_heads:]
    if not padding.is_meta and bool(padding.any()):
        raise ValueError(
            f"{key} has {flat.numel()} entries for {num_heads} heads and the "
            f"extra entries are not zero padding"
        )
    return flat[:num_heads]
