"""Kimi Linear family (moonshotai/Kimi-Linear-48B-A3B): owned implementation.

From-scratch port of Moonshot's hybrid linear-attention architecture
(arXiv:2510.26692), the attention foundation of Kimi K3. The parity
reference is the checkpoint's `modeling_kimi.py` (trust_remote_code, vendored
by `scripts/clone_kimi_linear_reference.py`) driving the FLA `kda` kernels;
see docs/plans/kimi-k3-spec.md for the pinned semantics.

Architecturally, a 27-layer decoder that interleaves two attention kinds at
~3:1 (per the config's 1-indexed `kda_layers` / `full_attn_layers` lists):

  - **KDA layers** (`blocks/kda.py`): q/k/v projections through short causal
    convs (kernel 4, SiLU), L2-normalized q/k, a per-channel decay gate
    `-exp(A_log) * softplus(low_rank(x) + dt_bias)`, the gated delta rule
    over a `(32, 128, 128)` matrix state, then a sigmoid-gated RMSNorm and
    output projection. Decode state is the matrix + three conv tails,
    constant per request.
  - **MLA layers**: DeepSeek-V3-shaped latent attention with `q_lora = None`
    and **NoPE**: no rotary embedding exists anywhere in this family; the
    `qk_rope_head_dim` split of q and the shared `kv_a` output is used
    UNROTATED (all position information comes from the KDA layers' decay
    and convolutions). Cache is the compressed `kv_a_proj_with_mqa` output
    (`512 + 64` per token), re-decompressed on read like `blocks/mla.py`.

The FFN is dense SwiGLU on layer 0 (`first_k_dense_replace = 1`) and a
256-expert top-8 MoE elsewhere. The router is sigmoid-scored with an
aux-loss-free `e_score_correction_bias`, BUT unlike DeepSeek-V3 / GLM
(`GlmNoAuxTcGate`, where the bias tilts selection only), the reference adds
the bias to the sigmoid scores IN PLACE, so the gathered expert weights are
the BIASED scores. `_KimiMoeGate` reproduces that faithfully; it is a
separate gate on purpose (HF code wins over the DeepSeek convention).

Serving: `USES_STATE_CACHE = True` routes the family to the per-request
`KimiStateCache` and the `StateCacheContinuousScheduler`, via the same
`forward_prefill_with_cache` / `forward_decode_with_cache[_ragged]` contract
as DeepSeek-V4. Because the whole family is position-free (NoPE + linear
attention), `forward_prefill_with_cache` also supports CONTINUATION from
`state_cache.start_pos > 0`, i.e. chunked prefill, which V4's state path
does not. Single-rank only for now (TP would shard the KDA conv channels
and the per-head state; a follow-up like Inkling). Cross-request prefix
sharing (`StatePrefixCache`) is V4-only; the KDA state-snapshot equivalent
is a follow-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional

from mini_infer.cache.kimi_state_cache import (
    KimiKdaLayerState,
    KimiKdaStateSpec,
    KimiMlaLayerState,
    KimiMlaStateSpec,
    KimiStateCache,
    KimiStateLayerSpec,
)
from mini_infer.models import register_model
from mini_infer.models.base import BaseCausalLM, KVCacheDims
from mini_infer.models.blocks.kda import (
    causal_conv1d_prefill,
    gated_rmsnorm,
    kda_chunkwise,
    kda_gate,
    kda_recurrent,
    l2norm,
)
from mini_infer.models.blocks.mixtral_moe import MixtralExpert
from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.swiglu import SwiGLU


@dataclass
class KimiLinearConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    # MLA (full-attention) layers' shape. `q_lora_rank` is always None and
    # `mla_use_nope` always True on this family; `from_hf` enforces both.
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
    intermediate_size: int  # dense-layer SwiGLU width
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

    @classmethod
    def from_hf(cls, hf_config: Any) -> KimiLinearConfig:
        linear_attn = getattr(hf_config, "linear_attn_config", None)
        if not linear_attn:
            raise ValueError(
                "KimiLinearConfig requires `linear_attn_config` (the hybrid "
                "KDA/MLA layout); a config without it is not this family"
            )
        if getattr(hf_config, "q_lora_rank", None) is not None:
            raise ValueError("Kimi Linear ships q_lora_rank=None; low-rank Q is not this family")
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
        num_experts = getattr(hf_config, "num_experts", None)
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
        # Per-head decay rate, checkpoint layout (1, 1, H, 1); init matches
        # the reference's log-uniform(1, 16) so a fresh model is finite.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(
                1, 1, -1, 1
            )
        )
        # The reference leaves dt_bias uninitialized (torch.empty); zeros keep
        # a fresh model finite, and checkpoints always overwrite it.
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        # Low-rank decay-gate and output-gate feature projections.
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
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

        decay = kda_gate(self.f_b_proj(self.f_a_proj(x)), self.A_log, self.dt_bias, self.head_dim)
        beta = self.b_proj(x).float().sigmoid()

        # L2-normalize q/k and scale q BEFORE the recurrence (the reference
        # kernels run with `use_qk_l2norm_in_kernel=True` and scale K^-0.5).
        q = l2norm(q_raw.view(batch, seq_len, self.num_heads, self.head_dim))
        q = q * self.head_dim**-0.5
        k = l2norm(k_raw.view(batch, seq_len, self.num_heads, self.head_dim))
        v = v_raw.view(batch, seq_len, self.num_heads, self.head_dim)

        initial = layer_state.recurrent_state if layer_state is not None else None
        # Reference mode split: fused_recurrent for spans <= 64, chunked
        # beyond. Both are the same math (pinned by test_kda_block).
        if seq_len <= 64:
            out, final_state = kda_recurrent(q, k, v, decay, beta, initial_state=initial)
        else:
            out, final_state = kda_chunkwise(q, k, v, decay, beta, initial_state=initial)

        if layer_state is not None:
            layer_state.recurrent_state.copy_(final_state)
            layer_state.conv_q.copy_(new_conv_q)
            layer_state.conv_k.copy_(new_conv_k)
            layer_state.conv_v.copy_(new_conv_v)

        gate = self.g_b_proj(self.g_a_proj(x)).view(batch, seq_len, self.num_heads, self.head_dim)
        out = gated_rmsnorm(out, gate, self.o_norm_weight, self.o_norm_eps)
        result: torch.Tensor = self.o_proj(out.reshape(batch, seq_len, -1))
        return result


class _KimiMlaAttention(nn.Module):
    """One NoPE MLA layer over the compressed-KV state buffer.

    Same decompression math as `blocks/mla.py` (cache the shared
    `kv_a_proj_with_mqa` output, rebuild per-head K/V on read) but NO rotary
    embedding anywhere, and reads/writes a dense per-request buffer instead
    of PagedKVCache streams.
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
        self.q_proj = nn.Linear(hidden, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
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
        queries = self.q_proj(x).view(batch, seq_len, self.num_heads, self.qk_head_dim)

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
        attn = torch.einsum("bhqk,bkhd->bqhd", weights, values)
        result: torch.Tensor = self.o_proj(attn.reshape(batch, seq_len, -1))
        return result


class _KimiMoeGate(nn.Module):
    """Kimi router: sigmoid (or softmax) scores + IN-PLACE correction bias.

    Faithful to the reference `KimiMoEGate`, which differs from the
    DeepSeek-V3 / GLM `noaux_tc` convention in two load-bearing ways:
      - the bias is added to the scores in place, so the gathered expert
        WEIGHTS are the biased scores (GLM gathers unbiased ones);
      - non-kept groups are masked to `0.0` (not `-inf`) before the top-k.
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
        self.weight = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Reference ships this uninitialized (torch.empty); zeros keep a
        # fresh model finite, checkpoints overwrite.
        self.e_score_correction_bias = nn.Parameter(torch.zeros(cfg.num_experts))

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route `(tokens, hidden)`; returns `(top_k_indices, top_k_weights)`,
        weights fp32, gathered from the BIASED scores, renormalized, scaled."""
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
        masked_choice = choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(masked_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = choice.gather(1, topk_indices)
        if self.top_k > 1 and self.moe_renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_indices, topk_weights * self.routed_scaling_factor


class _KimiMoe(nn.Module):
    """Sparse FFN: `_KimiMoeGate` + routed `MixtralExpert`s + shared SwiGLU.

    Expert tensors keep the checkpoint's names (`experts.N.w1/w2/w3`,
    `shared_experts.gate/up/down_proj`) so loading is a direct copy. Routed
    contributions accumulate in fp32, matching the reference's fp32 combine.
    """

    def __init__(self, cfg: KimiLinearConfig) -> None:
        super().__init__()
        assert cfg.num_experts is not None
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList(
            [
                MixtralExpert(cfg.hidden_size, cfg.moe_intermediate_size)
                for _ in range(cfg.num_experts)
            ]
        )
        self.gate = _KimiMoeGate(cfg)
        self.shared_experts: SwiGLU | None = (
            SwiGLU(cfg.hidden_size, cfg.moe_intermediate_size * cfg.num_shared_experts)
            if cfg.num_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        flat = x.view(-1, x.shape[-1])
        topk_indices, topk_weights = self.gate(flat)

        routed = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        expert_mask = functional.one_hot(topk_indices, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            weights = topk_weights[token_idx, top_k_pos, None]
            routed.index_add_(
                0, token_idx, self.experts[expert_idx](flat[token_idx]).float() * weights
            )
        out = routed.to(x.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(input_shape)


class _KimiDecoderLayer(nn.Module):
    """Pre-norm block: (KDA | NoPE-MLA) attention, then (SwiGLU | MoE) FFN."""

    def __init__(self, cfg: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = cfg.is_kda_layer(layer_idx)
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
            self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_state: KimiKdaLayerState | KimiMlaLayerState | None,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        if self.is_kda:
            assert layer_state is None or isinstance(layer_state, KimiKdaLayerState)
            x = self.self_attn(x, layer_state)
        else:
            assert layer_state is None or isinstance(layer_state, KimiMlaLayerState)
            x = self.self_attn(x, layer_state, row_starts)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.block_sparse_moe(x) if hasattr(self, "block_sparse_moe") else self.mlp(x)
        out: torch.Tensor = residual + x
        return out


class _KimiInnerModel(nn.Module):
    """Embedding + N decoder layers + final norm. HF prefix `model.`."""

    def __init__(self, cfg: KimiLinearConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [_KimiDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


# HF wraps each KDA conv in FLA's ShortConvolution (nn.Conv1d): (C, 1, W) -> (C, W).
_KDA_CONV_RE = re.compile(r"^(.*\.(?:q|k|v)_conv1d)\.weight$")
# FusedRMSNormGated's weight -> our flat parameter name.
_O_NORM_RE = re.compile(r"^(.*)\.o_norm\.weight$")


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
        for layer_idx, layer_module in enumerate(self.model.layers):
            layer_state = state_cache.layer(layer_idx) if state_cache is not None else None
            hidden = layer_module(hidden, layer_state, row_starts)
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
        remapped = _remap_kimi_state(hf_state_dict)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        missing_set = set(missing) - model.expected_missing_state_keys()
        if missing_set or unexpected:
            raise ValueError(
                f"weight load mismatch for KimiLinearForCausalLM: "
                f"missing={sorted(missing_set)}, unexpected={sorted(unexpected)}"
            )


def _remap_kimi_state(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF tensors -> our module keys.

    The module tree deliberately mirrors the reference names, so almost
    everything copies straight through. Two exceptions: the KDA conv weights
    lose HF's Conv1d singleton dim, and `o_norm.weight` maps to our flat
    `o_norm_weight` parameter.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        conv = _KDA_CONV_RE.match(key)
        if conv is not None:  # (C, 1, W) -> (C, W)
            remapped[f"{conv.group(1)}.weight"] = tensor.squeeze(1)
            continue
        o_norm = _O_NORM_RE.match(key)
        if o_norm is not None:
            remapped[f"{o_norm.group(1)}.o_norm_weight"] = tensor
            continue
        remapped[key] = tensor
    return remapped
