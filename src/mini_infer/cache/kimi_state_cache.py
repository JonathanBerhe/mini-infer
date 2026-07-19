"""Per-request state cache for hybrid KDA/MLA models (Kimi Linear, Kimi K3).

A Kimi-family layer carries one of two kinds of decode state, neither of
which fits the paged one-entry-per-token KV model:

  - **KDA layers** (linear attention): a fixed-size fp32 matrix state
    `(num_heads, head_dim_k, head_dim_v)` updated by the gated delta rule,
    plus three short-convolution tails `(conv_channels, kernel)` holding the
    last `kernel` RAW pre-conv inputs for the q/k/v convolutions (FLA cache
    layout, newest last). Constant size per request regardless of context
    length; this is where Kimi Linear's 75% cache reduction comes from.
  - **MLA layers** (the 1-in-4 full-attention layers): an append-only
    per-token buffer of the COMPRESSED `kv_a_proj_with_mqa` output
    (`kv_lora_rank + qk_rope_head_dim` wide, shared across heads). The
    per-head K/V are re-decompressed on read, exactly like `blocks/mla.py`;
    Kimi's MLA is NoPE so nothing position-dependent is baked in.

This mirrors `StateCache` (the DeepSeek-V4 per-request cache) in lifecycle:
one instance per request slot (or one batched instance whose rows are
slots), a global `start_pos` the model reads at decode time, and
`copy_row_from` so the continuous-batching scheduler can move a prefilled
request into a batch row. It is deliberately a separate class rather than
new fields on `StateCache`: the V4 state (SWA ring + compressor
accumulators) and the Kimi state (recurrent matrix + conv tails + dense MLA
history) share no tensors, and keeping each cache's indexing visible at its
own call sites is the point of the state-cache design.

Cross-request prefix sharing (`StatePrefixCache`) is V4-only for now; a
KDA-state snapshot cache (what Moonshot upstreamed to vLLM as "KDA with
prefill cache") is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KimiKdaStateSpec:
    """Static shape of one KDA layer's state.

    Attributes:
        num_heads: KDA heads (`linear_attn_config.num_heads`).
        head_dim: per-head key AND value dim (`linear_attn_config.head_dim`;
            Kimi uses square states).
        conv_channels: channels of each short conv (`num_heads * head_dim`).
        conv_kernel_size: taps per channel
            (`linear_attn_config.short_conv_kernel_size`).
    """

    num_heads: int
    head_dim: int
    conv_channels: int
    conv_kernel_size: int


@dataclass(frozen=True)
class KimiMlaStateSpec:
    """Static shape of one MLA layer's state.

    Attributes:
        kv_width: width of one cached entry, `kv_lora_rank +
            qk_rope_head_dim` (the raw `kv_a_proj_with_mqa` output).
        max_seq_len: dense per-request capacity. Sized by the caller to
            prompt + generation budget; unlike the paged path this is a
            hard bound, admission control must respect it.
    """

    kv_width: int
    max_seq_len: int


KimiStateLayerSpec = KimiKdaStateSpec | KimiMlaStateSpec


@dataclass
class KimiKdaLayerState:
    """Mutable KDA state. The attention block reads/writes these directly."""

    # Delta-rule matrix state, always fp32 (matches the FLA kernels'
    # `initial_state must be in float32` contract).
    recurrent_state: torch.Tensor  # (B, num_heads, head_dim, head_dim)
    # Last `kernel` raw pre-conv inputs per conv, newest at [..., -1].
    conv_q: torch.Tensor  # (B, conv_channels, kernel)
    conv_k: torch.Tensor  # (B, conv_channels, kernel)
    conv_v: torch.Tensor  # (B, conv_channels, kernel)


@dataclass
class KimiMlaLayerState:
    """Mutable MLA state: append-only compressed-KV history."""

    kv: torch.Tensor  # (B, max_seq_len, kv_width)


KimiLayerState = KimiKdaLayerState | KimiMlaLayerState


class KimiStateCache:
    """Per-request fixed-size pool for hybrid KDA/MLA attention state.

    Same lifecycle contract as `StateCache`: construct with per-layer specs
    and a batch size, prefill populates it, each decode step reads and
    extends it, and the caller advances `start_pos` after every forward.
    """

    def __init__(
        self,
        layer_specs: list[KimiStateLayerSpec],
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not layer_specs:
            raise ValueError("layer_specs must be non-empty")
        self.layer_specs = layer_specs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self._layers: list[KimiLayerState] = []
        for spec in layer_specs:
            if isinstance(spec, KimiKdaStateSpec):
                self._layers.append(
                    KimiKdaLayerState(
                        recurrent_state=torch.zeros(
                            batch_size,
                            spec.num_heads,
                            spec.head_dim,
                            spec.head_dim,
                            device=self.device,
                            dtype=torch.float32,
                        ),
                        conv_q=torch.zeros(
                            batch_size,
                            spec.conv_channels,
                            spec.conv_kernel_size,
                            device=self.device,
                            dtype=dtype,
                        ),
                        conv_k=torch.zeros(
                            batch_size,
                            spec.conv_channels,
                            spec.conv_kernel_size,
                            device=self.device,
                            dtype=dtype,
                        ),
                        conv_v=torch.zeros(
                            batch_size,
                            spec.conv_channels,
                            spec.conv_kernel_size,
                            device=self.device,
                            dtype=dtype,
                        ),
                    )
                )
            else:
                if spec.max_seq_len <= 0:
                    raise ValueError(f"max_seq_len must be positive, got {spec.max_seq_len}")
                self._layers.append(
                    KimiMlaLayerState(
                        kv=torch.zeros(
                            batch_size,
                            spec.max_seq_len,
                            spec.kv_width,
                            device=self.device,
                            dtype=dtype,
                        )
                    )
                )
        # Global token position (advanced by the caller after each forward).
        self.start_pos = 0

    def layer(self, layer_idx: int) -> KimiLayerState:
        """Per-layer state; the attention block reads/writes the tensors directly."""
        return self._layers[layer_idx]

    def advance_start_pos(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.start_pos += n

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def copy_row_from(self, src_cache: KimiStateCache, *, src_row: int, dst_row: int) -> None:
        """Copy one request's full per-layer state from `src_cache[src_row]`
        into this cache's row `dst_row` (the scheduler's admit-into-slot move)."""
        if src_cache.num_layers != self.num_layers:
            raise ValueError(
                f"layer count mismatch: src has {src_cache.num_layers}, dst has {self.num_layers}"
            )
        for layer_idx in range(self.num_layers):
            src = src_cache.layer(layer_idx)
            dst = self._layers[layer_idx]
            if isinstance(src, KimiKdaLayerState) != isinstance(dst, KimiKdaLayerState):
                raise ValueError(f"layer {layer_idx}: state kind mismatch between caches")
            if isinstance(src, KimiKdaLayerState) and isinstance(dst, KimiKdaLayerState):
                dst.recurrent_state[dst_row] = src.recurrent_state[src_row]
                dst.conv_q[dst_row] = src.conv_q[src_row]
                dst.conv_k[dst_row] = src.conv_k[src_row]
                dst.conv_v[dst_row] = src.conv_v[src_row]
            elif isinstance(src, KimiMlaLayerState) and isinstance(dst, KimiMlaLayerState):
                # Capacities may differ (per-request temp vs batched cache);
                # copy the overlap, which the scheduler bounds by max_seq_len.
                span = min(src.kv.shape[1], dst.kv.shape[1])
                dst.kv[dst_row, :span] = src.kv[src_row, :span]
