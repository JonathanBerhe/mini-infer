"""Per-request state cache for the Nemotron-H hybrid (Mamba-2 / attention / MoE).

A Nemotron-H layer holds exactly one mixer, and only two kinds carry decode
state:

  - **Mamba-2 layers**: a fixed-size fp32 SSM state `(num_heads, head_dim,
    state_size)` and one short-conv tail `(conv_channels, kernel)` with the
    last `kernel` raw pre-conv inputs, newest last. The conv runs over the
    concatenated x, B and C channels, so there is one tail per layer.
  - **Attention layers**: append-only per-token K and V buffers
    `(max_seq_len, num_kv_heads, head_dim)`. The family uses no rotary
    embedding, so the stored keys are the raw projections.

MoE and dense-MLP layers have no state; their slot holds None.

Same lifecycle as `Qwen35StateCache` and `KimiStateCache`: one batched
instance whose rows are request slots, a global `start_pos`, and
`copy_row_from` so the scheduler can move a prefilled request into a row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NemotronMambaStateSpec:
    """Static shape of one Mamba-2 layer's state."""

    num_heads: int
    head_dim: int
    state_size: int
    conv_channels: int
    conv_kernel_size: int


@dataclass(frozen=True)
class NemotronAttentionStateSpec:
    """Static shape of one attention layer's K/V buffers."""

    num_kv_heads: int
    head_dim: int
    max_seq_len: int


NemotronStateLayerSpec = NemotronMambaStateSpec | NemotronAttentionStateSpec | None


@dataclass
class NemotronMambaLayerState:
    """Mutable per-request state of one Mamba-2 layer."""

    ssm: torch.Tensor  # (B, num_heads, head_dim, state_size) fp32
    conv: torch.Tensor  # (B, conv_channels, kernel) raw pre-conv inputs, newest last


@dataclass
class NemotronAttentionLayerState:
    """Mutable per-request state of one attention layer."""

    k: torch.Tensor  # (B, max_seq_len, num_kv_heads, head_dim)
    v: torch.Tensor  # (B, max_seq_len, num_kv_heads, head_dim)


NemotronLayerState = NemotronMambaLayerState | NemotronAttentionLayerState | None


class NemotronHStateCache:
    """Per-request fixed-size pool for the Mamba-2 and attention layers of Nemotron-H."""

    def __init__(
        self,
        layer_specs: list[NemotronStateLayerSpec],
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
        self._layers: list[NemotronLayerState] = []
        for spec in layer_specs:
            if isinstance(spec, NemotronMambaStateSpec):
                self._layers.append(
                    NemotronMambaLayerState(
                        # The reference keeps the SSM state in fp32
                        # (`mamba_ssm_cache_dtype`), whatever the model dtype.
                        ssm=torch.zeros(
                            batch_size,
                            spec.num_heads,
                            spec.head_dim,
                            spec.state_size,
                            device=self.device,
                            dtype=torch.float32,
                        ),
                        conv=torch.zeros(
                            batch_size,
                            spec.conv_channels,
                            spec.conv_kernel_size,
                            device=self.device,
                            dtype=dtype,
                        ),
                    )
                )
            elif isinstance(spec, NemotronAttentionStateSpec):
                if spec.max_seq_len <= 0:
                    raise ValueError(f"max_seq_len must be positive, got {spec.max_seq_len}")
                shape = (batch_size, spec.max_seq_len, spec.num_kv_heads, spec.head_dim)
                self._layers.append(
                    NemotronAttentionLayerState(
                        k=torch.zeros(shape, device=self.device, dtype=dtype),
                        v=torch.zeros(shape, device=self.device, dtype=dtype),
                    )
                )
            else:
                self._layers.append(None)
        self.start_pos = 0

    def layer(self, layer_idx: int) -> NemotronLayerState:
        return self._layers[layer_idx]

    def advance_start_pos(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.start_pos += n

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def max_seq_len(self) -> int:
        """Positions the attention buffers can hold (a large sentinel without attention)."""
        lengths = [
            spec.max_seq_len
            for spec in self.layer_specs
            if isinstance(spec, NemotronAttentionStateSpec)
        ]
        return min(lengths) if lengths else 2**62

    def check_capacity(self, end_pos: int) -> None:
        """Raise before a write that would run past the K/V buffers."""
        limit = self.max_seq_len
        if end_pos > limit:
            raise ValueError(
                f"sequence position {end_pos} exceeds the cache's max_seq_len {limit}; "
                "admit fewer tokens or build the cache with a larger max_seq_len"
            )

    def copy_row_from(self, src_cache: NemotronHStateCache, *, src_row: int, dst_row: int) -> None:
        """Copy one request's state from `src_cache[src_row]` into row `dst_row`, in place."""
        if src_cache.num_layers != self.num_layers:
            raise ValueError(
                f"layer count mismatch: src has {src_cache.num_layers}, dst has {self.num_layers}"
            )
        for layer_idx in range(self.num_layers):
            src = src_cache.layer(layer_idx)
            dst = self._layers[layer_idx]
            if type(src) is not type(dst):
                raise ValueError(f"layer {layer_idx}: state kind mismatch between caches")
            if isinstance(src, NemotronMambaLayerState) and isinstance(
                dst, NemotronMambaLayerState
            ):
                dst.ssm[dst_row] = src.ssm[src_row]
                dst.conv[dst_row] = src.conv[src_row]
            elif isinstance(src, NemotronAttentionLayerState) and isinstance(
                dst, NemotronAttentionLayerState
            ):
                span = min(dst.k.shape[1], src.k.shape[1])
                dst.k[dst_row, :span] = src.k[src_row, :span]
                dst.v[dst_row, :span] = src.v[src_row, :span]
