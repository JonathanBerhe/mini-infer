"""Per-request state cache for the Qwen hybrid families (Qwen3.5, Qwen3.6, Qwen3.8).

A Qwen hybrid layer carries one of two kinds of decode state, neither of
which fits the paged one-entry-per-token KV model:

  - **Gated DeltaNet layers** (linear attention, 3 of every 4): a fixed-size
    fp32 matrix state `(num_v_heads, head_k_dim, head_v_dim)` updated by the
    gated delta rule, plus ONE short-convolution tail
    `(conv_channels, kernel)` holding the last `kernel` RAW pre-conv inputs
    (newest last). The layer fuses q, k and v into a single projection and
    runs a single depthwise convolution over all of those channels, so there
    is one tail here where Kimi Linear's KDA layers have three. Constant
    size per request regardless of context length; this is where the hybrid's
    long-context cache saving comes from.
  - **Gated full-attention layers** (1 of every 4): an append-only per-token
    buffer of ordinary per-head K and V, `(max_seq_len, num_kv_heads,
    head_dim)` each. This is plain grouped-query attention, NOT Kimi's
    compressed MLA buffer, so nothing is re-decompressed on read. K is
    stored AFTER its RMS norm and AFTER rotary embedding, which is the
    subtle part: the stored keys are already positioned, so a decode step
    only has to position its own new query.

This mirrors `KimiStateCache` in lifecycle: one instance per request slot
(or one batched instance whose rows are slots), a global `start_pos` the
model reads at decode time, and `copy_row_from` so the continuous-batching
scheduler can move a prefilled request into a batch row. It is deliberately
a separate class rather than new fields on `KimiStateCache`: the two share
a lifecycle but no tensors (one fused conv tail against three, plain GQA
buffers against compressed MLA), and keeping each cache's indexing visible
at its own call sites is the point of the state-cache design.

The head dimensions are kept as separate `head_k_dim` and `head_v_dim`
fields even though every checkpoint shipped so far has them equal, so that
a non-square release does not require reshaping this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QwenGdnStateSpec:
    """Static shape of one Gated DeltaNet layer's state.

    Attributes:
        num_v_heads: value heads, which is also the number of recurrent
            state matrices (q and k are broadcast up to this count before
            the recurrence).
        head_k_dim: key head dimension, the state's row count.
        head_v_dim: value head dimension, the state's column count.
        conv_channels: channels of the fused q/k/v convolution, equal to
            `2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim`.
        conv_kernel_size: taps per channel.
    """

    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_channels: int
    conv_kernel_size: int


@dataclass(frozen=True)
class QwenAttentionStateSpec:
    """Static shape of one gated full-attention layer's K/V buffers.

    Attributes:
        num_kv_heads: grouped-query KV heads.
        head_dim: per-head dimension.
        max_seq_len: hard bound on positions this cache can hold. Admission
            has to respect it, because unlike the linear layers these buffers
            grow with context.
    """

    num_kv_heads: int
    head_dim: int
    max_seq_len: int


QwenStateLayerSpec = QwenGdnStateSpec | QwenAttentionStateSpec


@dataclass
class QwenGdnLayerState:
    """Mutable per-request state of one Gated DeltaNet layer."""

    recurrent_state: torch.Tensor  # (B, num_v_heads, head_k_dim, head_v_dim) fp32
    conv: torch.Tensor  # (B, conv_channels, kernel) raw pre-conv inputs, newest last


@dataclass
class QwenAttentionLayerState:
    """Mutable per-request state of one gated full-attention layer."""

    k: torch.Tensor  # (B, max_seq_len, num_kv_heads, head_dim), post-norm and post-rope
    v: torch.Tensor  # (B, max_seq_len, num_kv_heads, head_dim)


QwenLayerState = QwenGdnLayerState | QwenAttentionLayerState


class Qwen35StateCache:
    """Per-request fixed-size pool for hybrid Gated DeltaNet / gated attention state.

    Same lifecycle contract as `KimiStateCache`: construct with per-layer
    specs and a batch size, prefill populates it, each decode step reads and
    extends it, and the caller advances `start_pos` after every forward.
    """

    def __init__(
        self,
        layer_specs: list[QwenStateLayerSpec],
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
        self._layers: list[QwenLayerState] = []
        for spec in layer_specs:
            if isinstance(spec, QwenGdnStateSpec):
                self._layers.append(
                    QwenGdnLayerState(
                        # The recurrence runs in fp32 regardless of the
                        # activation dtype; the checkpoint's own config asks
                        # for it via mamba_ssm_dtype float32.
                        recurrent_state=torch.zeros(
                            batch_size,
                            spec.num_v_heads,
                            spec.head_k_dim,
                            spec.head_v_dim,
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
            else:
                if spec.max_seq_len <= 0:
                    raise ValueError(f"max_seq_len must be positive, got {spec.max_seq_len}")
                shape = (batch_size, spec.max_seq_len, spec.num_kv_heads, spec.head_dim)
                self._layers.append(
                    QwenAttentionLayerState(
                        k=torch.zeros(shape, device=self.device, dtype=dtype),
                        v=torch.zeros(shape, device=self.device, dtype=dtype),
                    )
                )
        # Global token position (advanced by the caller after each forward).
        self.start_pos = 0

    def layer(self, layer_idx: int) -> QwenLayerState:
        """Per-layer state; the attention block reads/writes the tensors directly."""
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
        """Positions the full-attention buffers can hold.

        Returns a large sentinel when the model has no full-attention layer,
        since only those layers grow with context.
        """
        lengths = [
            spec.max_seq_len
            for spec in self.layer_specs
            if isinstance(spec, QwenAttentionStateSpec)
        ]
        return min(lengths) if lengths else 2**62

    def check_capacity(self, end_pos: int) -> None:
        """Raise before a write that would run past the K/V buffers.

        Called by the attention layers ahead of an append so that an
        over-long request fails with a clear message instead of an opaque
        indexing error deep inside a forward pass.
        """
        limit = self.max_seq_len
        if end_pos > limit:
            raise ValueError(
                f"sequence position {end_pos} exceeds the cache's max_seq_len {limit}; "
                "admit fewer tokens or build the cache with a larger max_seq_len"
            )

    def copy_row_from(self, src_cache: Qwen35StateCache, *, src_row: int, dst_row: int) -> None:
        """Copy one request's full per-layer state from `src_cache[src_row]`
        into this cache's row `dst_row` (the scheduler's admit-into-slot move)."""
        if src_cache.num_layers != self.num_layers:
            raise ValueError(
                f"layer count mismatch: src has {src_cache.num_layers}, dst has {self.num_layers}"
            )
        for layer_idx in range(self.num_layers):
            src = src_cache.layer(layer_idx)
            dst = self._layers[layer_idx]
            if isinstance(src, QwenGdnLayerState) != isinstance(dst, QwenGdnLayerState):
                raise ValueError(f"layer {layer_idx}: state kind mismatch between caches")
            if isinstance(src, QwenGdnLayerState) and isinstance(dst, QwenGdnLayerState):
                dst.recurrent_state[dst_row] = src.recurrent_state[src_row]
                dst.conv[dst_row] = src.conv[src_row]
            elif isinstance(src, QwenAttentionLayerState) and isinstance(
                dst, QwenAttentionLayerState
            ):
                copy_len = min(dst.k.shape[1], src.k.shape[1])
                dst.k[dst_row, :copy_len] = src.k[src_row, :copy_len]
                dst.v[dst_row, :copy_len] = src.v[src_row, :copy_len]
