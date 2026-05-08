"""Per-request state cache for DeepSeek-V4 attention (HCA / CSA).

The V4 attention has two pieces of cache state that don't fit cleanly
into PagedKVCache's "one entry per token" model:

  - **Sliding-window KV (SWA):** the most recent `n_win` raw KV entries,
    one per token. Always uncompressed. Kept as a circular buffer at
    inference time so each new decode step shifts the oldest entry out.
  - **Compressor accumulator state:** for the in-flight block — i.e.
    tokens that have arrived but the compressor hasn't yet emitted a
    compressed entry for (the next `m` boundary hasn't fallen yet).
    Holds the per-token KV and score values for the partial block.

The compressed entries themselves are large append-only history (one
entry per `m` tokens), so they could plug into PagedKVCache as a stream
the way Stage C3 wired MLA's `kv_latent`. We keep them in this same
per-request StateCache for now to make the C4c parity test self-contained;
moving them to PagedKVCache is one of the follow-up stages.

Stage C4c lands HCA decode. CSA decode needs an additional sub-cache
for the Lightning Indexer's own compressor — handled by a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass(frozen=True)
class StateLayerSpec:
    """Static configuration for one layer's state.

    Attributes:
        kv_head_dim: Width of one compressed/uncompressed entry. The shared
            MQA head dim — V4 uses one head, all `n_h` queries broadcast.
        compression_ratio: `m` (CSA) or `m'` (HCA). One compressed entry
            per `compression_ratio` raw tokens.
        n_win: Sliding-window size — number of raw entries kept uncompressed.
        max_n_compressed: Cap on the compressed history. The caller picks
            this based on the max sequence length they intend to support
            (`max_n_compressed >= ceil(max_seq_len / compression_ratio)`).
        overlap_mode: `True` only for CSA's main compressor (m=4 in V4).
            HCA uses `overlap_mode=False`. Stage C4c.A wires only the
            non-overlap path; the field is here so CSA can fill it later
            without changing the dataclass shape.
    """

    kv_head_dim: int
    compression_ratio: int
    n_win: int
    max_n_compressed: int
    overlap_mode: bool = False


@dataclass
class _LayerState:
    """Mutable per-layer state tensors.

    Caller (the attention block) reads/writes these directly. We don't
    bury the math in methods because the indexing needs to be visible
    at the call site for the parity test to be auditable.
    """

    # Circular SWA buffer: write position is `start_pos % n_win`.
    swa_kv: torch.Tensor  # (B, n_win, kv_head_dim)
    # Append-only compressed history: write index is `n_completed_blocks`.
    compressed_kv: torch.Tensor  # (B, max_n_compressed, kv_head_dim)
    # In-flight block accumulator (for the partial block waiting on its
    # next `m` boundary). Indexed by `start_pos % compression_ratio`.
    cmp_kv_state: torch.Tensor  # (B, compression_ratio, kv_head_dim)
    cmp_score_state: torch.Tensor  # (B, compression_ratio, kv_head_dim)
    # How many compressed entries have been written so far. The block
    # increments this on every flush.
    n_compressed_blocks: int = 0
    # How many raw tokens have been written into `swa_kv` (capped at n_win
    # for downstream "how much of the window is valid?" queries).
    swa_count: int = 0


class StateCache:
    """Per-request, per-layer fixed-size pool for V4 SWA + compressor state.

    One `_LayerState` per layer of the model. All layers share the same
    request batch dimension (one StateCache per request slot in a higher-
    level scheduler; this class itself is request-local).

    Lifecycle:
        - Construct with the per-layer specs and batch size.
        - Prefill: the attention block writes the prefill's SWA window
          and any flushed compressed entries into the per-layer state.
        - Decode: each step writes one new SWA entry (circular) + updates
          the compressor accumulator + maybe flushes one compressed entry.
        - `advance_start_pos(n)` moves the global token-position counter,
          which the attention block consults at decode time.
    """

    def __init__(
        self,
        layer_specs: list[StateLayerSpec],
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not layer_specs:
            raise ValueError("layer_specs must be non-empty")
        for i, spec in enumerate(layer_specs):
            if spec.n_win <= 0:
                raise ValueError(f"layer {i}: n_win must be positive, got {spec.n_win}")
            if spec.compression_ratio <= 0:
                raise ValueError(
                    f"layer {i}: compression_ratio must be positive, got {spec.compression_ratio}"
                )
            if spec.max_n_compressed <= 0:
                raise ValueError(
                    f"layer {i}: max_n_compressed must be positive, got {spec.max_n_compressed}"
                )
            if spec.overlap_mode:
                # CSA support is the next stage — explicit fail keeps surprise out.
                raise NotImplementedError(
                    f"layer {i}: overlap_mode=True (CSA) not yet supported in StateCache"
                )
        self.layer_specs = layer_specs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self._layers: list[_LayerState] = []
        for spec in layer_specs:
            self._layers.append(
                _LayerState(
                    swa_kv=torch.zeros(
                        batch_size, spec.n_win, spec.kv_head_dim, device=self.device, dtype=dtype
                    ),
                    compressed_kv=torch.zeros(
                        batch_size,
                        spec.max_n_compressed,
                        spec.kv_head_dim,
                        device=self.device,
                        dtype=dtype,
                    ),
                    cmp_kv_state=torch.zeros(
                        batch_size,
                        spec.compression_ratio,
                        spec.kv_head_dim,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    cmp_score_state=torch.full(
                        (batch_size, spec.compression_ratio, spec.kv_head_dim),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
        # Global token position (advanced by the model after each forward call).
        self.start_pos = 0

    def layer(self, layer_idx: int) -> _LayerState:
        """Per-layer state — caller reads/writes the tensors directly."""
        return self._layers[layer_idx]

    def advance_start_pos(self, n: int) -> None:
        """Advance the global token-position counter by `n` (called after a forward pass)."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.start_pos += n

    @property
    def num_layers(self) -> int:
        return len(self._layers)


# `field` is referenced indirectly via dataclass defaults on _LayerState elsewhere;
# importing it here keeps the import-list intent explicit.
_ = field
