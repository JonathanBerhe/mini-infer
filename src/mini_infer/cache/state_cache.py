"""Per-request state cache for DeepSeek-V4 attention (HCA + CSA).

The V4 attention has cache state that doesn't fit cleanly into the
"one entry per token" model PagedKVCache assumes:

  - **Sliding-window KV (SWA):** the most recent `n_win` raw KV entries,
    one per token. Always uncompressed. Kept as a circular buffer at
    decode time so each new step shifts the oldest entry out.
  - **Compressor accumulator state:** the in-flight block — tokens
    that have arrived but the compressor hasn't yet emitted a
    compressed entry for. Holds the per-token KV and score values
    for the partial block. Doubled in CSA's overlap mode (also holds
    the previous block's overlap data so the next flush's softmax
    spans `2m` slots).
  - **Lightning Indexer state (CSA only):** the indexer has its own
    compressor whose state lives separately from the main one because
    it operates at a different `head_dim`.

The compressed history (one entry per `m` tokens) is also held here
for now — moving it to PagedKVCache as a stream is a follow-up. Keeping
it local makes the parity tests self-contained.

Decode-step contract:
    1. Block writes new SWA entry at `swa_kv[:, start_pos % n_win]`.
    2. Block calls `compressor.forward_decode_step` with `cmp_kv_state`
       and `cmp_score_state`; on a block boundary the helper returns a
       compressed entry which the block appends to `compressed_kv`.
    3. CSA layer additionally drives its `LightningIndexer` against
       `indexer.compressed_kv[:n_compressed_blocks]` to pick top-k.
    4. Caller advances the global counter via `advance_start_pos(n)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndexerStateSpec:
    """Static configuration for one CSA layer's Lightning Indexer state.

    The indexer's compressor is always overlap mode (the V4 design fixes
    `compress_ratio = 4` for CSA), so we don't expose `overlap_mode` here.

    Attributes:
        head_dim: Per-head feature dim of the indexer's Q and compressed K.
            Independent of the parent attention's `kv_head_dim`. V4 uses
            128 for the indexer regardless of the main attention's head_dim.
    """

    head_dim: int


@dataclass(frozen=True)
class StateLayerSpec:
    """Static configuration for one layer's state.

    Attributes:
        kv_head_dim: Width of one compressed/uncompressed KV entry — the
            shared MQA head dim. All `n_h` queries broadcast across this.
        compression_ratio: `m` (CSA) or `m'` (HCA). One compressed entry
            per `compression_ratio` raw tokens. `0` marks a pure-SWA layer
            (sliding window only, no compressor or indexer).
        n_win: Sliding-window size; number of raw entries kept uncompressed.
        max_n_compressed: Cap on the compressed history. The caller picks
            this based on the max sequence length they intend to support
            (`>= ceil(max_seq_len / compression_ratio)`).
        overlap_mode: `True` only for CSA's main compressor. Doubles the
            compressor accumulator's slot count and feature width: slots
            `[0, m)` hold the previous block's data (so the next flush's
            `2m`-wide softmax covers it), slots `[m, 2m)` hold the
            in-flight current block. The doubled feature width comes from
            `kv_proj` / `weight_proj` emitting `2 * kv_head_dim` outputs
            in overlap mode.
        indexer: `IndexerStateSpec` populated for CSA layers, `None` for
            HCA layers. Drives whether the per-layer state allocates a
            second sub-cache for the Lightning Indexer.
    """

    kv_head_dim: int
    compression_ratio: int
    n_win: int
    max_n_compressed: int
    overlap_mode: bool = False
    indexer: IndexerStateSpec | None = None


@dataclass
class _IndexerState:
    """Mutable Lightning Indexer state for one CSA layer."""

    # Append-only history of compressed entries from the indexer's compressor.
    compressed_kv: torch.Tensor  # (B, max_n_compressed, indexer_head_dim)
    # In-flight block accumulator. Always overlap mode for CSA: shape
    # (B, 2 * compression_ratio, 2 * indexer_head_dim).
    cmp_kv_state: torch.Tensor
    cmp_score_state: torch.Tensor
    n_compressed_blocks: int = 0


@dataclass
class _LayerState:
    """Mutable per-layer state tensors.

    Caller (the attention block) reads/writes these directly. We don't
    bury the math in methods because the indexing is the math, and it
    needs to be visible at the call site for the parity tests to be
    auditable.
    """

    # Circular SWA buffer: write at `start_pos % n_win`.
    swa_kv: torch.Tensor  # (B, n_win, kv_head_dim)
    # Append-only compressed history: write at `n_compressed_blocks`.
    compressed_kv: torch.Tensor  # (B, max_n_compressed, kv_head_dim)
    # In-flight compressor block accumulator.
    # HCA (overlap_mode=False): (B, compression_ratio, kv_head_dim).
    # CSA (overlap_mode=True):  (B, 2 * compression_ratio, 2 * kv_head_dim).
    cmp_kv_state: torch.Tensor
    cmp_score_state: torch.Tensor
    # Set only when the layer's spec.indexer is populated (CSA layers).
    indexer: _IndexerState | None = None
    # How many compressed entries have been appended.
    n_compressed_blocks: int = 0
    # How many raw tokens have flowed into `swa_kv` (capped at n_win).
    swa_count: int = 0


class StateCache:
    """Per-request fixed-size pool for V4 attention state.

    One `_LayerState` per layer of the model. All layers share the same
    request batch dimension (the cache itself is request-local; a higher-
    level scheduler holds a `StateCache` per request slot).

    Lifecycle:
        - Construct with the per-layer specs and batch size.
        - Prefill: typically populated externally (e.g. the parity tests
          sync from the reference's kv_cache + compressor buffers).
        - Decode: each step writes one new SWA entry + updates the
          compressor accumulator + maybe flushes one compressed entry.
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
        for layer_idx, spec in enumerate(layer_specs):
            if spec.n_win <= 0:
                raise ValueError(f"layer {layer_idx}: n_win must be positive, got {spec.n_win}")
            if spec.compression_ratio < 0:
                raise ValueError(
                    f"layer {layer_idx}: compression_ratio must be non-negative, "
                    f"got {spec.compression_ratio}"
                )
            if spec.compression_ratio == 0:
                # Pure-SWA layer (V4's ratio-0 layers): sliding window only, no
                # compressor or indexer. Only `swa_kv` is used; the compressed
                # history and accumulator tensors allocate to zero width.
                if spec.overlap_mode:
                    raise ValueError(
                        f"layer {layer_idx}: compression_ratio == 0 (SWA) cannot use overlap_mode"
                    )
                if spec.indexer is not None:
                    raise ValueError(
                        f"layer {layer_idx}: compression_ratio == 0 (SWA) cannot have an indexer"
                    )
            elif spec.max_n_compressed <= 0:
                raise ValueError(
                    f"layer {layer_idx}: max_n_compressed must be positive, "
                    f"got {spec.max_n_compressed}"
                )
            if spec.indexer is not None and spec.indexer.head_dim <= 0:
                raise ValueError(
                    f"layer {layer_idx}: indexer.head_dim must be positive, "
                    f"got {spec.indexer.head_dim}"
                )
        self.layer_specs = layer_specs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self._layers: list[_LayerState] = []
        for spec in layer_specs:
            self._layers.append(self._build_layer_state(spec, batch_size, self.device, dtype))
        # Global token position (advanced by the model after each forward call).
        self.start_pos = 0

    @staticmethod
    def _build_layer_state(
        spec: StateLayerSpec,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _LayerState:
        """Allocate per-layer tensors at the shapes implied by the spec.

        Overlap mode doubles both the slot count (so `2m` slots cover the
        current + previous blocks) and the feature width (so `kv_proj`'s
        `2 * kv_head_dim`-wide outputs fit). Half of the `2 * kv_head_dim`
        is consumed by the "current" branch and half by the "overlap"
        branch — see `TokenLevelCompressor._overlap_transform` and the
        decode flush path.
        """
        coff = 2 if spec.overlap_mode else 1
        cmp_kv_state = torch.zeros(
            batch_size,
            spec.compression_ratio * coff,
            spec.kv_head_dim * coff,
            device=device,
            dtype=torch.float32,
        )
        cmp_score_state = torch.full(
            (batch_size, spec.compression_ratio * coff, spec.kv_head_dim * coff),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        layer_state = _LayerState(
            swa_kv=torch.zeros(
                batch_size, spec.n_win, spec.kv_head_dim, device=device, dtype=dtype
            ),
            compressed_kv=torch.zeros(
                batch_size,
                spec.max_n_compressed,
                spec.kv_head_dim,
                device=device,
                dtype=dtype,
            ),
            cmp_kv_state=cmp_kv_state,
            cmp_score_state=cmp_score_state,
        )
        if spec.indexer is not None:
            indexer_head_dim = spec.indexer.head_dim
            # Indexer's compressor is always overlap mode.
            layer_state.indexer = _IndexerState(
                compressed_kv=torch.zeros(
                    batch_size,
                    spec.max_n_compressed,
                    indexer_head_dim,
                    device=device,
                    dtype=dtype,
                ),
                cmp_kv_state=torch.zeros(
                    batch_size,
                    spec.compression_ratio * 2,
                    indexer_head_dim * 2,
                    device=device,
                    dtype=torch.float32,
                ),
                cmp_score_state=torch.full(
                    (batch_size, spec.compression_ratio * 2, indexer_head_dim * 2),
                    float("-inf"),
                    device=device,
                    dtype=torch.float32,
                ),
            )
        return layer_state

    def layer(self, layer_idx: int) -> _LayerState:
        """Per-layer state — caller reads/writes the tensors directly."""
        return self._layers[layer_idx]

    def advance_start_pos(self, n: int) -> None:
        """Advance the global token-position counter by `n`."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.start_pos += n

    @property
    def num_layers(self) -> int:
        return len(self._layers)
