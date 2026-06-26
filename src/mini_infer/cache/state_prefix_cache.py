"""Cross-request prefix sharing for the per-request `StateCache` (DeepSeek-V4).

When two requests share a prompt prefix (a common system prompt, few-shot
examples, a RAG document), the cache state after that prefix is identical, since
V4's compression / window / indexer state is a deterministic function of the
prefix tokens and their positions. This caches a snapshot of the full per-request
`StateCache` keyed by the prompt token ids, so a later request whose prompt
*extends* a cached one restores that state and only processes the new suffix,
skipping the shared prefill.

This v1 caches snapshots at the full-prompt boundary (after a prompt is
prefilled). It gives reuse for repeated and extended prompts; caching at
intermediate block boundaries (so a never-sent-alone prefix is still reusable) is
a follow-up. Snapshots are FIFO-capped to bound memory; a real deployment would
use a paged pool so shared blocks are pointed at, not copied (ADR-019).

The snapshot captures everything a subsequent decode reads: the per-layer
buffers, the scalar counters (`n_compressed_blocks`, `swa_count`) and the
indexer sub-state, the global `start_pos`, plus the logits that predict the token
right after the prefix (used directly on an exact-length hit, where there is no
suffix to replay).
"""

from __future__ import annotations

import dataclasses

import torch

from mini_infer.cache.state_cache import StateCache


@dataclasses.dataclass
class _LayerSnapshot:
    swa_kv: torch.Tensor
    compressed_kv: torch.Tensor
    cmp_kv_state: torch.Tensor
    cmp_score_state: torch.Tensor
    n_compressed_blocks: int
    swa_count: int
    indexer_compressed_kv: torch.Tensor | None
    indexer_cmp_kv_state: torch.Tensor | None
    indexer_cmp_score_state: torch.Tensor | None
    indexer_n_compressed_blocks: int


@dataclasses.dataclass
class PrefixSnapshot:
    """The full `StateCache` state after prefilling some prompt, plus the logits
    that predict the next token (for an exact-length hit)."""

    position: int
    next_logits: torch.Tensor  # (vocab,)
    layers: list[_LayerSnapshot]


class StatePrefixCache:
    """Maps a prompt token-id prefix to a `StateCache` snapshot for reuse.

    Single-request granularity: snapshots come from B=1 caches (row 0). FIFO
    eviction past `max_entries`.
    """

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max_entries = max_entries
        self._entries: dict[tuple[int, ...], PrefixSnapshot] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, prompt_ids: list[int]) -> tuple[int, PrefixSnapshot | None]:
        """Return `(matched_len, snapshot)` for the longest cached prompt that is
        a prefix of `prompt_ids` (matched_len <= len(prompt_ids)), else `(0, None)`."""
        best_len = 0
        best: PrefixSnapshot | None = None
        for key, snapshot in self._entries.items():
            klen = len(key)
            if klen <= len(prompt_ids) and klen > best_len and prompt_ids[:klen] == list(key):
                best_len, best = klen, snapshot
        return best_len, best

    def insert(self, prompt_ids: list[int], snapshot: PrefixSnapshot) -> None:
        key = tuple(prompt_ids)
        if key in self._entries:
            return
        if len(self._entries) >= self._max_entries:
            # FIFO: drop the oldest inserted entry.
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        self._entries[key] = snapshot

    @staticmethod
    def snapshot_from_cache(
        state_cache: StateCache, next_logits: torch.Tensor, *, row: int = 0
    ) -> PrefixSnapshot:
        """Clone `state_cache` row `row` into a reusable snapshot."""
        layers: list[_LayerSnapshot] = []
        for layer_idx in range(state_cache.num_layers):
            layer = state_cache.layer(layer_idx)
            indexer = layer.indexer
            layers.append(
                _LayerSnapshot(
                    # Only the valid compressed-block prefix is cloned; the buffer's
                    # full width depends on the source prompt's max_seq_len, which
                    # need not match the target cache restored into.
                    swa_kv=layer.swa_kv[row : row + 1].clone(),
                    compressed_kv=layer.compressed_kv[
                        row : row + 1, : layer.n_compressed_blocks
                    ].clone(),
                    cmp_kv_state=layer.cmp_kv_state[row : row + 1].clone(),
                    cmp_score_state=layer.cmp_score_state[row : row + 1].clone(),
                    n_compressed_blocks=layer.n_compressed_blocks,
                    swa_count=layer.swa_count,
                    indexer_compressed_kv=(
                        indexer.compressed_kv[row : row + 1, : indexer.n_compressed_blocks].clone()
                        if indexer
                        else None
                    ),
                    indexer_cmp_kv_state=(
                        indexer.cmp_kv_state[row : row + 1].clone() if indexer else None
                    ),
                    indexer_cmp_score_state=(
                        indexer.cmp_score_state[row : row + 1].clone() if indexer else None
                    ),
                    indexer_n_compressed_blocks=(indexer.n_compressed_blocks if indexer else 0),
                )
            )
        return PrefixSnapshot(
            position=state_cache.start_pos, next_logits=next_logits.clone(), layers=layers
        )

    @staticmethod
    def restore_into(snapshot: PrefixSnapshot, state_cache: StateCache, *, row: int = 0) -> None:
        """Copy a snapshot back into `state_cache` row `row`, restoring counters
        and `start_pos` so a subsequent decode continues correctly."""
        if state_cache.num_layers != len(snapshot.layers):
            raise ValueError("snapshot layer count does not match the target StateCache")
        for layer_idx, layer_snap in enumerate(snapshot.layers):
            layer = state_cache.layer(layer_idx)
            # compressed_kv holds only the valid block prefix; copy it into the
            # matching leading slots of the (possibly wider) target buffer.
            n_blocks = layer_snap.compressed_kv.shape[1]
            layer.swa_kv[row : row + 1].copy_(layer_snap.swa_kv)
            layer.compressed_kv[row : row + 1, :n_blocks].copy_(layer_snap.compressed_kv)
            layer.cmp_kv_state[row : row + 1].copy_(layer_snap.cmp_kv_state)
            layer.cmp_score_state[row : row + 1].copy_(layer_snap.cmp_score_state)
            layer.n_compressed_blocks = layer_snap.n_compressed_blocks
            layer.swa_count = layer_snap.swa_count
            if layer.indexer is not None and layer_snap.indexer_compressed_kv is not None:
                # The indexer sub-state fields are populated together in
                # snapshot_from_cache, so they are all set when compressed_kv is.
                assert layer_snap.indexer_cmp_kv_state is not None
                assert layer_snap.indexer_cmp_score_state is not None
                n_idx = layer_snap.indexer_compressed_kv.shape[1]
                layer.indexer.compressed_kv[row : row + 1, :n_idx].copy_(
                    layer_snap.indexer_compressed_kv
                )
                layer.indexer.cmp_kv_state[row : row + 1].copy_(layer_snap.indexer_cmp_kv_state)
                layer.indexer.cmp_score_state[row : row + 1].copy_(
                    layer_snap.indexer_cmp_score_state
                )
                layer.indexer.n_compressed_blocks = layer_snap.indexer_n_compressed_blocks
        state_cache.start_pos = snapshot.position
