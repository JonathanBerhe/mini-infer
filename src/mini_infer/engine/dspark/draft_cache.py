"""The drafter's own small, unpaged K/V cache.

DeepSpec's inference loop uses a plain HF `DynamicCache` for the drafter and
crops it back every round (`past_key_values_draft.crop(start)`), discarding
that round's mask-token K/V unconditionally while the target-context
injections it received via `update()` accumulate permanently. That accumulated
piece is what "grows with sequence length" here, at the same linear rate any
ordinary KV cache grows (see ADR-027); it is NOT a re-derivation over a
widening prompt window, which is why this doesn't need `PagedKVCache`'s block
paging. Five small layers, batch size 1, is not worth paging.

`truncate_to` mirrors `PagedKVCache.truncate_to`'s contract (idempotent at the
current length, raises on growth) so the two caches read the same way at a
call site, even though this one is plain tensor slicing with no block table.
"""

from __future__ import annotations

import torch


class DSparkDraftCache:
    """Per-layer `(keys, values)` accumulator for the DSpark drafter, batch size 1."""

    def __init__(self, num_layers: int) -> None:
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `(new_keys, new_values)` (`(1, num_kv_heads, new_len, head_dim)`) and
        return the full accumulated `(keys, values)` for this layer.

        Mirrors HF `DynamicCache.update`: the cache stores the concatenation and
        hands the caller the whole thing back, since attention needs to see the
        full K/V history, not just what was just appended.
        """
        cached_keys, cached_values = self._keys[layer_idx], self._values[layer_idx]
        if cached_keys is None:
            full_keys, full_values = new_keys, new_values
        else:
            assert cached_values is not None
            full_keys = torch.cat([cached_keys, new_keys], dim=2)
            full_values = torch.cat([cached_values, new_values], dim=2)
        self._keys[layer_idx] = full_keys
        self._values[layer_idx] = full_values
        return full_keys, full_values

    def get_seq_length(self) -> int:
        """Current cached length. Uniform across layers (every layer appends in lockstep)."""
        first = self._keys[0]
        return 0 if first is None else int(first.shape[2])

    def truncate_to(self, new_seq_len: int) -> None:
        """Roll every layer back to `new_seq_len` positions. Idempotent at the current length.

        Matches `deepspec`'s `DynamicLayer.crop`: keeps the prefix `[:new_seq_len]`,
        drops the rest. Unlike `PagedKVCache.truncate_to`, there is no published-
        prefix-cache guard: the drafter's cache is never shared across requests.
        """
        current = self.get_seq_length()
        if new_seq_len < 0:
            raise ValueError(f"new_seq_len={new_seq_len} must be non-negative")
        if new_seq_len > current:
            raise ValueError(
                f"truncate_to(new_seq_len={new_seq_len}) > current {current}; "
                "truncation only shrinks"
            )
        if new_seq_len == current:
            return
        for layer_idx, keys in enumerate(self._keys):
            if keys is None:
                continue
            self._keys[layer_idx] = keys[..., :new_seq_len, :]
            values = self._values[layer_idx]
            assert values is not None
            self._values[layer_idx] = values[..., :new_seq_len, :]
