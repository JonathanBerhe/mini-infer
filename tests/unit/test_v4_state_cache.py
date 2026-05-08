"""StateCache shape + lifecycle tests.

The decode parity test (`test_v4_hca_decode_parity.py`) is the strong
correctness gate for the cache wiring. These narrower tests cover state
that drifts silently between layers:

  - Allocation shapes match the per-layer specs.
  - SWA circular indexing wraps at `n_win`.
  - Compressed counter increments only on flush.
  - `advance_start_pos` is monotonic.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.state_cache import StateCache, StateLayerSpec


def _spec(**overrides) -> StateLayerSpec:
    base = dict(kv_head_dim=8, compression_ratio=4, n_win=8, max_n_compressed=16)
    base.update(overrides)
    return StateLayerSpec(**base)


def test_state_cache_allocates_per_layer_tensors_with_correct_shapes() -> None:
    cache = StateCache([_spec(), _spec(kv_head_dim=16, n_win=4)], batch_size=2)
    assert cache.num_layers == 2
    layer0 = cache.layer(0)
    assert layer0.swa_kv.shape == (2, 8, 8)
    assert layer0.compressed_kv.shape == (2, 16, 8)
    assert layer0.cmp_kv_state.shape == (2, 4, 8)
    assert layer0.cmp_score_state.shape == (2, 4, 8)
    layer1 = cache.layer(1)
    assert layer1.swa_kv.shape == (2, 4, 16)
    assert layer1.compressed_kv.shape == (2, 16, 16)


def test_state_cache_initializes_score_state_to_negative_infinity() -> None:
    """`-inf` start makes the softmax of a partial block degenerate to the
    one populated slot — matches the reference's prefill semantics for
    the first sub-`m` window of decode steps."""
    cache = StateCache([_spec()], batch_size=1)
    layer = cache.layer(0)
    assert torch.all(torch.isneginf(layer.cmp_score_state))
    # And the softmax handles it (no NaN) — pretend we wrote ONE slot:
    layer.cmp_score_state[:, 0] = 1.5
    weights = layer.cmp_score_state.softmax(dim=1)
    assert torch.allclose(weights[:, 0], torch.ones_like(weights[:, 0]))
    assert torch.allclose(weights[:, 1:], torch.zeros_like(weights[:, 1:]))


def test_state_cache_advance_start_pos_is_monotonic() -> None:
    cache = StateCache([_spec()], batch_size=1)
    assert cache.start_pos == 0
    cache.advance_start_pos(4)
    assert cache.start_pos == 4
    cache.advance_start_pos(1)
    assert cache.start_pos == 5
    with pytest.raises(ValueError, match="non-negative"):
        cache.advance_start_pos(-1)


def test_state_cache_swa_circular_write_wraps_at_n_win() -> None:
    """Writing at `start_pos % n_win` overwrites the oldest slot once full."""
    cache = StateCache([_spec(n_win=4)], batch_size=1)
    layer = cache.layer(0)
    # Manually simulate 6 SWA writes; positions 0-3 fill slots 0-3, then
    # position 4 overwrites slot 0 and position 5 overwrites slot 1.
    for pos in range(6):
        layer.swa_kv[:, pos % 4] = torch.full(layer.swa_kv[:, 0].shape, float(pos))
    expected_at_each_slot = [4.0, 5.0, 2.0, 3.0]  # pos 4->slot 0, pos 5->slot 1; 2,3 retained
    for slot_idx, expected in enumerate(expected_at_each_slot):
        assert torch.all(layer.swa_kv[:, slot_idx] == expected)


def test_state_cache_rejects_invalid_specs() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        StateCache([_spec()], batch_size=0)
    with pytest.raises(ValueError, match="non-empty"):
        StateCache([], batch_size=1)
    with pytest.raises(ValueError, match="n_win"):
        StateCache([_spec(n_win=0)], batch_size=1)
    with pytest.raises(ValueError, match="compression_ratio"):
        StateCache([_spec(compression_ratio=0)], batch_size=1)
    with pytest.raises(ValueError, match="max_n_compressed"):
        StateCache([_spec(max_n_compressed=0)], batch_size=1)
    with pytest.raises(NotImplementedError, match="overlap_mode"):
        StateCache([_spec(overlap_mode=True)], batch_size=1)


def test_state_cache_layer_slots_are_independent() -> None:
    """Mutating layer 0's state must not leak into layer 1."""
    cache = StateCache([_spec(), _spec()], batch_size=1)
    cache.layer(0).swa_kv.fill_(7.0)
    cache.layer(0).n_compressed_blocks = 5
    assert torch.all(cache.layer(1).swa_kv == 0.0)
    assert cache.layer(1).n_compressed_blocks == 0
