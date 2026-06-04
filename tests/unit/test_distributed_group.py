"""Tests for `distributed.group` accessors, focused on `replica_scope`.

`replica_scope()` is the fix for the PD deadlock: a rank that belongs to a
multi-rank process group (for the prefill/decode KV handoff) must still build
+ run its model as a single un-sharded replica, or the model's TP-aware layers
insert `all_reduce` collectives that deadlock against the handoff. These tests
simulate a live multi-rank PG (without spawning one) and assert the scope
forces `(world_size, rank) == (1, 0)` for its duration only.

Fast + model-free, so they run in the default CI lane; the end-to-end proof
lives in `test_workers_mp.py` (slow / requires_model).
"""

from __future__ import annotations

import pytest

import mini_infer.distributed.group as group


def _simulate_pg(monkeypatch: pytest.MonkeyPatch, *, world_size: int, rank: int) -> None:
    """Make the group module report a live `world_size`-rank PG at `rank`."""
    monkeypatch.setattr(group, "is_initialized", lambda: True)
    monkeypatch.setattr(group.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(group.dist, "get_rank", lambda: rank)


def test_replica_scope_forces_single_replica(monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_pg(monkeypatch, world_size=2, rank=1)

    # Outside the scope, the accessors reflect the (simulated) PG.
    assert group.get_world_size() == 2
    assert group.get_rank() == 1

    # Inside, the rank looks like a lone replica.
    with group.replica_scope():
        assert group.get_world_size() == 1
        assert group.get_rank() == 0

    # Restored afterward.
    assert group.get_world_size() == 2
    assert group.get_rank() == 1


def test_replica_scope_nests_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_pg(monkeypatch, world_size=4, rank=3)

    with group.replica_scope():
        assert group.get_world_size() == 1
        with group.replica_scope():
            assert group.get_world_size() == 1
        # Inner exit must NOT restore to the PG value while still nested.
        assert group.get_world_size() == 1
    assert group.get_world_size() == 4


def test_replica_scope_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_pg(monkeypatch, world_size=2, rank=0)

    with pytest.raises(RuntimeError, match="boom"), group.replica_scope():
        assert group.get_world_size() == 1
        raise RuntimeError("boom")

    # The flag is restored even though the scope exited via an exception.
    assert group.get_world_size() == 2


def test_no_pg_is_unaffected_by_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no PG, accessors already return (1, 0); the scope is a no-op view."""
    monkeypatch.setattr(group, "is_initialized", lambda: False)
    assert group.get_world_size() == 1
    assert group.get_rank() == 0
    with group.replica_scope():
        assert group.get_world_size() == 1
        assert group.get_rank() == 0
    assert group.get_world_size() == 1
