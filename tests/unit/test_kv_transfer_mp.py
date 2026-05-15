"""Focused multi-process test for the KV-transfer wire protocol.

Bypasses model loading: ranks construct a fake `KVHandoff` with known
shapes and send/recv it. If this passes but the full PD multi-process
test hangs, the issue is in the model-load / prefill side, not the
transport. Fast (~2 s) so it's a useful debugging stop-gap.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.workers.kv_handoff import KVHandoff
from mini_infer.workers.kv_transfer import recv_handoff, send_handoff
from tests.unit._distributed_test_utils import (
    is_multi_process_available,
    run_multi_process,
)


def _fake_pool_for_recv(num_layers: int, num_kv_heads: int, head_dim: int) -> object:
    """Tiny stand-in for `BlockPool` exposing only what `recv_handoff` reads.

    `recv_handoff` calls:
      - `pool.num_layers`
      - `pool.stream_names(layer_idx)`
      - `pool.stream_spec(layer_idx, name)` -> object with `.num_kv_heads`, `.head_dim`
      - `pool.storage_for_stream(layer_idx, name)` -> tensor with `.device`
      - `pool.dtype`
    """

    class _Spec:
        def __init__(self, n: int, d: int) -> None:
            self.num_kv_heads = n
            self.head_dim = d

    class _Pool:
        dtype = torch.float32

        def __init__(self) -> None:
            self._spec = _Spec(num_kv_heads, head_dim)
            self._dummy_storage = torch.empty(1, device="cpu")
            self._num_layers = num_layers

        @property
        def num_layers(self) -> int:
            return self._num_layers

        def stream_names(self, layer_idx: int) -> list[str]:
            return ["k", "v"]

        def stream_spec(self, layer_idx: int, stream_name: str) -> object:
            return self._spec

        def storage_for_stream(self, layer_idx: int, stream_name: str) -> torch.Tensor:
            return self._dummy_storage

    return _Pool()


def _kv_transfer_target(
    rank: int,
    world_size: int,
    *,
    num_layers: int,
    prefill_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> list[int] | None:
    """Per-rank entry. Rank 0 sends a fixed handoff; rank 1 receives and checks."""
    if rank == 0:
        # Construct deterministic KV: stream tensors of shape (prefill_len, h, d)
        # filled with `layer_idx * 100 + stream_idx * 10 + position`.
        kv_streams_per_layer: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            layer: dict[str, torch.Tensor] = {}
            for stream_idx, stream_name in enumerate(("k", "v")):
                t = torch.full(
                    (prefill_len, num_kv_heads, head_dim),
                    fill_value=float(layer_idx * 100 + stream_idx * 10),
                    dtype=torch.float32,
                )
                # Tag each position with its index for round-trip verification.
                for pos in range(prefill_len):
                    t[pos] += pos
                layer[stream_name] = t
            kv_streams_per_layer.append(layer)
        handoff = KVHandoff(
            request_id="dbg",
            kv_streams_per_layer=kv_streams_per_layer,
            prefill_len=prefill_len,
            first_sampled_token_id=42,
            sampling_params=SamplingParams(temperature=0.0),
            max_tokens=8,
            eos_token_id=0,
        )
        send_handoff(handoff, dst_rank=1)
        return None

    if rank == 1:
        pool = _fake_pool_for_recv(num_layers, num_kv_heads, head_dim)
        handoff = recv_handoff(src_rank=0, pool=pool)
        # Verify shape + content.
        assert handoff.prefill_len == prefill_len
        assert handoff.first_sampled_token_id == 42
        assert handoff.num_layers == num_layers
        for layer_idx in range(num_layers):
            for stream_idx, stream_name in enumerate(("k", "v")):
                t = handoff.kv_streams_per_layer[layer_idx][stream_name]
                assert t.shape == (prefill_len, num_kv_heads, head_dim)
                for pos in range(prefill_len):
                    expected = float(layer_idx * 100 + stream_idx * 10 + pos)
                    actual = t[pos, 0, 0].item()
                    assert actual == expected, (
                        f"layer={layer_idx} stream={stream_name} pos={pos}: "
                        f"got {actual}, expected {expected}"
                    )
        return [1]  # marker for "verified ok"

    raise ValueError(f"unexpected rank {rank}")


def test_kv_transfer_round_trip() -> None:
    """End-to-end protocol test: rank 0 sends, rank 1 receives + verifies."""
    if not is_multi_process_available():
        pytest.skip("multi-process / gloo unavailable in this environment")

    results = run_multi_process(
        2,
        _kv_transfer_target,
        num_layers=4,
        prefill_len=5,
        num_kv_heads=2,
        head_dim=3,
        timeout_sec=60.0,
    )
    # rank 0 returns None; rank 1 returns [1] when verification passes.
    assert results[1] == [1]
