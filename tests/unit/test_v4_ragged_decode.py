"""Self-consistency: ragged HCA decode == per-request scalar decode.

A ragged decode step takes B requests sitting at DIFFERENT positions and serves
them in one batched forward. This pins that new path against the scalar
`forward_decode`, which is itself bit-parity validated against the DeepSeek-V4
reference (`test_v4_hca_decode_parity.py`). No reference is needed here: if one
ragged step equals each request stepped alone, the ragged plumbing (per-row SWA
scatter, per-row compressor flush, per-row gather indices) is correct.

The batch deliberately mixes the interesting cases in one step: a row whose step
closes a block (flush) next to rows that do not, a partial sliding window next
to full (wrapped) windows, and rows with different amounts of compressed history.
"""

from __future__ import annotations

import torch
from torch.nn.functional import cosine_similarity

from mini_infer.cache.state_cache import StateCache, StateLayerSpec
from mini_infer.models.blocks import HCAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.blocks.swa import SWAAttention

_HIDDEN = 64
_KV_HEAD_DIM = 32
_ROPE_HEAD_DIM = 8
_N_WIN = 8
_M = 8
_MAX_SEQ = 64


def _build_block() -> HCAAttention:
    return HCAAttention(
        hidden_size=_HIDDEN,
        num_heads=4,
        q_lora_rank=32,
        kv_head_dim=_KV_HEAD_DIM,
        rope_head_dim=_ROPE_HEAD_DIM,
        num_groups=2,
        o_lora_rank=32,
        window_size=_N_WIN,
        compression_ratio=_M,
        rms_norm_eps=1e-6,
    ).eval()


def _make_cache(batch_size: int) -> StateCache:
    spec = StateLayerSpec(
        kv_head_dim=_KV_HEAD_DIM,
        compression_ratio=_M,
        n_win=_N_WIN,
        max_n_compressed=_MAX_SEQ // _M,
    )
    return StateCache([spec], batch_size=batch_size)


def _token_pe(rotary: RotaryEmbedding, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor(positions, dtype=torch.long).unsqueeze(1)  # (B, 1)
    return rotary(torch.zeros(len(positions), 1), pos)


def _block_pe(rotary: RotaryEmbedding, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return _token_pe(rotary, [(p // _M) * _M for p in positions])


def test_hca_forward_decode_ragged_matches_scalar_per_request() -> None:
    torch.manual_seed(0)
    block = _build_block()
    rotary = RotaryEmbedding(head_dim=_ROPE_HEAD_DIM, base=10000.0)

    # pos 3: partial window, no compressed blocks; pos 7: closes a block (flush)
    # AND window just wrapped; pos 17: full window, two compressed blocks, no flush.
    warmups = [3, 7, 17]
    batch_size = len(warmups)

    snapshots: list[dict[str, torch.Tensor]] = []
    test_hiddens: list[torch.Tensor] = []
    scalar_outs: list[torch.Tensor] = []

    for warmup in warmups:
        cache = _make_cache(1)
        # Warm this request to `warmup` by decoding positions 0..warmup-1 alone.
        for position in range(warmup):
            hidden = torch.randn(1, 1, _HIDDEN)
            block_pe = _block_pe(rotary, [position]) if (position + 1) % _M == 0 else None
            with torch.no_grad():
                block.forward_decode(
                    hidden,
                    start_pos=position,
                    state_cache=cache,
                    layer_idx=0,
                    token_position_embeddings=_token_pe(rotary, [position]),
                    block_position_embeddings=block_pe,
                )
        # Snapshot the warmed state BEFORE the scalar test step mutates it.
        layer = cache.layer(0)
        snapshots.append(
            {
                "swa_kv": layer.swa_kv.clone(),
                "compressed_kv": layer.compressed_kv.clone(),
                "cmp_kv_state": layer.cmp_kv_state.clone(),
                "cmp_score_state": layer.cmp_score_state.clone(),
            }
        )
        # Scalar test step at `warmup`.
        hidden_test = torch.randn(1, 1, _HIDDEN)
        test_hiddens.append(hidden_test)
        block_pe = _block_pe(rotary, [warmup]) if (warmup + 1) % _M == 0 else None
        with torch.no_grad():
            scalar_outs.append(
                block.forward_decode(
                    hidden_test,
                    start_pos=warmup,
                    state_cache=cache,
                    layer_idx=0,
                    token_position_embeddings=_token_pe(rotary, [warmup]),
                    block_position_embeddings=block_pe,
                )
            )

    # Assemble a batched cache whose row b is request b's warmed snapshot.
    batched = _make_cache(batch_size)
    batched_layer = batched.layer(0)
    for row, snapshot in enumerate(snapshots):
        batched_layer.swa_kv[row] = snapshot["swa_kv"][0]
        batched_layer.compressed_kv[row] = snapshot["compressed_kv"][0]
        batched_layer.cmp_kv_state[row] = snapshot["cmp_kv_state"][0]
        batched_layer.cmp_score_state[row] = snapshot["cmp_score_state"][0]

    positions = torch.tensor(warmups, dtype=torch.long)
    hidden_ragged = torch.cat(test_hiddens, dim=0)  # (B, 1, H)
    with torch.no_grad():
        ragged_out = block.forward_decode_ragged(
            hidden_ragged,
            positions=positions,
            state_cache=batched,
            layer_idx=0,
            token_position_embeddings=_token_pe(rotary, warmups),
            block_position_embeddings=_block_pe(rotary, warmups),
        )

    assert ragged_out.shape == (batch_size, 1, _HIDDEN)
    for row, warmup in enumerate(warmups):
        ours = ragged_out[row : row + 1]
        theirs = scalar_outs[row]
        cosine = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
        max_diff = (ours - theirs).abs().max().item()
        assert cosine > 0.99999, (
            f"request {row} (pos {warmup}): cos={cosine:.7f}, max={max_diff:.2e}"
        )
        torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-5)


def _make_swa_cache(batch_size: int) -> StateCache:
    spec = StateLayerSpec(
        kv_head_dim=_KV_HEAD_DIM,
        compression_ratio=0,  # pure SWA: window only, no compressor / compressed history
        n_win=_N_WIN,
        max_n_compressed=1,
    )
    return StateCache([spec], batch_size=batch_size)


def test_swa_forward_decode_ragged_matches_scalar_per_request() -> None:
    """Ragged SWA decode (window-only) equals per-request scalar decode.

    SWA has no compressor, so the only ragged state is each row's circular
    window written at `pos[b] % n_win`. Positions mix a partial window, the
    just-wrapped boundary, and a full window.
    """
    torch.manual_seed(1)
    block = SWAAttention(
        hidden_size=_HIDDEN,
        num_heads=4,
        q_lora_rank=32,
        kv_head_dim=_KV_HEAD_DIM,
        rope_head_dim=_ROPE_HEAD_DIM,
        num_groups=2,
        o_lora_rank=32,
        window_size=_N_WIN,
        rms_norm_eps=1e-6,
    ).eval()
    rotary = RotaryEmbedding(head_dim=_ROPE_HEAD_DIM, base=10000.0)
    warmups = [2, 7, 15]
    batch_size = len(warmups)

    swa_snapshots: list[torch.Tensor] = []
    test_hiddens: list[torch.Tensor] = []
    scalar_outs: list[torch.Tensor] = []

    for warmup in warmups:
        cache = _make_swa_cache(1)
        for position in range(warmup):
            with torch.no_grad():
                block.forward_decode(
                    torch.randn(1, 1, _HIDDEN),
                    start_pos=position,
                    state_cache=cache,
                    layer_idx=0,
                    token_position_embeddings=_token_pe(rotary, [position]),
                )
        swa_snapshots.append(cache.layer(0).swa_kv.clone())
        hidden_test = torch.randn(1, 1, _HIDDEN)
        test_hiddens.append(hidden_test)
        with torch.no_grad():
            scalar_outs.append(
                block.forward_decode(
                    hidden_test,
                    start_pos=warmup,
                    state_cache=cache,
                    layer_idx=0,
                    token_position_embeddings=_token_pe(rotary, [warmup]),
                )
            )

    batched = _make_swa_cache(batch_size)
    for row, swa_kv in enumerate(swa_snapshots):
        batched.layer(0).swa_kv[row] = swa_kv[0]

    positions = torch.tensor(warmups, dtype=torch.long)
    with torch.no_grad():
        ragged_out = block.forward_decode_ragged(
            torch.cat(test_hiddens, dim=0),
            positions=positions,
            state_cache=batched,
            layer_idx=0,
            token_position_embeddings=_token_pe(rotary, warmups),
        )

    assert ragged_out.shape == (batch_size, 1, _HIDDEN)
    for row in range(batch_size):
        torch.testing.assert_close(
            ragged_out[row : row + 1], scalar_outs[row], rtol=1e-4, atol=1e-5
        )
