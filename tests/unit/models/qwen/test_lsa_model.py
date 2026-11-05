"""Latent Sparse Attention inside the Qwen hybrid: the model-level contract.

A tiny random Qwen hybrid with small LSA sizes makes sparsity start after a
few dozen tokens, so every path runs on CPU:

  1. While every earlier key is selected, LSA output equals dense output.
  2. Past that, it equals dense attention masked to the selected keys.
  3. A prefill split into chunks equals the same prefill in one go.
  4. Decode steps that close key blocks equal the prefill at the same
     positions, alone and in a ragged batch.

The selection itself is pinned to the authors' reference in
`test_lsa_parity.py`, so the oracle in (2) reuses it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from mini_infer.models.qwen import qwen3_5
from mini_infer.models.qwen.lsa import LsaConfig
from mini_infer.models.qwen.qwen3_5 import (
    FULL_ATTENTION,
    Qwen35Config,
    Qwen35ForConditionalGeneration,
)

TEXT = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "max_position_embeddings": 512,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
}
# 2 KV heads of 16 - 4 rotary dims each.
INPUT_DIM = 24
# Blocks of 16 keys, 4 buckets each. A query reads at most 32 bucket keys,
# 16 window keys and 2 sinks. With 4 buckets picked, sparsity starts at
# position 32, where a query first sees 2 closed blocks.
SPARSE = LsaConfig(
    window_size=16,
    sink_size=2,
    top_k=32,
    top_p=4,
    index_dim=8,
    pool_size=2,
    bucket_size=4,
    bucket_cap=8,
)
MAX_SEQ_LEN = 256
ATOL = 1e-5


def _model(lsa: LsaConfig | None) -> Qwen35ForConditionalGeneration:
    """The same random weights every call, with LSA on when `lsa` is given."""
    torch.manual_seed(0)
    cfg = Qwen35Config.from_hf(SimpleNamespace(**TEXT))
    model = Qwen35ForConditionalGeneration(cfg).eval()
    if lsa is not None:
        generator = torch.Generator().manual_seed(1)
        full = [i for i, kind in enumerate(cfg.layer_types) if kind == FULL_ATTENTION]
        maps = {
            i: (
                torch.randn(INPUT_DIM, lsa.index_dim, generator=generator),
                torch.randn(INPUT_DIM, lsa.index_dim, generator=generator),
            )
            for i in full
        }
        model.enable_lsa(maps, lsa)
    return model


def _ids(length: int, seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TEXT["vocab_size"], (1, length), generator=generator)


@torch.inference_mode()
def _prefill(model, ids: torch.Tensor, chunks: tuple[int, ...] = ()) -> tuple[torch.Tensor, object]:
    """Prefill `ids`, split at the `chunks` offsets. Returns logits and the cache."""
    cache = model.build_state_cache(max_seq_len=MAX_SEQ_LEN, batch_size=ids.shape[0])
    logits = []
    for start, stop in zip((0, *chunks), (*chunks, ids.shape[1]), strict=True):
        logits.append(model.forward_prefill_with_cache(ids[:, start:stop], state_cache=cache))
        cache.advance_start_pos(stop - start)
    return torch.cat(logits, dim=1), cache


@torch.inference_mode()
def _decode(model, cache, ids: torch.Tensor, start: int) -> torch.Tensor:
    """Feed `ids[:, start:]` one token at a time after a prefill of `start` tokens."""
    logits = []
    for pos in range(start, ids.shape[1]):
        logits.append(
            model.forward_decode_with_cache(ids[:, pos : pos + 1], start_pos=pos, state_cache=cache)
        )
        cache.advance_start_pos(1)
    return torch.cat(logits, dim=1)


def _masked_dense_attention(query, k_buffer, v_buffer, selected, scale):
    """Oracle for `attend_selected`: dense attention over the whole buffer,
    masked to each query's selected positions."""
    batch, span, heads, _ = query.shape
    group = heads // k_buffer.shape[2]
    buffer_len = k_buffer.shape[1]
    # Padding goes to a spare last column, dropped after the scatter.
    allowed = torch.zeros(batch, span, buffer_len + 1, dtype=torch.bool)
    slots = torch.where(selected >= 0, selected.long(), buffer_len)
    allowed = allowed.scatter(-1, slots, True)[..., :buffer_len]
    keys = k_buffer.to(query.dtype).repeat_interleave(group, dim=2).transpose(1, 2)
    values = v_buffer.to(query.dtype).repeat_interleave(group, dim=2).transpose(1, 2)
    out = functional.scaled_dot_product_attention(
        query.transpose(1, 2), keys, values, attn_mask=allowed[:, None], scale=scale
    )
    return out.transpose(1, 2)


def test_lsa_equals_dense_while_every_earlier_key_is_selected():
    # With 16 buckets picked, a query up to position 49 still sees at most 12
    # buckets, so it reads every earlier key.
    lsa = LsaConfig(**{**SPARSE.__dict__, "top_p": 16})
    length = lsa.list_width
    ids = _ids(length)
    dense, dense_cache = _prefill(_model(None), ids[:, :40])
    sparse, sparse_cache = _prefill(_model(lsa), ids[:, :40])
    torch.testing.assert_close(sparse, dense, atol=ATOL, rtol=0)
    dense_steps = _decode(_model(None), dense_cache, ids, 40)
    sparse_steps = _decode(_model(lsa), sparse_cache, ids, 40)
    torch.testing.assert_close(sparse_steps, dense_steps, atol=ATOL, rtol=0)


def test_lsa_equals_dense_attention_masked_to_the_selected_keys(monkeypatch):
    ids = _ids(150)
    model = _model(SPARSE)
    # Sub-chunks of a few queries, so the chunk loop runs many times.
    monkeypatch.setattr(qwen3_5, "_LSA_CHUNK_BYTES", 20_000)
    sparse, _ = _prefill(model, ids)
    monkeypatch.setattr(qwen3_5, "attend_selected", _masked_dense_attention)
    oracle, _ = _prefill(model, ids)
    torch.testing.assert_close(sparse, oracle, atol=ATOL, rtol=0)

    # The comparison is not vacuous: past position 32 LSA reads fewer keys,
    # so it no longer equals dense attention.
    dense, _ = _prefill(_model(None), ids)
    torch.testing.assert_close(sparse[:, :32], dense[:, :32], atol=ATOL, rtol=0)
    assert float((sparse[:, 32:] - dense[:, 32:]).abs().max()) > 1e-3


def test_chunked_prefill_equals_one_prefill():
    ids = _ids(150)
    model = _model(SPARSE)
    whole, whole_cache = _prefill(model, ids)
    # Chunk edges inside blocks and on a block edge (48).
    chunked, chunked_cache = _prefill(model, ids, chunks=(7, 48, 101))
    torch.testing.assert_close(chunked, whole, atol=1e-4, rtol=0)
    for layer_idx, kind in enumerate(model.cfg.layer_types):
        if kind != FULL_ATTENTION:
            continue
        whole_state, chunked_state = whole_cache.layer(layer_idx), chunked_cache.layer(layer_idx)
        assert torch.equal(chunked_state.lsa_members, whole_state.lsa_members)
        torch.testing.assert_close(chunked_state.lsa_centroids, whole_state.lsa_centroids)


def test_decode_that_closes_blocks_equals_prefill_at_the_same_positions():
    ids = _ids(100)
    model = _model(SPARSE)
    whole, _ = _prefill(model, ids)
    # Decode from 40 to 99 closes the blocks ending at 47, 63, 79 and 95.
    _, cache = _prefill(model, ids[:, :40])
    steps = _decode(model, cache, ids, 40)
    torch.testing.assert_close(steps, whole[:, 40:], atol=1e-4, rtol=0)


@torch.inference_mode()
def test_ragged_decode_rows_equal_their_own_prefill():
    """Two requests at different positions in one batch, moved in by row copy."""
    model = _model(SPARSE)
    lengths = (45, 70)
    requests = [_ids(length + 20, seed=10 + i) for i, length in enumerate(lengths)]
    batch = model.build_state_cache(max_seq_len=MAX_SEQ_LEN, batch_size=2)
    for row, (ids, length) in enumerate(zip(requests, lengths, strict=True)):
        _, single = _prefill(model, ids[:, :length])
        batch.copy_row_from(single, src_row=0, dst_row=row)

    steps: list[list[torch.Tensor]] = [[], []]
    for offset in range(20):
        positions = torch.tensor([length + offset for length in lengths])
        tokens = torch.stack([ids[0, pos] for ids, pos in zip(requests, positions, strict=True)])
        logits = model.forward_decode_with_cache_ragged(
            tokens.view(2, 1), positions=positions, state_cache=batch
        )
        for row in range(2):
            steps[row].append(logits[row, 0])

    for row, (ids, length) in enumerate(zip(requests, lengths, strict=True)):
        whole, _ = _prefill(model, ids)
        torch.testing.assert_close(torch.stack(steps[row]), whole[0, length:], atol=1e-4, rtol=0)


def test_disable_lsa_restores_dense_attention():
    ids = _ids(100)
    dense, _ = _prefill(_model(None), ids)
    model = _model(SPARSE)
    _, sparse_cache = _prefill(model, ids)
    model.disable_lsa()
    assert model.lsa is None
    assert not any("lsa_projection" in name for name, _ in model.named_modules())
    restored, _ = _prefill(model, ids)
    torch.testing.assert_close(restored, dense, atol=ATOL, rtol=0)
    with pytest.raises(ValueError, match="rebuild the cache"):
        model.forward_prefill_with_cache(ids[:, :4], state_cache=sparse_cache)


def test_enable_lsa_refuses_missing_layers_and_wrong_shapes():
    model = _model(None)
    good = torch.zeros(INPUT_DIM, SPARSE.index_dim)
    with pytest.raises(ValueError, match="missing full-attention layers"):
        model.enable_lsa({3: (good, good)}, SPARSE)
    with pytest.raises(ValueError, match="key map"):
        model.enable_lsa({3: (good, good), 7: (good, torch.zeros(INPUT_DIM, 4))}, SPARSE)
    assert model.lsa is None


def test_lsa_refuses_a_cache_built_without_its_buckets():
    stale = _model(None).build_state_cache(max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="rebuild the cache"):
        _model(SPARSE).forward_prefill_with_cache(_ids(4), state_cache=stale)


@torch.inference_mode()
def test_static_steps_with_the_after_step_hook_equal_eager_decode():
    """What `DecodeCudaGraph` runs: `decode_step_static`, which cannot
    cluster, then `after_decode_step` with the same positions. Two rows at
    different positions close blocks on different steps."""
    model = _model(SPARSE)
    lengths = (45, 62)
    requests = [_ids(length + 30, seed=20 + i) for i, length in enumerate(lengths)]
    caches = {name: model.build_state_cache(max_seq_len=MAX_SEQ_LEN, batch_size=2) for name in "ab"}
    for row, (ids, length) in enumerate(zip(requests, lengths, strict=True)):
        _, single = _prefill(model, ids[:, :length])
        for cache in caches.values():
            cache.copy_row_from(single, src_row=0, dst_row=row)

    for offset in range(30):
        positions = torch.tensor([length + offset for length in lengths])
        tokens = torch.stack(
            [ids[0, pos] for ids, pos in zip(requests, positions, strict=True)]
        ).view(2, 1)
        eager = model.forward_decode_with_cache_ragged(
            tokens, positions=positions, state_cache=caches["a"]
        )
        static = model.decode_step_static(tokens, positions, caches["b"])
        model.after_decode_step(positions, caches["b"])
        torch.testing.assert_close(static, eager, atol=ATOL, rtol=0)

    for layer_idx, kind in enumerate(model.cfg.layer_types):
        if kind == FULL_ATTENTION:
            eager_state, static_state = caches["a"].layer(layer_idx), caches["b"].layer(layer_idx)
            assert torch.equal(static_state.lsa_members, eager_state.lsa_members)


def test_lsa_maps_follow_the_model_but_stay_out_of_its_state_dict():
    model = _model(SPARSE)
    assert not any("lsa" in name for name in model.state_dict())
    model = model.to(torch.float64)
    projection = model.model.layers[3].self_attn.lsa_projection
    assert projection is not None and projection.key_map.dtype == torch.float64
    # The cast is exact, so the index keys do not change.
    keys = torch.randn(5, INPUT_DIM)
    torch.testing.assert_close(
        projection.index_keys(keys),
        _model(SPARSE).model.layers[3].self_attn.lsa_projection.index_keys(keys),
    )
