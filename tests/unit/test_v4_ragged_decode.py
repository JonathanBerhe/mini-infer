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

import pytest
import torch
from torch.nn.functional import cosine_similarity

from mini_infer.cache.state_cache import StateCache, StateLayerSpec
from mini_infer.models.blocks import HCAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.blocks.swa import SWAAttention
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)

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


# ---------- model-level ragged decode (full hybrid: SWA + CSA + HCA) ----------


def _model_config(
    *, use_moe_ffn: bool = False, use_hyper_connections: bool = False
) -> DeepseekV4Config:
    """Full hybrid stack: layer ratios (0, 4, 8, 4) = SWA, CSA, HCA, CSA.

    Exercises every V4 attention mode in the ragged decode, including the CSA
    LightningIndexer (ratio 4) with its per-row masked top-k.
    """
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 4, 8, 4),
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_moe_ffn=use_moe_ffn,
        moe_intermediate_size=64 if use_moe_ffn else 0,
        num_routed_experts=4 if use_moe_ffn else 0,
        num_activated_experts=2 if use_moe_ffn else 0,
        num_hash_routed_layers=2 if use_moe_ffn else 0,
        moe_score_func="softmax",
        moe_route_scale=1.0,
        n_shared_experts=1 if use_moe_ffn else 0,
        use_hyper_connections=use_hyper_connections,
        hc_mult=4 if use_hyper_connections else 0,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


def _clone_all_layers(cache: StateCache) -> list[dict[str, torch.Tensor]]:
    """Clone every per-layer buffer (incl. the CSA indexer sub-state) so a
    prefilled B=1 cache can be reassembled into one row of a batched cache."""
    snapshot: list[dict[str, torch.Tensor]] = []
    for i in range(cache.num_layers):
        layer = cache.layer(i)
        entry = {
            "swa_kv": layer.swa_kv.clone(),
            "compressed_kv": layer.compressed_kv.clone(),
            "cmp_kv_state": layer.cmp_kv_state.clone(),
            "cmp_score_state": layer.cmp_score_state.clone(),
        }
        if layer.indexer is not None:
            entry["indexer.compressed_kv"] = layer.indexer.compressed_kv.clone()
            entry["indexer.cmp_kv_state"] = layer.indexer.cmp_kv_state.clone()
            entry["indexer.cmp_score_state"] = layer.indexer.cmp_score_state.clone()
        snapshot.append(entry)
    return snapshot


@pytest.mark.parametrize(
    "use_moe_ffn,use_hyper_connections",
    [(False, False), (True, False), (False, True)],
    ids=["vanilla", "moe", "hyper_connections"],
)
def test_model_forward_decode_ragged_matches_per_request_scalar(
    use_moe_ffn: bool, use_hyper_connections: bool
) -> None:
    """Ragged batched decode == per-request scalar decode, end to end.

    Three prompts of DIFFERENT lengths prefill into their own caches (so they
    sit at different positions), then decode greedily: scalar (each alone) vs
    ragged (all in one batched forward). The token streams must match exactly,
    across the vanilla / MoE / Hyper-Connections backbones.
    """
    cfg = _model_config(use_moe_ffn=use_moe_ffn, use_hyper_connections=use_hyper_connections)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    prompts = [list(range(8)), list(range(2, 18)), list(range(24))]  # lengths 8, 16, 24
    n_decode = 5
    batch_size = len(prompts)
    max_seq_len = max(len(p) for p in prompts) + n_decode + 1

    prefill_snapshots: list[list[dict[str, torch.Tensor]]] = []
    first_tokens: list[int] = []
    positions0: list[int] = []
    scalar_tokens: list[list[int]] = []

    for prompt in prompts:
        cache = StateCache(
            build_state_cache_layer_specs(cfg, max_seq_len=max_seq_len), batch_size=1
        )
        with torch.inference_mode():
            logits = model.forward_prefill_with_cache(
                torch.tensor([prompt], dtype=torch.long), state_cache=cache
            )
        cache.advance_start_pos(len(prompt))
        prefill_snapshots.append(_clone_all_layers(cache))  # state BEFORE any decode
        first_tokens.append(int(logits[0, -1].argmax()))
        positions0.append(len(prompt))

        # Scalar greedy decode, this request alone.
        tokens: list[int] = []
        nxt = first_tokens[-1]
        for _ in range(n_decode):
            with torch.inference_mode():
                step_logits = model.forward_decode_with_cache(
                    torch.tensor([[nxt]], dtype=torch.long),
                    start_pos=cache.start_pos,
                    state_cache=cache,
                )
            cache.advance_start_pos(1)
            nxt = int(step_logits[0, -1].argmax())
            tokens.append(nxt)
        scalar_tokens.append(tokens)

    # Assemble a batched cache: row b = request b's post-prefill snapshot.
    batched = StateCache(
        build_state_cache_layer_specs(cfg, max_seq_len=max_seq_len), batch_size=batch_size
    )
    for row, snapshot in enumerate(prefill_snapshots):
        for layer_idx, buffers in enumerate(snapshot):
            layer = batched.layer(layer_idx)
            layer.swa_kv[row] = buffers["swa_kv"][0]
            layer.compressed_kv[row] = buffers["compressed_kv"][0]
            layer.cmp_kv_state[row] = buffers["cmp_kv_state"][0]
            layer.cmp_score_state[row] = buffers["cmp_score_state"][0]
            if "indexer.compressed_kv" in buffers:
                assert layer.indexer is not None
                layer.indexer.compressed_kv[row] = buffers["indexer.compressed_kv"][0]
                layer.indexer.cmp_kv_state[row] = buffers["indexer.cmp_kv_state"][0]
                layer.indexer.cmp_score_state[row] = buffers["indexer.cmp_score_state"][0]

    positions = torch.tensor(positions0, dtype=torch.long)
    nxt_batched = torch.tensor(first_tokens, dtype=torch.long).unsqueeze(1)  # (B, 1)
    ragged_tokens: list[list[int]] = [[] for _ in prompts]
    for _ in range(n_decode):
        with torch.inference_mode():
            step_logits = model.forward_decode_with_cache_ragged(
                nxt_batched, positions=positions, state_cache=batched
            )
        nxt_batched = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        for row in range(batch_size):
            ragged_tokens[row].append(int(nxt_batched[row, 0]))
        positions = positions + 1

    for row, prompt in enumerate(prompts):
        assert ragged_tokens[row] == scalar_tokens[row], (
            f"request {row} (prompt len {len(prompt)}): "
            f"ragged {ragged_tokens[row]} != scalar {scalar_tokens[row]}"
        )
