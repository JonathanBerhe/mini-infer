"""Parity: the port's simulated activation quant vs the reference's quantized numerics.

The other V4 parity tests stub the reference's quantizers to identity, so
they pin the port's default full-precision mode. Here the reference runs
with real FP8 / FP4 round trips and a real Hadamard transform
(`install_real_quant_stubs`), and our blocks run with
`act_quant=SimulatedActQuant(...)`. Each test drives one attention type
through prefill and several decode steps (crossing compressed-block
flushes) and checks, against the reference:

  - the standalone `forward` and the cache-aware prefill outputs,
  - the quantized KV each side stores (window, compressed, indexer keys),
  - every decode-step output,
  - for CSA, the Lightning Indexer's top-k picks.

Config: the synthetic small config of the other V4 parity tests, with two
dims widened so the real kernels accept them (they assert the quantized
width splits into whole blocks): `head_dim=72`, so the non-RoPE width is
64 (one FP8 block), and `index_head_dim=32` (one FP4 block, a power of two
for the Hadamard transform).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.nn.functional import cosine_similarity

from mini_infer.cache.state_cache import IndexerStateSpec, StateCache, StateLayerSpec
from mini_infer.models.blocks import CSAAttention, HCAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding
from mini_infer.models.blocks.swa import SWAAttention
from mini_infer.models.blocks.v4 import SimulatedActQuant
from tests.unit._v4_reference_helpers import install_real_quant_stubs

_T_PREFILL = 16
_N_DECODE = 8
_BATCH = 2
# Same bar as the identity-stub V4 parity tests.
_COSINE_MIN = 0.999
# Much tighter bar on `max|ours - theirs| / max|theirs|`. The flag-on path
# clears it by orders of magnitude; the flag-off path fails it (pinned by
# `test_flag_off_misses_the_quantized_reference`), so it has teeth where the
# cosine bar alone would also pass a full-precision port on prefill.
_RELATIVE_ERROR_MAX = 1e-5
# Non-RoPE width of the synthetic config (`head_dim - rope_head_dim`).
_NOPE_DIM = 64

_LAYER_KINDS = {"swa": 0, "csa": 4, "hca": 8}


def _build_args(reference_module: Any, compress_ratio: int) -> Any:
    return reference_module.ModelArgs(
        max_batch_size=_BATCH,
        max_seq_len=64,
        dtype="bf16",
        dim=64,
        n_layers=1,
        n_heads=4,
        q_lora_rank=32,
        head_dim=72,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(compress_ratio,),
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
    )


def _init_reference_attention(ref_attn: Any, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in ref_attn.parameters():
        with torch.no_grad():
            parameter.data = (
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.02
            )
    for buffer in ref_attn.buffers():
        if buffer.dtype.is_floating_point:
            buffer.zero_()


def _build_our_block(
    args: Any, act_quant: SimulatedActQuant | None
) -> SWAAttention | CSAAttention | HCAAttention:
    common = dict(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        rms_norm_eps=args.norm_eps,
        act_quant=act_quant,
    )
    ratio = args.compress_ratios[0]
    if ratio == 0:
        return SWAAttention(**common)  # type: ignore[arg-type]
    if ratio == 4:
        return CSAAttention(  # type: ignore[arg-type]
            **common,
            compression_ratio=ratio,
            index_num_heads=args.index_n_heads,
            index_head_dim=args.index_head_dim,
            index_top_k=args.index_topk,
        )
    return HCAAttention(**common, compression_ratio=ratio)  # type: ignore[arg-type]


def _sync_weights(our_block: Any, ref_attn: Any) -> None:
    with torch.no_grad():
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)
        compressor_pairs = []
        if ref_attn.compress_ratio:
            compressor_pairs.append((our_block.compressor, ref_attn.compressor))
        if getattr(ref_attn, "indexer", None) is not None:
            our_block.indexer.wq_b.weight.copy_(ref_attn.indexer.wq_b.weight)
            our_block.indexer.weights_proj.weight.copy_(ref_attn.indexer.weights_proj.weight)
            compressor_pairs.append((our_block.indexer.compressor, ref_attn.indexer.compressor))
        for ours, theirs in compressor_pairs:
            ours.kv_proj.weight.copy_(theirs.wkv.weight)
            ours.weight_proj.weight.copy_(theirs.wgate.weight)
            ours.position_bias.copy_(theirs.ape)
            ours.norm.weight.copy_(theirs.norm.weight)


def _make_state_cache(args: Any) -> StateCache:
    ratio = args.compress_ratios[0]
    spec = StateLayerSpec(
        kv_head_dim=args.head_dim,
        compression_ratio=ratio,
        n_win=args.window_size,
        max_n_compressed=args.max_seq_len // ratio if ratio else 1,
        overlap_mode=ratio == 4,
        indexer=IndexerStateSpec(head_dim=args.index_head_dim) if ratio == 4 else None,
    )
    return StateCache([spec], batch_size=_BATCH)


def _position_embeddings(
    rotary: RotaryEmbedding, positions: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)`, each `(B, len(positions), rope_head_dim)`, same positions per row."""
    pos = torch.tensor(positions, dtype=torch.long).unsqueeze(0).expand(_BATCH, -1)
    return rotary(torch.zeros(_BATCH, len(positions)), pos)


def _relative_error(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    return (ours - theirs).abs().max().item() / max(theirs.abs().max().item(), 1e-12)


def _assert_matches(ours: torch.Tensor, theirs: torch.Tensor, label: str) -> None:
    assert ours.shape == theirs.shape, f"{label}: {tuple(ours.shape)} vs {tuple(theirs.shape)}"
    cosine = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0).item()
    relative_error = _relative_error(ours, theirs)
    message = f"{label}: cosine={cosine:.9f}, relative_error={relative_error:.3e}"
    assert cosine > _COSINE_MIN, message
    assert relative_error < _RELATIVE_ERROR_MAX, message


def _assert_stored_kv_matches(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> None:
    """Quantized dims must land on the same grid points; RoPE dims stay unquantized."""
    if name == "indexer keys":
        # Hadamard-rotated and FP4-quantized on every dim.
        assert torch.equal(ours, theirs), f"stored {name} differ"
        return
    assert torch.equal(ours[..., :_NOPE_DIM], theirs[..., :_NOPE_DIM]), f"stored {name} FP8 dims"
    # Neither side quantizes the RoPE dims: only last-bit RoPE rounding differs.
    torch.testing.assert_close(
        ours[..., _NOPE_DIM:], theirs[..., _NOPE_DIM:], rtol=0, atol=1e-6, msg=f"{name} RoPE"
    )


def _run_case(
    reference_module: Any, layer_kind: str, act_quant: SimulatedActQuant | None
) -> dict[str, Any]:
    """Prefill + decode on the reference and on our block; collect everything to compare."""
    torch.manual_seed(0)
    args = _build_args(reference_module, _LAYER_KINDS[layer_kind])
    ratio = args.compress_ratios[0]
    n_win = args.window_size
    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=42)
    our_block = _build_our_block(args, act_quant)
    _sync_weights(our_block, ref_attn)
    state_cache = _make_state_cache(args)
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)

    prefill_input = torch.randn(_BATCH, _T_PREFILL, args.dim) * 0.5
    decode_inputs = [torch.randn(_BATCH, 1, args.dim) * 0.5 for _ in range(_N_DECODE)]
    token_pe = _position_embeddings(rotary, list(range(_T_PREFILL)))
    compressed_pe = _position_embeddings(
        rotary, [block * ratio for block in range(_T_PREFILL // ratio)] if ratio else []
    )

    result: dict[str, Any] = {"args": args, "decode": []}
    with torch.no_grad():
        result["ref_prefill"] = ref_attn(prefill_input, start_pos=0)
        result["our_forward"] = our_block(prefill_input, token_pe, compressed_pe)
        result["our_prefill"] = our_block.forward_prefill_with_cache(
            prefill_input,
            token_position_embeddings=token_pe,
            compressed_position_embeddings=compressed_pe,
            state_cache=state_cache,
            layer_idx=0,
        )
        state_cache.advance_start_pos(_T_PREFILL)
        layer = state_cache.layer(0)
        result["kv_after_prefill"] = [("window", layer.swa_kv, ref_attn.kv_cache[:, :n_win])]
        if ratio:
            n_blocks = _T_PREFILL // ratio
            result["kv_after_prefill"].append(
                (
                    "compressed",
                    layer.compressed_kv[:, :n_blocks],
                    ref_attn.kv_cache[:, n_win : n_win + n_blocks],
                )
            )
        if ratio == 4:
            assert layer.indexer is not None
            result["kv_after_prefill"].append(
                (
                    "indexer keys",
                    layer.indexer.compressed_kv[:, :n_blocks],
                    ref_attn.indexer.kv_cache[:, :n_blocks],
                )
            )
        # Snapshot now: decode below keeps writing into both caches.
        result["kv_after_prefill"] = [
            (name, ours.clone(), theirs.clone())
            for name, ours, theirs in result["kv_after_prefill"]
        ]

        for step, decode_input in enumerate(decode_inputs):
            position = _T_PREFILL + step
            block_pe = None
            if ratio and (position + 1) % ratio == 0:
                block_pe = _position_embeddings(rotary, [(position // ratio) * ratio])
            theirs = ref_attn(decode_input, start_pos=position)
            ours = our_block.forward_decode(
                decode_input,
                start_pos=position,
                state_cache=state_cache,
                layer_idx=0,
                token_position_embeddings=_position_embeddings(rotary, [position]),
                block_position_embeddings=block_pe,
            )
            result["decode"].append((position, ours, theirs))
    return result


@pytest.mark.parametrize("layer_kind", list(_LAYER_KINDS))
def test_simulated_act_quant_matches_quantized_reference(
    reference_module: Any, monkeypatch: pytest.MonkeyPatch, layer_kind: str
) -> None:
    """Flag on vs reference with real quantizers: prefill, stored KV, every decode step."""
    install_real_quant_stubs(reference_module, monkeypatch, scale_fmt="ue8m0")
    result = _run_case(reference_module, layer_kind, SimulatedActQuant(fp8_scale_fmt="ue8m0"))

    _assert_matches(result["our_forward"], result["ref_prefill"], "standalone forward")
    _assert_matches(result["our_prefill"], result["ref_prefill"], "cache-aware prefill")
    for name, ours, theirs in result["kv_after_prefill"]:
        _assert_stored_kv_matches(name, ours, theirs)
    for position, ours, theirs in result["decode"]:
        _assert_matches(ours, theirs, f"decode start_pos={position}")


def test_simulated_act_quant_matches_reference_with_fp32_scales(
    reference_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`scale_fmt=None` (plain FP32 FP8 scales) on both sides, CSA (all three round trips)."""
    install_real_quant_stubs(reference_module, monkeypatch, scale_fmt=None)
    result = _run_case(reference_module, "csa", SimulatedActQuant(fp8_scale_fmt=None))

    _assert_matches(result["our_prefill"], result["ref_prefill"], "cache-aware prefill")
    for name, ours, theirs in result["kv_after_prefill"]:
        _assert_stored_kv_matches(name, ours, theirs)
    for position, ours, theirs in result["decode"]:
        _assert_matches(ours, theirs, f"decode start_pos={position}")


@pytest.mark.parametrize("layer_kind", list(_LAYER_KINDS))
def test_flag_off_misses_the_quantized_reference(
    reference_module: Any, monkeypatch: pytest.MonkeyPatch, layer_kind: str
) -> None:
    """Guard against a vacuous pass: without the flag, the checks above fail.

    The real stubs move the reference's stored KV by far more than the RoPE
    tolerance, and its outputs by far more than `_RELATIVE_ERROR_MAX`, so the
    flag-on matches are not an accident of loose bounds.
    """
    install_real_quant_stubs(reference_module, monkeypatch, scale_fmt="ue8m0")
    result = _run_case(reference_module, layer_kind, act_quant=None)
    for name, ours, theirs in result["kv_after_prefill"]:
        assert (ours - theirs).abs().max().item() > 1e-3, f"{name} KV barely changed"
    assert _relative_error(result["our_prefill"], result["ref_prefill"]) > _RELATIVE_ERROR_MAX


def test_indexer_top_k_matches_quantized_reference(
    reference_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Lightning Indexer picks the same compressed blocks as the quantized reference."""
    install_real_quant_stubs(reference_module, monkeypatch, scale_fmt="ue8m0")
    torch.manual_seed(0)
    args = _build_args(reference_module, _LAYER_KINDS["csa"])
    ratio = args.compress_ratios[0]
    seqlen = args.max_seq_len
    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=7)
    our_block = _build_our_block(args, SimulatedActQuant(fp8_scale_fmt="ue8m0"))
    assert isinstance(our_block, CSAAttention)
    _sync_weights(our_block, ref_attn)
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)

    x = torch.randn(_BATCH, seqlen, args.dim) * 0.5
    with torch.no_grad():
        ref_attn(x, start_pos=0)  # wires the reference indexer's cache + RoPE table
        q_lora_latent = ref_attn.q_norm(ref_attn.wq_a(x))
        theirs = ref_attn.indexer(x, q_lora_latent, 0, seqlen)
        ours = our_block.indexer(
            hidden_states=x,
            q_lora_latent=q_lora_latent,
            token_position_embeddings=_position_embeddings(rotary, list(range(seqlen))),
            compressed_position_embeddings=_position_embeddings(
                rotary, [block * ratio for block in range(seqlen // ratio)]
            ),
            compressed_offset=seqlen,
        )
    # Same set per query; order among equal scores is not part of the contract.
    assert torch.equal(ours.sort(dim=-1).values, theirs.long().sort(dim=-1).values)
