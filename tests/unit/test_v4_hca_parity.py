"""HF parity: HCAAttention vs DeepSeek-V4-Pro reference inference code.

The strong correctness gate for Stage C4a. Builds the reference's
unified `Attention` module on a small synthetic config (compress_ratio
in `(128,)` forces the HCA path with no Indexer), patches the tilelang
kernel module to PyTorch equivalents, syncs weights tensor-by-tensor
to our `HCAAttention`, runs both forwards on the same input, and
asserts cosine-sim > 0.999.

What this catches:
    - Compressor formula correctness (positional bias, softmax axis,
      RoPE position assignment for compressed entries).
    - Attention-sink semantics (per-head logit added to softmax denom).
    - Partial RoPE on the right dim slice; inverse output RoPE direction.
    - Q-norm + K-norm placement relative to RoPE.
    - Grouped output projection (per-group einsum vs monolithic).

The reference is vendored under `third_party/deepseek_v4_reference/`
by `scripts/clone_v4_reference.py`. We add that directory to `sys.path`
and import its `model` module after registering a `kernel` stub in
`sys.modules` (so the reference's `from kernel import ...` resolves
to PyTorch equivalents instead of the unavailable tilelang kernels).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn.functional import cosine_similarity

from mini_infer.models.blocks import HCAAttention
from mini_infer.models.blocks.rope import RotaryEmbedding

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "deepseek_v4_reference"


def _stub_kernel_module() -> types.ModuleType:
    """Replace the tilelang kernels with PyTorch equivalents.

    The reference's `model.py` does `from kernel import act_quant,
    fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn`
    at module top-level. Registering a `kernel` ModuleType in
    `sys.modules` BEFORE that import resolves all of those to our stubs.

    Stubs we actually exercise on the HCA path:
      - `act_quant(..., inplace=True)` — used as quant-aware fp8
        round-trip on non-rope KV dims; no-op at fp32.
      - `sparse_attn(q, kv, attn_sink, topk_idxs, scale)` — gather +
        per-head softmax with sink baked into the denominator.

    Stubs that exist but raise (we want a loud failure if the test
    config accidentally drifts onto a non-HCA path):
      - `fp8_gemm`, `fp4_gemm`, `fp4_act_quant` (Indexer / quant paths)
      - `hc_split_sinkhorn` (Hyper-Connections)
    """
    kernel = types.ModuleType("kernel")

    def act_quant(
        x: torch.Tensor,
        block_size: int = 128,
        scale_fmt: Any = None,
        scale_dtype: Any = torch.float32,
        inplace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # QAT (quant + dequant) is approximately identity at fp32.
        if inplace:
            return x
        return x, x.new_ones((*x.shape[:-1], x.shape[-1] // block_size))

    def fp4_act_quant(
        x: torch.Tensor, block_size: int = 32, inplace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inplace:
            return x
        return x, x.new_ones((*x.shape[:-1], x.shape[-1] // block_size))

    def fp8_gemm(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fp8_gemm hit on supposedly fp32 HCA parity path")

    def fp4_gemm(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fp4_gemm hit on supposedly fp32 HCA parity path")

    def sparse_attn(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        # PyTorch reference for the tilelang kernel. Mirrors our
        # `hca_mqa_with_sink` (intentionally — the parity test checks
        # that the two implementations agree, so they share math but
        # NOT code, to avoid testing a stub against itself).
        b, m, h, d = q.shape
        is_pad = topk_idxs < 0
        safe_idxs = topk_idxs.clamp(min=0).long()
        # gather kv: (b, m, n_topk, d)
        expanded = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, d)
        kv_expanded = kv.unsqueeze(1).expand(-1, m, -1, -1)
        gathered = torch.gather(kv_expanded, dim=2, index=expanded)
        scores = torch.einsum("bmhd,bmkd->bhmk", q.float(), gathered.float()) * softmax_scale
        scores = scores.masked_fill(is_pad.unsqueeze(1), float("-inf"))
        sink_col = attn_sink.float().view(1, h, 1, 1).expand(b, -1, m, 1)
        scores = torch.cat([scores, sink_col], dim=-1)
        weights = scores.softmax(dim=-1)[..., :-1]
        out = torch.einsum("bhmk,bmkd->bmhd", weights, gathered.float())
        return out.to(q.dtype)

    def hc_split_sinkhorn(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("hc_split_sinkhorn not exercised by HCA-only parity test")

    kernel.act_quant = act_quant  # type: ignore[attr-defined]
    kernel.fp4_act_quant = fp4_act_quant  # type: ignore[attr-defined]
    kernel.fp8_gemm = fp8_gemm  # type: ignore[attr-defined]
    kernel.fp4_gemm = fp4_gemm  # type: ignore[attr-defined]
    kernel.sparse_attn = sparse_attn  # type: ignore[attr-defined]
    kernel.hc_split_sinkhorn = hc_split_sinkhorn  # type: ignore[attr-defined]
    return kernel


@pytest.fixture(scope="module")
def reference_module() -> Any:
    """Import the vendored DeepSeek-V4 reference `model.py` with stubbed kernels."""
    if not REFERENCE_DIR.exists():
        pytest.skip(
            f"DeepSeek-V4 reference not vendored at {REFERENCE_DIR}; "
            "run `uv run python scripts/clone_v4_reference.py`"
        )
    # Register the kernel stub BEFORE importing the reference's `model`.
    sys.modules["kernel"] = _stub_kernel_module()
    sys.path.insert(0, str(REFERENCE_DIR))

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        if "model" in sys.modules:
            del sys.modules["model"]
        ref = importlib.import_module("model")
        ref.default_dtype = torch.float32
        yield ref
    finally:
        torch.set_default_dtype(prev_dtype)
        # Don't unload `model` — `precompute_freqs_cis`'s lru_cache is
        # keyed on its args; subsequent imports would just rebuild it.


def _build_synthetic_args(reference_module: Any) -> Any:
    """Small HCA-only config that exercises every code path on CPU in <1s."""
    return reference_module.ModelArgs(
        max_batch_size=2,
        max_seq_len=512,
        dtype="bf16",  # not used (default_dtype patched to fp32)
        dim=128,
        n_layers=1,
        n_heads=4,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=64,
        window_size=32,
        compress_ratios=(128,),
        original_seq_len=0,  # disable YaRN
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
    )


def _init_reference_attention(ref_attn: Any, seed: int) -> None:
    """Replace `torch.empty(...)` parameters with deterministic random values.

    The reference uses `nn.Parameter(torch.empty(...))` everywhere — those
    tensors hold uninitialized memory. For a meaningful parity test, we
    seed-fill them so the same module configuration produces the same
    weights every run, and so we have non-NaN values to copy into our
    HCAAttention.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for p in ref_attn.parameters():
        with torch.no_grad():
            p.data = torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.02
    # Reset stateful buffers used by the compressor's incremental decode path
    # (irrelevant to our prefill test but want them deterministic).
    for buf in ref_attn.buffers():
        if buf.dtype.is_floating_point:
            buf.zero_()


def _sync_weights(our_block: HCAAttention, ref_attn: Any) -> None:
    """Copy reference parameters into our HCAAttention, name-by-name."""
    with torch.no_grad():
        # --- Q low-rank ---
        our_block.q_a_proj.weight.copy_(ref_attn.wq_a.weight)
        our_block.q_a_layernorm.weight.copy_(ref_attn.q_norm.weight)
        our_block.q_b_proj.weight.copy_(ref_attn.wq_b.weight)
        # --- SWA K=V branch ---
        our_block.swa_kv_proj.weight.copy_(ref_attn.wkv.weight)
        our_block.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        # --- Compressor ---
        our_block.compressor.kv_proj.weight.copy_(ref_attn.compressor.wkv.weight)
        our_block.compressor.weight_proj.weight.copy_(ref_attn.compressor.wgate.weight)
        our_block.compressor.position_bias.copy_(ref_attn.compressor.ape)
        our_block.compressor.norm.weight.copy_(ref_attn.compressor.norm.weight)
        # --- Sink ---
        our_block.sink.sink_logits.copy_(ref_attn.attn_sink)
        # --- Grouped output projection ---
        our_block.grouped_output.wo_a.copy_(ref_attn.wo_a.weight)
        our_block.grouped_output.wo_b.weight.copy_(ref_attn.wo_b.weight)


def _build_position_embeddings(
    rotary: RotaryEmbedding,
    bsz: int,
    seqlen: int,
    compression_ratio: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Compute (cos, sin) for both raw token positions and compressed positions."""
    token_positions = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    cos_t, sin_t = rotary(torch.zeros(bsz, seqlen, device=device), token_positions)

    n_compressed = seqlen // compression_ratio
    compressed_positions = (
        (torch.arange(n_compressed, device=device) * compression_ratio).unsqueeze(0).expand(bsz, -1)
    )
    cos_c, sin_c = rotary(torch.zeros(bsz, n_compressed, device=device), compressed_positions)
    return (cos_t, sin_t), (cos_c, sin_c)


def test_hca_block_matches_v4_reference(reference_module: Any) -> None:
    """Cosine-sim > 0.999 between our HCAAttention and the reference Attention."""
    torch.manual_seed(0)
    args = _build_synthetic_args(reference_module)

    ref_attn = reference_module.Attention(0, args)
    _init_reference_attention(ref_attn, seed=42)

    our_block = HCAAttention(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        compression_ratio=args.compress_ratios[0],
        rms_norm_eps=args.norm_eps,
    )
    _sync_weights(our_block, ref_attn)

    bsz, seqlen = 2, args.max_seq_len
    x = torch.randn(bsz, seqlen, args.dim) * 0.5

    # Build position embeddings for our block via the matching RoPE table.
    rotary = RotaryEmbedding(head_dim=args.rope_head_dim, base=args.rope_theta)
    token_pe, compressed_pe = _build_position_embeddings(
        rotary, bsz, seqlen, args.compress_ratios[0], device=x.device
    )

    with torch.no_grad():
        ours = our_block(x, token_pe, compressed_pe)
        theirs = ref_attn(x, start_pos=0)

    assert ours.shape == theirs.shape == (bsz, seqlen, args.dim)
    cos_sim = cosine_similarity(ours.flatten().float(), theirs.flatten().float(), dim=0)
    max_diff = (ours - theirs).abs().max().item()
    rel_err = max_diff / max(theirs.abs().max().item(), 1e-9)
    assert cos_sim > 0.999, (
        f"cosine_sim={cos_sim:.6f}, max_abs_diff={max_diff:.3e}, "
        f"rel_err={rel_err:.3e} (target > 0.999)"
    )


def test_hca_block_handles_uneven_seqlen_assertion(reference_module: Any) -> None:
    """Our standalone block requires `seqlen % m == 0` and rejects otherwise.

    Different from the reference (which handles tail tokens via state buffers
    during incremental compression). Documents the standalone-block contract.
    """
    args = _build_synthetic_args(reference_module)
    our_block = HCAAttention(
        hidden_size=args.dim,
        num_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        num_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
        window_size=args.window_size,
        compression_ratio=args.compress_ratios[0],
        rms_norm_eps=args.norm_eps,
    )
    bsz, seqlen = 1, args.compress_ratios[0] + 1  # not a multiple of m
    x = torch.randn(bsz, seqlen, args.dim)
    # Just any cos/sin — the compressor's seqlen check fires before RoPE.
    cos = torch.zeros(bsz, seqlen, args.rope_head_dim)
    sin = torch.zeros(bsz, seqlen, args.rope_head_dim)
    with pytest.raises(ValueError, match="multiple of compression_ratio"):
        our_block(x, (cos, sin), (cos[:, :1], sin[:, :1]))
