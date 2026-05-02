"""Modal probe: parity test for our `kv_quant="nvfp4"` cache path.

Builds two `BlockPool` + `PagedKVCache` instances with identical
geometry — one bf16 + flash-attn-equivalent (we route through
FlashInfer for both so the kernel is the same), one nvfp4 +
flashinfer. Writes the SAME packed K/V into both via
`PagedKVCache.append_kv_packed`, then runs
`flashinfer_attention_forward` on both. Reports cosine similarity
between the attention outputs.

If cos_sim is high (>0.9), the cache writes/reads are correct and
the broken token output we see in the full bench must come from
something downstream (sampler, scheduler, or compounding precision
loss across 28 layers + 128 decode steps). If cos_sim is low, the
nvfp4 write/read path is broken and we have a concrete starting
point for debugging.

Run with:

    MINI_INFER_BENCH_GPU=B200 uv run modal run scripts/modal_nvfp4_parity.py
"""

import os
from pathlib import Path

import modal

app = modal.App("mini-infer-nvfp4-parity")

_BENCH_GPU = os.environ.get("MINI_INFER_BENCH_GPU", "B200")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.6", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.40", "triton>=3.0")
    .pip_install("flashinfer-python>=0.6.10rc1")
    .add_local_file(
        str(Path(__file__).parent / "data" / "technical_passage.md"),
        "/root/scripts/data/technical_passage.md",
    )
    .add_local_python_source("mini_infer")
)


@app.function(image=image, gpu=_BENCH_GPU, timeout=600)
def parity() -> str:
    import traceback

    import torch

    lines: list[str] = []
    lines.append(f"GPU: {torch.cuda.get_device_name()}")
    lines.append(f"Compute capability: {torch.cuda.get_device_capability()}")
    lines.append(f"PyTorch: {torch.__version__}")
    try:
        import flashinfer

        lines.append(f"FlashInfer: {flashinfer.__version__}")
    except ImportError as exc:
        return f"FlashInfer import failed: {exc}"

    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.flashinfer_backend import flashinfer_attention_forward
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    # Geometry: matches Qwen2.5-1.5B head dims (head_dim=128, num_kv_heads=2)
    # but small block count to keep the test focused. block_size=16 satisfies
    # both NVFP4 constraints (block_size % 4 == 0; head_dim % 64 == 0).
    num_blocks = 8
    block_size = 16
    num_layers = 1  # single layer is enough; the bug would surface at any depth
    num_kv_heads = 2
    head_dim = 128
    num_q_heads = 12  # GQA group_size = 6
    seq_len = 64  # 4 blocks worth — enough to exercise multi-page quant
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(0)
    # Random bf16 K/V for the prompt. Same data fed to both pools.
    packed_k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
    packed_v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
    q = torch.randn(1, num_q_heads, head_dim, dtype=dtype, device=device) * 0.1

    def _build_cache(kv_quant: str | None) -> PagedKVCache:
        pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device="cuda",
            kv_quant=kv_quant,
            attention_backend="flashinfer",
        )
        cache = PagedKVCache(pool)
        cache.add_request_slot()
        # Allocate enough blocks for seq_len tokens
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        cache._block_ids[0] = [pool.allocate() for _ in range(num_blocks_needed)]
        cache._num_tokens[0] = seq_len
        return cache

    def _write_kv(cache: PagedKVCache) -> None:
        cu_seqlens_q_new = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        cache._write_packed_kv(0, packed_k, packed_v, cu_seqlens_q_new)

    def _attention_out(cache: PagedKVCache) -> torch.Tensor:
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        return flashinfer_attention_forward(
            q,
            cache,
            layer_idx=0,
            cu_seqlens_q=cu_seqlens_q,
            softmax_scale=1.0 / (head_dim**0.5),
        )

    lines.append("")
    lines.append("=== Parity setup ===")
    lines.append(
        f"  geom: blocks={num_blocks} bs={block_size} layers={num_layers} "
        f"kv_heads={num_kv_heads} head_dim={head_dim} q_heads={num_q_heads}"
    )
    lines.append(f"  prompt: seq_len={seq_len} ({(seq_len + block_size - 1) // block_size} pages)")

    # bf16 baseline.
    try:
        bf16_cache = _build_cache(kv_quant=None)
        _write_kv(bf16_cache)
        out_bf16 = _attention_out(bf16_cache)
        lines.append(f"  bf16 attn out: shape={tuple(out_bf16.shape)} dtype={out_bf16.dtype}")
    except Exception as exc:
        lines.append(f"  bf16 EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))
        return "\n".join(lines)

    # nvfp4 path.
    try:
        nvfp4_cache = _build_cache(kv_quant="nvfp4")
        _write_kv(nvfp4_cache)
        out_nvfp4 = _attention_out(nvfp4_cache)
        lines.append(f"  nvfp4 attn out: shape={tuple(out_nvfp4.shape)} dtype={out_nvfp4.dtype}")
    except Exception as exc:
        lines.append(f"  nvfp4 EXCEPTION: {type(exc).__name__}: {exc}")
        lines.append("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))
        return "\n".join(lines)

    cos = float(
        torch.nn.functional.cosine_similarity(
            out_bf16.float().flatten(), out_nvfp4.float().flatten(), dim=0
        ).item()
    )
    rel_err = float((out_bf16.float() - out_nvfp4.float()).abs().mean().item()) / max(
        float(out_bf16.float().abs().mean().item()), 1e-9
    )
    lines.append("")
    lines.append("=== Result ===")
    lines.append(f"  cos_sim:  {cos:.6f}")
    lines.append(f"  rel_err:  {rel_err:.6f}")
    lines.append(f"  bf16  out norm: {float(out_bf16.float().norm().item()):.4f}")
    lines.append(f"  nvfp4 out norm: {float(out_nvfp4.float().norm().item()):.4f}")
    lines.append(f"  bf16  out[0,0,:8]: {out_bf16[0, 0, :8].float().tolist()}")
    lines.append(f"  nvfp4 out[0,0,:8]: {out_nvfp4[0, 0, :8].float().tolist()}")

    # Also dump some pool-state diagnostics so we know the writes happened.
    pool = nvfp4_cache._pool
    assert pool._nvfp4_storage is not None
    assert pool._nvfp4_global_sf is not None
    assert pool._nvfp4_initialized is not None
    lines.append("")
    lines.append("=== nvfp4 pool state ===")
    lines.append(f"  initialized[L=0]: {pool._nvfp4_initialized[0].tolist()}")
    lines.append(f"  global_sf[L=0]:   {pool._nvfp4_global_sf[0].tolist()}")
    lines.append(f"  storage[L=0,K,B=0,S=0]: {pool._nvfp4_storage[0, 0, 0, 0, 0, :8].tolist()}")
    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    print(parity.remote())
