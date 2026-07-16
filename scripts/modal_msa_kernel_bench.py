"""MSA block-sparse decode kernel: GPU parity + microbench + end-to-end A/B (A10).

Three questions, one cheap GPU run:

1. **Parity**: does `msa_paged_decode_triton` match the pure-torch reference
   (itself CPU-validated against the dense-mask oracle) on bf16 paged data,
   across topk / context / batch sweeps including -1 padding and partial last
   blocks? Also a sparsity probe: perturbing K/V ONLY at non-selected blocks
   must not change the kernel output at all (proof it reads just the selected
   blocks).
2. **Op microbench**: kernel vs the shipped torch path (materialize full K/V +
   dense mask + SDPA) at M3's real head geometry (64q/4kv/128, topk 16 + local,
   index block 128), context sweep. The torch path is O(context) per step; the
   kernel is O(topk * 128).
3. **End-to-end mini A/B**: a synthetic-weights M3-shaped model (16q/1kv/128,
   group 16 like M3) decoding at long context, kernel on vs off: tokens must be
   identical and the tok/s ratio is the honest end-to-end signal (the indexer's
   O(context) re-scoring stays on both arms, Amdahl-capping the win).

Run with:
    uv run modal run scripts/modal_msa_kernel_bench.py
"""

import modal

_GPU = "A10"

app = modal.App("mini-infer-msa-kernel-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.6.0", extra_index_url="https://download.pytorch.org/whl/cu124")
    # mini_infer.cache imports transformers at module level (DynamicCache).
    .pip_install("transformers>=4.40", "safetensors>=0.4", "huggingface_hub>=0.20")
    .add_local_python_source("mini_infer")
)


def _build_cache(model, num_blocks: int, block_size: int, device, dtype):  # type: ignore[no-untyped-def]
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=dtype,
        device=device,
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    return cache


def _parity_sweep() -> list[str]:
    """Kernel vs torch reference on random paged data; returns report lines."""
    import torch

    from mini_infer.cache.block_pool import BlockPool, StreamSpec
    from mini_infer.cache.msa_paged_attention import (
        msa_paged_decode_torch,
        msa_paged_decode_triton,
    )
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    lines = []
    device = "cuda"
    for num_kv_heads, group, topk, seq_lens in [
        (4, 16, 16, [4096]),  # M3 geometry, long context
        (4, 16, 16, [1000, 3333]),  # ragged batch, partial last blocks
        (1, 16, 4, [700]),  # small topk, one kv head
        (4, 16, 16, [130]),  # short context: fewer blocks than topk -> -1 padding
    ]:
        torch.manual_seed(0)
        head_dim, index_block, pool_block = 128, 128, 16
        num_q_heads = num_kv_heads * group
        max_tokens = max(seq_lens)
        num_blocks = (max_tokens // pool_block + 2) * len(seq_lens) + 4
        streams = [
            [StreamSpec("k", num_kv_heads, head_dim), StreamSpec("v", num_kv_heads, head_dim)]
        ]
        pool = BlockPool(
            num_blocks=num_blocks,
            block_size=pool_block,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=device,
            layer_streams=streams,
            attention_backend="torch",
        )
        cache = PagedKVCache(pool)
        for _ in seq_lens:
            cache.add_request_slot()
        total = sum(seq_lens)
        cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32)
        k = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        cache.append_stream_packed(k, cu, 0, "k")
        cache.append_stream_packed(v, cu, 0, "v")

        q = torch.randn(len(seq_lens), num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
        tables = cache.block_tables_per_request_tensor(device)
        # Random selections incl. the local (last) block, -1 padded; one
        # independent row per KV head (per-head selection, transformers 5.14).
        selections = []
        for n in seq_lens:
            nb = -(-n // index_block)
            last = nb - 1
            per_head = torch.full((num_kv_heads, topk), -1, dtype=torch.int64)
            for kv_h in range(num_kv_heads):
                others = torch.randperm(max(nb - 1, 1))[: max(min(topk, nb) - 1, 0)]
                per_head[kv_h, 0] = last
                per_head[kv_h, 1 : 1 + others.numel()] = others
            selections.append(per_head.to(device))

        k_pool = pool.storage_for_stream(0, "k")
        v_pool = pool.storage_for_stream(0, "v")
        ref = msa_paged_decode_torch(
            q, k_pool, v_pool, tables, seq_lens, selections, index_block_size=index_block
        )
        got = msa_paged_decode_triton(
            q, k_pool, v_pool, tables, seq_lens, selections, index_block_size=index_block
        )
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), got.float().flatten(), dim=0
        ).item()
        max_abs = (ref.float() - got.float()).abs().max().item()
        ok = cos > 0.999 and max_abs < 2e-2
        lines.append(
            f"parity kv={num_kv_heads} topk={topk} seqs={seq_lens}: "
            f"cos={cos:.6f} max_abs={max_abs:.2e} {'PASS' if ok else 'FAIL'}"
        )

        # Sparsity probe: noise in NON-selected blocks must not move the output.
        keep = torch.zeros(total, dtype=torch.bool)
        starts = [0]
        for n in seq_lens:
            starts.append(starts[-1] + n)
        for r, n in enumerate(seq_lens):
            # A token is perturbable only if NO head selected its block.
            for b in selections[r].flatten().tolist():
                if b >= 0:
                    lo = starts[r] + b * index_block
                    hi = starts[r] + min((b + 1) * index_block, n)
                    keep[lo:hi] = True
        noise_k = k.clone()
        noise_v = v.clone()
        noise_k[~keep] = torch.randn_like(noise_k[~keep])
        noise_v[~keep] = torch.randn_like(noise_v[~keep])
        pool2 = BlockPool(
            num_blocks=num_blocks,
            block_size=pool_block,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=device,
            layer_streams=streams,
            attention_backend="torch",
        )
        cache2 = PagedKVCache(pool2)
        for _ in seq_lens:
            cache2.add_request_slot()
        cache2.append_stream_packed(noise_k, cu, 0, "k")
        cache2.append_stream_packed(noise_v, cu, 0, "v")
        got2 = msa_paged_decode_triton(
            q,
            pool2.storage_for_stream(0, "k"),
            pool2.storage_for_stream(0, "v"),
            cache2.block_tables_per_request_tensor(device),
            seq_lens,
            selections,
            index_block_size=index_block,
        )
        identical = torch.equal(got, got2)
        lines.append(
            f"  sparsity probe (noise outside selection): "
            f"{'PASS (bit-identical)' if identical else 'FAIL (output moved)'}"
        )
    return lines


def _op_microbench() -> list[str]:
    """Kernel vs shipped torch path (materialize + mask + SDPA) at M3 geometry."""
    import time

    import torch

    from mini_infer.cache.block_pool import BlockPool, StreamSpec
    from mini_infer.cache.msa_paged_attention import msa_paged_decode_triton
    from mini_infer.cache.packed_attention import packed_attention_torch
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.models.blocks.minimax_m3_indexer import MiniMaxM3Indexer

    lines = []
    device = "cuda"
    head_dim, index_block, pool_block = 128, 128, 16
    num_kv_heads, num_q_heads, topk = 4, 64, 16
    idxer = (
        MiniMaxM3Indexer(
            hidden_size=256,
            num_heads=4,
            head_dim=head_dim,
            block_size=index_block,
            topk_blocks=topk,
            num_query_heads=num_q_heads,
            local_blocks=1,
        )
        .to(device)
        .eval()
    )

    for ctx in [1024, 4096, 16384, 65536]:
        for batch in [1, 8]:
            torch.manual_seed(0)
            seq_lens = [ctx] * batch
            total = ctx * batch
            num_blocks = total // pool_block + batch + 2
            streams = [
                [StreamSpec("k", num_kv_heads, head_dim), StreamSpec("v", num_kv_heads, head_dim)]
            ]
            pool = BlockPool(
                num_blocks=num_blocks,
                block_size=pool_block,
                num_layers=1,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=torch.bfloat16,
                device=device,
                layer_streams=streams,
                attention_backend="torch",
            )
            cache = PagedKVCache(pool)
            for _ in seq_lens:
                cache.add_request_slot()
            cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32)
            k = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
            cache.append_stream_packed(k, cu, 0, "k")
            cache.append_stream_packed(v, cu, 0, "v")

            q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
            tables = cache.block_tables_per_request_tensor(device)
            nb = -(-ctx // index_block)
            sel = []
            for _ in range(batch):
                s = torch.full((num_kv_heads, topk), -1, dtype=torch.int64, device=device)
                for kv_h in range(num_kv_heads):
                    pick = torch.randperm(nb, device=device)[: min(topk, nb)]
                    pick[0] = nb - 1  # force local
                    s[kv_h, : pick.numel()] = pick
                sel.append(s)

            def kernel_arm(q=q, pool=pool, tables=tables, seq_lens=seq_lens, sel=sel):  # type: ignore[no-untyped-def]
                return msa_paged_decode_triton(
                    q,
                    pool.storage_for_stream(0, "k"),
                    pool.storage_for_stream(0, "v"),
                    tables,
                    seq_lens,
                    sel,
                    index_block_size=index_block,
                )

            cu_dec = torch.arange(batch + 1, dtype=torch.int32)

            def torch_arm(q=q, cache=cache, sel=sel, ctx=ctx, batch=batch, cu_dec=cu_dec):  # type: ignore[no-untyped-def]
                keys_full, cu_k, _ = cache.materialize_packed_stream(0, "k")
                vals_full, _, _ = cache.materialize_packed_stream(0, "v")
                masks = []
                for r in range(batch):
                    pos = torch.tensor([[ctx - 1]], device=device)
                    m = idxer.build_block_mask(
                        sel[r].view(1, num_kv_heads, 1, -1), ctx, pos, dtype=torch.bfloat16
                    )
                    masks.append(m[0].transpose(0, 1))
                return packed_attention_torch(
                    q, keys_full, vals_full, cu_dec, cu_k, head_dim**-0.5, block_mask=masks
                )

            for name, fn in [("kernel", kernel_arm), ("torch", torch_arm)]:
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                iters = 20
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / iters * 1e6
                if name == "kernel":
                    t_kernel = dt
                else:
                    t_torch = dt
            lines.append(
                f"op ctx={ctx:6d} B={batch}: kernel {t_kernel:9.1f}us  "
                f"torch {t_torch:9.1f}us  speedup {t_torch / t_kernel:6.2f}x"
            )
    return lines


def _e2e_ab(ctx_len: int, decode_steps: int) -> list[str]:
    """Synthetic M3-shaped model, chunked prefill to ctx_len, timed decode A/B."""
    import time

    import torch

    from mini_infer.models.minimax_m3 import MiniMaxM3Config, MiniMaxM3ForCausalLM

    device = "cuda"
    cfg = MiniMaxM3Config(
        vocab_size=4096,
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=1,  # group 16, same as M3
        head_dim=128,
        dense_intermediate_size=2048,
        moe_intermediate_size=512,
        shared_intermediate_size=512,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.0,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rotary_dim=64,
        tie_word_embeddings=False,
        index_block_size=128,
        index_topk_blocks=16,
        index_n_heads=4,
        index_head_dim=128,
        index_local_blocks=1,
        first_dense_layers=1,
    )
    torch.manual_seed(0)
    model = MiniMaxM3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()

    lines = []
    results = {}
    for arm in ["torch", "kernel"]:
        model.set_decode_kernel(arm == "kernel")
        torch.manual_seed(1)
        pool_blocks = (ctx_len + decode_steps) // 16 + 8
        cache = _build_cache(model, pool_blocks, 16, device, torch.bfloat16)
        prompt = torch.randint(0, cfg.vocab_size, (ctx_len,), generator=None)

        with torch.inference_mode():
            # Chunked prefill (the materialized path; both arms identical here).
            chunk = 1024
            done = 0
            while done < ctx_len:
                n = min(chunk, ctx_len - done)
                ids = prompt[done : done + n].unsqueeze(0).to(device)
                pos = torch.arange(done, done + n, device=device).unsqueeze(0)
                logits = model(
                    input_ids=ids,
                    position_ids=pos,
                    past_key_values=cache,
                    cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32),
                )
                done += n
            nxt = int(logits[0, -1].argmax())
            tokens = [nxt]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cache_len = ctx_len
            for _ in range(decode_steps - 1):
                logits = model(
                    input_ids=torch.tensor([[nxt]], device=device),
                    position_ids=torch.tensor([[cache_len]], device=device),
                    past_key_values=cache,
                    cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
                )
                cache_len += 1
                nxt = int(logits[0, -1].argmax())
                tokens.append(nxt)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        results[arm] = (tokens, decode_steps / dt)
        lines.append(f"e2e ctx={ctx_len} arm={arm}: {decode_steps / dt:8.2f} tok/s")

    same = results["torch"][0] == results["kernel"][0]
    lines.append(
        f"e2e ctx={ctx_len}: token identity {'PASS' if same else 'FAIL'}; "
        f"speedup {results['kernel'][1] / results['torch'][1]:.2f}x"
    )
    if not same:
        t_toks, k_toks = results["torch"][0], results["kernel"][0]
        first = next(i for i, (a, b) in enumerate(zip(t_toks, k_toks, strict=True)) if a != b)
        lines.append(f"  first divergence at decode step {first}")
    return lines


@app.function(gpu=_GPU, image=image, timeout=1500)
def bench() -> str:
    import torch

    report = [f"GPU: {torch.cuda.get_device_name(0)}"]
    report.append("== parity ==")
    report.extend(_parity_sweep())
    report.append("== op microbench (64q/4kv/128, topk 16, index block 128) ==")
    report.extend(_op_microbench())
    report.append("== end-to-end A/B (synthetic M3-shaped, 16q/1kv/128) ==")
    report.extend(_e2e_ab(ctx_len=32768, decode_steps=64))
    return "\n".join(report)


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
