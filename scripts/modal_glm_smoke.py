"""GLM-MoE-DSA correctness smoke on a real GPU (Modal).

GLM-5.2 ships only at ~753B, so there's no small checkpoint to load on a single
GPU. This smoke instead validates the *code path* on real CUDA with a synthetic
tiny config: the from-scratch `GlmMoeDsaForCausalLM` must match HF
`GlmMoeDsaForCausalLM` on GPU for (1) a full prefill (logits + greedy argmax)
and (2) multi-step greedy decode through the PagedKVCache + index_k stream. It
also runs a bf16 generation pass (the production dtype) to confirm the GPU
kernels (SDPA, the indexer einsums, the cache) produce finite, sane output.

What this catches that the CPU tests can't: CUDA-specific behavior of the
materialized SDPA path, the index_k cache stream on device, and the bf16 path.

Run with:
    uv run modal run scripts/modal_glm_smoke.py
"""

import modal

_GPU = "L4"  # cheap, modern, bf16-capable; the synthetic model is tiny

app = modal.App("mini-infer-glm-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.6.2",  # the version that ships modeling_glm_moe_dsa
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
        "numpy",
    )
    .add_local_python_source("mini_infer")
)


def _build_models(device: str, dtype):  # type: ignore[no-untyped-def]
    """Tiny HF GlmMoeDsa + a weight-synced mini-infer model, both on `device`."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig as HFConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM as HFModel,
    )

    from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    hf_cfg = HFConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        q_lora_rank=24,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        index_topk=4,
        index_n_heads=2,
        index_head_dim=16,
        indexer_types=["full", "shared", "full", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        hidden_act="silu",
    )
    hf_cfg._attn_implementation = "eager"
    hf_model = HFModel(hf_cfg).to(device=device, dtype=dtype).eval()
    mini = (
        GlmMoeDsaForCausalLM(GlmMoeDsaConfig.from_hf(hf_cfg)).to(device=device, dtype=dtype).eval()
    )
    GlmMoeDsaForCausalLM.load_weights(mini, hf_model.state_dict())
    return hf_cfg, hf_model, mini


def _make_cache(mini, device: str, dtype, num_slots: int = 1, num_blocks: int = 32):  # type: ignore[no-untyped-def]

    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=mini.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=mini.cfg.kv_lora_rank,
        dtype=dtype,
        device=device,
        layer_streams=mini.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    for _ in range(num_slots):
        cache.add_request_slot()
    return cache


def _cos_sim(a, b):  # type: ignore[no-untyped-def]
    import torch

    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().float(), b.flatten().float(), dim=0
        ).item()
    )


@app.function(image=image, gpu=_GPU, timeout=1800)
def smoke() -> dict:
    """Run fp32 prefill + decode parity and a bf16 generation pass on GPU."""
    import torch

    device = "cuda"
    assert torch.cuda.is_available(), "no CUDA device in the container"
    gpu_name = torch.cuda.get_device_name(0)
    result: dict = {"gpu": gpu_name}

    # ---- fp32 full-model prefill + greedy decode parity vs HF ----
    torch.manual_seed(0)
    _, hf_model, mini = _build_models(device, torch.float32)
    prompt = [3, 1, 4, 1, 5]
    n_new = 6

    # HF incremental greedy.
    hf_tokens = list(prompt)
    past = None
    cur = torch.tensor([prompt], device=device, dtype=torch.long)
    with torch.inference_mode():
        for _ in range(n_new):
            out = hf_model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            hf_tokens.append(nxt)
            cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
        hf_prefill_logits = hf_model(
            input_ids=torch.tensor([prompt], device=device, dtype=torch.long), use_cache=False
        ).logits

    # mini: prefill then incremental decode through the PagedKVCache.
    plen = len(prompt)
    cache = _make_cache(mini, device, torch.float32)
    mini_tokens = list(prompt)
    with torch.inference_mode():
        logits = mini(
            input_ids=torch.tensor([prompt], device=device, dtype=torch.long),
            position_ids=torch.arange(plen, device=device).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], device=device, dtype=torch.int32),
        )
        mini_prefill_logits = logits
        nxt = int(logits[0, -1].argmax())
        mini_tokens.append(nxt)
        cache_len = plen
        for _ in range(n_new - 1):
            logits = mini(
                input_ids=torch.tensor([[nxt]], device=device, dtype=torch.long),
                position_ids=torch.tensor([[cache_len]], device=device, dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            )
            cache_len += 1
            nxt = int(logits[0, -1].argmax())
            mini_tokens.append(nxt)

    result["fp32_prefill_cos_sim"] = _cos_sim(hf_prefill_logits, mini_prefill_logits)
    result["fp32_prefill_max_abs_diff"] = float(
        (hf_prefill_logits - mini_prefill_logits).abs().max().item()
    )
    result["fp32_prefill_argmax_match"] = bool(
        torch.equal(hf_prefill_logits.argmax(-1), mini_prefill_logits.argmax(-1))
    )
    result["hf_tokens"] = hf_tokens
    result["mini_tokens"] = mini_tokens
    result["fp32_decode_tokens_match"] = mini_tokens == hf_tokens

    # ---- bf16 production-dtype generation smoke (runs + finite + sane) ----
    torch.manual_seed(0)
    _, _, mini_bf16 = _build_models(device, torch.bfloat16)
    cache_bf16 = _make_cache(mini_bf16, device, torch.bfloat16)
    with torch.inference_mode():
        logits = mini_bf16(
            input_ids=torch.tensor([prompt], device=device, dtype=torch.long),
            position_ids=torch.arange(plen, device=device).unsqueeze(0),
            past_key_values=cache_bf16,
            cu_seqlens_q=torch.tensor([0, plen], device=device, dtype=torch.int32),
        )
        result["bf16_prefill_finite"] = bool(torch.isfinite(logits).all().item())
        bf16_tokens = [int(logits[0, -1].argmax())]
        cache_len = plen
        for _ in range(n_new - 1):
            logits = mini_bf16(
                input_ids=torch.tensor([[bf16_tokens[-1]]], device=device, dtype=torch.long),
                position_ids=torch.tensor([[cache_len]], device=device, dtype=torch.long),
                past_key_values=cache_bf16,
                cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            )
            cache_len += 1
            bf16_tokens.append(int(logits[0, -1].argmax()))
        result["bf16_decode_finite"] = bool(torch.isfinite(logits).all().item())
        result["bf16_tokens"] = bf16_tokens

    return result


@app.local_entrypoint()
def main() -> None:
    r = smoke.remote()
    print("=" * 60)
    print(f"GPU: {r['gpu']}")
    print("--- fp32 parity vs HF GlmMoeDsa ---")
    print(f"  prefill cosine sim:     {r['fp32_prefill_cos_sim']:.6f}")
    print(f"  prefill max abs diff:   {r['fp32_prefill_max_abs_diff']:.2e}")
    print(f"  prefill argmax match:   {r['fp32_prefill_argmax_match']}")
    print(f"  decode tokens match:    {r['fp32_decode_tokens_match']}")
    print(f"    HF:   {r['hf_tokens']}")
    print(f"    mini: {r['mini_tokens']}")
    print("--- bf16 production-dtype smoke ---")
    print(f"  prefill finite:         {r['bf16_prefill_finite']}")
    print(f"  decode finite:          {r['bf16_decode_finite']}")
    print(f"  bf16 tokens:            {r['bf16_tokens']}")
    print("=" * 60)
    ok = (
        r["fp32_prefill_cos_sim"] > 0.999
        and r["fp32_prefill_argmax_match"]
        and r["fp32_decode_tokens_match"]
        and r["bf16_prefill_finite"]
        and r["bf16_decode_finite"]
    )
    print("RESULT:", "PASS" if ok else "FAIL")
