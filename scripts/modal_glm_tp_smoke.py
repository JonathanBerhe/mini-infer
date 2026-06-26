"""GLM-MoE-DSA tensor-parallel smoke on 2 GPUs under real NCCL (Modal).

Validates that the GLM attention + DSA-indexer path shards correctly under TP on
real hardware: the world_size=2 forward must produce logits identical across
ranks (TP consistency, via the indexer all-reduce + o_proj row-parallel
all-reduce) and matching a world_size=1 reference (correctness). The CPU gloo
test (`test_glm_dsa_indexer_tp_parity`) covers the indexer selection in
isolation; this exercises the full model on the NCCL interconnect.

Config: dense layers 0-2 + a sparse (MoE) layer, so this exercises the
expert-parallel weight load (`load_weights`'s global->local expert remap) on top
of the MLA + DSA-indexer TP path. Each rank holds its slice of routed experts;
the all-reduces (indexer score, MoE routed sum, o_proj) must agree across ranks.

Run with:
    uv run modal run scripts/modal_glm_tp_smoke.py
"""

import modal

_GPU = "L4:2"

app = modal.App("mini-infer-glm-tp-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.6.2",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
        "numpy",
    )
    .add_local_python_source("mini_infer")
)

_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]  # 8 tokens, exercises index_topk=4 selection


def _make_hf_config():  # type: ignore[no-untyped-def]
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig as HFConfig,
    )

    hf_cfg = HFConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,  # 4 / 2 = 2 heads per rank
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
        index_n_heads=2,  # 2 / 2 = 1 indexer head per rank
        index_head_dim=16,
        # Dense layers 0-2 + a sparse MoE layer: exercises expert-parallel load.
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        indexer_types=["full", "shared", "full", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        hidden_act="silu",
    )
    hf_cfg._attn_implementation = "eager"
    return hf_cfg


def _build_mini(hf_cfg, device: str):  # type: ignore[no-untyped-def]
    import torch

    from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    cfg = GlmMoeDsaConfig.from_hf(hf_cfg)
    return GlmMoeDsaForCausalLM(cfg).to(device=device, dtype=torch.float32).eval()


_STATE_PATH = "/tmp/glm_hf_state.pt"


def _prefill_logits_list(mini, device: str) -> list:  # type: ignore[no-untyped-def]
    """Run one prefill of _PROMPT; return logits[0] as a plain nested list.

    Returning a Python list (not a tensor) is deliberate: tensors crossing an
    mp.Queue use torch's shared-memory resource_sharer, which breaks when the
    producing process exits (ConnectionResetError). Plain lists pickle cleanly.
    """
    import torch

    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache

    pool = BlockPool(
        num_blocks=16,
        block_size=4,
        num_layers=mini.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=mini.cfg.kv_lora_rank,
        dtype=torch.float32,
        device=device,
        layer_streams=mini.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    plen = len(_PROMPT)
    with torch.inference_mode():
        logits = mini(
            input_ids=torch.tensor([_PROMPT], device=device, dtype=torch.long),
            position_ids=torch.arange(plen, device=device).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], device=device, dtype=torch.int32),
        )
    return logits[0].detach().float().cpu().tolist()


def _run_rank(rank: int, world_size: int, state_path: str) -> list:
    """Per-rank TP forward: build sharded model, load full weights from file, prefill."""
    import torch
    import torch.distributed as dist

    from mini_infer.distributed.group import destroy_distributed, init_distributed
    from mini_infer.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    init_distributed(
        world_size=world_size,
        rank=rank,
        backend="nccl",
        master_addr="127.0.0.1",
        master_port=29500,
    )
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        state_dict = torch.load(state_path, map_location="cpu")
        mini = _build_mini(_make_hf_config(), device)
        # load_weights shards the full HF state_dict per rank: column/row/vocab
        # parallel for attention/indexer/lm_head, expert-parallel for routed experts.
        GlmMoeDsaForCausalLM.load_weights(mini, state_dict)
        return _prefill_logits_list(mini, device)
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def _child_entry(rank: int, world_size: int, state_path: str, queue) -> None:  # type: ignore[no-untyped-def]
    try:
        queue.put(("ok", rank, _run_rank(rank, world_size, state_path)))
    except Exception:
        import traceback

        queue.put(("err", rank, traceback.format_exc()))


@app.function(image=image, gpu=_GPU, timeout=1800)
def smoke() -> dict:
    import torch
    import torch.multiprocessing as mp
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM as HFModel,
    )

    from mini_infer.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    torch.manual_seed(0)
    hf_cfg = _make_hf_config()
    hf_model = HFModel(hf_cfg).to("cpu", torch.float32).eval()
    state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}
    torch.save(state_dict, _STATE_PATH)  # share with workers via file, not mp args

    # world_size=1 reference on cuda:0 (no process group), freed before the
    # workers spawn. Computing the reference on GPU (not CPU) makes ws=2-vs-ws=1
    # isolate tensor parallelism: the MoE expert dispatch has a GPU-vs-CPU fp32
    # gap (index_add_ / scaled matmuls) that would otherwise mask TP parity.
    ref_mini = _build_mini(hf_cfg, "cuda:0")
    GlmMoeDsaForCausalLM.load_weights(ref_mini, state_dict)
    ref_logits = torch.tensor(_prefill_logits_list(ref_mini, "cuda:0"))
    del ref_mini
    torch.cuda.empty_cache()

    # world_size=2 under NCCL on the 2 GPUs. Workers return plain lists, not
    # tensors (tensors over an mp.Queue break when the producer exits).
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_child_entry, args=(rank, 2, _STATE_PATH, queue)) for rank in range(2)
    ]
    for p in procs:
        p.start()
    try:
        raw = [queue.get(timeout=900) for _ in range(2)]
    finally:
        for p in procs:
            p.join(timeout=10)
    errors = [payload for status, _, payload in raw if status == "err"]
    if errors:
        raise RuntimeError("\n".join(errors))
    by_rank = {rank: torch.tensor(payload) for _, rank, payload in raw}
    rank0, rank1 = by_rank[0], by_rank[1]

    def _cos(a, b):  # type: ignore[no-untyped-def]
        return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))

    return {
        "ranks_consistent": bool(torch.allclose(rank0, rank1, atol=1e-4)),
        "rank0_vs_rank1_max_abs_diff": float((rank0 - rank1).abs().max()),
        "rank0_vs_ref_cos_sim": _cos(rank0, ref_logits),
        "rank0_vs_ref_argmax_match": bool(torch.equal(rank0.argmax(-1), ref_logits.argmax(-1))),
        "rank0_vs_ref_max_abs_diff": float((rank0 - ref_logits).abs().max()),
    }


@app.local_entrypoint()
def main() -> None:
    r = smoke.remote()
    print("=" * 60)
    print("GLM-MoE-DSA tensor-parallel smoke (2x L4, NCCL, MoE config)")
    print(f"  ranks consistent (rank0==rank1):  {r['ranks_consistent']}")
    print(f"    max abs diff rank0 vs rank1:    {r['rank0_vs_rank1_max_abs_diff']:.2e}")
    print(f"  rank0 vs ws=1 ref cosine sim:     {r['rank0_vs_ref_cos_sim']:.6f}")
    print(f"  rank0 vs ws=1 ref argmax match:   {r['rank0_vs_ref_argmax_match']}")
    print(f"    max abs diff rank0 vs ref:      {r['rank0_vs_ref_max_abs_diff']:.2e}")
    print("=" * 60)
    ok = (
        r["ranks_consistent"]
        and r["rank0_vs_ref_cos_sim"] > 0.999
        and r["rank0_vs_ref_argmax_match"]
    )
    print("RESULT:", "PASS" if ok else "FAIL")
