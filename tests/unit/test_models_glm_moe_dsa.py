"""GLM-MoE-DSA owned model: registry, structure, forward, IndexShare, parity.

The 753B real checkpoint is far out of reach for CPU CI, so these tests use a
synthetic mini-config exercising the structurally distinctive bits: MLA streams,
the DSA indexer on "full" layers, heterogeneous dense/MoE FFN, IndexShare top-k
reuse across "shared" layers (which carry no indexer weights), and the
stacked-3D expert weight load. Full-model bit-parity vs HF runs on the same
tiny config (no checkpoint download).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models import REGISTRY
from mini_infer.models.blocks import GlmMoeFFN, SwiGLU
from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM


def _make_cfg(
    *,
    num_hidden_layers: int = 4,
    mlp_layer_types: tuple[str, ...] = ("dense", "dense", "dense", "sparse"),
    indexer_types: tuple[str, ...] = ("full", "full", "full", "full"),
    index_topk: int = 8,
    tie_word_embeddings: bool = False,
) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
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
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=tie_word_embeddings,
        index_topk=index_topk,
        index_head_dim=16,
        index_n_heads=2,
        mlp_layer_types=mlp_layer_types,
        indexer_types=indexer_types,
    )


def _make_cache(
    model: GlmMoeDsaForCausalLM, num_slots: int = 1, num_blocks: int = 8
) -> PagedKVCache:
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=model.cfg.kv_lora_rank,
        dtype=torch.float32,
        device="cpu",
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    for _ in range(num_slots):
        cache.add_request_slot()
    return cache


def _make_paged_cache(model: GlmMoeDsaForCausalLM) -> PagedKVCache:
    return _make_cache(model)


def test_registry_has_glm_moe_dsa() -> None:
    assert REGISTRY.lookup("GlmMoeDsaForCausalLM") is GlmMoeDsaForCausalLM


def test_per_layer_streams_reports_mla_shape() -> None:
    # full/shared/full/shared: full layers carry an extra index_k cache stream.
    cfg = _make_cfg(indexer_types=("full", "shared", "full", "shared"))
    model = GlmMoeDsaForCausalLM(cfg)
    streams = model.per_layer_streams()
    assert len(streams) == cfg.num_hidden_layers
    for layer_idx, layer_streams in enumerate(streams):
        by_name = {s.name: s for s in layer_streams}
        # MLA streams are always present.
        assert by_name["kv_latent"].head_dim == cfg.kv_lora_rank
        assert by_name["k_rope"].head_dim == cfg.qk_rope_head_dim
        if cfg.indexer_is_shared(layer_idx):
            assert "index_k" not in by_name
        else:
            # index_k goes first so layer 0 triggers block allocation.
            assert layer_streams[0].name == "index_k"
            assert by_name["index_k"].head_dim == cfg.index_head_dim


def test_required_attention_backend_is_torch() -> None:
    assert GlmMoeDsaForCausalLM(_make_cfg()).required_attention_backend() == "torch"


def test_dense_and_sparse_ffn_dispatch() -> None:
    """mlp_layer_types controls the FFN type per layer."""
    cfg = _make_cfg(
        mlp_layer_types=("dense", "dense", "dense", "sparse"),
    )
    model = GlmMoeDsaForCausalLM(cfg)
    assert isinstance(model.model.layers[0].mlp, SwiGLU)
    assert isinstance(model.model.layers[2].mlp, SwiGLU)
    assert isinstance(model.model.layers[3].mlp, GlmMoeFFN)


def test_config_helpers() -> None:
    cfg = _make_cfg(
        mlp_layer_types=("dense", "dense", "dense", "sparse"),
        indexer_types=("full", "shared", "shared", "shared"),
    )
    assert cfg.is_moe_layer(0) is False
    assert cfg.is_moe_layer(3) is True
    assert cfg.indexer_is_shared(0) is False
    assert cfg.indexer_is_shared(1) is True


def test_forward_runs_through_paged_cache() -> None:
    """End-to-end forward: dense + MoE layers, MLA + DSA, lm_head; finite logits."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(cfg).to(torch.float32).eval()
    cache = _make_paged_cache(model)

    total_q = 5
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )
    assert logits.shape == (1, total_q, cfg.vocab_size)
    assert torch.all(torch.isfinite(logits))


def _run_and_count_indexer_calls(cfg: GlmMoeDsaConfig) -> list[int]:
    """Run one forward, returning how many times each layer's indexer fired.

    A "full" layer should compute its own top-k (count 1); a "shared" layer
    should reuse the prior selection (count 0).
    """
    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(cfg).to(torch.float32).eval()
    cache = _make_paged_cache(model)
    total_q = 5
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)

    # Count indexer firings per layer by wrapping each layer's compute_dsa_topk.
    counts = [0] * cfg.num_hidden_layers
    for i, layer in enumerate(model.model.layers):

        def make_wrapper(idx: int, fn):  # type: ignore[no-untyped-def]
            def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                counts[idx] += 1
                return fn(*args, **kwargs)

            return wrapper

        layer.self_attn.compute_dsa_topk = make_wrapper(i, layer.self_attn.compute_dsa_topk)  # type: ignore[method-assign]

    with torch.inference_mode():
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )
    return counts


def test_indexshare_shared_layers_reuse_full_layer_selection() -> None:
    """full,shared,shared,shared: only the full layer runs the indexer."""
    cfg = _make_cfg(
        num_hidden_layers=4,
        mlp_layer_types=("dense", "dense", "dense", "dense"),
        indexer_types=("full", "shared", "shared", "shared"),
    )
    assert _run_and_count_indexer_calls(cfg) == [1, 0, 0, 0]


def test_indexshare_recomputes_at_each_full_layer() -> None:
    """full,shared,full,shared: each full layer recomputes; shared layers reuse."""
    cfg = _make_cfg(
        num_hidden_layers=4,
        mlp_layer_types=("dense", "dense", "dense", "dense"),
        indexer_types=("full", "shared", "full", "shared"),
    )
    assert _run_and_count_indexer_calls(cfg) == [1, 0, 1, 0]


def test_full_model_parity_vs_hf() -> None:
    """Round-trip HF state_dict through load_weights, then full-model logit parity.

    Exercises everything at once: from_hf, the stacked-3D expert weight load,
    dense+MoE layers, MLA+DSA with selective top-k, and IndexShare across
    full/shared layers. The greedy argmax matching at every position is the
    correctness gate; logits also match numerically in fp32.
    """
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig as HFConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM as HFModel,
    )

    torch.manual_seed(0)
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
        index_topk=4,  # selective: DSA actually drops keys
        index_n_heads=2,
        index_head_dim=16,
        # full/shared/full/shared exercises IndexShare reuse end to end.
        indexer_types=["full", "shared", "full", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        hidden_act="silu",
    )
    hf_cfg._attn_implementation = "eager"
    hf_model = HFModel(hf_cfg).to(torch.float32).eval()

    my_cfg = GlmMoeDsaConfig.from_hf(hf_cfg)
    my_model = GlmMoeDsaForCausalLM(my_cfg).to(torch.float32).eval()
    GlmMoeDsaForCausalLM.load_weights(my_model, hf_model.state_dict())

    total_q = 6
    input_ids = torch.randint(0, my_cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)

    with torch.inference_mode():
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        cache = _make_paged_cache(my_model)
        my_logits = my_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )

    assert hf_logits.shape == my_logits.shape
    cs = float(
        torch.nn.functional.cosine_similarity(hf_logits.flatten(), my_logits.flatten(), dim=0)
    )
    assert cs > 0.999, f"full-model logit parity failed: cos_sim={cs:.6f}"
    # Greedy decode trajectory must be identical at every position.
    assert torch.equal(hf_logits.argmax(dim=-1), my_logits.argmax(dim=-1))
    assert torch.allclose(hf_logits, my_logits, atol=1e-3), (
        f"max_abs_diff={(hf_logits - my_logits).abs().max().item():.6f}"
    )


def _build_hf_and_mine() -> tuple[Any, GlmMoeDsaForCausalLM]:
    """A tiny HF model + a weight-synced mini-infer model (shared by parity tests)."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig as HFConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM as HFModel,
    )

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
    hf_model = HFModel(hf_cfg).to(torch.float32).eval()
    my_model = GlmMoeDsaForCausalLM(GlmMoeDsaConfig.from_hf(hf_cfg)).to(torch.float32).eval()
    GlmMoeDsaForCausalLM.load_weights(my_model, hf_model.state_dict())
    return hf_model, my_model


def _hf_greedy(hf_model: Any, prompt: list[int], n_new: int) -> list[int]:
    """HF incremental greedy decode (its own DynamicCache + indexer key cache)."""
    tokens = list(prompt)
    past = None
    cur = torch.tensor([prompt], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(n_new):
            out = hf_model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long)
    return tokens


def _mine_batched_generate(
    model: GlmMoeDsaForCausalLM, prompts: list[list[int]], n_new: int
) -> list[list[int]]:
    """Greedy-generate all prompts together through one shared PagedKVCache.

    Prefill is one ragged packed forward; each decode step is one packed
    forward of B tokens (one per request) at their own positions. Mirrors how
    the continuous-batching scheduler drives `forward_step_packed`.
    """
    batch = len(prompts)
    cache = _make_cache(model, num_slots=batch, num_blocks=32)
    gen = [list(p) for p in prompts]
    cur_len = [len(p) for p in prompts]

    packed = [tok for p in prompts for tok in p]
    cu = [0]
    pos: list[int] = []
    for p in prompts:
        cu.append(cu[-1] + len(p))
        pos.extend(range(len(p)))

    with torch.inference_mode():
        logits = model(
            input_ids=torch.tensor([packed], dtype=torch.long),
            position_ids=torch.tensor([pos], dtype=torch.long),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        )
        nxt = [int(logits[0, cu[b + 1] - 1].argmax()) for b in range(batch)]
        for b in range(batch):
            gen[b].append(nxt[b])

        for _ in range(n_new - 1):
            logits = model(
                input_ids=torch.tensor([nxt], dtype=torch.long),
                position_ids=torch.tensor([[cur_len[b] for b in range(batch)]], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor(list(range(batch + 1)), dtype=torch.int32),
            )
            nxt = [int(logits[0, b].argmax()) for b in range(batch)]
            for b in range(batch):
                gen[b].append(nxt[b])
                cur_len[b] += 1
    return gen


def test_greedy_decode_parity_vs_hf() -> None:
    """Multi-step greedy decode matches HF token-for-token.

    HF decodes incrementally with its DynamicCache + the indexer's own key
    cache; mini-infer decodes through the PagedKVCache including the new
    index_k stream. Matching tokens proves decode-time indexer caching is
    correct (index_topk=4 < context, so selection bites as the context grows).
    """
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()

    prompt = [3, 1, 4, 1, 5]
    n_new = 6

    hf_tokens = _hf_greedy(hf_model, prompt, n_new)

    # mini-infer: prefill, then one-token decode steps through the PagedKVCache.
    my_tokens = list(prompt)
    cache = _make_paged_cache(my_model)
    plen = len(prompt)
    with torch.inference_mode():
        logits = my_model(
            input_ids=torch.tensor([prompt], dtype=torch.long),
            position_ids=torch.arange(plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], dtype=torch.int32),
        )
        nxt = int(logits[0, -1].argmax())
        my_tokens.append(nxt)
        cache_len = plen
        for _ in range(n_new - 1):
            logits = my_model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long),
                position_ids=torch.tensor([[cache_len]], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            )
            cache_len += 1
            nxt = int(logits[0, -1].argmax())
            my_tokens.append(nxt)

    assert my_tokens == hf_tokens, f"HF {hf_tokens} vs ours {my_tokens}"


def test_batched_decode_matches_hf() -> None:
    """Continuous-batching: two ragged-length prompts decoded together match HF.

    Exercises per-request cu_seqlens in the indexer and SDPA, the index_k stream
    per cache slot, and block allocation across slots. Each request's tokens
    must equal HF run on that prompt alone.
    """
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()

    prompts = [[3, 1, 4, 1, 5], [2, 7, 1, 8]]  # different lengths
    n_new = 5

    hf_gen = [_hf_greedy(hf_model, p, n_new) for p in prompts]
    my_gen = _mine_batched_generate(my_model, prompts, n_new)

    assert my_gen == hf_gen, f"HF {hf_gen} vs ours {my_gen}"
