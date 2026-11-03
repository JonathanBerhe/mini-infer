"""Block-level parity vs HF's Qwen2 reference modules.

Each owned block is checked against the corresponding HF Qwen2 module
on shared random input + matching weights. Cosine similarity > 0.999
catches drift in the math (RoPE half-rotation, RMSNorm fp32 promotion,
SwiGLU activation order). These run on CPU; no model download needed.
"""

import pytest
import torch

from mini_infer.models.blocks import RMSNorm, RotaryEmbedding, SwiGLU, apply_rotary_pos_emb


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    return float(torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item())


def test_rmsnorm_matches_hf_qwen2() -> None:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(0)
    hidden = 64
    x = torch.randn(1, 8, hidden, dtype=torch.float32)

    ours = RMSNorm(hidden, eps=1e-6)
    theirs = Qwen2RMSNorm(hidden, eps=1e-6)
    # Sync weights.
    with torch.no_grad():
        theirs.weight.copy_(ours.weight)

    out_ours = ours(x)
    out_theirs = theirs(x)
    assert _cos_sim(out_ours, out_theirs) > 0.999
    # Element-wise close too (RMSNorm is small enough that drift is suspicious).
    assert torch.allclose(out_ours, out_theirs, atol=1e-5)


def test_rope_matches_hf_qwen2() -> None:
    """`RotaryEmbedding` + `apply_rotary_pos_emb` round-trip vs HF's helper."""
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb as hf_apply_rope,
    )

    torch.manual_seed(0)
    head_dim = 64
    num_heads = 4
    total_q = 12
    base = 10000.0

    rope = RotaryEmbedding(head_dim, base=base)
    # `hidden_states` is just used for dtype/device; positions drive the math.
    h = torch.zeros(1, total_q, num_heads * head_dim, dtype=torch.float32)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cos, sin = rope(h, position_ids)

    q = torch.randn(1, num_heads, total_q, head_dim, dtype=torch.float32)
    k = torch.randn(1, num_heads, total_q, head_dim, dtype=torch.float32)

    q_ours, k_ours = apply_rotary_pos_emb(q, k, cos, sin)
    q_theirs, k_theirs = hf_apply_rope(q, k, cos, sin)

    assert _cos_sim(q_ours, q_theirs) > 0.999
    assert _cos_sim(k_ours, k_theirs) > 0.999
    assert torch.allclose(q_ours, q_theirs, atol=1e-5)
    assert torch.allclose(k_ours, k_theirs, atol=1e-5)


def test_rmsnorm_with_scale_false_matches_hf_gemma4() -> None:
    """`RMSNorm(with_scale=False)` matches HF `Gemma4RMSNorm(with_scale=False)`.

    Used by Gemma 4's `v_norm`: a pure RMS rescale with no learnable
    weight. Keeping element-wise parity with HF here matters because any
    drift propagates straight into the V tensor before SDPA.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    torch.manual_seed(0)
    hidden = 64
    x = torch.randn(1, 8, 4, hidden, dtype=torch.float32)  # (B, T, num_kv, head_dim)

    ours = RMSNorm(hidden, eps=1e-6, with_scale=False)
    theirs = Gemma4RMSNorm(hidden, eps=1e-6, with_scale=False)

    # No weight to sync (with_scale=False ⇒ no parameter).
    assert not hasattr(ours, "weight"), "with_scale=False should not allocate `weight`"
    assert not hasattr(theirs, "weight")

    out_ours = ours(x)
    out_theirs = theirs(x)
    assert _cos_sim(out_ours, out_theirs) > 0.999
    assert torch.allclose(out_ours, out_theirs, atol=1e-5)


def test_gqa_attention_k_eq_v_projects_q_and_k_only() -> None:
    """`GroupedQueryAttention(attention_k_eq_v=True)` builds a merged projection
    with q and k rows and no v rows.

    Gemma 4 full layers reuse the post-projection k tensor as V. The loader
    relies on `attention_k_eq_v` to look for no `v_proj` tensor, and the
    Gemma 4 model file drops the checkpoint copies of it.
    """
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.models.blocks import GroupedQueryAttention

    gqa = GroupedQueryAttention(
        hidden_size=64,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
        layer_idx=0,
        attention_k_eq_v=True,
    )
    assert not gqa.qkv_proj.with_v
    assert gqa.qkv_proj.split_sizes == (4 * 16, 2 * 16)
    assert gqa.qkv_proj.weight.shape == ((4 + 2) * 16, 64)
    assert set(gqa.state_dict()) == {"qkv_proj.weight", "o_proj.weight"}

    # The block runs with k standing in for v.
    pool = BlockPool(
        num_blocks=4,
        block_size=4,
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    x = torch.randn(1, 3, 64)
    cos, sin = RotaryEmbedding(16)(x, torch.tensor([[0, 1, 2]]))
    with torch.inference_mode():
        out = gqa(x, (cos, sin), cache, torch.tensor([0, 3], dtype=torch.int32))
    assert out.shape == (1, 3, 64)

    # The homogeneous (default) block projects all three.
    gqa_default = GroupedQueryAttention(
        hidden_size=64,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
        layer_idx=0,
    )
    assert gqa_default.qkv_proj.with_v
    assert gqa_default.qkv_proj.split_sizes == (4 * 16, 2 * 16, 2 * 16)
    assert gqa_default.qkv_proj.weight.shape == ((4 + 2 + 2) * 16, 64)


def test_swiglu_matches_hf_qwen2() -> None:
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(0)
    hidden = 32
    intermediate = 96
    x = torch.randn(1, 4, hidden, dtype=torch.float32)

    cfg = Qwen2Config(hidden_size=hidden, intermediate_size=intermediate)
    ours = SwiGLU(hidden, intermediate)
    theirs = Qwen2MLP(cfg)
    # Sync weights.
    with torch.no_grad():
        ours.gate_proj.weight.copy_(theirs.gate_proj.weight)
        ours.up_proj.weight.copy_(theirs.up_proj.weight)
        ours.down_proj.weight.copy_(theirs.down_proj.weight)

    out_ours = ours(x)
    out_theirs = theirs(x)
    assert _cos_sim(out_ours, out_theirs) > 0.999
    assert torch.allclose(out_ours, out_theirs, atol=1e-5)


def test_fused_rms_norm_knob_reaches_every_norm_of_the_shared_block() -> None:
    """`fused_rms_norm` on the Llama, Qwen2 and Qwen3 configs sets `fused` on every
    RMSNorm the model builds, the per-head q and k norms included. On the CPU the
    flag changes nothing, because the kernel needs CUDA."""
    import dataclasses

    from mini_infer.models.llama.llama import LlamaConfig, LlamaForCausalLM
    from mini_infer.models.qwen.qwen2 import Qwen2Config, Qwen2ForCausalLM
    from mini_infer.models.qwen.qwen3 import Qwen3Config, Qwen3ForCausalLM

    dims = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    families = (
        (LlamaConfig, LlamaForCausalLM, 5),
        (Qwen2Config, Qwen2ForCausalLM, 5),
        (Qwen3Config, Qwen3ForCausalLM, 9),  # plus a q norm and a k norm per layer
    )
    for config_cls, model_cls, expected_norms in families:
        default = config_cls(**dims)
        assert default.fused_rms_norm is True
        for fused in (True, False):
            model = model_cls(dataclasses.replace(default, fused_rms_norm=fused))
            norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
            assert len(norms) == expected_norms, config_cls.__name__
            assert all(norm.fused is fused for norm in norms), config_cls.__name__

    norm_fused = RMSNorm(32, eps=1e-6, fused=True)
    norm_plain = RMSNorm(32, eps=1e-6)
    x = torch.randn(3, 5, 32)
    assert torch.equal(norm_fused(x), norm_plain(x))


def test_gqa_fused_rotary_write_matches_the_torch_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the kernel gate forced on and a PyTorch stand-in for the kernel, the block
    gives the same output and the same cache content as its PyTorch path."""
    import mini_infer.models.blocks.gqa as gqa_module
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.models.blocks import GroupedQueryAttention

    calls: list[int] = []

    def stand_in(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        block_ids: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        q_rot, k_rot = apply_rotary_pos_emb(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2),
            cos.unsqueeze(0),
            sin.unsqueeze(0),
        )
        k_pool[block_ids, slots] = k_rot.transpose(1, 2).squeeze(0)
        v_pool[block_ids, slots] = v
        calls.append(1)
        return q_rot.transpose(1, 2).squeeze(0).contiguous()

    torch.manual_seed(0)
    block = GroupedQueryAttention(
        hidden_size=32, num_q_heads=4, num_kv_heads=2, head_dim=8, qkv_bias=True, layer_idx=0
    )
    rotary = RotaryEmbedding(8)

    def pool() -> BlockPool:
        return BlockPool(
            num_blocks=8,
            block_size=4,
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            attention_backend="torch",
        )

    def run(cache: PagedKVCache) -> torch.Tensor:
        cache.add_request_slot()
        cache.add_request_slot()
        x = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(1))
        # The engine runs every forward in inference mode; the cache write is in place.
        with torch.inference_mode():
            cos, sin = rotary(x, torch.tensor([[0, 1, 2, 0, 1]]))
            out: torch.Tensor = block(
                x, (cos, sin), cache, torch.tensor([0, 3, 5], dtype=torch.int32)
            )
        return out

    plain_cache = PagedKVCache(pool())
    expected = run(plain_cache)
    assert calls == []

    monkeypatch.setattr(gqa_module, "supports_rope_write_kernel", lambda *tensors: True)
    monkeypatch.setattr(gqa_module, "rope_and_write_kv", stand_in)
    fused_cache = PagedKVCache(pool())
    got = run(fused_cache)
    assert calls == [1]
    assert torch.equal(got, expected)
    plain_k, plain_v, _, _ = plain_cache.materialize_packed_kv(0)
    fused_k, fused_v, _, _ = fused_cache.materialize_packed_kv(0)
    assert torch.equal(fused_k, plain_k)
    assert torch.equal(fused_v, plain_v)


_TINY_QWEN2_DIMS = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def test_load_weights_merges_the_hf_q_k_v_tensors() -> None:
    """`load_weights` concatenates the HF `q_proj`, `k_proj` and `v_proj` weights and
    biases into `qkv_proj`, q then k then v, and a block with one of them missing
    fails the load loudly."""
    from mini_infer.models.qwen.qwen2 import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(**_TINY_QWEN2_DIMS)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    hidden, q_size, kv_size = 32, 4 * 8, 2 * 8
    hf_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    hf_parts: dict[str, torch.Tensor] = {}
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}.self_attn."
        for suffix in ("weight", "bias"):
            del hf_state[f"{prefix}qkv_proj.{suffix}"]
            for name, rows in (("q_proj", q_size), ("k_proj", kv_size), ("v_proj", kv_size)):
                shape = (rows, hidden) if suffix == "weight" else (rows,)
                hf_parts[f"{prefix}{name}.{suffix}"] = torch.randn(shape)
    hf_state.update(hf_parts)

    Qwen2ForCausalLM.load_weights(model, hf_state)

    assert not any(".q_proj." in key or ".k_proj." in key or ".v_proj." in key for key in hf_state)
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}.self_attn."
        layer = model.model.layers[i].self_attn.qkv_proj
        for suffix, param in (("weight", layer.weight), ("bias", layer.bias)):
            expected = torch.cat(
                [hf_parts[f"{prefix}{name}.{suffix}"] for name in ("q_proj", "k_proj", "v_proj")]
            )
            assert torch.equal(param.detach(), expected), f"{prefix}{suffix}"

    fresh = Qwen2ForCausalLM(cfg)
    partial = {key: value.detach().clone() for key, value in fresh.state_dict().items()}
    del partial["model.layers.0.self_attn.qkv_proj.weight"]
    partial["model.layers.0.self_attn.q_proj.weight"] = torch.randn(q_size, hidden)
    partial["model.layers.0.self_attn.k_proj.weight"] = torch.randn(kv_size, hidden)
    with pytest.raises(ValueError, match=r"qkv_proj\.weight"):
        Qwen2ForCausalLM.load_weights(fresh, partial)


def test_gqa_splits_the_output_of_a_replaced_projection_layer() -> None:
    """The int8 quantizer swaps `qkv_proj` for an `Int8Linear` that only gives the
    merged output; the block splits it into q, k and v itself."""
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.models.blocks import GroupedQueryAttention
    from mini_infer.quant.int8 import Int8Linear, quantize_model_to_int8

    torch.manual_seed(0)
    block = GroupedQueryAttention(
        hidden_size=32, num_q_heads=4, num_kv_heads=2, head_dim=8, qkv_bias=False, layer_idx=0
    )
    assert quantize_model_to_int8(block) == 2  # qkv_proj and o_proj
    assert isinstance(block.qkv_proj, Int8Linear)

    x = torch.randn(1, 5, 32)
    q, k, v = block._project(x)
    assert v is not None
    assert (q.shape, k.shape, v.shape) == ((1, 5, 32), (1, 5, 16), (1, 5, 16))

    pool = BlockPool(
        num_blocks=4,
        block_size=4,
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    cos, sin = RotaryEmbedding(8)(x, torch.tensor([[0, 1, 2, 3, 4]]))
    with torch.inference_mode():
        out = block(x, (cos, sin), cache, torch.tensor([0, 5], dtype=torch.int32))
    assert out.shape == (1, 5, 32)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3", "mixtral"])
def test_decoder_layers_with_folded_adds_match_the_plain_loop(family: str) -> None:
    """`run_decoder_layers` folds each residual add into the next norm. On the CPU
    every sum is the same `residual + x` as in the plain loop, so the logits and
    the Qwen3 hidden-state taps are bit-equal."""
    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.models.llama.llama import LlamaConfig, LlamaForCausalLM
    from mini_infer.models.mistral.mixtral import MixtralConfig, MixtralForCausalLM
    from mini_infer.models.qwen.qwen2 import Qwen2Config, Qwen2ForCausalLM
    from mini_infer.models.qwen.qwen3 import Qwen3Config, Qwen3ForCausalLM

    dims = {**_TINY_QWEN2_DIMS, "num_hidden_layers": 3}
    builders = {
        "llama": lambda: LlamaForCausalLM(LlamaConfig(**dims)),
        "qwen2": lambda: Qwen2ForCausalLM(Qwen2Config(**dims)),
        "qwen3": lambda: Qwen3ForCausalLM(Qwen3Config(**dims)),
        "mixtral": lambda: MixtralForCausalLM(
            MixtralConfig(**dims, num_local_experts=4, num_experts_per_tok=2)
        ),
    }
    torch.manual_seed(0)
    model = builders[family]().eval()
    # Random norm weights, so a norm that skipped its residual would show.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, RMSNorm) and module.with_scale:
                module.weight.copy_(torch.rand_like(module.weight) + 0.5)

    def cache() -> PagedKVCache:
        dims_kv = model.kv_cache_dims
        pool = BlockPool(
            num_blocks=8,
            block_size=4,
            num_layers=dims_kv.num_layers,
            num_kv_heads=dims_kv.num_kv_heads,
            head_dim=dims_kv.head_dim,
            dtype=torch.float32,
            device="cpu",
            attention_backend="torch",
        )
        made = PagedKVCache(pool)
        made.add_request_slot()
        made.add_request_slot()
        return made

    input_ids = torch.tensor([[3, 9, 27, 5, 11]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    taps = frozenset({0, 2}) if family == "qwen3" else None

    with torch.inference_mode():
        sink: dict[int, torch.Tensor] = {}
        kwargs = {"tap_layers": taps, "hidden_state_sink": sink} if taps else {}
        got = model(input_ids, position_ids, cache(), cu_seqlens, **kwargs)

        plain_cache = cache()
        x = model.model.embed_tokens(input_ids)
        position_embeddings = model.rotary_emb(x, position_ids)
        expected_taps = {}
        for i, layer in enumerate(model.model.layers):
            x = layer(x, position_embeddings, plain_cache, cu_seqlens)
            if taps and i in taps:
                expected_taps[i] = x
        expected = model.lm_head(model.model.norm(x))

    assert torch.equal(got, expected), family
    assert sink.keys() == expected_taps.keys()
    for i, tapped in expected_taps.items():
        assert torch.equal(sink[i], tapped), (family, i)
