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


def test_gqa_attention_k_eq_v_skips_v_proj() -> None:
    """`GroupedQueryAttention(attention_k_eq_v=True)` does not construct v_proj.

    Gemma 4 full layers reuse the post-`k_proj` tensor as V. The model
    file relies on `self.v_proj is None` to know it should not build a
    parameter and to filter the corresponding safetensors keys at load.
    """
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
    assert gqa.v_proj is None
    # state_dict should not contain v_proj keys.
    assert not any(k.startswith("v_proj.") for k in gqa.state_dict())

    # Sanity-check the homogeneous (default) path still wires v_proj.
    gqa_default = GroupedQueryAttention(
        hidden_size=64,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
        layer_idx=0,
    )
    assert gqa_default.v_proj is not None
    assert any(k.startswith("v_proj.") for k in gqa_default.state_dict())


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
