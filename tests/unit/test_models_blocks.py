"""Block-level parity vs HF's Qwen2 reference modules.

Each owned block is checked against the corresponding HF Qwen2 module
on shared random input + matching weights. Cosine similarity > 0.999
catches drift in the math (RoPE half-rotation, RMSNorm fp32 promotion,
SwiGLU activation order). These run on CPU; no model download needed.
"""

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
