"""Kimi K3 structural tests: wrapper config, checkpoint translation, K3 blocks.

Reference-free (no vendored code needed): these pin the `KimiK3Config`
contract on the raw `config.json` shape (`text_config` as a nested dict),
the loader's tower dropping / prefix stripping / MXFP4 expert dequant, and
the K3-only block math (SiTU, the bounded KDA gate, AttnRes) against direct
transcriptions of the reference formulas.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mini_infer.models import REGISTRY
from mini_infer.models.blocks.activations import situ
from mini_infer.models.blocks.kda import kda_lower_bound_gate
from mini_infer.models.kimi_linear.attn_res import attn_res_mix
from mini_infer.models.kimi_linear.kimi_k3 import (
    KimiK3Config,
    KimiK3ForConditionalGeneration,
)
from mini_infer.models.kimi_linear.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _tiny_text_config(**overrides: Any) -> dict[str, Any]:
    """The K3 `text_config` block at toy scale, as `config.json` spells it."""
    fields: dict[str, Any] = dict(
        model_type="kimi_linear",
        architectures=["KimiLinearForCausalLM"],
        vocab_size=96,
        hidden_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        intermediate_size=64,
        # Expert widths are multiples of 32 so the MXFP4 block scale applies.
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_token=3,
        num_shared_experts=2,
        routed_scaling_factor=1.0,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
        q_lora_rank=12,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=2,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    fields.update(overrides)
    return fields


def _tiny_hf_namespace(**text_overrides: Any) -> SimpleNamespace:
    """The wrapper config as `_raw_config_namespace` yields it: top-level
    attributes, `text_config` left as a plain dict."""
    return SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        model_type="kimi_k3",
        text_config=_tiny_text_config(**text_overrides),
        vision_config={"vt_hidden_size": 8},
    )


def _tiny_model(seed: int = 0) -> KimiK3ForConditionalGeneration:
    torch.manual_seed(seed)
    return KimiK3ForConditionalGeneration(KimiK3Config.from_hf(_tiny_hf_namespace())).float().eval()


def _pack_mxfp4(
    nibbles: torch.Tensor, scale_bytes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack `(out, in)` E2M1 nibble indices two per byte (low nibble first)."""
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scale_bytes.to(torch.uint8)


def _expected_mxfp4(
    nibbles: torch.Tensor, scale_bytes: torch.Tensor, block: int = 32
) -> torch.Tensor:
    table = torch.tensor(_E2M1)
    scale = torch.exp2(scale_bytes.float() - 127.0).repeat_interleave(block, dim=-1)
    return (table[nibbles.long()] * scale).to(torch.bfloat16)


def test_registry_has_kimi_k3() -> None:
    assert REGISTRY.lookup("KimiK3ForConditionalGeneration") is KimiK3ForConditionalGeneration
    assert issubclass(KimiK3ForConditionalGeneration, KimiLinearForCausalLM)
    assert KimiK3ForConditionalGeneration.USES_STATE_CACHE is True


def test_from_hf_reads_text_config_dict() -> None:
    cfg = KimiK3Config.from_hf(_tiny_hf_namespace())
    text = cfg.text
    assert text.hidden_act == "situ"
    assert (text.situ_beta, text.situ_linear_beta) == (4.0, 25.0)
    assert text.q_lora_rank == 12
    assert text.mla_use_output_gate is True
    assert text.kda_use_full_rank_gate is True
    assert text.kda_gate_lower_bound == -5.0
    assert text.routed_expert_hidden_size == 32
    assert text.latent_moe_use_norm is True
    assert text.attn_res_block_size == 2
    assert text.num_shared_experts == 2
    # The K3 modeling file gathers UNBIASED scores as expert weights.
    assert text.moe_weights_from_biased_scores is False


def test_from_hf_accepts_text_config_object() -> None:
    hf = _tiny_hf_namespace()
    hf.text_config = SimpleNamespace(**hf.text_config)
    assert KimiK3Config.from_hf(hf).text.attn_res_block_size == 2


def test_from_hf_requires_text_config() -> None:
    with pytest.raises(ValueError, match="text_config"):
        KimiK3Config.from_hf(SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]))


def test_from_hf_rejects_unknown_hidden_act() -> None:
    with pytest.raises(ValueError, match="hidden_act"):
        KimiK3Config.from_hf(_tiny_hf_namespace(hidden_act="gelu"))


def test_from_hf_rejects_non_positive_attn_res_block() -> None:
    with pytest.raises(ValueError, match="attn_res_block_size"):
        KimiK3Config.from_hf(_tiny_hf_namespace(attn_res_block_size=0))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"q_lora_rank": 0}, "q_lora_rank"),
        ({"routed_expert_hidden_size": 0}, "routed_expert_hidden_size"),
        ({"routed_expert_hidden_size": -4}, "routed_expert_hidden_size"),
    ],
)
def test_from_hf_rejects_non_positive_widths(override: dict[str, Any], match: str) -> None:
    """A zero or negative width must fail in `from_hf`, before any weight loads,
    not as a shape error in the first forward."""
    with pytest.raises(ValueError, match=match):
        KimiK3Config.from_hf(_tiny_hf_namespace(**override))


@pytest.mark.parametrize("lower_bound", [0.0, 2.5])
def test_from_hf_rejects_non_negative_gate_lower_bound(lower_bound: float) -> None:
    linear_attn = dict(_tiny_text_config()["linear_attn_config"], gate_lower_bound=lower_bound)
    with pytest.raises(ValueError, match="gate_lower_bound"):
        KimiK3Config.from_hf(_tiny_hf_namespace(linear_attn_config=linear_attn))


def test_attn_res_query_norm_is_weight_only() -> None:
    """The AttnRes `*_res_norm` modules hold only the scoring weight; calling
    one must fail rather than normalize the candidates a second time."""
    model = _tiny_model()
    norm = model.model.layers[2].mlp_res_norm
    assert [name for name, _ in norm.named_parameters()] == ["weight"]
    with pytest.raises(NotImplementedError):
        norm(torch.zeros(1, 48))


def test_k3_modules_exist_with_checkpoint_names() -> None:
    model = _tiny_model()
    names = set(model.state_dict())
    # KDA layer: full-rank output gate, flat A_log, no low-rank gate pair.
    assert "model.layers.0.self_attn.g_proj.weight" in names
    assert "model.layers.0.self_attn.g_a_proj.weight" not in names
    assert model.model.layers[0].self_attn.A_log.shape == (2,)
    # MLA layer: low-rank Q + output gate.
    for suffix in ("q_a_proj", "q_a_layernorm", "q_b_proj", "g_proj"):
        assert f"model.layers.3.self_attn.{suffix}.weight" in names
    # Latent MoE projections + norm, and the shared experts.
    for suffix in ("routed_expert_down_proj", "routed_expert_up_proj", "routed_expert_norm"):
        assert f"model.layers.1.block_sparse_moe.{suffix}.weight" in names
    # Experts live in the latent width (32), the shared experts in the hidden
    # width (48) at `num_shared_experts * moe_intermediate_size` (64).
    assert model.model.layers[1].block_sparse_moe.experts[0].w1.weight.shape == (32, 32)
    assert model.model.layers[1].block_sparse_moe.routed_expert_down_proj.weight.shape == (32, 48)
    assert model.model.layers[1].block_sparse_moe.shared_experts.gate_proj.weight.shape == (64, 48)
    # AttnRes queries per layer and at the output.
    for suffix in (
        "self_attention_res_norm",
        "self_attention_res_proj",
        "mlp_res_norm",
        "mlp_res_proj",
    ):
        assert f"model.layers.2.{suffix}.weight" in names
    assert "model.output_attn_res_norm.weight" in names
    assert model.model.output_attn_res_proj.weight.shape == (1, 48)


def test_stateless_forward_shape_and_finiteness() -> None:
    model = _tiny_model()
    ids = torch.randint(0, 96, (2, 9))
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (2, 9, 96)
    assert torch.isfinite(logits).all()


def test_decode_matches_stateless_prefill() -> None:
    """Prefill-then-decode through the KimiStateCache reproduces the
    stateless full-sequence logits for the last token (AttnRes carries no
    cross-token state, so the cache contract is unchanged)."""
    model = _tiny_model()
    torch.manual_seed(3)
    ids = torch.randint(0, 96, (1, 7))
    with torch.inference_mode():
        full = model(ids)
        cache = model.build_state_cache(max_seq_len=16)
        model.forward_prefill_with_cache(ids[:, :6], state_cache=cache)
        cache.advance_start_pos(6)
        step = model.forward_decode_with_cache(ids[:, 6:], start_pos=6, state_cache=cache)
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-4)


def _checkpoint_state(
    model: KimiLinearForCausalLM, prefix: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The model's tensors in the CHECKPOINT layout under `prefix`: Conv1d
    `(C, 1, W)` weights, `o_norm.weight`, and random MXFP4 experts
    (`weight_packed` + `weight_scale`). Returns the state and the BF16
    values each expert must dequantize to."""
    torch.manual_seed(5)
    state: dict[str, torch.Tensor] = {}
    expected_experts: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        name = prefix + key
        if key.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            state[name] = value.unsqueeze(1).clone()
        elif key.endswith("o_norm_weight"):
            state[name.replace("o_norm_weight", "o_norm.weight")] = value.clone()
        elif ".experts." in key and key.endswith(".weight"):
            out_dim, in_dim = value.shape
            nibbles = torch.randint(0, 16, (out_dim, in_dim))
            scale_bytes = torch.randint(125, 130, (out_dim, in_dim // 32))
            packed, scale = _pack_mxfp4(nibbles, scale_bytes)
            state[name[: -len(".weight")] + ".weight_packed"] = packed
            state[name[: -len(".weight")] + ".weight_scale"] = scale
            expected_experts[key] = _expected_mxfp4(nibbles, scale_bytes)
        else:
            state[name] = value.clone()
    assert expected_experts, "the tiny config must have MXFP4-shaped experts"
    return state, expected_experts


def test_load_weights_translates_the_checkpoint_layout() -> None:
    """A state_dict in the released CHECKPOINT's layout loads: `language_model.`
    prefix, vision tower + projector present (dropped), and MXFP4 experts
    that must dequantize to the expected BF16 values."""
    model = _tiny_model()
    state, expected_experts = _checkpoint_state(model, "language_model.")
    state["vision_tower.encoder.blocks.0.mlp.fc0.weight"] = torch.zeros(4, 4)
    state["mm_projector.proj.0.weight"] = torch.zeros(4, 4)

    KimiK3ForConditionalGeneration.load_weights(model, state)
    loaded = model.state_dict()
    for key, expected in expected_experts.items():
        assert torch.equal(loaded[key], expected.float()), key
    assert loaded["model.layers.0.self_attn.A_log"].shape == (2,)


def test_text_only_extraction_loads_through_kimi_linear() -> None:
    """A re-export that names the text stack's own architecture
    (`KimiLinearForCausalLM`, no wrapper prefix) still carries MXFP4
    experts; the text stack's loader must dequantize them too."""
    torch.manual_seed(0)
    cfg = KimiLinearConfig.from_hf(SimpleNamespace(**_tiny_text_config()))
    model = KimiLinearForCausalLM(cfg).float().eval()
    state, expected_experts = _checkpoint_state(model, "")

    KimiLinearForCausalLM.load_weights(model, state)
    loaded = model.state_dict()
    for key, expected in expected_experts.items():
        assert torch.equal(loaded[key], expected.float()), key


def test_load_weights_accepts_text_only_keys() -> None:
    """Keys without the `language_model.` prefix (an extracted text model)
    load unchanged."""
    model = _tiny_model()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    KimiK3ForConditionalGeneration.load_weights(model, state)


def test_load_weights_rejects_orphan_packed_weight() -> None:
    model = _tiny_model()
    state = {"language_model." + k: v.clone() for k, v in model.state_dict().items()}
    key = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
    tensor = state.pop(key)
    state[key.replace(".weight", ".weight_packed")] = torch.zeros(
        tensor.shape[0], tensor.shape[1] // 2, dtype=torch.uint8
    )
    with pytest.raises(ValueError, match="weight_packed / weight_scale"):
        KimiK3ForConditionalGeneration.load_weights(model, state)


def test_load_weights_rejects_unknown_keys() -> None:
    model = _tiny_model()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["model.layers.0.self_attn.bogus.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        KimiK3ForConditionalGeneration.load_weights(model, state)


def test_situ_matches_reference_formula() -> None:
    """`situ` == the reference `SituAndMul` on a concatenated gate-up tensor."""
    torch.manual_seed(0)
    gate = torch.randn(5, 7, dtype=torch.bfloat16) * 6
    up = torch.randn(5, 7, dtype=torch.bfloat16) * 40
    beta, linear_beta = 4.0, 25.0

    x = torch.cat([gate, up], dim=-1)
    half = x.shape[-1] // 2
    g32, u32 = x[..., :half].float(), x[..., half:].float()
    situ_a = beta * torch.tanh(g32 / beta) * torch.sigmoid(g32)
    u32 = linear_beta * torch.tanh(u32 / linear_beta)
    expected = (situ_a * u32).to(x.dtype)

    out = situ(gate, up, beta=beta, linear_beta=linear_beta)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)
    # Without linear_beta the up branch passes through.
    plain = situ(gate, up, beta=beta, linear_beta=None)
    assert torch.equal(plain, (situ_a * x[..., half:].float()).to(x.dtype))


def test_lower_bound_gate_is_bounded_and_matches_formula() -> None:
    torch.manual_seed(0)
    raw = torch.randn(2, 3, 4 * 8) * 3
    a_log = torch.randn(4)
    dt_bias = torch.randn(4 * 8)
    out = kda_lower_bound_gate(raw, a_log, dt_bias, 8, -5.0)
    assert out.shape == (2, 3, 4, 8)
    assert out.dtype == torch.float32
    assert bool((out > -5.0).all()) and bool((out < 0.0).all())
    expected = -5.0 * torch.sigmoid(
        a_log.exp().view(4, 1) * (raw.view(2, 3, 4, 8) + dt_bias.view(4, 8))
    )
    assert torch.allclose(out, expected)
    with pytest.raises(ValueError, match="negative"):
        kda_lower_bound_gate(raw, a_log, dt_bias, 8, 0.5)


def test_attn_res_with_no_blocks_is_identity() -> None:
    """A single candidate (the running sum alone) gets softmax weight 1."""
    torch.manual_seed(0)
    prefix = torch.randn(2, 5, 16)
    blocks = prefix.new_zeros(2, 5, 0, 16)
    out = attn_res_mix(prefix, blocks, torch.randn(1, 16), torch.rand(16), 1e-5)
    assert torch.allclose(out, prefix)


def test_attn_res_matches_reference_formula() -> None:
    """`attn_res_mix` over `(batch, seq, blocks, hidden)` == the reference
    `_apply_attn_res` transcription over flattened tokens."""
    torch.manual_seed(1)
    batch, seq, hidden, num_blocks = 2, 3, 16, 4
    prefix = torch.randn(batch, seq, hidden, dtype=torch.bfloat16)
    blocks = torch.randn(batch, seq, num_blocks, hidden, dtype=torch.bfloat16)
    proj_w = torch.randn(1, hidden)
    norm_w = torch.rand(hidden) + 0.5
    eps = 1e-5

    v = torch.cat((blocks.view(-1, num_blocks, hidden), prefix.view(-1, 1, hidden)), dim=1)
    v_float = v.float()
    k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + eps)
    score_weight = norm_w.float() * proj_w.squeeze(0).float()
    probs = (k * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    expected = torch.matmul(probs, v_float).squeeze(1).to(v.dtype).view(batch, seq, hidden)

    out = attn_res_mix(prefix, blocks, proj_w, norm_w, eps)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), expected.float(), atol=1e-2)
