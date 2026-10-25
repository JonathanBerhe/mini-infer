"""GLM-5.3 runs on the GLM-5.2 port: the released config, parsed unchanged.

`tests/fixtures/glm53/config.json` is a byte-for-byte copy of `config.json` from
zai-org/GLM-5.3 at revision ba32f8c19c8738e2c57c71798c601d477a7bf844 (sha256
3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69). It differs
from the GLM-5.2-FP8 config (revision f33c6dc501ee5a2c7e35155653b1b1abbc320951)
in one field, `transformers_version`. The two `model.safetensors.index.json`
files at those revisions are identical.

The tests cover three things:

  - `GlmMoeDsaConfig.from_hf` reads the real config the way the engine does.
  - The config fields that `from_hf` does not read hold the values that the
    port hardcodes. A future release that flips one of them fails here.
  - The real geometry (78 layers, 256 routed experts, block-FP8 layout) loads
    through `load_weights` on the meta device, so no weights are needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from mini_infer.models import _raw_config_namespace
from mini_infer.models.glm.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "glm53"

# Text tensors in the released index: 117060, of which 58266 are `weight_scale_inv`.
_RELEASED_TEXT_TENSORS = 117060
_RELEASED_SCALE_TENSORS = 58266
# Weights that ship in BF16 with no scale (the released index has no scale for them).
_UNSCALED_WEIGHTS = ("embed_tokens", "lm_head", "mlp.gate.weight", "indexer.weights_proj")


@pytest.fixture(scope="module")
def raw_config() -> dict[str, Any]:
    with (FIXTURE_DIR / "config.json").open() as config_file:
        loaded: dict[str, Any] = json.load(config_file)
    return loaded


@pytest.fixture(scope="module")
def hf_config() -> object:
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(FIXTURE_DIR))


def test_from_hf_reads_the_released_config(hf_config: object) -> None:
    cfg = GlmMoeDsaConfig.from_hf(hf_config)
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size) == (78, 6144, 154880)
    assert (cfg.intermediate_size, cfg.moe_intermediate_size) == (12288, 2048)
    assert cfg.num_attention_heads == 64
    assert (cfg.q_lora_rank, cfg.kv_lora_rank) == (2048, 512)
    assert (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim) == (192, 64, 256)
    assert (cfg.n_routed_experts, cfg.n_shared_experts, cfg.num_experts_per_tok) == (256, 1, 8)
    assert (cfg.n_group, cfg.topk_group) == (1, 1)
    assert cfg.routed_scaling_factor == 2.5
    assert cfg.norm_topk_prob is True
    assert cfg.rms_norm_eps == 1e-5
    assert (cfg.index_topk, cfg.index_head_dim, cfg.index_n_heads) == (2048, 128, 32)
    assert cfg.tie_word_embeddings is False
    assert cfg.attention_bias is False
    # Plain RoPE with theta 8e6 and no YaRN, so the softmax scale is the plain one.
    assert (cfg.rope_type, cfg.rope_theta, cfg.rope_mscale_all_dim) == ("default", 8e6, 0.0)
    assert cfg.attention_softmax_scale() == 256**-0.5
    # Block-FP8 checkpoint: routed experts stay e4m3-resident.
    assert cfg.expert_dtype == "fp8"
    assert [cfg.is_moe_layer(i) for i in range(5)] == [False, False, False, True, True]
    assert sum(cfg.is_moe_layer(i) for i in range(78)) == 75


def test_raw_config_route_matches_autoconfig(hf_config: object) -> None:
    """The fallback reader in `load_model` gives the same port config as AutoConfig."""
    via_auto = GlmMoeDsaConfig.from_hf(hf_config)
    via_raw = GlmMoeDsaConfig.from_hf(_raw_config_namespace(str(FIXTURE_DIR)))
    assert via_raw == via_auto


def test_explicit_indexer_types_match_freq_and_offset(raw_config: dict[str, Any]) -> None:
    """`indexer_types` is explicit, so `index_topk_freq` and the offset are never read.

    They would only matter if `indexer_types` were absent. Check that they
    describe the same schedule, so ignoring them is safe.
    """
    freq = int(raw_config["index_topk_freq"])
    offset = int(raw_config["index_skip_topk_offset"])
    layers = int(raw_config["num_hidden_layers"])
    assert raw_config["index_topk_pattern"] is None
    schedule = ["full" if max(i - offset + 1, 0) % freq == 0 else "shared" for i in range(layers)]
    assert raw_config["indexer_types"] == schedule
    cfg = GlmMoeDsaConfig.from_hf(_raw_config_namespace(str(FIXTURE_DIR)))
    assert list(cfg.indexer_types) == schedule
    full_layers = [i for i, kind in enumerate(schedule) if kind == "full"]
    assert full_layers[:4] == [0, 1, 2, 6]
    assert len(full_layers) == 21
    # Layer 0 must run the indexer: a shared layer has no top-k to reuse.
    assert cfg.indexer_types[0] == "full"


def test_unread_fields_match_what_the_port_hardcodes(raw_config: dict[str, Any]) -> None:
    """Each field below is not read by `from_hf`. The port fixes its value in code."""
    # Main MLA and indexer both use interleaved RoPE (blocks/mla.py use_interleaved_rope,
    # glm/dsa_indexer.py apply_interleaved_rotary_pos_emb).
    assert raw_config["rope_interleave"] is True
    assert raw_config["indexer_rope_interleave"] is True
    # Router: fp32 sigmoid scores with the aux-loss-free selection bias (GlmNoAuxTcGate).
    assert raw_config["scoring_func"] == "sigmoid"
    assert raw_config["topk_method"] == "noaux_tc"
    assert raw_config["moe_router_dtype"] == "float32"
    # One MTP draft layer at index 78. `load_weights` drops it.
    assert raw_config["num_nextn_predict_layers"] == 1
    assert raw_config["index_share_for_mtp_iteration"] is True
    # The routed-expert and dequant code assume e4m3 with 128x128 scale blocks.
    quant = raw_config["quantization_config"]
    assert isinstance(quant, dict)
    assert (quant["quant_method"], quant["fmt"]) == ("fp8", "e4m3")
    assert quant["weight_block_size"] == [128, 128]
    rope = raw_config["rope_parameters"]
    assert isinstance(rope, dict)
    assert rope == {"rope_theta": 8000000, "rope_type": "default"}
    assert raw_config["architectures"] == ["GlmMoeDsaForCausalLM"]


def _released_layout_on_meta(hf_config: object) -> dict[str, torch.Tensor]:
    """The released checkpoint's tensors as meta tensors.

    Names and shapes come from the transformers reference built on the meta
    device. The reference stacks routed experts in 3D, so each stack splits into
    the per-expert `gate/up/down_proj` layout that the released index uses.
    Every 2-D weight except `_UNSCALED_WEIGHTS` is e4m3 with a ceil-block
    `weight_scale_inv`. The MTP layer (index 78) is added as extra keys.
    """
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM as HFModel,
    )

    with torch.device("meta"):
        reference = HFModel(hf_config)
    shapes: dict[str, tuple[int, ...]] = {}
    for name, tensor in reference.state_dict().items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name[: -len("gate_up_proj")]
            count, fused, hidden = tensor.shape
            for j in range(count):
                shapes[f"{prefix}{j}.gate_proj.weight"] = (fused // 2, hidden)
                shapes[f"{prefix}{j}.up_proj.weight"] = (fused // 2, hidden)
        elif name.endswith(".mlp.experts.down_proj"):
            prefix = name[: -len("down_proj")]
            for j in range(tensor.shape[0]):
                shapes[f"{prefix}{j}.down_proj.weight"] = tuple(tensor.shape[1:])
        else:
            shapes[name] = tuple(tensor.shape)

    def meta(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(*shape, dtype=dtype, device="meta")

    def block_scale(shape: tuple[int, ...]) -> torch.Tensor:
        return meta((math.ceil(shape[0] / 128), math.ceil(shape[1] / 128)), torch.float32)

    layout: dict[str, torch.Tensor] = {}
    for name, shape in shapes.items():
        is_fp8 = (
            name.endswith(".weight")
            and len(shape) == 2
            and not any(part in name for part in _UNSCALED_WEIGHTS)
        )
        if is_fp8:
            layout[name] = meta(shape, torch.float8_e4m3fn)
            layout[name + "_scale_inv"] = block_scale(shape)
        elif name.endswith("e_score_correction_bias"):
            layout[name] = meta(shape, torch.float32)
        else:
            layout[name] = meta(shape, torch.bfloat16)
    # Pin the synthesized layout to the counts read from the released index.
    assert len(layout) == _RELEASED_TEXT_TENSORS
    assert sum(k.endswith(".weight_scale_inv") for k in layout) == _RELEASED_SCALE_TENSORS
    # MTP layer: present in the index, never used by the port.
    gate = layout["model.layers.3.mlp.experts.0.gate_proj.weight"].shape
    layout["model.layers.78.eh_proj.weight"] = meta((6144, 12288), torch.bfloat16)
    layout["model.layers.78.enorm.weight"] = meta((6144,), torch.bfloat16)
    layout["model.layers.78.shared_head.norm.weight"] = meta((6144,), torch.bfloat16)
    layout["model.layers.78.mlp.experts.0.gate_proj.weight"] = meta(
        tuple(gate), torch.float8_e4m3fn
    )
    layout["model.layers.78.mlp.experts.0.gate_proj.weight_scale_inv"] = block_scale(tuple(gate))
    return layout


@pytest.fixture(scope="module")
def released_layout(hf_config: object) -> dict[str, torch.Tensor]:
    return _released_layout_on_meta(hf_config)


def test_released_layout_loads_on_meta_device(
    hf_config: object, released_layout: dict[str, torch.Tensor]
) -> None:
    """Every tensor the released checkpoint carries has a home in the port, with the right shape."""
    cfg = GlmMoeDsaConfig.from_hf(hf_config)
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(cfg)
    # No exception means: nothing missing, nothing unexpected, every shape matches.
    GlmMoeDsaForCausalLM.load_weights(model, dict(released_layout))


def test_meta_load_rejects_an_indexer_on_a_shared_layer(
    hf_config: object, released_layout: dict[str, torch.Tensor]
) -> None:
    """Control: the meta load fails when the checkpoint disagrees with `indexer_types`."""
    cfg = GlmMoeDsaConfig.from_hf(hf_config)
    assert cfg.indexer_is_shared(4)
    broken = dict(released_layout)
    broken["model.layers.4.self_attn.indexer.k_norm.bias"] = torch.empty(
        128, dtype=torch.bfloat16, device="meta"
    )
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(cfg)
    with pytest.raises(ValueError, match="unexpected"):
        GlmMoeDsaForCausalLM.load_weights(model, broken)
