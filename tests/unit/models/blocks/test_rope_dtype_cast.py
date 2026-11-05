"""`RotaryEmbedding.inv_freq` stays fp32 when a model is cast to a low dtype.

`load_model` builds each model with `model_cls(cfg).to(device=..., dtype=...)`.
`nn.Module.to(dtype)` casts every floating point buffer, and a bf16 `inv_freq`
moves the RoPE angle at position p by p times its rounding error: about 1.8 rad
at 1000 tokens and 14 rad at 8000 with head_dim 128 and theta 500000.
transformers builds the table in fp32 when it loads a checkpoint in bf16. These
tests check that the cast keeps the fp32 table, that device moves still apply,
and that the bf16 cos/sin tables at long positions match the fp32 reference.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from mini_infer.models.blocks.rope import RotaryEmbedding

# Llama 3.1 RoPE values: the widest angles of the families in the registry.
_HEAD_DIM = 128
_THETA = 500000.0
_LONG_POSITIONS = (0, 1, 100, 1000, 8000, 10000)

_TINY_DIMS = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}

_CASTS: dict[str, Callable[[nn.Module], nn.Module]] = {
    "to_bf16": lambda module: module.to(dtype=torch.bfloat16),
    "to_device_and_bf16": lambda module: module.to(device="cpu", dtype=torch.bfloat16),
    "to_fp16": lambda module: module.to(torch.float16),
    "half": lambda module: module.half(),
    "bfloat16": lambda module: module.bfloat16(),
}


def _tiny_model(family: str) -> nn.Module:
    from mini_infer.models.gemma.gemma3 import Gemma3Config, Gemma3ForCausalLM
    from mini_infer.models.llama.llama import LlamaConfig, LlamaForCausalLM
    from mini_infer.models.mistral.mixtral import MixtralConfig, MixtralForCausalLM
    from mini_infer.models.qwen.qwen2 import Qwen2Config, Qwen2ForCausalLM
    from mini_infer.models.qwen.qwen3 import Qwen3Config, Qwen3ForCausalLM

    dims = {**_TINY_DIMS, "rope_theta": _THETA}
    if family == "llama":
        return LlamaForCausalLM(LlamaConfig(**dims))
    if family == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(**dims))
    if family == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(**dims))
    if family == "mixtral":
        return MixtralForCausalLM(MixtralConfig(**dims, num_local_experts=4, num_experts_per_tok=2))
    if family == "gemma3":
        return Gemma3ForCausalLM(
            Gemma3Config(
                **_TINY_DIMS,
                rope_theta_local=10000.0,
                rope_theta_global=1000000.0,
                sliding_window=4,
                layer_types=["sliding_attention", "full_attention"],
                query_pre_attn_scalar=8,
            )
        )
    raise ValueError(family)


def _rotary_tables(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.inv_freq
        for name, module in model.named_modules()
        if isinstance(module, RotaryEmbedding)
    }


@pytest.mark.parametrize("cast", sorted(_CASTS))
@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3", "mixtral", "gemma3"])
def test_dtype_cast_keeps_every_rotary_table_fp32(family: str, cast: str) -> None:
    """The weights take the new dtype; each `inv_freq` keeps its fp32 values."""
    model = _tiny_model(family)
    before = {name: table.clone() for name, table in _rotary_tables(model).items()}
    assert before, f"{family} has no RotaryEmbedding"

    result = _CASTS[cast](model)

    assert result is model
    assert {param.dtype for param in model.parameters()} <= {torch.bfloat16, torch.float16}
    after = _rotary_tables(model)
    assert after.keys() == before.keys()
    for name, table in after.items():
        assert table.dtype == torch.float32, (family, name)
        assert torch.equal(table, before[name]), (family, name)


def test_fp32_round_trip_keeps_the_exact_table() -> None:
    """bf16 -> fp32 does not bring back bf16-rounded values."""
    rotary = RotaryEmbedding(_HEAD_DIM, base=_THETA)
    expected = rotary.inv_freq.clone()
    rotary.bfloat16().float().half().to(torch.float32)
    assert rotary.inv_freq.dtype == torch.float32
    assert torch.equal(rotary.inv_freq, expected)


def test_meta_construction_keeps_the_table_fp32() -> None:
    """A model built on `meta` (the DeepSeek-V4 loader) keeps an fp32 table
    through a dtype cast and `to_empty`, so `rematerialize_rotary_tables`
    fills a buffer of the right dtype."""
    with torch.device("meta"):
        rotary = RotaryEmbedding(_HEAD_DIM, base=_THETA)
    rotary.to(dtype=torch.bfloat16)
    assert rotary.inv_freq.is_meta
    assert rotary.inv_freq.dtype == torch.float32

    rotary.to_empty(device="cpu")
    assert rotary.inv_freq.device == torch.device("cpu")
    assert rotary.inv_freq.dtype == torch.float32


def _tables_at(
    rotary: RotaryEmbedding, dtype: torch.dtype, positions: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.zeros(1, len(positions), 4, dtype=dtype)
    position_ids = torch.tensor([positions])
    return rotary(hidden, position_ids)


def test_bf16_tables_at_long_positions_match_the_fp32_reference() -> None:
    """After a bf16 cast, cos/sin at position 10000 differ from the fp32
    reference only by the rounding of the output to bf16."""
    reference = RotaryEmbedding(_HEAD_DIM, base=_THETA)
    cos_ref, sin_ref = _tables_at(reference, torch.float32, _LONG_POSITIONS)

    model = nn.Sequential(RotaryEmbedding(_HEAD_DIM, base=_THETA)).to(dtype=torch.bfloat16)
    rotary = model[0]
    assert isinstance(rotary, RotaryEmbedding)
    cos, sin = _tables_at(rotary, torch.bfloat16, _LONG_POSITIONS)

    # The tables come out in the activation dtype, as the fused RoPE + KV
    # write kernel needs, and equal the fp32 tables rounded once to bf16.
    assert cos.dtype == sin.dtype == torch.bfloat16
    assert torch.equal(cos, cos_ref.to(torch.bfloat16))
    assert torch.equal(sin, sin_ref.to(torch.bfloat16))
    # Half a bf16 step for values in [0.5, 1) is 2^-9.
    torch.testing.assert_close(cos.float(), cos_ref, rtol=0.0, atol=2**-9)
    torch.testing.assert_close(sin.float(), sin_ref, rtol=0.0, atol=2**-9)

    # A bf16 inv_freq (the old behavior) misses by far more than the rounding.
    rounded = RotaryEmbedding(_HEAD_DIM, base=_THETA)
    rounded.inv_freq = rounded.inv_freq.to(torch.bfloat16)
    cos_rounded, _ = _tables_at(rounded, torch.bfloat16, _LONG_POSITIONS)
    assert (cos_rounded.float() - cos_ref).abs().max().item() > 0.5


# The llama3 rescale of the released Llama 3.1 8B / 70B / 405B configs.
_LLAMA31_ROPE = {
    "rope_type": "llama3",
    "rope_theta": _THETA,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


@pytest.mark.parametrize("rope_type", ["default", "llama3"])
def test_load_model_rotary_tables_equal_transformers(tmp_path: Path, rope_type: str) -> None:
    """`load_model` in bf16 gives the same fp32 `inv_freq` and the same bf16
    cos/sin at long positions as transformers' `from_pretrained` in bf16."""
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM as HFLlamaForCausalLM

    from mini_infer.models import load_model

    rope_kwargs: dict[str, object] = (
        {"rope_parameters": dict(_LLAMA31_ROPE)}
        if rope_type == "llama3"
        else {"rope_theta": _THETA}
    )
    torch.manual_seed(0)
    hf_config = HFLlamaConfig(
        vocab_size=64,
        hidden_size=_HEAD_DIM,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        tie_word_embeddings=False,
        **rope_kwargs,
    )
    HFLlamaForCausalLM(hf_config).save_pretrained(tmp_path)

    hf_model = HFLlamaForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    model = load_model(str(tmp_path), dtype=torch.bfloat16, device="cpu")

    rotary = model.rotary_emb
    assert isinstance(rotary, RotaryEmbedding)
    assert next(model.parameters()).dtype == torch.bfloat16
    hf_inv_freq = hf_model.model.rotary_emb.inv_freq
    assert hf_inv_freq.dtype == rotary.inv_freq.dtype == torch.float32
    assert torch.equal(rotary.inv_freq, hf_inv_freq)

    hidden = torch.zeros(1, len(_LONG_POSITIONS), _HEAD_DIM, dtype=torch.bfloat16)
    position_ids = torch.tensor([_LONG_POSITIONS])
    cos, sin = rotary(hidden, position_ids)
    hf_cos, hf_sin = hf_model.model.rotary_emb(hidden, position_ids)
    assert torch.equal(cos, hf_cos)
    assert torch.equal(sin, hf_sin)
