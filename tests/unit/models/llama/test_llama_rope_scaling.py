"""The llama3 RoPE rescale of Llama 3.1, 3.2 and 3.3 matches transformers.

Llama 3.1+ configs set `rope_type: "llama3"`, which divides the low-frequency
`inv_freq` entries by `factor` and blends a middle band. The tiny model here
uses `original_max_position_embeddings=64` and `head_dim=16`, so its eight
frequencies fall in all three bands (one kept, two blended, five divided).
RoPE depends only on the distance between positions, so the 192-token
prompt (not an offset on the positions) is what puts query-key distances
past the 64-token wavelength. The prompt runs in two chunks through the
paged cache, so the second chunk starts at position 96. Peaked attention
(large q/k weights) makes the logits feel each change of rotation angle: the
same weights with plain RoPE miss transformers by far more than the tolerance.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models.blocks import Llama3RopeScaling, RotaryEmbedding
from mini_infer.models.llama.llama import LlamaConfig, LlamaForCausalLM

transformers = pytest.importorskip("transformers")
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

VOCAB = 128
HEAD_DIM = 16
SEQ_LEN = 192
CHUNK = 96

_LLAMA3_TINY = {
    "rope_type": "llama3",
    "rope_theta": 10000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}
# The values of the released Llama 3.1 8B / 70B / 405B and Llama 3.3 70B configs.
_LLAMA31 = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
# Llama 3.2 1B / 3B use factor 32.
_LLAMA32 = {**_LLAMA31, "factor": 32.0}


def _hf_config(rope_parameters: dict[str, object] | None) -> transformers.LlamaConfig:
    kwargs: dict[str, object] = {}
    if rope_parameters is not None:
        kwargs["rope_parameters"] = dict(rope_parameters)
    return transformers.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        **kwargs,
    )


def _pair(
    rope_parameters: dict[str, object] | None,
) -> tuple[transformers.LlamaForCausalLM, LlamaForCausalLM]:
    torch.manual_seed(0)
    hf_cfg = _hf_config(rope_parameters)
    hf_model = transformers.LlamaForCausalLM(hf_cfg).eval().float()
    hf_model.config._attn_implementation = "eager"
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" in name:
                param.uniform_(0.5, 1.5)
            elif name.endswith(("q_proj.weight", "k_proj.weight")):
                # Large q/k weights give peaked attention, so the logits
                # feel every change of the rotation angle.
                param.uniform_(-0.5, 0.5)
            else:
                param.uniform_(-0.2, 0.2)
    ours = LlamaForCausalLM(LlamaConfig.from_hf(hf_cfg)).eval().float()
    state = {name: tensor.clone() for name, tensor in hf_model.state_dict().items()}
    LlamaForCausalLM.load_weights(ours, state)
    return hf_model, ours


def _paged_logits(model: LlamaForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    """Logits of `ids` (1, SEQ_LEN) in two prefill chunks through a paged cache."""
    dims = model.kv_cache_dims
    pool = BlockPool(
        num_blocks=SEQ_LEN // 16 + 1,
        block_size=16,
        num_layers=dims.num_layers,
        num_kv_heads=dims.num_kv_heads,
        head_dim=dims.head_dim,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    chunks = []
    with torch.inference_mode():
        for start in range(0, SEQ_LEN, CHUNK):
            chunk_ids = ids[:, start : start + CHUNK]
            positions = torch.arange(start, start + chunk_ids.shape[1]).unsqueeze(0)
            cu_seqlens = torch.tensor([0, chunk_ids.shape[1]], dtype=torch.int32)
            chunks.append(model(chunk_ids, positions, cache, cu_seqlens))
    return torch.cat(chunks, dim=1)


@pytest.mark.parametrize("rope_parameters", [_LLAMA31, _LLAMA32, _LLAMA3_TINY])
def test_llama3_inv_freq_equals_transformers(rope_parameters: dict[str, object]) -> None:
    hf_cfg = _hf_config(rope_parameters)
    expected, _ = ROPE_INIT_FUNCTIONS["llama3"](hf_cfg, "cpu")
    cfg = LlamaConfig.from_hf(hf_cfg)
    got = RotaryEmbedding(cfg.head_dim, base=cfg.rope_theta, llama3_scaling=cfg.rope_scaling)
    assert torch.equal(got.inv_freq, expected.float())


def test_llama3_scaled_logits_match_transformers() -> None:
    hf_model, ours = _pair(_LLAMA3_TINY)
    assert ours.cfg.rope_scaling == Llama3RopeScaling(
        factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=64
    )
    ids = torch.randint(0, VOCAB, (1, SEQ_LEN), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = hf_model(input_ids=ids).logits
    got = _paged_logits(ours, ids)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    # The same weights with plain RoPE drift far past the tolerance, so the
    # comparison above does see the rescaled frequencies.
    plain = LlamaForCausalLM(dataclasses.replace(ours.cfg, rope_scaling=None)).eval().float()
    plain.load_state_dict(ours.state_dict())
    drift = (_paged_logits(plain, ids) - expected).abs().max().item()
    assert drift > 0.5, drift


def test_plain_llama2_config_is_unchanged() -> None:
    """A config without rope scaling builds the table it built before."""
    hf_model, ours = _pair(None)
    assert ours.cfg.rope_scaling is None
    assert ours.cfg.rope_theta == 10000.0
    assert torch.equal(ours.rotary_emb.inv_freq, RotaryEmbedding(HEAD_DIM, base=10000.0).inv_freq)
    ids = torch.randint(0, VOCAB, (1, SEQ_LEN), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = hf_model(input_ids=ids).logits
    torch.testing.assert_close(_paged_logits(ours, ids), expected, atol=1e-4, rtol=1e-4)


def _raw_config(**fields: object) -> SimpleNamespace:
    """A `config.json` read as a namespace, as `load_model` does without AutoConfig."""
    base: dict[str, object] = {
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    return SimpleNamespace(**{**base, **fields})


def test_raw_config_reads_rope_scaling_next_to_rope_theta() -> None:
    """transformers 4.x and raw config.json files keep the scaling in `rope_scaling`."""
    scaling = {key: value for key, value in _LLAMA31.items() if key != "rope_theta"}
    cfg = LlamaConfig.from_hf(_raw_config(rope_theta=500000.0, rope_scaling=scaling))
    assert cfg.rope_theta == 500000.0
    assert cfg.rope_scaling == Llama3RopeScaling(
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )
    plain = LlamaConfig.from_hf(_raw_config(rope_theta=500000.0, rope_scaling=None))
    assert plain.rope_theta == 500000.0
    assert plain.rope_scaling is None


@pytest.mark.parametrize(
    "hf_config",
    [
        _raw_config(rope_theta=10000.0, rope_scaling={"type": "linear", "factor": 2.0}),
        _raw_config(rope_theta=10000.0, rope_scaling={"type": "dynamic", "factor": 2.0}),
        _hf_config(
            {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 128,
            }
        ),
    ],
    ids=["linear", "dynamic", "yarn"],
)
def test_unknown_rope_type_raises(hf_config: object) -> None:
    with pytest.raises(NotImplementedError, match="rope_type"):
        LlamaConfig.from_hf(hf_config)


def test_llama3_with_a_missing_field_raises() -> None:
    scaling = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0}
    with pytest.raises(ValueError, match="high_freq_factor"):
        LlamaConfig.from_hf(_raw_config(rope_theta=500000.0, rope_scaling=scaling))


def test_llama3_and_yarn_cannot_combine() -> None:
    scaling = Llama3RopeScaling(
        factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=64
    )
    with pytest.raises(ValueError, match="pick one"):
        RotaryEmbedding(HEAD_DIM, llama3_scaling=scaling, yarn_original_seq_len=64)
