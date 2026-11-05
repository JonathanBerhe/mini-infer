"""Kimi K3.1 Pro: the DSA config, the indexer, the sparse MLA layer and the checkpoint layout.

The checkpoint ships no modeling code, so there is no reference to run. The
tests pin the port in three other ways:

  - the indexer against a direct transcription of its formula
    (`dsa_indexer.py`), and the DSA layer against a loop over the keys
    that each query keeps;
  - the model against Kimi K3: while every query keeps all of its keys, K3.1
    with the same weights must equal K3 bit for bit;
  - the cached paths (decode, chunked prefill, ragged decode) against the
    stateless forward, with a top-k small enough that the selection runs.

`tests/fixtures/kimi_k31/config.json` is a byte-for-byte copy of `config.json`
from moonshotai/Kimi-K3.1-Pro at revision
6d0a0202956796543364d666ab19c6ad1a494ee1 (sha256
d95037631e3f23bf527a9f476f25209b231b46853a84877b9c56d5672b8d1202).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn import functional

from mini_infer.models import REGISTRY, _raw_config_namespace
from mini_infer.models.kimi_linear.dsa_indexer import KimiDsaIndexer
from mini_infer.models.kimi_linear.kimi_k3 import (
    KimiK3Config,
    KimiK3ForConditionalGeneration,
    KimiK31Config,
    KimiK31ForConditionalGeneration,
)
from mini_infer.models.kimi_linear.state_cache import KimiMlaLayerState
from tests.unit.models.kimi_linear.test_models_kimi_k3 import (
    _checkpoint_state,
    _tiny_hf_namespace,
    _tiny_text_config,
    _write_k3_checkpoint,
)

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "kimi_k31"

# The tiny K3 config has one MLA layer, the last of four (1-indexed 4).
_DSA_LAYER = 3
# Eight indexer heads: a score is exactly 0 only when every head's ReLU is
# off, so with few heads exact ties at the top-k boundary are common.
_DSA = {
    "dsa_layers": [4],
    "index_topk": 3,
    "index_num_heads": 8,
    "index_head_dim": 8,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}
_INDEXER_SHAPES = {
    "wq_b.weight": (64, 12),
    "wk.weight": (8, 48),
    "weights_proj.weight": (8, 48),
    "k_norm.weight": (8,),
    "k_norm.bias": (8,),
}


def _k31_namespace(dsa: dict[str, Any] | None = None, **text_overrides: Any) -> SimpleNamespace:
    """The K3.1 wrapper config at toy scale: the K3 one plus `dsa_config`."""
    return SimpleNamespace(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=_tiny_text_config(dsa_config={**_DSA, **(dsa or {})}, **text_overrides),
        vision_config={"vt_hidden_size": 8},
    )


def _k31_model(seed: int = 0, **dsa: Any) -> KimiK31ForConditionalGeneration:
    torch.manual_seed(seed)
    cfg = KimiK31Config.from_hf(_k31_namespace(dsa))
    return KimiK31ForConditionalGeneration(cfg).float().eval()


def _k3_twin(model: KimiK31ForConditionalGeneration) -> KimiK3ForConditionalGeneration:
    """Kimi K3 with the K3.1 model's weights, the indexers left out."""
    k3 = KimiK3ForConditionalGeneration(KimiK3Config.from_hf(_tiny_hf_namespace())).float().eval()
    state = {k: v for k, v in model.state_dict().items() if ".indexer." not in k}
    k3.load_state_dict(state, strict=True)
    return k3


# ---- config ----


def test_registry_has_kimi_k31() -> None:
    assert REGISTRY.lookup("KimiK31ForConditionalGeneration") is KimiK31ForConditionalGeneration
    assert issubclass(KimiK31ForConditionalGeneration, KimiK3ForConditionalGeneration)
    assert KimiK31ForConditionalGeneration.USES_STATE_CACHE is True


def test_from_hf_reads_the_dsa_config() -> None:
    text = KimiK31Config.from_hf(_k31_namespace()).text
    assert text.uses_dsa is True
    assert (text.dsa_index_topk, text.dsa_index_num_heads, text.dsa_index_head_dim) == (3, 8, 8)
    assert KimiK3Config.from_hf(_tiny_hf_namespace()).text.uses_dsa is False


def test_from_hf_accepts_a_dsa_config_object() -> None:
    hf = _k31_namespace()
    hf.text_config["dsa_config"] = SimpleNamespace(**hf.text_config["dsa_config"])
    assert KimiK31Config.from_hf(hf).text.dsa_index_topk == 3


def test_k31_config_requires_the_dsa_config() -> None:
    with pytest.raises(ValueError, match="dsa_config"):
        KimiK31Config.from_hf(_tiny_hf_namespace())


@pytest.mark.parametrize(
    ("dsa", "match"),
    [
        ({"index_topk_freq": 2}, "index_topk_freq"),
        ({"index_rope_dim": 64}, "index_rope_dim"),
        ({"index_k_norm_type": "rmsnorm"}, "index_k_norm_type"),
        ({"dsa_layers": [3]}, "dsa_layers"),
        ({"dsa_layers": None}, "dsa_layers"),
        ({"index_topk": 0}, "index_topk"),
        ({"index_num_heads": 0}, "index_num_heads"),
        ({"index_head_dim": -8}, "index_head_dim"),
    ],
)
def test_from_hf_rejects_unsupported_dsa_settings(dsa: dict[str, Any], match: str) -> None:
    """Each setting that the port does not implement fails before the weights load."""
    with pytest.raises(ValueError, match=match):
        KimiK31Config.from_hf(_k31_namespace(dsa))


def test_from_hf_rejects_dsa_without_a_query_latent() -> None:
    with pytest.raises(ValueError, match="q_lora_rank"):
        KimiK31Config.from_hf(_k31_namespace(q_lora_rank=None))


def test_released_config_parses() -> None:
    """The released config: 24 DSA layers on exactly the MLA layers, and the
    settings that `from_hf` checks hold the values that the port implements."""
    raw = json.loads((FIXTURE_DIR / "config.json").read_text())
    assert raw["architectures"] == ["KimiK31ForConditionalGeneration"]
    dsa = raw["text_config"]["dsa_config"]
    assert dsa["index_topk_freq"] == 1 and dsa["index_rope_dim"] == 0
    assert dsa["index_k_norm_type"] == "layernorm" and dsa["index_skip_topk_offset"] == 1

    text = KimiK31Config.from_hf(_raw_config_namespace(str(FIXTURE_DIR))).text
    assert (text.dsa_index_topk, text.dsa_index_num_heads, text.dsa_index_head_dim) == (
        2048,
        32,
        128,
    )
    assert len(text.full_attn_layers) == 24 and sorted(dsa["dsa_layers"]) == text.full_attn_layers
    assert (text.num_hidden_layers, text.q_lora_rank, text.rms_norm_eps) == (93, 1536, 1e-5)
    assert text.mxfp4_weights is True
    assert text.moe_weights_from_biased_scores is False


# ---- modules and state ----


def test_dsa_layers_carry_the_checkpoint_names() -> None:
    model = _k31_model()
    state = model.state_dict()
    prefix = f"model.layers.{_DSA_LAYER}.self_attn.indexer."
    indexer = {k[len(prefix) :]: tuple(v.shape) for k, v in state.items() if k.startswith(prefix)}
    assert indexer == _INDEXER_SHAPES
    assert sum(".indexer." in k for k in state) == len(_INDEXER_SHAPES)
    attn = model.model.layers[_DSA_LAYER].self_attn
    assert isinstance(attn.indexer, KimiDsaIndexer)
    assert attn.indexer.k_norm.eps == 1e-5  # rms_norm_eps, as SGLang sets it


def test_state_cache_keeps_indexer_keys_on_dsa_layers_only() -> None:
    cache = _k31_model().build_state_cache(max_seq_len=16, batch_size=2)
    mla = cache.layer(_DSA_LAYER)
    assert isinstance(mla, KimiMlaLayerState)
    assert mla.index_k is not None and tuple(mla.index_k.shape) == (2, 16, 8)
    k3_cache = _k3_twin(_k31_model()).build_state_cache(max_seq_len=16)
    k3_mla = k3_cache.layer(_DSA_LAYER)
    assert isinstance(k3_mla, KimiMlaLayerState) and k3_mla.index_k is None
    with pytest.raises(ValueError, match="indexer keys"):
        k3_cache.copy_row_from(cache, src_row=0, dst_row=0)


# ---- the indexer and the layer ----


def _formula_scores(
    indexer: KimiDsaIndexer, x: torch.Tensor, q_lora: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """`score[b, t, j]` from the indexer formula, one dot product at a time."""
    batch, seq_len, _ = x.shape
    q = indexer.wq_b(q_lora).view(batch, seq_len, indexer.num_heads, indexer.head_dim)
    weights = indexer.weights_proj(x) * indexer.num_heads**-0.5
    score = torch.zeros(batch, seq_len, keys.shape[1])
    scale = indexer.softmax_scale
    for b in range(batch):
        for t in range(seq_len):
            for j in range(keys.shape[1]):
                score[b, t, j] = sum(
                    weights[b, t, h] * max(float(q[b, t, h] @ keys[b, j]) * scale, 0.0)
                    for h in range(indexer.num_heads)
                )
    return score


def _assert_top_k(keep: torch.Tensor, score: torch.Tensor, valid: torch.Tensor, k: int) -> None:
    """Each query keeps `min(k, visible)` visible keys, none scoring below a dropped
    one. Ties at the boundary may go either way, as in every engine."""
    for b in range(keep.shape[0]):
        for t in range(keep.shape[1]):
            kept, visible = keep[b, t], valid[b, t]
            assert not bool((kept & ~visible).any())
            assert int(kept.sum()) == min(k, int(visible.sum()))
            dropped = visible & ~kept
            if bool(dropped.any()):
                assert float(score[b, t][kept].min()) >= float(score[b, t][dropped].max()) - 1e-6


def test_indexer_keeps_the_top_scores_of_its_formula() -> None:
    torch.manual_seed(1)
    indexer = KimiDsaIndexer(
        hidden_size=10, q_lora_rank=6, num_heads=3, head_dim=4, index_topk=2, norm_eps=1e-5
    )
    batch, seq_len, total_k = 2, 3, 6
    x = torch.randn(batch, seq_len, 10)
    q_lora = torch.randn(batch, seq_len, 6)
    keys = torch.randn(batch, total_k, 4)
    # Queries at the last three positions of a six-token history.
    valid = torch.arange(total_k).view(1, 1, total_k) <= torch.arange(3, 6).view(1, seq_len, 1)
    valid = valid.expand(batch, -1, -1)

    with torch.no_grad():
        keep = indexer.select(x, q_lora, keys, valid)
        _assert_top_k(keep, _formula_scores(indexer, x, q_lora, keys), valid, 2)


def test_indexer_keys_use_an_fp32_layer_norm() -> None:
    torch.manual_seed(2)
    indexer = KimiDsaIndexer(
        hidden_size=10, q_lora_rank=6, num_heads=2, head_dim=4, index_topk=2, norm_eps=1e-5
    )
    x = torch.randn(1, 3, 10)
    norm = indexer.k_norm
    expected = functional.layer_norm(indexer.wk(x), (4,), norm.weight, norm.bias, 1e-5)
    assert torch.allclose(indexer.keys(x), expected)
    # In bf16 the norm still runs in fp32, and only its output is rounded.
    bf16 = indexer.to(torch.bfloat16)
    x16 = x.bfloat16()
    expected16 = functional.layer_norm(
        bf16.wk(x16).float(), (4,), norm.weight.float(), norm.bias.float(), 1e-5
    ).bfloat16()
    assert torch.equal(bf16.keys(x16), expected16)


def _reference_dsa_layer(attn: Any, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """The DSA layer as a loop: query `t` attends to the keys `keep[b, t]`."""
    batch, seq_len, _ = x.shape
    heads, nope, rope = attn.num_heads, attn.qk_nope_head_dim, attn.qk_rope_head_dim
    v_dim = attn.v_head_dim
    q_lora = attn.q_a_layernorm(attn.q_a_proj(x))
    q = attn.q_b_proj(q_lora).view(batch, seq_len, heads, nope + rope)
    latent, k_rot = attn.kv_a_proj_with_mqa(x).split([attn.kv_lora_rank, rope], dim=-1)
    decompressed = attn.kv_b_proj(attn.kv_a_layernorm(latent)).view(batch, seq_len, heads, -1)
    k_nope, values = decompressed.split([nope, v_dim], dim=-1)
    keys = torch.cat([k_nope, k_rot.unsqueeze(2).expand(-1, -1, heads, -1)], dim=-1)
    out = torch.zeros(batch, seq_len, heads, v_dim)
    for b in range(batch):
        for t in range(seq_len):
            kept = torch.nonzero(keep[b, t]).flatten().tolist()
            for h in range(heads):
                logits = torch.stack([q[b, t, h] @ keys[b, j, h] for j in kept])
                probs = (logits * attn.softmax_scale).softmax(dim=-1)
                out[b, t, h] = sum(p * values[b, j, h] for p, j in zip(probs, kept, strict=True))
    gated = out.reshape(batch, seq_len, -1) * torch.sigmoid(attn.g_proj(x))
    result: torch.Tensor = attn.o_proj(gated)
    return result


def test_dsa_layer_matches_a_loop_over_the_kept_keys() -> None:
    """With top-k 3 over 10 tokens, the later queries read only 3 earlier tokens."""
    attn = _k31_model(seed=4).model.layers[_DSA_LAYER].self_attn
    torch.manual_seed(5)
    x = torch.randn(2, 10, 48)
    causal = torch.ones(10, 10, dtype=torch.bool).tril().expand(2, -1, -1)
    with torch.no_grad():
        got = attn(x, None, torch.zeros(2, dtype=torch.long))
        q_lora = attn.q_a_layernorm(attn.q_a_proj(x))
        keys = attn.indexer.keys(x)
        keep = attn.indexer.select(x, q_lora, keys, causal)
        _assert_top_k(keep, _formula_scores(attn.indexer, x, q_lora, keys), causal, 3)
        expected = _reference_dsa_layer(attn, x, keep)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    # The selection matters: the dense K3 layer gives another output.
    k3_attn = _k3_twin(_k31_model(seed=4)).model.layers[_DSA_LAYER].self_attn
    with torch.no_grad():
        dense = k3_attn(x, None, torch.zeros(2, dtype=torch.long))
    assert not torch.allclose(got[:, 3:], dense[:, 3:], atol=1e-3)
    torch.testing.assert_close(got[:, :3], dense[:, :3], rtol=0, atol=0)


# ---- the model ----


def test_k31_equals_k3_while_every_key_is_kept() -> None:
    """With top-k 64 over 9 tokens the indexer keeps every key: K3.1 is K3."""
    model = _k31_model(index_topk=64)
    k3 = _k3_twin(model)
    torch.manual_seed(6)
    ids = torch.randint(0, 96, (2, 9))
    with torch.inference_mode():
        assert torch.equal(model(ids), k3(ids))
        caches = [m.build_state_cache(max_seq_len=16, batch_size=2) for m in (model, k3)]
        steps = []
        for m, cache in zip((model, k3), caches, strict=True):
            m.forward_prefill_with_cache(ids[:, :8], state_cache=cache)
            cache.advance_start_pos(8)
            steps.append(m.forward_decode_with_cache(ids[:, 8:], start_pos=8, state_cache=cache))
    assert torch.equal(steps[0], steps[1])


def test_sparse_decode_and_chunked_prefill_match_the_stateless_forward() -> None:
    model = _k31_model(seed=7)
    torch.manual_seed(8)
    ids = torch.randint(0, 96, (1, 10))
    with torch.inference_mode():
        full = model(ids)
        cache = model.build_state_cache(max_seq_len=16)
        first = model.forward_prefill_with_cache(ids[:, :4], state_cache=cache)
        cache.advance_start_pos(4)
        second = model.forward_prefill_with_cache(ids[:, 4:7], state_cache=cache)
        cache.advance_start_pos(3)
        steps = []
        for position in range(7, 10):
            steps.append(
                model.forward_decode_with_cache(
                    ids[:, position : position + 1], start_pos=position, state_cache=cache
                )
            )
            cache.advance_start_pos(1)
    cached = torch.cat([first, second, *steps], dim=1)
    torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-4)


def test_ragged_decode_keeps_each_rows_own_keys() -> None:
    """Two requests of different lengths in one batched cache: the shorter row's
    unused buffer entries must not enter its top-k."""
    model = _k31_model(seed=9)
    torch.manual_seed(10)
    prompts = [torch.randint(0, 96, (1, 7)), torch.randint(0, 96, (1, 5))]
    with torch.inference_mode():
        own = []
        batched = model.build_state_cache(max_seq_len=16, batch_size=2)
        for row, prompt in enumerate(prompts):
            cache = model.build_state_cache(max_seq_len=16)
            model.forward_prefill_with_cache(prompt[:, :-1], state_cache=cache)
            cache.advance_start_pos(prompt.shape[1] - 1)
            batched.copy_row_from(cache, src_row=0, dst_row=row)
            own.append(
                model.forward_decode_with_cache(
                    prompt[:, -1:], start_pos=prompt.shape[1] - 1, state_cache=cache
                )
            )
        ragged = model.forward_decode_with_cache_ragged(
            torch.cat([prompts[0][:, -1:], prompts[1][:, -1:]]),
            positions=torch.tensor([6, 4]),
            state_cache=batched,
        )
    torch.testing.assert_close(ragged, torch.cat(own), rtol=1e-5, atol=1e-5)


def test_dsa_layer_needs_a_cache_with_indexer_keys() -> None:
    model = _k31_model()
    k3_cache = _k3_twin(model).build_state_cache(max_seq_len=16)
    with pytest.raises(ValueError, match="indexer keys"), torch.inference_mode():
        model.forward_prefill_with_cache(torch.zeros(1, 3, dtype=torch.long), state_cache=k3_cache)


# ---- checkpoint ----


def test_load_weights_translates_the_k31_checkpoint_layout() -> None:
    """The K3 checkpoint layout plus the five indexer tensors per DSA layer."""
    source = _k31_model(seed=11)
    state, expected_experts = _checkpoint_state(source, "language_model.")
    prefix = f"language_model.model.layers.{_DSA_LAYER}.self_attn.indexer."
    assert {k[len(prefix) :] for k in state if k.startswith(prefix)} == set(_INDEXER_SHAPES)
    state["vision_tower.encoder.blocks.0.mlp.fc0.weight"] = torch.zeros(4, 4)

    model = _k31_model(seed=12)
    KimiK31ForConditionalGeneration.load_weights(model, state)
    loaded = model.state_dict()
    for name in _INDEXER_SHAPES:
        key = f"model.layers.{_DSA_LAYER}.self_attn.indexer.{name}"
        assert torch.equal(loaded[key], state[f"language_model.{key}"]), key
    for key, expected in expected_experts.items():
        assert torch.equal(loaded[key], expected.float()), key


def test_load_weights_rejects_a_checkpoint_without_indexers() -> None:
    """A K3 checkpoint under the K3.1 architecture misses the indexer tensors."""
    state, _ = _checkpoint_state(_k31_model(), "language_model.")
    state = {k: v for k, v in state.items() if ".indexer." not in k}
    with pytest.raises(ValueError, match=r"missing=.*indexer"):
        KimiK31ForConditionalGeneration.load_weights(_k31_model(), state)


def test_from_checkpoint_streams_the_indexers(tmp_path: Path) -> None:
    """K3.1 inherits the shard-by-shard load. It fills the indexers and gives
    the model of `load_weights`, also where the selection runs (6 tokens, top-k 3)."""
    state, _ = _checkpoint_state(_k31_model(), "language_model.")
    _write_k3_checkpoint(tmp_path, state)
    (tmp_path / "config.json").write_text(json.dumps(vars(_k31_namespace())))
    in_memory = _k31_model(seed=1)
    KimiK31ForConditionalGeneration.load_weights(in_memory, state)

    streamed = KimiK31ForConditionalGeneration.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.float32
    )

    assert isinstance(streamed, KimiK31ForConditionalGeneration)
    in_memory_state = in_memory.state_dict()
    for key, value in streamed.state_dict().items():
        assert torch.equal(value, in_memory_state[key]), key
    ids = torch.randint(0, 96, (1, 6))
    with torch.inference_mode():
        assert torch.equal(streamed(ids), in_memory(ids))
