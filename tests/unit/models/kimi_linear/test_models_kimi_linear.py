"""Kimi Linear structural tests: config validation, cache layout, serving path.

Reference-free (no vendored code needed): these pin the `from_hf` contract,
the KimiStateCache per-layer layout, weight-load failure modes, and the
end-to-end serving smoke through `StateCacheGenerator` +
`StateCacheContinuousScheduler` against the single-request oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

import mini_infer.models as models_module
from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models import load_model
from mini_infer.models.kimi_linear.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM
from mini_infer.models.kimi_linear.state_cache import (
    KimiKdaLayerState,
    KimiMlaLayerState,
    KimiStateCache,
)
from mini_infer.scheduler.request_state import Request
from mini_infer.scheduler.state_cache_continuous_scheduler import (
    StateCacheContinuousScheduler,
)
from tests.unit._kimi_tokenizer_files import write_kimi_tokenizer_files


def _tiny_hf_namespace(**overrides: Any) -> SimpleNamespace:
    """The raw-config shape `Config.from_hf` consumes (attributes + a plain
    dict for linear_attn_config, exactly what `_raw_config_namespace` yields)."""
    fields: dict[str, Any] = dict(
        vocab_size=96,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=None,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 3,
        },
        intermediate_size=48,
        moe_intermediate_size=16,
        num_experts=8,
        num_experts_per_token=3,
        num_shared_experts=1,
        routed_scaling_factor=2.446,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _tiny_model(seed: int = 0) -> KimiLinearForCausalLM:
    torch.manual_seed(seed)
    cfg = KimiLinearConfig.from_hf(_tiny_hf_namespace())
    return KimiLinearForCausalLM(cfg).float().eval()


class _FakeTokenizer:
    """Just enough surface for the scheduler: encode / decode / eos."""

    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 96 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_from_hf_rejects_bad_layer_partition() -> None:
    """kda_layers + full_attn_layers must partition 1..num_layers; a silent
    mismatch would build the wrong hybrid."""
    bad = _tiny_hf_namespace(
        linear_attn_config={
            "kda_layers": [1, 2],  # layer 3 missing from both lists
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 3,
        }
    )
    with pytest.raises(ValueError, match="partition"):
        KimiLinearConfig.from_hf(bad)


def test_from_hf_rejects_roped_mla() -> None:
    with pytest.raises(ValueError, match="NoPE"):
        KimiLinearConfig.from_hf(_tiny_hf_namespace(mla_use_nope=False))


def test_from_hf_accepts_low_rank_q() -> None:
    """`q_lora_rank` (K3) builds the `q_a_proj -> norm -> q_b_proj` path on
    the MLA layers; Kimi Linear's `None` keeps the plain `q_proj`."""
    cfg = KimiLinearConfig.from_hf(_tiny_hf_namespace(q_lora_rank=12))
    model = KimiLinearForCausalLM(cfg)
    mla = model.model.layers[3].self_attn
    assert mla.q_a_proj.weight.shape == (12, 32)
    assert mla.q_b_proj.weight.shape == (2 * (8 + 4), 12)
    assert not hasattr(mla, "q_proj")
    plain = KimiLinearForCausalLM(KimiLinearConfig.from_hf(_tiny_hf_namespace()))
    assert hasattr(plain.model.layers[3].self_attn, "q_proj")


def test_from_hf_keeps_kimi_linear_router_convention() -> None:
    """A config without any K3-only field is the Kimi Linear modeling file,
    whose expert weights are the BIASED scores."""
    cfg = KimiLinearConfig.from_hf(_tiny_hf_namespace())
    assert cfg.moe_weights_from_biased_scores is True
    assert cfg.hidden_act == "silu"
    assert cfg.attn_res_block_size is None


def test_layer_kind_indexing_is_one_indexed() -> None:
    cfg = KimiLinearConfig.from_hf(_tiny_hf_namespace())
    # 1-indexed lists [1,2,3] KDA + [4] full -> 0-indexed layers 0-2 KDA, 3 MLA.
    assert [cfg.is_kda_layer(i) for i in range(4)] == [True, True, True, False]
    # first_k_dense_replace=1 -> layer 0 dense, the rest MoE.
    assert [cfg.is_moe_layer(i) for i in range(4)] == [False, True, True, True]


def test_state_cache_layout_matches_layer_kinds() -> None:
    model = _tiny_model()
    cache = model.build_state_cache(max_seq_len=16, batch_size=2)
    kinds = [type(cache.layer(i)) for i in range(4)]
    assert kinds == [KimiKdaLayerState, KimiKdaLayerState, KimiKdaLayerState, KimiMlaLayerState]
    kda = cache.layer(0)
    assert isinstance(kda, KimiKdaLayerState)
    assert kda.recurrent_state.shape == (2, 2, 8, 8)
    assert kda.recurrent_state.dtype == torch.float32  # FLA state contract
    assert kda.conv_q.shape == (2, 16, 3)
    mla = cache.layer(3)
    assert isinstance(mla, KimiMlaLayerState)
    assert mla.kv.shape == (2, 16, 16 + 4)  # kv_lora_rank + qk_rope_head_dim


def test_copy_row_rejects_mismatched_caches() -> None:
    model = _tiny_model()
    dst = model.build_state_cache(max_seq_len=16, batch_size=2)
    src = KimiStateCache(dst.layer_specs[:2], batch_size=1)
    with pytest.raises(ValueError, match="layer count"):
        dst.copy_row_from(src, src_row=0, dst_row=0)


def test_mla_buffer_overflow_raises() -> None:
    """Writing past max_seq_len must fail loudly (the dense buffer is a hard
    admission bound, unlike the paged path)."""
    model = _tiny_model()
    cache = model.build_state_cache(max_seq_len=4)
    ids = torch.randint(0, 96, (1, 5))
    with pytest.raises(ValueError, match="overflow"):
        model.forward_prefill_with_cache(ids, state_cache=cache)


def test_load_weights_rejects_unknown_keys() -> None:
    model = _tiny_model()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["model.layers.0.self_attn.bogus.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        KimiLinearForCausalLM.load_weights(model, state)


def test_load_weights_roundtrip_with_hf_shapes() -> None:
    """A state_dict in the CHECKPOINT's layout (Conv1d `(C, 1, W)` weights,
    `o_norm.weight`) loads cleanly through the remap."""
    model = _tiny_model()
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            state[key] = value.unsqueeze(1).clone()  # our (C, W) -> HF (C, 1, W)
        elif key.endswith("o_norm_weight"):
            state[key.replace("o_norm_weight", "o_norm.weight")] = value.clone()
        else:
            state[key] = value.clone()
    KimiLinearForCausalLM.load_weights(model, state)


def test_uses_state_cache_flag() -> None:
    assert KimiLinearForCausalLM.USES_STATE_CACHE is True


def test_stateless_forward_shape_and_finiteness() -> None:
    model = _tiny_model()
    ids = torch.randint(0, 96, (2, 9))
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (2, 9, 96)
    assert torch.isfinite(logits).all()


@pytest.mark.slow
def test_scheduler_serving_matches_scalar_oracle() -> None:
    """End-to-end serving smoke: two concurrent requests through the
    (generalized) StateCacheContinuousScheduler equal running each alone
    through StateCacheGenerator, the same self-consistency contract the V4
    scheduler tests pin."""
    model = _tiny_model()
    generator = StateCacheGenerator(
        model, cast(Tokenizer, _FakeTokenizer()), device="cpu", dtype=torch.float32
    )
    prompts = ["hello kimi", "different prompt!"]
    budgets = [6, 5]
    expected = [
        generator.tokenizer.decode(
            generator.generate_ids(generator.tokenizer.encode(p), max_new_tokens=n)
        )
        for p, n in zip(prompts, budgets, strict=True)
    ]

    scheduler = StateCacheContinuousScheduler(generator, max_batch_size=2, max_seq_len=64)
    scheduler.start()
    try:
        handles = [
            scheduler.submit(
                Request(prompt=p, max_tokens=n, sampling_params=SamplingParams(temperature=0.0))
            )
            for p, n in zip(prompts, budgets, strict=True)
        ]
        results = [handle.wait() for handle in handles]
    finally:
        scheduler.stop()

    assert [r.text for r in results] == expected


def _checkpoint_layout(model: KimiLinearForCausalLM) -> dict[str, torch.Tensor]:
    """The model's tensors as the released checkpoint stores them.

    Conv1d `(C, 1, W)` taps, `o_norm.weight`, `A_log` as `(1, 1, H, 1)`, the
    `A_log` / `dt_bias` pair in fp32 and every other tensor in bf16. A tied
    model stores the embedding only.
    """
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key == "lm_head.weight" and model.cfg.tie_word_embeddings:
            continue
        if key.endswith(("A_log", "dt_bias")):
            value = value.float()
            if key.endswith("A_log"):
                value = value.reshape(1, 1, -1, 1)
            state[key] = value.clone()
            continue
        value = value.to(torch.bfloat16)
        if key.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            state[key] = value.unsqueeze(1).clone()
        elif key.endswith("o_norm_weight"):
            state[key.replace("o_norm_weight", "o_norm.weight")] = value.clone()
        else:
            state[key] = value.clone()
    return state


def _write_checkpoint(
    directory: Path, hf_config: dict[str, Any], state: dict[str, torch.Tensor]
) -> None:
    """`config.json`, two safetensors shards and the index, as `iter_safetensors_shards` reads."""
    (directory / "config.json").write_text(json.dumps(hf_config))
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {key: shard_names[index % 2] for index, key in enumerate(sorted(state))}
    for shard in shard_names:
        save_file(
            {key: tensor.contiguous() for key, tensor in state.items() if weight_map[key] == shard},
            str(directory / shard),
        )
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _tiny_checkpoint(directory: Path, **overrides: Any) -> KimiLinearForCausalLM:
    """Write a tiny Kimi Linear checkpoint with random weights; returns the source model."""
    hf_config = {
        **vars(_tiny_hf_namespace(**overrides)),
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
    }
    torch.manual_seed(3)
    source = KimiLinearForCausalLM(KimiLinearConfig.from_hf(_tiny_hf_namespace(**overrides)))
    with torch.no_grad():
        for param in source.parameters():
            param.uniform_(-0.2, 0.2)
    _write_checkpoint(directory, hf_config, _checkpoint_layout(source))
    return source


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_from_checkpoint_matches_load_model(tmp_path: Path, tie_word_embeddings: bool) -> None:
    """The shard-by-shard load gives the same bf16 model and logits as `load_model`."""
    _tiny_checkpoint(tmp_path, tie_word_embeddings=tie_word_embeddings)

    reference = load_model(str(tmp_path), dtype=torch.bfloat16, device="cpu")
    streamed = KimiLinearForCausalLM.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.bfloat16
    )

    assert isinstance(streamed, KimiLinearForCausalLM)
    assert not streamed.training
    # `A_log` and `dt_bias` are fp32 in the constructor and in the checkpoint.
    assert {param.dtype for param in streamed.parameters()} == {torch.bfloat16}
    assert {param.dtype for param in reference.parameters()} == {torch.bfloat16}
    assert torch.get_default_dtype() == torch.float32
    if tie_word_embeddings:
        assert streamed.lm_head.weight is streamed.model.embed_tokens.weight
    reference_state = reference.state_dict()
    for key, value in streamed.state_dict().items():
        assert torch.equal(value, reference_state[key]), key
    ids = torch.randint(0, 96, (2, 7))
    with torch.inference_mode():
        assert torch.equal(streamed(ids), reference(ids))


def test_from_checkpoint_fails_on_a_missing_tensor(tmp_path: Path) -> None:
    source = _tiny_checkpoint(tmp_path)
    state = _checkpoint_layout(source)
    del state["model.layers.3.self_attn.kv_b_proj.weight"]
    _write_checkpoint(tmp_path, json.loads((tmp_path / "config.json").read_text()), state)
    missing = r"missing=\['model.layers.3.self_attn.kv_b_proj.weight'\]"
    with pytest.raises(ValueError, match=missing):
        KimiLinearForCausalLM.from_checkpoint(str(tmp_path), device="cpu", dtype=torch.float32)


def test_state_cache_generator_loads_kimi_without_hub_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`StateCacheGenerator.from_pretrained` serves a Kimi checkpoint as the API server does.

    The weights load through `from_checkpoint` (not `load_model`), and the
    tokenizer through the fast Kimi tokenizer (no hub code). The greedy text
    equals the text of a generator around the `load_model` model.
    """
    ranks = write_kimi_tokenizer_files(tmp_path)
    _tiny_checkpoint(tmp_path, vocab_size=len(ranks) + 258)
    reference_model = load_model(str(tmp_path), dtype=torch.float32, device="cpu")

    def _no_load_model(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Kimi Linear must load through from_checkpoint")

    monkeypatch.setattr(models_module, "load_model", _no_load_model)
    generator = StateCacheGenerator.from_pretrained(
        str(tmp_path), device="cpu", dtype=torch.float32
    )

    assert isinstance(generator.model, KimiLinearForCausalLM)
    reference = StateCacheGenerator(
        cast(Any, reference_model), generator.tokenizer, device="cpu", dtype=torch.float32
    )
    prompt = "The capital of France is"
    text = generator.generate(prompt, max_new_tokens=6)
    assert text == reference.generate(prompt, max_new_tokens=6)
