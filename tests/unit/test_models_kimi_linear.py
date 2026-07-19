"""Kimi Linear structural tests: config validation, cache layout, serving path.

Reference-free (no vendored code needed): these pin the `from_hf` contract,
the KimiStateCache per-layer layout, weight-load failure modes, and the
end-to-end serving smoke through `StateCacheGenerator` +
`StateCacheContinuousScheduler` against the single-request oracle.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mini_infer.cache.kimi_state_cache import (
    KimiKdaLayerState,
    KimiMlaLayerState,
    KimiStateCache,
)
from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM
from mini_infer.scheduler.request_state import Request
from mini_infer.scheduler.state_cache_continuous_scheduler import (
    StateCacheContinuousScheduler,
)


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


def test_from_hf_rejects_q_lora() -> None:
    with pytest.raises(ValueError, match="q_lora_rank"):
        KimiLinearConfig.from_hf(_tiny_hf_namespace(q_lora_rank=64))


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
