"""Slice 1: single-request greedy generation for V4's StateCache path.

Covered here (CPU, synthetic configs, no GPU, no reference):

  - Model-level `forward_prefill_with_cache` reproduces the standalone
    `forward` last-position logits for aligned input. This is the strongest
    local oracle: the attention-layer prefill-with-cache is already
    bit-parity validated against the DeepSeek-V4 reference
    (`test_v4_prefill_cache_aware.py`); this confirms the decoder-layer +
    model wrappers around it are wired correctly, across the vanilla, MoE,
    and Hyper-Connections backbones.
  - `StateCacheGenerator.generate_ids`: finite, in-range, deterministic,
    EOS-terminated, count-capped, and consistent with a hand-written
    prefill+greedy loop. Handles prompt lengths that are not a multiple of
    every compression ratio (the cache-aware prefill supports it).
  - SWA (`compression_ratio == 0`) layers, which real V4-Flash uses at the
    head of its stack, generate end-to-end through the cache path.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.state_cache import StateCache
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)


def _make_config(
    *,
    use_moe_ffn: bool = False,
    use_hyper_connections: bool = False,
    num_hidden_layers: int = 4,
    compress_ratios: tuple[int, ...] | None = None,
) -> DeepseekV4Config:
    """Small 4-layer hybrid (CSA, HCA, CSA, HCA) by default; fits a laptop CPU.

    `compress_ratios` defaults to `(4, 8, 4, 8)` (no SWA layers), so any prompt
    length that is a multiple of 8 is aligned for every layer. Pass an explicit
    tuple to include a pure-SWA (`0`) layer, e.g. `(0, 4, 8, 4)`.
    """
    if compress_ratios is None:
        compress_ratios = tuple(4 if i % 2 == 0 else 8 for i in range(num_hidden_layers))
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_head_dim=32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=compress_ratios,
        index_num_heads=2,
        index_head_dim=16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_moe_ffn=use_moe_ffn,
        moe_intermediate_size=64 if use_moe_ffn else 0,
        num_routed_experts=4 if use_moe_ffn else 0,
        num_activated_experts=2 if use_moe_ffn else 0,
        num_hash_routed_layers=(num_hidden_layers // 2) if use_moe_ffn else 0,
        moe_score_func="softmax",
        moe_route_scale=1.0,
        n_shared_experts=1 if use_moe_ffn else 0,
        use_hyper_connections=use_hyper_connections,
        hc_mult=4 if use_hyper_connections else 0,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


class _FakeTokenizer:
    """Minimal duck-typed tokenizer for the string-path wrapper test."""

    def __init__(self, prompt_ids: list[int], eos_token_id: int | None = None) -> None:
        self._prompt_ids = prompt_ids
        self._eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        return list(self._prompt_ids)

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id


# ---------- forward_prefill_with_cache equivalence vs standalone forward ----------


@pytest.mark.parametrize(
    "use_moe_ffn,use_hyper_connections",
    [(False, False), (True, False), (False, True)],
    ids=["vanilla", "moe", "hyper_connections"],
)
def test_model_prefill_with_cache_matches_standalone_forward(
    use_moe_ffn: bool, use_hyper_connections: bool
) -> None:
    """Cache-aware prefill produces the same logits as `forward` for aligned input.

    The attention sublayer's prefill-with-cache path is already reference-
    validated; this confirms the new decoder-layer + model wrappers (residual,
    FFN, Hyper-Connections, head reduction) thread it correctly.
    """
    cfg = _make_config(use_moe_ffn=use_moe_ffn, use_hyper_connections=use_hyper_connections)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    batch_size, seqlen = 1, 16  # multiple of lcm(4, 8) == 8
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seqlen), dtype=torch.long)

    state_cache = StateCache(
        build_state_cache_layer_specs(cfg, max_n_compressed=8),
        batch_size=batch_size,
    )
    with torch.inference_mode():
        standalone_logits = model(input_ids)
        prefill_logits = model.forward_prefill_with_cache(input_ids, state_cache=state_cache)

    assert prefill_logits.shape == standalone_logits.shape
    torch.testing.assert_close(prefill_logits, standalone_logits, rtol=1e-4, atol=1e-5)


def test_model_prefill_with_cache_populates_cache_state() -> None:
    """After prefill, per-layer counters reflect the prompt (not a zeroed cache)."""
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    seqlen = 16
    input_ids = torch.randint(0, cfg.vocab_size, (1, seqlen), dtype=torch.long)
    state_cache = StateCache(build_state_cache_layer_specs(cfg, max_n_compressed=8), batch_size=1)
    with torch.inference_mode():
        model.forward_prefill_with_cache(input_ids, state_cache=state_cache)

    for layer_idx, compression_ratio in enumerate(cfg.compress_ratios):
        layer_state = state_cache.layer(layer_idx)
        assert layer_state.n_compressed_blocks == seqlen // compression_ratio
        assert layer_state.swa_count == min(seqlen, cfg.window_size)
        # Some compressed entry must be non-zero (the prompt actually wrote state).
        assert torch.any(layer_state.compressed_kv != 0)


def test_model_prefill_with_cache_rejects_layer_count_mismatch() -> None:
    cfg = _make_config(num_hidden_layers=4)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    too_few = StateCache(build_state_cache_layer_specs(cfg, max_n_compressed=8)[:2], batch_size=1)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
    with pytest.raises(ValueError, match="layers"):
        model.forward_prefill_with_cache(input_ids, state_cache=too_few)


# ---------- StateCacheGenerator.generate_ids ----------


def test_generate_ids_returns_finite_in_range_tokens() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(8))  # aligned length
    out = gen.generate_ids(prompt_ids, max_new_tokens=6)

    assert len(out) == 6
    assert all(0 <= t < cfg.vocab_size for t in out)


def test_generate_ids_is_deterministic() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(8))
    first = gen.generate_ids(prompt_ids, max_new_tokens=6)
    second = gen.generate_ids(prompt_ids, max_new_tokens=6)
    assert first == second


def test_generate_ids_first_token_matches_standalone_forward_argmax() -> None:
    """The first generated token is argmax of `forward`'s last-position logits.

    Ties the generator's prefill back to the standalone forward the smoke
    validates, without needing the reference.
    """
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(8))
    with torch.inference_mode():
        forward_logits = model(torch.tensor([prompt_ids], dtype=torch.long))
    expected_first = int(forward_logits[0, -1, :].argmax().item())

    out = gen.generate_ids(prompt_ids, max_new_tokens=1)
    assert out == [expected_first]


def test_generate_ids_matches_manual_prefill_decode_loop() -> None:
    """generate_ids equals an explicit prefill + greedy decode loop.

    Guards the position / advance_start_pos bookkeeping against off-by-one.
    """
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(8))
    max_new_tokens = 5

    # Manual reference loop using the same primitives.
    state_cache = StateCache(build_state_cache_layer_specs(cfg, max_n_compressed=16), batch_size=1)
    manual: list[int] = []
    with torch.inference_mode():
        logits = model.forward_prefill_with_cache(
            torch.tensor([prompt_ids], dtype=torch.long), state_cache=state_cache
        )
        state_cache.advance_start_pos(len(prompt_ids))
        next_token = int(logits[0, -1, :].argmax().item())
        for _ in range(max_new_tokens):
            manual.append(next_token)
            step_logits = model.forward_decode_with_cache(
                torch.tensor([[next_token]], dtype=torch.long),
                start_pos=state_cache.start_pos,
                state_cache=state_cache,
            )
            state_cache.advance_start_pos(1)
            next_token = int(step_logits[0, -1, :].argmax().item())

    assert gen.generate_ids(prompt_ids, max_new_tokens=max_new_tokens) == manual


def test_generate_ids_respects_eos() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(8))
    full = gen.generate_ids(prompt_ids, max_new_tokens=5)

    # Generation stops right before the first occurrence of the EOS token, and
    # the EOS token itself is not emitted. `.index` keeps the expectation
    # correct even if `full` happens to repeat a token.
    for eos in (full[0], full[2]):
        expected = full[: full.index(eos)]
        assert gen.generate_ids(prompt_ids, max_new_tokens=5, eos_token_id=eos) == expected


def test_generate_ids_handles_unaligned_prompt() -> None:
    """Prompt length need not be a multiple of every compression ratio."""
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(10))  # 10 is not a multiple of 8 (HCA layers)
    out = gen.generate_ids(prompt_ids, max_new_tokens=6)
    assert len(out) == 6
    assert all(0 <= t < cfg.vocab_size for t in out)


def test_generate_ids_validates_inputs() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    with pytest.raises(ValueError, match="non-empty"):
        gen.generate_ids([], max_new_tokens=4)
    with pytest.raises(ValueError, match="max_new_tokens"):
        gen.generate_ids([1, 2, 3], max_new_tokens=0)


def test_generate_string_path_uses_tokenizer() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    tokenizer = _FakeTokenizer(prompt_ids=list(range(8)))
    gen = StateCacheGenerator(model, tokenizer)  # type: ignore[arg-type]

    text = gen.generate("ignored by the fake tokenizer", max_new_tokens=4)
    # decode() joins ids with spaces; 4 tokens -> 4 space-separated ints.
    assert len(text.split()) == 4


def test_generate_requires_tokenizer() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)  # no tokenizer
    with pytest.raises(ValueError, match="no tokenizer"):
        gen.generate("hello", max_new_tokens=4)


def test_generator_infers_device_and_dtype_from_model() -> None:
    """device/dtype default to the model's own, not a hardcoded cpu/float32."""
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval().to(torch.float64)
    gen = StateCacheGenerator(model)
    assert gen.dtype == torch.float64
    assert gen.device == "cpu"

    # An explicit argument still overrides the inference.
    override = StateCacheGenerator(model, device="cpu", dtype=torch.float32)
    assert override.dtype == torch.float32


# ---------- build_state_cache_layer_specs per-layer sizing ----------


def test_build_state_cache_layer_specs_sizes_per_layer_from_max_seq_len() -> None:
    cfg = _make_config()  # ratios (4, 8, 4, 8)
    specs = build_state_cache_layer_specs(cfg, max_seq_len=32)
    # ceil(32 / 4) == 8 for the CSA layers, ceil(32 / 8) == 4 for the HCA layers.
    assert [spec.max_n_compressed for spec in specs] == [8, 4, 8, 4]


def test_build_state_cache_layer_specs_requires_exactly_one_sizing_arg() -> None:
    cfg = _make_config()
    with pytest.raises(ValueError, match="exactly one"):
        build_state_cache_layer_specs(cfg)
    with pytest.raises(ValueError, match="exactly one"):
        build_state_cache_layer_specs(cfg, max_n_compressed=8, max_seq_len=32)


# ---------- SWA (compression_ratio == 0) layers ----------


def test_generate_ids_runs_with_swa_layer() -> None:
    """A config whose head layer is pure-SWA (ratio 0), like real V4-Flash, generates."""
    cfg = _make_config(compress_ratios=(0, 4, 8, 4))
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    out = gen.generate_ids(list(range(8)), max_new_tokens=6)
    assert len(out) == 6
    assert all(0 <= t < cfg.vocab_size for t in out)


def test_model_prefill_with_cache_matches_forward_with_swa_layer() -> None:
    """Cache-aware prefill matches standalone forward when an SWA layer is present."""
    cfg = _make_config(compress_ratios=(0, 4, 8, 4))
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
    state_cache = StateCache(build_state_cache_layer_specs(cfg, max_seq_len=64), batch_size=1)
    with torch.inference_mode():
        standalone_logits = model(input_ids)
        prefill_logits = model.forward_prefill_with_cache(input_ids, state_cache=state_cache)
    torch.testing.assert_close(prefill_logits, standalone_logits, rtol=1e-4, atol=1e-5)


# ---------- from_checkpoint (real-model load path, synthetic round-trip) ----------


def test_from_checkpoint_round_trips_a_synthetic_v4_checkpoint(tmp_path) -> None:
    """The real-model load path round-trips a synthetic V4 checkpoint and generates.

    Exercises the same sequence the 2x B200 smoke runs per rank
    (config.json -> meta construction -> load_weights -> rotary rebuild), on a
    CPU-sized config that includes an SWA (ratio 0) layer, so meta
    construction, the weight load, and rotary `inv_freq` rematerialization are
    all covered without the real 158 GB checkpoint or a GPU.
    """
    import dataclasses
    import json

    from safetensors.torch import save_file

    cfg = _make_config(compress_ratios=(0, 4, 8, 4))
    torch.manual_seed(0)
    source = DeepseekV4ForCausalLM(cfg).eval()

    # HF-style config.json (from_hf reads our field names) + a weights shard.
    config_json = {**dataclasses.asdict(cfg), "architectures": ["DeepseekV4ForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config_json))
    save_file(
        {k: v.detach().clone().contiguous() for k, v in source.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )

    loaded = DeepseekV4ForCausalLM.from_checkpoint(str(tmp_path), device="cpu", dtype=torch.float32)

    # inv_freq is a non-persistent buffer (absent from the checkpoint); meta
    # construction left it meta, so from_checkpoint must have rebuilt it.
    assert not loaded.rotary_emb.inv_freq.is_meta

    # Weights loaded correctly: logits match the source model.
    input_ids = torch.tensor([list(range(8))], dtype=torch.long)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(input_ids), source(input_ids), rtol=1e-4, atol=1e-5)

    # And the loaded model generates through the StateCache path.
    out = StateCacheGenerator(loaded).generate_ids(list(range(8)), max_new_tokens=4)
    assert len(out) == 4


# ---------- generate_ids_batched (lockstep cohort batching) ----------


@pytest.mark.parametrize(
    "use_moe_ffn,use_hyper_connections",
    [(False, False), (True, False), (False, True)],
    ids=["vanilla", "moe", "hyper_connections"],
)
def test_generate_ids_batched_matches_sequential(
    use_moe_ffn: bool, use_hyper_connections: bool
) -> None:
    """A batched cohort decodes token-for-token like N separate generate_ids calls.

    This is the Phase 0 oracle: lockstep batching must not cross-contaminate
    sequences, so the batched result equals running each prompt alone. The
    per-block batched attention math is separately bit-parity validated against
    the DeepSeek-V4 reference at B=2; this pins the generator loop (prefill +
    decode + sampling) at the cohort level, across the vanilla, MoE, and
    Hyper-Connections backbones.
    """
    cfg = _make_config(use_moe_ffn=use_moe_ffn, use_hyper_connections=use_hyper_connections)
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompts = [list(range(8)), list(range(8, 16)), [3, 1, 4, 1, 5, 9, 2, 6]]
    batched = gen.generate_ids_batched(prompts, max_new_tokens=6)
    sequential = [gen.generate_ids(prompt_ids, max_new_tokens=6) for prompt_ids in prompts]

    assert batched == sequential
    assert [len(out) for out in batched] == [6, 6, 6]


def test_generate_ids_batched_respects_eos_per_sequence() -> None:
    """Each sequence stops at its own EOS; the batch keeps stepping for the rest.

    With a shared EOS token, sequences finish at different steps. The batched
    result must still equal each prompt run alone with the same EOS.
    """
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompts = [list(range(8)), list(range(8, 16)), [3, 1, 4, 1, 5, 9, 2, 6]]
    # Pick an EOS partway through the first sequence's output, so at least one
    # sequence terminates early while others keep going.
    eos = gen.generate_ids(prompts[0], max_new_tokens=6)[2]

    batched = gen.generate_ids_batched(prompts, max_new_tokens=6, eos_token_id=eos)
    sequential = [
        gen.generate_ids(prompt_ids, max_new_tokens=6, eos_token_id=eos) for prompt_ids in prompts
    ]
    assert batched == sequential
    # EOS is never echoed.
    assert all(eos not in out for out in batched)


def test_generate_ids_batched_matches_sequential_with_swa_layer() -> None:
    """Lockstep cohort matches sequential when a pure-SWA (ratio 0) layer is present."""
    cfg = _make_config(compress_ratios=(0, 4, 8, 4))
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompts = [list(range(8)), [7, 7, 7, 7, 1, 2, 3, 4]]
    batched = gen.generate_ids_batched(prompts, max_new_tokens=5)
    sequential = [gen.generate_ids(prompt_ids, max_new_tokens=5) for prompt_ids in prompts]
    assert batched == sequential


def test_generate_ids_batched_single_prompt_equals_generate_ids() -> None:
    """B=1 cohort is exactly generate_ids (the degenerate lockstep case)."""
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    prompt_ids = list(range(8))
    assert gen.generate_ids_batched([prompt_ids], max_new_tokens=6) == [
        gen.generate_ids(prompt_ids, max_new_tokens=6)
    ]


def test_generate_ids_batched_validates_inputs() -> None:
    cfg = _make_config()
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    gen = StateCacheGenerator(model)

    with pytest.raises(ValueError, match="non-empty"):
        gen.generate_ids_batched([], max_new_tokens=4)
    with pytest.raises(ValueError, match="non-empty"):
        gen.generate_ids_batched([[1, 2, 3], []], max_new_tokens=4)
    with pytest.raises(ValueError, match="equal-length"):
        gen.generate_ids_batched([[1, 2, 3], [1, 2]], max_new_tokens=4)
    with pytest.raises(ValueError, match="max_new_tokens"):
        gen.generate_ids_batched([[1, 2, 3]], max_new_tokens=0)
