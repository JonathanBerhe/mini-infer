"""Full-model bit-parity for the Qwen hybrid family against transformers.

The reference is the pinned transformers implementation of the architecture
the released checkpoints actually declare (`Qwen3_5ForCausalLM` is the
text-only half of `Qwen3_5ForConditionalGeneration`). Weights are pushed
through our production `load_weights` with CHECKPOINT-shaped keys, so the
prefix strip, the vision/MTP drop and the conv squeeze are all exercised
rather than bypassed.

The synthetic config uses 6 linear value heads over 2 linear key heads, a 3x
broadcast ratio, because that is what Qwen3.8-27B ships; the 4B and 9B
siblings use 2x, so hardcoding either would hide a bug in the other.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.qwen3_5 import Qwen35Config, Qwen35ForConditionalGeneration

transformers = pytest.importorskip("transformers")
Qwen3_5TextConfig = transformers.Qwen3_5TextConfig
Qwen3_5ForCausalLM = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5"
).Qwen3_5ForCausalLM

VOCAB = 101
HEAD_DIM = 32
PARTIAL = 0.25


def _hf_config() -> Qwen3_5TextConfig:
    half = int(HEAD_DIM * PARTIAL) // 2
    return Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,  # 2 repeats of [linear, linear, linear, full]
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": PARTIAL,
            "mrope_interleaved": True,
            "mrope_section": [half // 3, half // 3, half - 2 * (half // 3)],
        },
    )


def _checkpoint_state(hf_model) -> dict[str, torch.Tensor]:
    """Re-key a reference state dict the way the released checkpoint stores it."""
    state: dict[str, torch.Tensor] = {}
    for key, tensor in hf_model.state_dict().items():
        name = "model.language_model." + key[len("model.") :] if key.startswith("model.") else key
        if name.endswith("conv1d.weight") and tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        state[name] = tensor.clone()
    return state


@pytest.fixture
def pair():
    torch.manual_seed(0)
    hf_cfg = _hf_config()
    hf_model = Qwen3_5ForCausalLM(hf_cfg).eval().float()
    for param in hf_model.parameters():
        with torch.no_grad():
            param.uniform_(-0.08, 0.08)
    ours = Qwen35ForConditionalGeneration(Qwen35Config.from_hf(hf_cfg)).eval().float()
    Qwen35ForConditionalGeneration.load_weights(ours, _checkpoint_state(hf_model))
    return hf_model, ours


def test_hybrid_layout_is_three_to_one(pair):
    _, ours = pair
    kinds = ours.cfg.layer_types
    assert kinds == ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + [
        "full_attention"
    ]


def test_prefill_matches_reference(pair):
    hf_model, ours = pair
    ids = torch.randint(0, VOCAB, (1, 12))
    with torch.no_grad():
        expected = hf_model(input_ids=ids).logits
        got = ours(ids)
    assert torch.equal(expected.argmax(-1), got.argmax(-1))
    assert torch.allclose(expected, got, atol=1e-4)


def test_greedy_decode_matches_reference_across_chunk_boundary(pair):
    """Prompt longer than the 64-token chunk, then step token by token."""
    hf_model, ours = pair
    prompt = torch.randint(0, VOCAB, (1, 70))
    cache = ours.build_state_cache(max_seq_len=256, batch_size=1)
    with torch.no_grad():
        ours_logits = ours.forward_prefill_with_cache(prompt, state_cache=cache)
        hf_out = hf_model(input_ids=prompt, use_cache=True)
    cache.advance_start_pos(prompt.shape[1])
    past = hf_out.past_key_values
    ours_tokens = [int(ours_logits[0, -1].argmax())]
    hf_tokens = [int(hf_out.logits[0, -1].argmax())]
    for _ in range(12):
        with torch.no_grad():
            step = ours.forward_decode_with_cache(
                torch.tensor([[ours_tokens[-1]]]),
                start_pos=cache.start_pos,
                state_cache=cache,
            )
            hf_step = hf_model(
                input_ids=torch.tensor([[hf_tokens[-1]]]), past_key_values=past, use_cache=True
            )
        cache.advance_start_pos(1)
        past = hf_step.past_key_values
        ours_tokens.append(int(step[0, -1].argmax()))
        hf_tokens.append(int(hf_step.logits[0, -1].argmax()))
    assert ours_tokens == hf_tokens


def test_chunked_prefill_splits_mid_conv_kernel(pair):
    """Splitting a prefill must not disturb the conv tails or the rope offsets."""
    _, ours = pair
    prompt = torch.randint(0, VOCAB, (1, 70))
    one_shot_cache = ours.build_state_cache(max_seq_len=128, batch_size=1)
    with torch.no_grad():
        one_shot = ours.forward_prefill_with_cache(prompt, state_cache=one_shot_cache)
    chunked_cache = ours.build_state_cache(max_seq_len=128, batch_size=1)
    with torch.no_grad():
        for chunk in (prompt[:, :6], prompt[:, 6:9], prompt[:, 9:]):
            chunked = ours.forward_prefill_with_cache(chunk, state_cache=chunked_cache)
            chunked_cache.advance_start_pos(chunk.shape[1])
    assert torch.allclose(chunked[0, -1], one_shot[0, -1], atol=1e-4)


def test_ragged_decode_matches_single_request_oracle(pair):
    """Two prompts of different lengths in one batch, after copy_row_from."""
    _, ours = pair
    prompts = [torch.randint(0, VOCAB, (1, 9)), torch.randint(0, VOCAB, (1, 5))]
    singles = []
    for prompt in prompts:
        cache = ours.build_state_cache(max_seq_len=64, batch_size=1)
        with torch.no_grad():
            logits = ours.forward_prefill_with_cache(prompt, state_cache=cache)
        cache.advance_start_pos(prompt.shape[1])
        singles.append((cache, int(logits[0, -1].argmax())))

    batched = ours.build_state_cache(max_seq_len=64, batch_size=2)
    for row, (cache, _) in enumerate(singles):
        batched.copy_row_from(cache, src_row=0, dst_row=row)
    positions = torch.tensor([prompts[0].shape[1], prompts[1].shape[1]])
    tokens = torch.tensor([[singles[0][1]], [singles[1][1]]])
    with torch.no_grad():
        ragged = ours.forward_decode_with_cache_ragged(
            tokens, positions=positions, state_cache=batched
        )
        oracle = [
            ours.forward_decode_with_cache(
                torch.tensor([[token]]), start_pos=prompt.shape[1], state_cache=cache
            )[0, -1]
            for (cache, token), prompt in zip(singles, prompts, strict=True)
        ]
    for row in range(2):
        assert torch.allclose(ragged[row, -1], oracle[row], atol=1e-4)


def test_text_only_rope_equals_reference_multi_stream_rope(pair):
    """mrope collapses to plain 1D partial rope when the streams are identical.

    The text backbone's rotary embedding is nominally three-component. For
    text-only input all three streams carry the same positions, and the
    interleave then rewrites each section with identical values. Feeding the
    reference explicit identical streams must therefore match its own default
    path, and both must match what our plain partial rope produces (which the
    prefill parity test already covers end to end).
    """
    hf_model, ours = pair
    ids = torch.randint(0, VOCAB, (1, 11))
    explicit = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, ids.shape[1])
    with torch.no_grad():
        default_path = hf_model(input_ids=ids).logits
        multi_stream = hf_model(input_ids=ids, position_ids=explicit).logits
        got = ours(ids)
    assert torch.allclose(default_path, multi_stream, atol=1e-6)
    assert torch.allclose(got, multi_stream, atol=1e-4)


def test_load_weights_drops_vision_tower_and_mtp_head(pair):
    """The released checkpoints bundle both; this port serves text only."""
    hf_model, _ = pair
    state = _checkpoint_state(hf_model)
    state["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(3, 3)
    state["model.visual.merger.norm.bias"] = torch.zeros(3)
    state["mtp.fc.weight"] = torch.zeros(2, 2)
    state["mtp.layers.0.self_attn.q_proj.weight"] = torch.zeros(2, 2)
    fresh = Qwen35ForConditionalGeneration(Qwen35Config.from_hf(_hf_config())).eval().float()
    Qwen35ForConditionalGeneration.load_weights(fresh, state)
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        assert torch.allclose(fresh(ids), hf_model(input_ids=ids).logits, atol=1e-4)


def _write_checkpoint(directory, text_config, state: dict[str, torch.Tensor]) -> None:
    """Write `state` as two safetensors shards plus the multimodal config.json."""
    import json

    from safetensors.torch import save_file

    config = transformers.Qwen3_5Config(
        text_config=text_config.to_dict(), tie_word_embeddings=text_config.tie_word_embeddings
    )
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.save_pretrained(directory)
    keys = sorted(state)
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {key: shard_names[i % 2] for i, key in enumerate(keys)}
    for shard in shard_names:
        save_file(
            {key: state[key].contiguous() for key in keys if weight_map[key] == shard},
            str(directory / shard),
        )
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_from_checkpoint_matches_in_memory_load(tmp_path, tie_word_embeddings):
    """The streaming, built-on-device load gives the same model as `load_weights`.

    The checkpoint has two shards and carries vision and MTP tensors, as the
    released ones do. The 4B ties its embeddings and the 27B does not, so both
    layouts are covered.
    """
    torch.manual_seed(0)
    text_config = _hf_config()
    text_config.tie_word_embeddings = tie_word_embeddings
    hf_model = Qwen3_5ForCausalLM(text_config).eval().float()
    for param in hf_model.parameters():
        with torch.no_grad():
            param.uniform_(-0.08, 0.08)
    state = _checkpoint_state(hf_model)
    if tie_word_embeddings:
        # Tied checkpoints store the embedding only.
        state.pop("lm_head.weight", None)
    state["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(3, 3)
    state["mtp.fc.weight"] = torch.zeros(2, 2)
    _write_checkpoint(tmp_path, text_config, state)

    in_memory = Qwen35ForConditionalGeneration(Qwen35Config.from_hf(text_config)).eval().float()
    Qwen35ForConditionalGeneration.load_weights(in_memory, state)
    streamed = Qwen35ForConditionalGeneration.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.float32
    )

    assert streamed.cfg.tie_word_embeddings is tie_word_embeddings
    if tie_word_embeddings:
        assert streamed.lm_head.weight is streamed.model.embed_tokens.weight
    ids = torch.randint(0, VOCAB, (1, 9))
    with torch.no_grad():
        assert torch.equal(streamed(ids), in_memory(ids))
        assert torch.allclose(streamed(ids), hf_model(input_ids=ids).logits, atol=1e-4)


def test_from_checkpoint_builds_in_the_requested_dtype(tmp_path):
    """Parameters come out in `dtype`; the rotary table stays fp32."""
    torch.manual_seed(0)
    text_config = _hf_config()
    hf_model = Qwen3_5ForCausalLM(text_config).eval()
    _write_checkpoint(tmp_path, text_config, _checkpoint_state(hf_model))

    model = Qwen35ForConditionalGeneration.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.bfloat16
    )

    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}
    rotary_tables = [buf for name, buf in model.named_buffers() if name.endswith("inv_freq")]
    assert rotary_tables
    assert all(table.dtype == torch.float32 for table in rotary_tables)
    assert torch.get_default_dtype() == torch.float32
