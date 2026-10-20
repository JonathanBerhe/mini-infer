"""Nemotron-H parity against the pinned transformers implementation (fp32, CPU).

Weights reach our model through the production `load_weights` with
checkpoint-shaped keys (`backbone.` prefix, one tensor per routed expert,
`(channels, 1, kernel)` conv taps, an `mtp.` layer), so the remapping is
tested together with the math.

Which reference to use: up to transformers 5.14, the CPU fallback for
Mamba-2 (`torch_forward`, also in Zamba2) sums the inter-chunk recurrence
over the target chunk instead of the source chunk, so its one-shot prefill is
wrong for spans longer than `chunk_size` (the CUDA kernels are correct).
transformers 5.15.0 fixed it as part of a refactor
(huggingface/transformers#47452). The token-by-token decode path shares no
code with the chunked scan, so the tests compare spans of one chunk against
the one-shot reference and longer spans against the reference fed one token
at a time, which holds on every version.
`test_reference_chunked_fallback_matches_its_version` pins which behavior
the installed transformers has.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

transformers = pytest.importorskip("transformers")
HFNemotronHForCausalLM = pytest.importorskip(
    "transformers.models.nemotron_h.modeling_nemotron_h"
).NemotronHForCausalLM

VOCAB = 97
CHUNK = 8
HYBRID = ["mamba", "moe", "attention", "mamba", "moe", "attention"]


def _hf_config(layer_types: list[str] = HYBRID, **overrides: object) -> object:
    kwargs: dict[str, object] = {
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "layers_block_type": layer_types,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "mamba_num_heads": 8,
        "mamba_head_dim": 8,
        "ssm_state_size": 16,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": CHUNK,
        "intermediate_size": 48,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "moe_shared_expert_intermediate_size": 40,
        "routed_scaling_factor": 2.5,
        "use_mamba_kernels": False,
        "num_nextn_predict_layers": 0,
        "tie_word_embeddings": False,
    }
    kwargs.update(overrides)
    return transformers.NemotronHConfig(**kwargs)


def _randomize(hf: torch.nn.Module) -> None:
    """Spread the weights so routing and decay differences show in the logits.

    `dt_bias` stays positive, which keeps every step size well above the
    reference CPU path's `time_step_min` clamp (see the model docstring).
    """
    with torch.no_grad():
        for name, param in hf.named_parameters():
            if name.endswith("dt_bias"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("A_log"):
                param.uniform_(-1.0, 1.0)
            else:
                param.uniform_(-0.15, 0.15)
        for name, buffer in hf.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.uniform_(-0.05, 0.05)


def _checkpoint_state(hf: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The reference state dict in the released checkpoint's layout."""
    state: dict[str, torch.Tensor] = {}
    for key, tensor in hf.state_dict().items():
        name = "backbone." + key[len("model.") :] if key.startswith("model.") else key
        if name.endswith(("mixer.experts.up_proj", "mixer.experts.down_proj")):
            base, proj = name.rsplit(".", 1)
            for expert in range(tensor.shape[0]):
                state[f"{base}.{expert}.{proj}.weight"] = tensor[expert].clone()
            continue
        state[name] = tensor.clone()
    return state


def _pair(layer_types: list[str] = HYBRID, **overrides: object):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    hf_cfg = _hf_config(layer_types, **overrides)
    hf = HFNemotronHForCausalLM(hf_cfg).eval().float()
    _randomize(hf)
    ours = NemotronHForCausalLM(NemotronHConfig.from_hf(hf_cfg)).eval().float()
    NemotronHForCausalLM.load_weights(ours, _checkpoint_state(hf))
    return hf, ours


@pytest.fixture
def pair():  # type: ignore[no-untyped-def]
    return _pair()


def _reference_token_by_token(hf: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Reference logits `(T, vocab)` for a one-row prompt, fed one token at a time."""
    with torch.no_grad():
        out = hf(input_ids=ids[:, :1], use_cache=True)
        past = out.past_key_values
        logits = [out.logits[0, -1]]
        for t in range(1, ids.shape[1]):
            out = hf(input_ids=ids[:, t : t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1])
    return torch.stack(logits)


def test_layer_kinds_come_from_the_config(pair) -> None:  # type: ignore[no-untyped-def]
    _, ours = pair
    assert ours.cfg.layer_types == HYBRID


def test_prefill_within_one_chunk_matches_reference(pair) -> None:  # type: ignore[no-untyped-def]
    hf, ours = pair
    ids = torch.randint(0, VOCAB, (2, CHUNK))
    with torch.no_grad():
        expected = hf(input_ids=ids, use_cache=False).logits
        got = ours(ids)
    assert torch.equal(expected.argmax(-1), got.argmax(-1))
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_prefill_across_chunks_matches_reference_token_by_token(pair) -> None:  # type: ignore[no-untyped-def]
    hf, ours = pair
    ids = torch.randint(0, VOCAB, (1, 3 * CHUNK - 3))
    expected = _reference_token_by_token(hf, ids)
    with torch.no_grad():
        got = ours(ids)[0]
    assert torch.equal(expected.argmax(-1), got.argmax(-1))
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_reference_chunked_fallback_matches_its_version(pair) -> None:  # type: ignore[no-untyped-def]
    """Up to 5.14 the one-shot reference is off past one chunk; from 5.15.0 it is right."""
    from packaging.version import Version

    hf, _ = pair
    ids = torch.randint(0, VOCAB, (1, 3 * CHUNK - 3))
    with torch.no_grad():
        one_shot = hf(input_ids=ids, use_cache=False).logits[0]
    gap = float((one_shot - _reference_token_by_token(hf, ids)).abs().max())
    if Version(transformers.__version__) >= Version("5.15.0"):
        assert gap < 1e-5
    else:
        assert gap > 1e-6


def test_greedy_decode_matches_reference(pair) -> None:  # type: ignore[no-untyped-def]
    """Prefill across a chunk boundary, then decode greedily through the cache."""
    hf, ours = pair
    prompt = torch.randint(0, VOCAB, (1, CHUNK + 3))
    steps = 10
    with torch.no_grad():
        cache = ours.build_state_cache(max_seq_len=64, batch_size=1)
        logits = ours.forward_prefill_with_cache(prompt, state_cache=cache)
        cache.advance_start_pos(prompt.shape[1])
        tokens = [int(logits[0, -1].argmax())]
        for _ in range(steps):
            step = ours.forward_decode_with_cache(
                torch.tensor([[tokens[-1]]]), start_pos=cache.start_pos, state_cache=cache
            )
            cache.advance_start_pos(1)
            tokens.append(int(step[0, -1].argmax()))
    sequence = torch.cat([prompt, torch.tensor([tokens[:-1]])], dim=1)
    expected = _reference_token_by_token(hf, sequence)
    reference_tokens = expected[prompt.shape[1] - 1 :].argmax(-1).tolist()
    assert tokens == reference_tokens


def test_chunked_prefill_matches_one_shot(pair) -> None:  # type: ignore[no-untyped-def]
    _, ours = pair
    prompt = torch.randint(0, VOCAB, (1, 3 * CHUNK - 3))
    with torch.no_grad():
        one_shot_cache = ours.build_state_cache(max_seq_len=64, batch_size=1)
        one_shot = ours.forward_prefill_with_cache(prompt, state_cache=one_shot_cache)
        chunked_cache = ours.build_state_cache(max_seq_len=64, batch_size=1)
        for piece in (prompt[:, :5], prompt[:, 5:13], prompt[:, 13:]):
            chunked = ours.forward_prefill_with_cache(piece, state_cache=chunked_cache)
            chunked_cache.advance_start_pos(piece.shape[1])
    torch.testing.assert_close(chunked[0, -1], one_shot[0, -1], rtol=1e-5, atol=1e-5)


def test_ragged_decode_matches_single_request_oracle(pair) -> None:  # type: ignore[no-untyped-def]
    _, ours = pair
    prompts = [torch.randint(0, VOCAB, (1, 11)), torch.randint(0, VOCAB, (1, 4))]
    singles = []
    with torch.no_grad():
        for prompt in prompts:
            cache = ours.build_state_cache(max_seq_len=32, batch_size=1)
            logits = ours.forward_prefill_with_cache(prompt, state_cache=cache)
            cache.advance_start_pos(prompt.shape[1])
            singles.append((cache, int(logits[0, -1].argmax())))
    batched = ours.build_state_cache(max_seq_len=32, batch_size=2)
    for row, (cache, _) in enumerate(singles):
        batched.copy_row_from(cache, src_row=0, dst_row=row)
    positions = torch.tensor([11, 4])
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
        torch.testing.assert_close(ragged[row, -1], oracle[row], rtol=1e-5, atol=1e-5)


def test_dense_mlp_and_latent_moe_layers_match_reference() -> None:
    """The Bolt-style latent MoE and the dense MLP layer kind, within one chunk."""
    hf, ours = _pair(["mamba", "mlp", "moe", "attention"], moe_latent_size=24)
    assert ours.cfg.layer_types == ["mamba", "mlp", "moe", "attention"]
    ids = torch.randint(0, VOCAB, (2, CHUNK))
    with torch.no_grad():
        expected = hf(input_ids=ids, use_cache=False).logits
        got = ours(ids)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_load_drops_mtp_and_accepts_the_legacy_embedding_name(pair) -> None:  # type: ignore[no-untyped-def]
    hf, ours = pair
    state = _checkpoint_state(hf)
    state["backbone.embedding.weight"] = state.pop("backbone.embeddings.weight")
    state["mtp.layers.0.eh_proj.weight"] = torch.zeros(3, 3)
    state["mtp.layers.0.mixer.experts.0.up_proj.weight"] = torch.zeros(3, 3)
    fresh = NemotronHForCausalLM(ours.cfg).eval().float()
    NemotronHForCausalLM.load_weights(fresh, state)
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        assert torch.equal(fresh(ids), ours(ids))


def test_load_rejects_a_missing_expert(pair) -> None:  # type: ignore[no-untyped-def]
    hf, ours = pair
    state = _checkpoint_state(hf)
    del state["backbone.layers.1.mixer.experts.3.down_proj.weight"]
    with pytest.raises(ValueError, match="incomplete experts"):
        NemotronHForCausalLM.load_weights(NemotronHForCausalLM(ours.cfg), state)
