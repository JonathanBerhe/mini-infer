"""Hash-routed MoE gate: shape contract, math, bit-parity vs V4 reference.

Six (mode x score_function) cells covered:

    | mode        | softmax | sigmoid | softplus_sqrt |
    |-------------|---------|---------|---------------|
    | hash        | ✓       | ✓       | ✓             |
    | score_topk  | ✓       | ✓       | ✓             |

Plus shape / validation tests (vocab_size required for hash mode, etc.).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.models.blocks.hash_routed_gate import HashRoutedGate

# ---------- Constructor validation ----------


def test_hash_mode_requires_positive_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        HashRoutedGate(
            hidden_size=8,
            num_routed_experts=4,
            num_activated_experts=2,
            routing_mode="hash",
        )


def test_score_topk_mode_does_not_require_vocab_size() -> None:
    HashRoutedGate(
        hidden_size=8,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="score_topk",
    )


def test_rejects_invalid_score_func() -> None:
    with pytest.raises(ValueError, match="score_func"):
        HashRoutedGate(
            hidden_size=8,
            num_routed_experts=4,
            num_activated_experts=2,
            routing_mode="score_topk",
            score_func="cosine",  # type: ignore[arg-type]
        )


def test_rejects_num_activated_greater_than_routed() -> None:
    with pytest.raises(ValueError, match="num_activated_experts"):
        HashRoutedGate(
            hidden_size=8,
            num_routed_experts=4,
            num_activated_experts=8,  # > num_routed_experts
            routing_mode="score_topk",
        )


# ---------- Hash mode behavior ----------


def test_hash_mode_indices_come_from_lookup_table() -> None:
    torch.manual_seed(0)
    vocab_size = 16
    num_routed = 4
    num_activated = 2
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=num_routed,
        num_activated_experts=num_activated,
        routing_mode="hash",
        vocab_size=vocab_size,
    )
    # Set a known lookup: token 5 routes to experts [1, 3], token 7 routes to [0, 2].
    with torch.no_grad():
        gate.tid2eid.zero_()
        gate.tid2eid[5] = torch.tensor([1, 3], dtype=torch.int32)
        gate.tid2eid[7] = torch.tensor([0, 2], dtype=torch.int32)

    hidden_states = torch.randn(2, 8)
    input_ids = torch.tensor([5, 7], dtype=torch.long)
    _, indices = gate(hidden_states, input_ids)
    assert indices.dtype == torch.int64
    assert indices.shape == (2, num_activated)
    assert torch.equal(indices[0], torch.tensor([1, 3]))
    assert torch.equal(indices[1], torch.tensor([0, 2]))


def test_hash_mode_requires_input_ids() -> None:
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="hash",
        vocab_size=16,
    )
    hidden_states = torch.randn(2, 8)
    with pytest.raises(ValueError, match="input_ids"):
        gate(hidden_states)


# ---------- Score-topk mode behavior ----------


def test_score_topk_mode_indices_match_argmax_with_bias() -> None:
    torch.manual_seed(0)
    num_routed = 4
    num_activated = 2
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=num_routed,
        num_activated_experts=num_activated,
        routing_mode="score_topk",
        score_func="softmax",
    )
    # Set an extreme bias that forces specific experts to be picked despite
    # the score function: bias[0] = +100 dominates softmax's [0, 1] range.
    with torch.no_grad():
        gate.bias.zero_()
        gate.bias[0] = 100.0
        gate.bias[2] = 50.0

    hidden_states = torch.randn(3, 8)
    _, indices = gate(hidden_states)
    # Expert 0 has the largest bias, expert 2 is next; both should be selected for every token.
    for token_idx in range(3):
        assert set(indices[token_idx].tolist()) == {0, 2}


def test_score_topk_weights_unaffected_by_bias() -> None:
    """Bias shifts top-k selection but not the gathered weights — the
    weights come from the unbiased score function output."""
    torch.manual_seed(0)
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="score_topk",
        score_func="softmax",
    )
    hidden_states = torch.randn(3, 8)
    # Take a snapshot of weights with bias=0.
    with torch.no_grad():
        gate.bias.zero_()
    weights_no_bias, indices_no_bias = gate(hidden_states)
    # Now add bias that doesn't change the topk indices ranking.
    with torch.no_grad():
        gate.bias.fill_(0.001)  # uniform tiny bias — shouldn't reorder topk
    weights_with_uniform_bias, indices_uniform = gate(hidden_states)
    # Indices should match (uniform bias doesn't change ranking).
    assert torch.equal(indices_no_bias, indices_uniform)
    # And weights should be identical (bias doesn't enter weights).
    torch.testing.assert_close(weights_no_bias, weights_with_uniform_bias)


# ---------- Score function math ----------


def test_softmax_does_not_renormalize_after_gather() -> None:
    """Softmax already integrates over ALL experts; the gathered top-k subset
    naturally doesn't sum to 1, and we don't fix that."""
    torch.manual_seed(0)
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=8,
        num_activated_experts=2,
        routing_mode="score_topk",
        score_func="softmax",
    )
    with torch.no_grad():
        gate.bias.zero_()
    hidden_states = torch.randn(4, 8)
    weights, _ = gate(hidden_states)
    # Sum over the activated experts is < 1.0 (the rest of the softmax mass
    # went to unselected experts).
    sums = weights.sum(dim=-1)
    assert torch.all(sums < 1.0)
    assert torch.all(sums > 0.0)


def test_sigmoid_renormalizes_to_sum_one_per_token() -> None:
    torch.manual_seed(0)
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=8,
        num_activated_experts=3,
        routing_mode="score_topk",
        score_func="sigmoid",
    )
    with torch.no_grad():
        gate.bias.zero_()
    hidden_states = torch.randn(5, 8)
    weights, _ = gate(hidden_states)
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-6, atol=1e-6)


def test_softplus_sqrt_renormalizes_to_sum_one_per_token() -> None:
    torch.manual_seed(0)
    gate = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=8,
        num_activated_experts=3,
        routing_mode="score_topk",
        score_func="softplus_sqrt",
    )
    with torch.no_grad():
        gate.bias.zero_()
    hidden_states = torch.randn(5, 8)
    weights, _ = gate(hidden_states)
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-6, atol=1e-6)


def test_route_scale_multiplies_weights() -> None:
    torch.manual_seed(0)
    gate_unit_scale = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="score_topk",
        score_func="softmax",
        route_scale=1.0,
    )
    gate_double_scale = HashRoutedGate(
        hidden_size=8,
        num_routed_experts=4,
        num_activated_experts=2,
        routing_mode="score_topk",
        score_func="softmax",
        route_scale=2.0,
    )
    # Sync weights so the only difference is route_scale.
    with torch.no_grad():
        gate_double_scale.weight.copy_(gate_unit_scale.weight)
        gate_double_scale.bias.copy_(gate_unit_scale.bias)
    hidden_states = torch.randn(3, 8)
    weights_unit, _ = gate_unit_scale(hidden_states)
    weights_double, _ = gate_double_scale(hidden_states)
    torch.testing.assert_close(weights_double, weights_unit * 2.0)


# ---------- Bit-parity vs the V4 reference's `Gate` ----------


def _build_reference_args(reference_module: Any, *, score_func: str, n_hash_layers: int) -> Any:
    """Construct ModelArgs configured for a Gate parity test."""
    return reference_module.ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        dtype="bf16",
        dim=64,
        n_layers=4,  # enough to span both hash and non-hash layer ids
        n_heads=4,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(4,) * 4,
        original_seq_len=0,
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        n_routed_experts=8,
        n_activated_experts=2,
        score_func=score_func,
        route_scale=1.5,
        n_hash_layers=n_hash_layers,
        vocab_size=32,
    )


def _seed_fill_module(module: Any, seed: int) -> None:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in module.parameters():
        with torch.no_grad():
            if parameter.dtype.is_floating_point:
                parameter.data = (
                    torch.randn(parameter.shape, generator=rng, dtype=torch.float32) * 0.02
                )
            elif parameter.dtype == torch.int32:
                # tid2eid lookup: random expert ids in [0, num_routed_experts).
                parameter.data = torch.randint(
                    0, 8, parameter.shape, generator=rng, dtype=torch.int32
                )


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "softplus_sqrt"])
@pytest.mark.parametrize("routing_mode", ["hash", "score_topk"])
def test_gate_matches_v4_reference(
    reference_module: Any, score_func: str, routing_mode: str
) -> None:
    """Bit-parity vs reference Gate across all (mode, score_func) cells."""
    # Reference's `score_func` field uses the same string for our "softplus_sqrt"
    # but expressed differently in the reference. Map:
    reference_score_func = {
        "softmax": "softmax",
        "sigmoid": "sigmoid",
        "softplus_sqrt": "softplus",  # the reference's else-branch is softplus.sqrt()
    }[score_func]
    n_hash_layers = 4 if routing_mode == "hash" else 0
    layer_id = 0  # always hash if n_hash_layers > 0, never if 0
    args = _build_reference_args(
        reference_module,
        score_func=reference_score_func,
        n_hash_layers=n_hash_layers,
    )
    ref_gate = reference_module.Gate(layer_id, args)
    _seed_fill_module(ref_gate, seed=42)

    our_gate = HashRoutedGate(
        hidden_size=args.dim,
        num_routed_experts=args.n_routed_experts,
        num_activated_experts=args.n_activated_experts,
        routing_mode=routing_mode,  # type: ignore[arg-type]
        score_func=score_func,  # type: ignore[arg-type]
        route_scale=args.route_scale,
        vocab_size=args.vocab_size if routing_mode == "hash" else None,
    )
    with torch.no_grad():
        our_gate.weight.copy_(ref_gate.weight)
        if routing_mode == "hash":
            our_gate.tid2eid.copy_(ref_gate.tid2eid)
        else:
            assert our_gate.bias is not None
            our_gate.bias.copy_(ref_gate.bias)

    torch.manual_seed(0)
    num_tokens = 12
    hidden_states = torch.randn(num_tokens, args.dim) * 0.5
    input_ids = torch.randint(0, args.vocab_size, (num_tokens,), dtype=torch.long)

    with torch.no_grad():
        ours_weights, ours_indices = our_gate(hidden_states, input_ids)
        theirs_weights, theirs_indices = ref_gate(
            hidden_states, input_ids if routing_mode == "hash" else None
        )

    # Indices: should match exactly (same selection logic).
    assert torch.equal(ours_indices, theirs_indices.to(torch.int64))
    # Weights: should match within fp32 tolerance.
    torch.testing.assert_close(ours_weights, theirs_weights.to(torch.float32), rtol=1e-5, atol=1e-6)
