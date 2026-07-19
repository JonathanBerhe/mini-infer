"""Kimi Linear owned model: full-model bit-parity vs the vendored reference.

The 48B checkpoint is out of reach for CPU CI, so this uses a tiny-random
config exercising the distinctive bits: a 3:1 KDA/MLA hybrid (kda_layers
[1,2,3], full_attn_layers [4], both 1-indexed like the real config), the
short conv kernel (4) shorter than the prompts so tails matter, NoPE MLA
(the rope-dim splits used unrotated), a dense layer 0 + MoE elsewhere, and
the Kimi router's in-place-bias semantics (expert weights gathered from
BIASED sigmoid scores; the bias is randomized so a GLM-style unbiased gate
cannot pass). Identical weights are loaded into both models via
`load_weights` on the reference's state_dict, so any divergence is a math
bug, not a loading one.

The oracle is the checkpoint's own `modeling_kimi.py` (pinned revision,
vendored by `scripts/clone_kimi_linear_reference.py`) running on the FLA
naive-reference semantics (`_kimi_reference_helpers`); skips cleanly when
the reference isn't vendored. Beyond one-shot prefill parity, the decode
tests pin the serving path: greedy, batched-ragged (two requests through
one KimiStateCache, the continuous-batching shape), and chunked prefill
that crosses a conv-kernel boundary mid-prompt.

The reference leaves `dt_bias` and `e_score_correction_bias` uninitialized
(`torch.empty`); the builder ALWAYS randomizes them before any forward, both
to keep the oracle finite and to make the tests depend on those parameters.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.models import REGISTRY
from mini_infer.models.kimi_linear import (
    KimiLinearConfig,
    KimiLinearForCausalLM,
)


def _tiny_reference_config() -> Any:
    from kimi_linear_reference.configuration_kimi import KimiLinearConfig as ReferenceConfig

    # Mirrors the real 48B shape at toy scale: 3 KDA layers to 1 MLA layer,
    # conv kernel 4 (< the 12-token prompts, so conv tails carry context),
    # dense layer 0 (first_k_dense_replace=1), 8-expert top-3 MoE with a
    # shared expert and the real routed_scaling_factor.
    return ReferenceConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=96,
        moe_intermediate_size=32,
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
        q_lora_rank=None,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        },
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )


def _randomize_empty_parameters(reference_model: Any) -> None:
    """The reference constructs `dt_bias` and `e_score_correction_bias` with
    `torch.empty` and never initializes them (its `_init_weights` covers only
    Linear/Embedding), so a fresh model may hold inf/NaN. Randomizing also
    makes the tests DEPEND on these parameters: a port that dropped the gate
    bias or the KDA dt_bias could not pass. `o_norm` weights leave HF's
    all-ones init for the same reason."""
    with torch.no_grad():
        for name, param in reference_model.named_parameters():
            if name.endswith("dt_bias") or name.endswith("e_score_correction_bias"):
                param.normal_(0.0, 0.5)
            elif name.endswith("o_norm.weight"):
                param.uniform_(0.7, 1.3)


def _build_reference_and_mine(kimi_reference: Any) -> tuple[Any, KimiLinearForCausalLM]:
    torch.manual_seed(0)
    ref_cfg = _tiny_reference_config()
    reference_model = kimi_reference.KimiLinearForCausalLM(ref_cfg).to(torch.float32).eval()
    # KimiLinearModel.__init__ force-overrides to flash_attention_2 (GPU
    # serving assumption); flip back to eager AFTER construction so the MLA
    # layers run on CPU. The layers read the config at call time.
    reference_model.config._attn_implementation = "eager"
    _randomize_empty_parameters(reference_model)

    my_model = KimiLinearForCausalLM(KimiLinearConfig.from_hf(ref_cfg)).to(torch.float32).eval()
    KimiLinearForCausalLM.load_weights(my_model, reference_model.state_dict())
    return reference_model, my_model


def test_registry_has_kimi_linear() -> None:
    assert REGISTRY.lookup("KimiLinearForCausalLM") is KimiLinearForCausalLM


def test_full_model_parity_vs_reference(kimi_reference: Any) -> None:
    """Load the reference state_dict through load_weights, then full-model
    prefill logit parity (12 tokens crossing the conv kernel)."""
    reference_model, my_model = _build_reference_and_mine(kimi_reference)

    total = 12
    input_ids = torch.randint(0, my_model.cfg.vocab_size, (1, total), dtype=torch.long)
    with torch.inference_mode():
        ref_logits = reference_model(input_ids=input_ids, use_cache=False).logits
        my_logits = my_model(input_ids)

    assert ref_logits.shape == my_logits.shape
    cos = float(
        torch.nn.functional.cosine_similarity(ref_logits.flatten(), my_logits.flatten(), dim=0)
    )
    assert cos > 0.999, f"full-model logit parity failed: cos_sim={cos:.6f}"
    assert torch.equal(ref_logits.argmax(dim=-1), my_logits.argmax(dim=-1))
    assert torch.allclose(ref_logits, my_logits, atol=1e-3), (
        f"max_abs_diff={(ref_logits - my_logits).abs().max().item():.6f}"
    )


def _reference_greedy(reference_model: Any, prompt: list[int], n_new: int) -> list[int]:
    """Reference incremental greedy decode (its own KimiDynamicCache: conv
    tails + recurrent state + MLA KV)."""
    tokens = list(prompt)
    past = None
    current = torch.tensor([prompt], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(n_new):
            out = reference_model(input_ids=current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = int(out.logits[0, -1].argmax())
            tokens.append(next_token)
            current = torch.tensor([[next_token]], dtype=torch.long)
    return tokens


def test_greedy_decode_parity_vs_reference(kimi_reference: Any) -> None:
    """Multi-step greedy decode matches the reference token-for-token.

    The reference decodes through its KimiDynamicCache; mini-infer through
    the KimiStateCache (delta-rule matrix + conv tails + dense MLA buffer).
    Decode extends well past the conv kernel so the rolling tails, not the
    prompt, carry the conv context."""
    reference_model, my_model = _build_reference_and_mine(kimi_reference)

    prompt = [3, 1, 4, 1, 5, 9]
    n_new = 8

    ref_tokens = _reference_greedy(reference_model, prompt, n_new)

    my_tokens = list(prompt)
    cache = my_model.build_state_cache(max_seq_len=32)
    with torch.inference_mode():
        logits = my_model.forward_prefill_with_cache(
            torch.tensor([prompt], dtype=torch.long), state_cache=cache
        )
        cache.advance_start_pos(len(prompt))
        next_token = int(logits[0, -1].argmax())
        my_tokens.append(next_token)
        for _ in range(n_new - 1):
            logits = my_model.forward_decode_with_cache(
                torch.tensor([[next_token]], dtype=torch.long),
                start_pos=cache.start_pos,
                state_cache=cache,
            )
            cache.advance_start_pos(1)
            next_token = int(logits[0, -1].argmax())
            my_tokens.append(next_token)

    assert my_tokens == ref_tokens, f"reference {ref_tokens} vs ours {my_tokens}"


def test_batched_ragged_decode_parity_vs_reference(kimi_reference: Any) -> None:
    """Two prompts of different lengths decode together through ONE batched
    KimiStateCache: per-request prefill into B=1 caches, `copy_row_from`
    into the batch (the scheduler's admit move), then ragged steps with
    per-row positions. Token-equal with the reference run per prompt."""
    reference_model, my_model = _build_reference_and_mine(kimi_reference)

    prompts = [[3, 1, 4, 1, 5, 9], [2, 7, 2, 6]]
    n_new = 6
    expected = [_reference_greedy(reference_model, p, n_new) for p in prompts]

    batch = len(prompts)
    batched = my_model.build_state_cache(max_seq_len=32, batch_size=batch)
    next_tokens: list[int] = []
    positions: list[int] = []
    generated = [list(p) for p in prompts]
    with torch.inference_mode():
        for row, prompt in enumerate(prompts):
            single = my_model.build_state_cache(max_seq_len=32, batch_size=1)
            logits = my_model.forward_prefill_with_cache(
                torch.tensor([prompt], dtype=torch.long), state_cache=single
            )
            batched.copy_row_from(single, src_row=0, dst_row=row)
            next_tokens.append(int(logits[0, -1].argmax()))
            positions.append(len(prompt))
            generated[row].append(next_tokens[row])

        for _ in range(n_new - 1):
            logits = my_model.forward_decode_with_cache_ragged(
                torch.tensor(next_tokens, dtype=torch.long).unsqueeze(1),
                positions=torch.tensor(positions, dtype=torch.long),
                state_cache=batched,
            )
            for row in range(batch):
                next_tokens[row] = int(logits[row, -1].argmax())
                generated[row].append(next_tokens[row])
                positions[row] += 1

    assert generated == expected, f"reference {expected} vs ours {generated}"


def test_chunked_prefill_matches_reference(kimi_reference: Any) -> None:
    """Prefilling 12 tokens as 7 + 5 gives the reference's full-sequence
    logits for the tail. The chunk boundary lands inside the conv kernel's
    reach, so the second chunk's convs must pick up their tails from the
    cached state, the KDA layers their carried matrix state, and the MLA
    layers the first chunk's buffer entries. V4's state path cannot do this;
    Kimi's position-free math makes it natural."""
    reference_model, my_model = _build_reference_and_mine(kimi_reference)

    total = 12
    torch.manual_seed(1)
    ids = torch.randint(0, my_model.cfg.vocab_size, (1, total), dtype=torch.long)

    with torch.inference_mode():
        ref_logits = reference_model(input_ids=ids, use_cache=False).logits

        cache = my_model.build_state_cache(max_seq_len=32)
        split = 7
        my_model.forward_prefill_with_cache(ids[:, :split], state_cache=cache)
        cache.advance_start_pos(split)
        tail_logits = my_model.forward_prefill_with_cache(ids[:, split:], state_cache=cache)
        cache.advance_start_pos(total - split)

    assert torch.allclose(ref_logits[:, split:], tail_logits, atol=1e-3), (
        f"max_abs_diff={(ref_logits[:, split:] - tail_logits).abs().max().item():.6f}"
    )


def test_moe_gate_matches_reference(kimi_reference: Any) -> None:
    """Component parity: our `_KimiMoeGate` vs the reference `KimiMoEGate` on
    the same weights, including the in-place bias semantics (expert weights
    gathered from BIASED scores). Selection order is unspecified
    (`sorted=False`), so compare per-token (index, weight) pairs sorted by
    index."""
    from mini_infer.models.kimi_linear import _KimiMoeGate

    torch.manual_seed(0)
    ref_cfg = _tiny_reference_config()
    ref_gate = kimi_reference.KimiMoEGate(ref_cfg).float().eval()
    with torch.no_grad():
        ref_gate.weight.normal_(0.0, 0.5)
        ref_gate.e_score_correction_bias.normal_(0.0, 0.5)

    my_gate = _KimiMoeGate(KimiLinearConfig.from_hf(ref_cfg)).float()
    my_gate.load_state_dict(ref_gate.state_dict())

    tokens = torch.randn(16, ref_cfg.hidden_size)
    with torch.inference_mode():
        ref_idx, ref_weights = ref_gate(tokens.unsqueeze(0))
        my_idx, my_weights = my_gate(tokens)

    ref_order = ref_idx.argsort(dim=-1)
    my_order = my_idx.argsort(dim=-1)
    assert torch.equal(ref_idx.gather(1, ref_order), my_idx.gather(1, my_order))
    assert torch.allclose(
        ref_weights.gather(1, ref_order), my_weights.gather(1, my_order), atol=1e-6
    )


def test_gate_weights_are_biased_scores(kimi_reference: Any) -> None:
    """Regression pin for the Kimi-vs-DeepSeek router difference: with a
    nonzero correction bias, the reference's expert weights are NOT the
    unbiased sigmoid scores (the GLM/DeepSeek-V3 convention). Guards against
    'simplifying' our gate to `GlmNoAuxTcGate` later."""
    torch.manual_seed(0)
    ref_cfg = _tiny_reference_config()
    ref_gate = kimi_reference.KimiMoEGate(ref_cfg).float().eval()
    with torch.no_grad():
        ref_gate.weight.normal_(0.0, 0.5)
        ref_gate.e_score_correction_bias.normal_(0.0, 1.0)

    tokens = torch.randn(8, ref_cfg.hidden_size)
    with torch.inference_mode():
        idx, weights = ref_gate(tokens.unsqueeze(0))
        # Recompute the UNBIASED convention on the same selection.
        scores = torch.sigmoid(tokens.float() @ ref_gate.weight.float().t())
        unbiased = scores.gather(1, idx)
        unbiased = unbiased / (unbiased.sum(-1, keepdim=True) + 1e-20)
        unbiased = unbiased * ref_cfg.routed_scaling_factor

    assert not torch.allclose(weights, unbiased, atol=1e-4), (
        "reference gate returned unbiased weights; the in-place-bias pin no longer holds"
    )
