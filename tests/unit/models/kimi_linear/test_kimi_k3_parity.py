"""Kimi K3 text stack: full-model bit-parity vs the vendored K3 reference.

The 2.8T checkpoint is out of reach, so this uses a tiny-random config that
switches on every K3 delta at toy scale: SiTU-GLU with the real betas
(4.0 / 25.0), low-rank Q plus the MLA output gate, the full-rank KDA output
gate with the bounded decay (`gate_lower_bound = -5.0`), latent MoE with
its RMSNorm (hidden 64 -> latent 48), two shared experts, and AttnRes with
a block size of 2 over 6 layers (snapshots at layers 0, 2 and 4, so the
mixers see 1 to 3 blocks). The hybrid is 5 KDA layers to 1 MLA layer
(`kda_layers` [1,2,3,5,6], `full_attn_layers` [4], 1-indexed like the real
config) with conv kernel 4 shorter than the prompts. Identical weights are
loaded into both models via `load_weights` on the reference's state_dict.

The oracle is the checkpoint's own `modeling_kimi_linear.py` (pinned
revision, vendored by `scripts/kimi_linear/clone_kimi_k3_reference.py`) on the
FLA naive-reference semantics (`_kimi_reference_helpers`, including the
in-kernel gate and `sigmoid(beta)` forms K3 asks the kernels for); skips
cleanly when the reference isn't vendored. The decode tests pin the serving
path exactly as the Kimi Linear suite does: greedy, batched-ragged, and
chunked prefill across a conv-kernel boundary.

Uninitialized reference parameters (`dt_bias`, `e_score_correction_bias`)
and the all-ones norms that would make a test insensitive (`o_norm`, the
AttnRes query norms, the latent-MoE norm) are randomized before any forward.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.models import REGISTRY
from mini_infer.models.kimi_linear.kimi_k3 import KimiK3ForConditionalGeneration
from mini_infer.models.kimi_linear.kimi_linear import (
    KimiLinearConfig,
    KimiLinearForCausalLM,
)


def _tiny_reference_config() -> Any:
    from kimi_k3_reference.configuration_kimi_k3 import KimiLinearConfig as ReferenceConfig

    return ReferenceConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_token=3,
        num_shared_experts=2,
        routed_scaling_factor=1.0,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        routed_expert_hidden_size=48,
        latent_moe_use_norm=True,
        q_lora_rank=24,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=2,
        linear_attn_config={
            "kda_layers": [1, 2, 3, 5, 6],
            "full_attn_layers": [4],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )


def _randomize_degenerate_parameters(reference_model: Any) -> None:
    """Randomize what the reference leaves uninitialized (`torch.empty`) or at
    an all-ones init that would hide a port bug: a dropped gate bias, a
    dropped KDA dt_bias, an AttnRes mixer that ignores its query, or a
    latent-MoE norm that is skipped."""
    with torch.no_grad():
        for name, param in reference_model.named_parameters():
            if name.endswith(("dt_bias", "e_score_correction_bias")):
                param.normal_(0.0, 0.5)
            elif name.endswith(("o_norm.weight", "res_norm.weight", "routed_expert_norm.weight")):
                param.uniform_(0.7, 1.3)
            elif name.endswith("res_proj.weight"):
                param.normal_(0.0, 1.0)


def _build_reference_and_mine(kimi_k3_reference: Any) -> tuple[Any, KimiLinearForCausalLM]:
    torch.manual_seed(0)
    ref_cfg = _tiny_reference_config()
    reference_model = kimi_k3_reference.KimiLinearForCausalLM(ref_cfg).to(torch.float32).eval()
    # KimiLinearModel.__init__ force-overrides to flash_attention_2; flip
    # back to eager AFTER construction so the MLA layer runs on CPU.
    reference_model.config._attn_implementation = "eager"
    _randomize_degenerate_parameters(reference_model)

    my_model = KimiLinearForCausalLM(KimiLinearConfig.from_hf(ref_cfg)).to(torch.float32).eval()
    KimiLinearForCausalLM.load_weights(my_model, reference_model.state_dict())
    return reference_model, my_model


def test_registry_has_kimi_k3() -> None:
    assert REGISTRY.lookup("KimiK3ForConditionalGeneration") is KimiK3ForConditionalGeneration


def test_from_hf_switches_on_the_k3_delta(kimi_k3_reference: Any) -> None:
    cfg = KimiLinearConfig.from_hf(_tiny_reference_config())
    assert cfg.hidden_act == "situ"
    assert (cfg.situ_beta, cfg.situ_linear_beta) == (4.0, 25.0)
    assert cfg.q_lora_rank == 24
    assert cfg.mla_use_output_gate is True
    assert cfg.kda_use_full_rank_gate is True
    assert cfg.kda_gate_lower_bound == -5.0
    assert cfg.routed_expert_hidden_size == 48
    assert cfg.latent_moe_use_norm is True
    assert cfg.attn_res_block_size == 2
    assert cfg.moe_weights_from_biased_scores is False


def test_full_model_parity_vs_reference(kimi_k3_reference: Any) -> None:
    """Load the reference state_dict through load_weights, then full-model
    prefill logit parity (12 tokens crossing the conv kernel)."""
    reference_model, my_model = _build_reference_and_mine(kimi_k3_reference)

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


def test_greedy_decode_parity_vs_reference(kimi_k3_reference: Any) -> None:
    """Multi-step greedy decode matches the reference token-for-token through
    the KimiStateCache (AttnRes mixes per token, so it adds no state)."""
    reference_model, my_model = _build_reference_and_mine(kimi_k3_reference)

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


def test_batched_ragged_decode_parity_vs_reference(kimi_k3_reference: Any) -> None:
    """Two prompts of different lengths decode together through ONE batched
    KimiStateCache, token-equal with the reference run per prompt."""
    reference_model, my_model = _build_reference_and_mine(kimi_k3_reference)

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


def test_chunked_prefill_matches_reference(kimi_k3_reference: Any) -> None:
    """Prefilling 12 tokens as 7 + 5 gives the reference's full-sequence
    logits for the tail (conv tails, KDA state and MLA buffer carried)."""
    reference_model, my_model = _build_reference_and_mine(kimi_k3_reference)

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


def test_moe_gate_matches_reference(kimi_k3_reference: Any) -> None:
    """Component parity: our gate in the K3 convention vs the reference
    `KimiMoEGate`, with a randomized correction bias."""
    from mini_infer.models.kimi_linear.kimi_linear import _KimiMoeGate

    torch.manual_seed(0)
    ref_cfg = _tiny_reference_config()
    ref_gate = kimi_k3_reference.KimiMoEGate(ref_cfg).float().eval()
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


def test_k3_gate_weights_are_unbiased_scores(kimi_k3_reference: Any) -> None:
    """Regression pin for the K3-vs-Kimi-Linear router difference: the K3
    reference adds the bias out of place, so its expert weights ARE the
    unbiased sigmoid scores (the opposite of the Kimi Linear pin)."""
    torch.manual_seed(0)
    ref_cfg = _tiny_reference_config()
    ref_gate = kimi_k3_reference.KimiMoEGate(ref_cfg).float().eval()
    with torch.no_grad():
        ref_gate.weight.normal_(0.0, 0.5)
        ref_gate.e_score_correction_bias.normal_(0.0, 1.0)

    tokens = torch.randn(8, ref_cfg.hidden_size)
    with torch.inference_mode():
        idx, weights = ref_gate(tokens.unsqueeze(0))
        scores = torch.sigmoid(tokens.float() @ ref_gate.weight.float().t())
        unbiased = scores.gather(1, idx)
        unbiased = unbiased / (unbiased.sum(-1, keepdim=True) + 1e-20)
        unbiased = unbiased * ref_cfg.routed_scaling_factor

    assert torch.allclose(weights, unbiased, atol=1e-6), (
        "K3 reference gate returned biased weights; the convention pin no longer holds"
    )
