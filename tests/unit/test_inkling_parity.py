"""Inkling owned model: full-model bit-parity vs the HF reference.

The 975B checkpoint (and the 276B Inkling-Small) are out of reach for CPU CI,
so this uses a tiny-random config exercising the distinctive bits: hybrid
sliding (window 8, 4 KV heads) vs global (2 KV heads) attention layers, the
learned relative bias with a rel_extent (6) SHORTER than the context so the
zero-beyond-extent path fires on global layers, log length scaling (n_floor 4,
so tau > 1 for most positions), all four per-layer short convolutions, dense
(with global_scale) vs MoE MLPs, the shared-expert-sink router, and the muP
logits divide + unpadded-vocab slice. Identical weights are loaded into both
models (via load_weights on HF's state_dict), so any divergence is a math bug,
not a loading one. Skips cleanly when transformers < 5.14 (no inkling module).

Beyond one-shot prefill parity, the decode tests pin the serving path: greedy
and batched ragged decode step through the PagedKVCache, where the SConv tails
are gathered from the conv_* streams, token-equal with HF's incremental
decode (which carries its own per-layer conv states). Chunked prefill crosses
a conv/window boundary mid-prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models import REGISTRY
from mini_infer.models.inkling import InklingConfig, InklingForCausalLM


def _make_cache(
    model: InklingForCausalLM, *, num_blocks: int = 64, num_slots: int = 1
) -> PagedKVCache:
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
        layer_attention=model.per_layer_attention(),
        layer_kv_shape=model.per_layer_kv_shape(),
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    for _ in range(num_slots):
        cache.add_request_slot()
    return cache


def test_registry_has_inkling() -> None:
    assert REGISTRY.lookup("InklingForConditionalGeneration") is InklingForCausalLM


def _tiny_hf_config():
    from transformers.models.inkling.configuration_inkling import InklingTextConfig

    # Both layer types (2 sliding + 2 global) and both MLP types (2 dense +
    # 2 sparse). Window 8 < the 12-token prefill so sliding layers clip;
    # rel_extent 6 < context so the global bias zeroes beyond the extent;
    # n_floor 4 so log scaling is active from position 4 on. Sliding layers
    # have MORE KV heads (4) than global ones (2), like the real config
    # (16 vs 8), which pins the per-layer heterogeneous pool shapes.
    return InklingTextConfig(
        vocab_size=128,
        unpadded_vocab_size=120,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=6,
        log_scaling_n_floor=4,
        log_scaling_alpha=0.1,
        layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        intermediate_size=96,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=4,
        n_shared_experts=2,
        route_scale=8.0,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
        logits_mup_width_multiplier=24.0,
    )


def _randomize_scales_and_biases(hf_model: Any) -> None:
    """HF's _init_weights zeroes the router bias and ones the global scales,
    which would let a port that DROPS those parameters still pass. Randomize
    them so the tests actually depend on them."""
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                param.normal_(0.0, 0.5)
            elif name.endswith("global_scale"):
                param.uniform_(0.7, 1.3)


def _build_hf_and_mine() -> tuple[Any, InklingForCausalLM]:
    from transformers.models.inkling.modeling_inkling import (
        InklingForCausalLM as HFModel,
    )

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_config()
    # The relative bias flows through HF's attention interface as a
    # `position_bias` kwarg that ONLY the eager path consumes; sdpa would
    # silently drop it (HF ships the family with flash-attn disabled for
    # the same reason).
    hf_cfg._attn_implementation = "eager"
    hf_model = HFModel(hf_cfg).to(torch.float32).eval()
    _randomize_scales_and_biases(hf_model)

    my_model = InklingForCausalLM(InklingConfig.from_hf(hf_cfg)).to(torch.float32).eval()
    InklingForCausalLM.load_weights(my_model, hf_model.state_dict())
    return hf_model, my_model


def test_full_model_parity_vs_hf() -> None:
    """Load the HF state_dict through load_weights, then full-model logit parity."""
    pytest.importorskip("transformers.models.inkling.modeling_inkling")
    hf_model, my_model = _build_hf_and_mine()

    total_q = 12
    input_ids = torch.randint(0, my_model.cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32)

    with torch.inference_mode():
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        cache = _make_cache(my_model)
        my_logits = my_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            cu_seqlens_q=cu_seqlens_q,
        )

    # Both sides slice to unpadded_vocab_size (120 of 128).
    assert hf_logits.shape == my_logits.shape, f"{hf_logits.shape} vs {my_logits.shape}"
    assert my_logits.shape[-1] == 120
    cs = float(
        torch.nn.functional.cosine_similarity(hf_logits.flatten(), my_logits.flatten(), dim=0)
    )
    assert cs > 0.999, f"full-model logit parity failed: cos_sim={cs:.6f}"
    assert torch.equal(hf_logits.argmax(dim=-1), my_logits.argmax(dim=-1))
    assert torch.allclose(hf_logits, my_logits, atol=1e-3), (
        f"max_abs_diff={(hf_logits - my_logits).abs().max().item():.6f}"
    )


def _hf_greedy(hf_model: Any, prompt: list[int], n_new: int) -> list[int]:
    """HF incremental greedy decode (its own cache, conv states included)."""
    tokens = list(prompt)
    past = None
    cur = torch.tensor([prompt], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(n_new):
            out = hf_model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long)
    return tokens


def test_greedy_decode_parity_vs_hf() -> None:
    """Multi-step greedy decode matches HF token-for-token.

    HF decodes incrementally with its own conv-state cache; mini-infer decodes
    through the PagedKVCache, rebuilding each step's conv outputs from the
    conv_* stream tails. The context grows past the sliding window (8) and the
    global rel_extent (6), so both clipping paths are active during decode.
    """
    pytest.importorskip("transformers.models.inkling.modeling_inkling")
    hf_model, my_model = _build_hf_and_mine()

    prompt = [3, 1, 4, 1, 5]
    n_new = 8  # context reaches 13 tokens > window 8 > rel_extent 6

    hf_tokens = _hf_greedy(hf_model, prompt, n_new)

    my_tokens = list(prompt)
    cache = _make_cache(my_model)
    plen = len(prompt)
    with torch.inference_mode():
        logits = my_model(
            input_ids=torch.tensor([prompt], dtype=torch.long),
            position_ids=torch.arange(plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], dtype=torch.int32),
        )
        nxt = int(logits[0, -1].argmax())
        my_tokens.append(nxt)
        cache_len = plen
        for _ in range(n_new - 1):
            logits = my_model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long),
                position_ids=torch.tensor([[cache_len]], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            )
            cache_len += 1
            nxt = int(logits[0, -1].argmax())
            my_tokens.append(nxt)

    assert my_tokens == hf_tokens, f"HF {hf_tokens} vs ours {my_tokens}"


def test_batched_ragged_decode_parity_vs_hf() -> None:
    """Two prompts of different lengths decode together through ONE shared cache.

    Prefill is one ragged packed forward; each decode step is one packed
    forward of two tokens at their own positions, the continuous-batching
    shape. Each stream's per-request conv tails must stay separated."""
    pytest.importorskip("transformers.models.inkling.modeling_inkling")
    hf_model, my_model = _build_hf_and_mine()

    # Second prompt is exactly kernel_size long: transformers 5.14's
    # update_conv_state path crashes on prompts SHORTER than the conv kernel
    # (shape mismatch in InklingShortConvolution.forward), so 4 is the
    # shortest raggedness we can reference-check against HF.
    prompts = [[3, 1, 4, 1, 5], [2, 7, 2, 6]]
    n_new = 6

    expected = [_hf_greedy(hf_model, p, n_new) for p in prompts]

    batch = len(prompts)
    cache = _make_cache(my_model, num_slots=batch)
    gen = [list(p) for p in prompts]
    cur_len = [len(p) for p in prompts]

    packed = [tok for p in prompts for tok in p]
    cu = [0]
    pos: list[int] = []
    for p in prompts:
        cu.append(cu[-1] + len(p))
        pos.extend(range(len(p)))

    with torch.inference_mode():
        logits = my_model(
            input_ids=torch.tensor([packed], dtype=torch.long),
            position_ids=torch.tensor([pos], dtype=torch.long),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        )
        nxt = [int(logits[0, cu[b + 1] - 1].argmax()) for b in range(batch)]
        for b in range(batch):
            gen[b].append(nxt[b])

        for _ in range(n_new - 1):
            logits = my_model(
                input_ids=torch.tensor([nxt], dtype=torch.long),
                position_ids=torch.tensor([[cur_len[b] for b in range(batch)]], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor(list(range(batch + 1)), dtype=torch.int32),
            )
            nxt = [int(logits[0, b].argmax()) for b in range(batch)]
            for b in range(batch):
                gen[b].append(nxt[b])
                cur_len[b] += 1

    assert gen == expected, f"HF {expected} vs ours {gen}"


def test_chunked_prefill_matches_one_shot() -> None:
    """Prefilling 12 tokens as 7 + 5 gives the same logits as one shot.

    The chunk boundary lands inside the conv kernel's reach AND inside the
    sliding window, so the second chunk's SConvs must pick up their tails
    from the conv_* streams and its attention must see the first chunk's
    K/V. Compared against HF's full-sequence logits for the tail positions."""
    pytest.importorskip("transformers.models.inkling.modeling_inkling")
    hf_model, my_model = _build_hf_and_mine()

    total_q = 12
    torch.manual_seed(1)
    ids = torch.randint(0, my_model.cfg.vocab_size, (total_q,), dtype=torch.long)

    with torch.inference_mode():
        hf_logits = hf_model(input_ids=ids.unsqueeze(0), use_cache=False).logits

        cache = _make_cache(my_model)
        split = 7
        my_model(
            input_ids=ids[:split].unsqueeze(0),
            position_ids=torch.arange(split, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, split], dtype=torch.int32),
        )
        tail_logits = my_model(
            input_ids=ids[split:].unsqueeze(0),
            position_ids=torch.arange(split, total_q, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, total_q - split], dtype=torch.int32),
        )

    assert torch.allclose(hf_logits[:, split:], tail_logits, atol=1e-3), (
        f"max_abs_diff={(hf_logits[:, split:] - tail_logits).abs().max().item():.6f}"
    )


def test_gate_matches_hf_router() -> None:
    """Component parity: our InklingGate vs HF's InklingTopkRouter on the same
    weights: indices, routed weights, and shared gammas all equal."""
    pytest.importorskip("transformers.models.inkling.modeling_inkling")
    from transformers.models.inkling.modeling_inkling import InklingTopkRouter

    from mini_infer.models.blocks.inkling_moe import InklingGate

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_config()
    hf_router = InklingTopkRouter(hf_cfg)
    with torch.no_grad():
        hf_router.weight.normal_(0.0, 0.5)
        hf_router.e_score_correction_bias.normal_(0.0, 0.5)
        hf_router.global_scale.uniform_(0.7, 1.3)

    gate = InklingGate(
        hidden_size=hf_cfg.hidden_size,
        n_routed_experts=hf_cfg.n_routed_experts,
        n_shared_experts=hf_cfg.n_shared_experts,
        top_k=hf_cfg.num_experts_per_tok,
        route_scale=hf_cfg.route_scale,
    )
    gate.load_state_dict(hf_router.state_dict())

    tokens = torch.randn(16, hf_cfg.hidden_size)
    with torch.inference_mode():
        _, hf_weights, hf_indices, hf_gammas = hf_router(tokens)
        my_weights, my_indices, my_gammas = gate(tokens)

    assert torch.equal(hf_indices, my_indices)
    assert torch.allclose(hf_weights, my_weights, atol=1e-6)
    assert torch.allclose(hf_gammas, my_gammas, atol=1e-6)


def test_sconv_prefill_decode_consistency() -> None:
    """Component check: our SConv over a full sequence equals running it
    step-by-step with the (kernel_size - 1)-token tails a decode step gathers."""
    from mini_infer.models.blocks.inkling_sconv import InklingShortConv

    torch.manual_seed(0)
    channels, kernel_size, seq = 6, 4, 10
    conv = InklingShortConv(channels, kernel_size)
    with torch.no_grad():
        conv.weight.normal_(0.0, 0.5)
    x = torch.randn(seq, channels)

    with torch.inference_mode():
        full = conv(x, None)
        stepped = torch.empty_like(full)
        for t in range(seq):
            tail = x[max(0, t - (kernel_size - 1)) : t]
            stepped[t] = conv(x[t : t + 1], tail)

    assert torch.allclose(full, stepped, atol=1e-6)
