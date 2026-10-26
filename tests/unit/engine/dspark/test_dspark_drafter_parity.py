"""DSpark Qwen3 drafter: micro-config bit-parity vs DeepSpec's reference math.

DeepSpec (github.com/deepseek-ai/DeepSpec) is not an installed package (it's
not on PyPI), so unlike `test_minimax_m3_parity.py` there's no HF modeling
code to import as the oracle. Each `_ref_*` function below is instead a
direct, literal re-transcription of the exact DeepSpec source it cites,
written independently of `mini_infer.engine.dspark`'s implementation. Both
sides read weights from the SAME `Qwen3DSparkDrafter` instance (via its
submodules), so any divergence is a math bug, not a loading or seeding one.

Mechanics and citations: `docs/decisions/ADR-027-dspark-drafter-port.md`.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from mini_infer.engine.dspark import Qwen3DSparkConfig, Qwen3DSparkDrafter
from mini_infer.engine.dspark.attention import apply_dspark_rotary_pos_emb
from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.models.blocks.rope import RotaryEmbedding


def _tiny_config() -> Qwen3DSparkConfig:
    return Qwen3DSparkConfig(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        target_layer_ids=[0, 1],
        mask_token_id=47,
        block_size=3,
        markov_rank=5,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )


def _ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _ref_apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`deepspec/modeling/dspark/qwen3/modeling.py`'s local `apply_rotary_pos_emb`
    override (NOT the plain HF one): q is rotated with only the LAST `q_len`
    rows of cos/sin, k with the full table."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_ref_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_ref_rotate_half(k) * sin)
    return q_embed, k_embed


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm's compute order (fp32 variance, rsqrt, cast back)."""
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight * x32.to(input_dtype)


def _ref_attention_forward(
    self_attn, hidden_states: torch.Tensor, target_hidden_states: torch.Tensor, cos, sin, cfg
) -> torch.Tensor:
    """`Qwen3DSparkAttention.forward`, re-derived independently of `DSparkAttention`."""
    bsz, q_len = hidden_states.shape[:2]
    ctx_len = target_hidden_states.shape[1]
    q = self_attn.q_proj(hidden_states).view(bsz, q_len, cfg.num_attention_heads, cfg.head_dim)
    q = _ref_rms_norm(q, self_attn.q_norm.weight, cfg.rms_norm_eps).transpose(1, 2)
    k_ctx = self_attn.k_proj(target_hidden_states)
    k_noise = self_attn.k_proj(hidden_states)
    v_ctx = self_attn.v_proj(target_hidden_states)
    v_noise = self_attn.v_proj(hidden_states)
    k = torch.cat([k_ctx, k_noise], dim=1).view(
        bsz, ctx_len + q_len, cfg.num_key_value_heads, cfg.head_dim
    )
    v = torch.cat([v_ctx, v_noise], dim=1).view(
        bsz, ctx_len + q_len, cfg.num_key_value_heads, cfg.head_dim
    )
    k = _ref_rms_norm(k, self_attn.k_norm.weight, cfg.rms_norm_eps).transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = _ref_apply_rotary_pos_emb(q, k, cos, sin)
    groups = cfg.num_attention_heads // cfg.num_key_value_heads
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scale = cfg.head_dim**-0.5
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    attn = torch.softmax(scores, dim=-1) @ v.float()
    attn = attn.to(hidden_states.dtype).transpose(1, 2).reshape(bsz, q_len, -1)
    out: torch.Tensor = self_attn.o_proj(attn)
    return out


def _ref_swiglu(mlp, x: torch.Tensor) -> torch.Tensor:
    gate = functional.silu(mlp.gate_proj(x))
    up = mlp.up_proj(x)
    out: torch.Tensor = mlp.down_proj(gate * up)
    return out


def _ref_forward_backbone(
    drafter: Qwen3DSparkDrafter,
    noise_embedding: torch.Tensor,
    target_hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """`Qwen3DSparkModel._forward_backbone`, re-derived from the SAME weights `drafter` holds."""
    cfg = drafter.cfg
    hidden_states = noise_embedding
    target_hidden_states = _ref_rms_norm(
        drafter.fc(target_hidden_states), drafter.hidden_norm.weight, cfg.rms_norm_eps
    )
    cos, sin = drafter.rotary_emb(hidden_states, position_ids)
    for layer in drafter.layers:
        residual = hidden_states
        x = _ref_rms_norm(hidden_states, layer.input_layernorm.weight, cfg.rms_norm_eps)
        x = _ref_attention_forward(layer.self_attn, x, target_hidden_states, cos, sin, cfg)
        hidden_states = residual + x

        residual = hidden_states
        x = _ref_rms_norm(hidden_states, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x = _ref_swiglu(layer.mlp, x)
        hidden_states = residual + x
    return _ref_rms_norm(hidden_states, drafter.norm.weight, cfg.rms_norm_eps)


def _ref_sample_block_tokens(
    markov_head, base_logits: torch.Tensor, *, first_prev_token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`VanillaMarkov.sample_block_tokens` at temperature 0 (greedy): re-derived independently."""
    _batch_size, proposal_len = base_logits.shape[:2]
    sampled_tokens = []
    corrected_logits = []
    prev_token_ids = first_prev_token_ids.long()
    for step_idx in range(proposal_len):
        prev_embed = markov_head.markov_w1(prev_token_ids)
        bias = markov_head.markov_w2(prev_embed)
        step_logits = base_logits[:, step_idx, :] + bias
        corrected_logits.append(step_logits.unsqueeze(1))
        next_token_ids = step_logits.argmax(dim=-1)
        sampled_tokens.append(next_token_ids)
        prev_token_ids = next_token_ids
    return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def test_dspark_rotary_pos_emb_matches_reference() -> None:
    torch.manual_seed(0)
    bsz, num_heads, num_kv_heads, head_dim = 1, 4, 2, 8
    ctx_len, q_len = 5, 3
    q = torch.randn(bsz, num_heads, q_len, head_dim)
    k = torch.randn(bsz, num_kv_heads, ctx_len + q_len, head_dim)
    rotary = RotaryEmbedding(head_dim, base=10000.0)
    position_ids = torch.arange(ctx_len + q_len).unsqueeze(0)
    cos, sin = rotary(q, position_ids)

    q_ours, k_ours = apply_dspark_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
    q_ref, k_ref = _ref_apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
    assert torch.allclose(q_ours, q_ref, atol=1e-6)
    assert torch.allclose(k_ours, k_ref, atol=1e-6)


def test_dspark_backbone_and_logits_match_reference() -> None:
    torch.manual_seed(1)
    cfg = _tiny_config()
    drafter = Qwen3DSparkDrafter(cfg)
    drafter.eval()

    bsz, ctx_len = 1, 5
    target_hidden = torch.randn(bsz, ctx_len, len(cfg.target_layer_ids) * cfg.hidden_size)
    draft_input_ids = torch.full((bsz, cfg.block_size), cfg.mask_token_id, dtype=torch.long)
    draft_input_ids[:, 0] = 3
    noise_embed = drafter.embed_tokens(draft_input_ids)
    position_ids = torch.arange(ctx_len + cfg.block_size).unsqueeze(0)

    with torch.no_grad():
        hidden_ours = drafter.forward_backbone(
            noise_embedding=noise_embed,
            target_hidden_states=target_hidden,
            position_ids=position_ids,
        )
        hidden_ref = _ref_forward_backbone(drafter, noise_embed, target_hidden, position_ids)
        logits_ours = drafter.compute_logits(hidden_ours)
        logits_ref = drafter.lm_head(hidden_ref)

    assert torch.allclose(hidden_ours, hidden_ref, atol=1e-5), (
        (hidden_ours - hidden_ref).abs().max()
    )
    assert torch.allclose(logits_ours, logits_ref, atol=1e-5), (
        (logits_ours - logits_ref).abs().max()
    )


def test_dspark_markov_sample_block_tokens_matches_reference() -> None:
    torch.manual_seed(2)
    cfg = _tiny_config()
    drafter = Qwen3DSparkDrafter(cfg)
    drafter.eval()
    assert drafter.markov_head is not None

    base_logits = torch.randn(1, cfg.block_size, cfg.vocab_size)
    first_prev_token_ids = torch.tensor([3])

    with torch.no_grad():
        sampled_ours, corrected_ours = drafter.sample_draft_tokens(
            base_logits, first_prev_token_ids=first_prev_token_ids, temperature=0.0
        )
        sampled_ref, corrected_ref = _ref_sample_block_tokens(
            drafter.markov_head, base_logits, first_prev_token_ids=first_prev_token_ids
        )

    assert torch.equal(sampled_ours, sampled_ref)
    assert torch.allclose(corrected_ours, corrected_ref, atol=1e-6)


def test_dspark_confidence_head_matches_reference() -> None:
    torch.manual_seed(3)
    cfg = _tiny_config()
    drafter = Qwen3DSparkDrafter(cfg)
    drafter.eval()
    assert drafter.confidence_head is not None
    assert drafter.markov_head is not None

    hidden = torch.randn(1, cfg.block_size, cfg.hidden_size)
    prev_token_ids = torch.tensor([[3, 7, 11]])

    with torch.no_grad():
        conf_ours = drafter.predict_confidence_step(hidden, prev_token_ids=prev_token_ids)
        prev_embed_ref = drafter.markov_head.markov_w1(prev_token_ids.long())
        features_ref = torch.cat([hidden, prev_embed_ref], dim=-1)
        conf_ref = drafter.confidence_head.proj(features_ref).squeeze(-1).float()

    assert conf_ours is not None
    assert torch.allclose(conf_ours, conf_ref, atol=1e-6)
    # No sigmoid inside the module: raw logits can be outside [0, 1].
    assert not torch.all((conf_ours >= 0.0) & (conf_ours <= 1.0))


def test_dspark_draft_cache_incremental_matches_single_shot() -> None:
    """Two cached rounds must equal one uncached forward over the concatenated context.

    Confirms the position-id contiguous-slice scheme (ADR-027) and the
    cache's accumulate-context / discard-block behavior are self-consistent:
    fc + RMSNorm are per-token operations, so projecting round 1's and
    round 2's context together (single shot) or separately then
    concatenating (incremental via cache) must agree exactly.
    """
    torch.manual_seed(4)
    cfg = _tiny_config()
    drafter = Qwen3DSparkDrafter(cfg)
    drafter.eval()
    cache = DSparkDraftCache(cfg.num_hidden_layers)

    prompt_len = 5
    ctx1 = torch.randn(1, prompt_len, len(cfg.target_layer_ids) * cfg.hidden_size)
    round1_ids = torch.full((1, cfg.block_size), cfg.mask_token_id, dtype=torch.long)
    round1_ids[:, 0] = 3
    start_1 = prompt_len
    position_ids_1 = torch.arange(0, start_1 + cfg.block_size).unsqueeze(0)

    with torch.no_grad():
        drafter.forward_backbone(
            noise_embedding=drafter.embed_tokens(round1_ids),
            target_hidden_states=ctx1,
            position_ids=position_ids_1,
            past_key_values=cache,
        )
    # Crop back to round 1's OWN start: discards round 1's block K/V
    # entirely, keeps only the 5 ctx1 rows (matching `crop(start)` using the
    # round's start value BEFORE it advances, not after).
    cache.truncate_to(start_1)
    assert cache.get_seq_length() == prompt_len

    accepted = 2  # 2 of 3 draft tokens accepted this round
    start_2 = start_1 + accepted + 1
    ctx2_len = accepted + 1
    ctx2 = torch.randn(1, ctx2_len, len(cfg.target_layer_ids) * cfg.hidden_size)
    round2_ids = torch.full((1, cfg.block_size), cfg.mask_token_id, dtype=torch.long)
    round2_ids[:, 0] = 9
    position_ids_2 = torch.arange(cache.get_seq_length(), start_2 + cfg.block_size).unsqueeze(0)

    with torch.no_grad():
        hidden_incremental = drafter.forward_backbone(
            noise_embedding=drafter.embed_tokens(round2_ids),
            target_hidden_states=ctx2,
            position_ids=position_ids_2,
            past_key_values=cache,
        )

        # Single shot: no cache, context = ALL of round 1's kept ctx1 rows
        # plus round 2's ctx2 rows, query = only round 2's block, one
        # absolute position span covering everything. Valid because fc +
        # RMSNorm are per-token ops: projecting ctx1 and ctx2 together here
        # or separately across the two incremental calls must agree exactly.
        combined_ctx = torch.cat([ctx1, ctx2], dim=1)
        position_ids_combined = torch.arange(0, prompt_len + ctx2_len + cfg.block_size).unsqueeze(0)
        hidden_single_shot = drafter.forward_backbone(
            noise_embedding=drafter.embed_tokens(round2_ids),
            target_hidden_states=combined_ctx,
            position_ids=position_ids_combined,
            past_key_values=None,
        )

    assert torch.allclose(hidden_incremental, hidden_single_shot, atol=1e-5), (
        (hidden_incremental - hidden_single_shot).abs().max()
    )
