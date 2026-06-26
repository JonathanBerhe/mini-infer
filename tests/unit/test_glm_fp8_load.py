"""GLM-MoE-DSA block-FP8 + per-expert checkpoint loading.

The published GLM-5.2-FP8 checkpoint (confirmed via the safetensors index) stores
routed experts PER-EXPERT (`experts.{j}.{gate,up,down}_proj.weight`) and FP8-quant
most 2-D weights in `[128,128]` blocks (`e4m3` weight + `.weight_scale_inv` scale),
including partial-block shapes like `kv_a_proj_with_mqa` (576 rows). These tests
exercise `GlmMoeDsaForCausalLM.load_weights`'s ceil-aware dequant + per-expert
rename on a synthetic checkpoint, with no download and no GPU.
"""

from __future__ import annotations

import math
import re

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM, _dequant_block_fp8

PROMPT = [3, 1, 4, 1, 5, 9]
# Weights the real checkpoint FP8-quantizes (everything else stays BF16).
_FP8_PROJ = (
    ".q_a_proj.",
    ".q_b_proj.",
    ".kv_a_proj_with_mqa.",
    ".kv_b_proj.",
    ".o_proj.",
    ".gate_proj.",
    ".up_proj.",
    ".down_proj.",
    ".wq_b.",
    ".wk.",
)
_W_TO_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_EXPERT_W_RE = re.compile(r"\.(experts\.\d+|shared_experts)\.(w[123])\.weight$")


def _make_cfg() -> GlmMoeDsaConfig:
    # Dims chosen so FP8 weights span ÷128 blocks AND a partial block:
    # kv_a_proj_with_mqa is (kv_lora+qk_rope=192, hidden=128) -> ceil(192/128)=2.
    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        kv_lora_rank=128,
        q_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        index_topk=4,
        index_head_dim=128,
        index_n_heads=2,
        mlp_layer_types=("dense", "dense", "dense", "sparse"),
        indexer_types=("full", "shared", "full", "shared"),
    )


def _mini_to_hf_name(name: str) -> str:
    """Reverse our w1/w2/w3 expert naming back to the checkpoint's *_proj names."""
    m = _EXPERT_W_RE.search(name)
    if m is None:
        return name
    return name[: m.start(2)] + _W_TO_PROJ[m.group(2)] + ".weight"


def _quant_block_fp8(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a 2-D BF16 weight to e4m3 + per-tile (ceil) scale."""
    rows, cols = w.shape
    nbm, nbn = math.ceil(rows / block), math.ceil(cols / block)
    q = torch.zeros((rows, cols), dtype=torch.float32)
    scale = torch.zeros((nbm, nbn), dtype=torch.float32)
    for bi in range(nbm):
        for bj in range(nbn):
            r0, r1 = bi * block, min((bi + 1) * block, rows)
            c0, c1 = bj * block, min((bj + 1) * block, cols)
            blk = w[r0:r1, c0:c1].float()
            s = float(blk.abs().max()) / 448.0  # e4m3 max ~ 448
            s = s if s > 0 else 1.0
            scale[bi, bj] = s
            q[r0:r1, c0:c1] = blk / s
    return q.to(torch.float8_e4m3fn), scale


def _to_hf_state_dict(mini: GlmMoeDsaForCausalLM, *, fp8: bool) -> dict[str, torch.Tensor]:
    """Emit a checkpoint-format state_dict: per-expert *_proj names, optional FP8."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in mini.state_dict().items():
        hf_name = _mini_to_hf_name(name)
        eligible = hf_name.endswith(".weight") and any(p in hf_name for p in _FP8_PROJ)
        if fp8 and eligible and tensor.ndim == 2:
            q, scale = _quant_block_fp8(tensor)
            out[hf_name] = q
            out[hf_name + "_scale_inv"] = scale
        else:
            out[hf_name] = tensor
    return out


def _prefill_logits(mini: GlmMoeDsaForCausalLM) -> torch.Tensor:
    pool = BlockPool(
        num_blocks=8,
        block_size=4,
        num_layers=mini.cfg.num_hidden_layers,
        num_kv_heads=1,
        head_dim=mini.cfg.kv_lora_rank,
        dtype=torch.float32,
        device="cpu",
        layer_streams=mini.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    plen = len(PROMPT)
    with torch.inference_mode():
        return mini(
            input_ids=torch.tensor([PROMPT], dtype=torch.long),
            position_ids=torch.arange(plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], dtype=torch.int32),
        )


def test_dequant_block_fp8_handles_partial_blocks() -> None:
    """The ceil-aware dequant round-trips a quantized weight, including a row
    count that is not a multiple of the block size (192 = 128 + 64)."""
    torch.manual_seed(0)
    w = torch.randn(192, 128)
    q, scale = _quant_block_fp8(w)
    assert scale.shape == (2, 1)  # ceil(192/128)=2, ceil(128/128)=1
    recovered = _dequant_block_fp8(q, scale)
    assert recovered.shape == (192, 128)
    cs = torch.nn.functional.cosine_similarity(recovered.flatten().float(), w.flatten(), dim=0)
    assert float(cs) > 0.99  # e4m3 is lossy but should track closely


def test_per_expert_bf16_load_is_exact() -> None:
    """Per-expert (checkpoint) layout, BF16: load must reconstruct the model
    exactly (validates the gate/up/down -> w1/w3/w2 rename in isolation)."""
    torch.manual_seed(0)
    ref = GlmMoeDsaForCausalLM(_make_cfg()).to(torch.float32).eval()
    ref_logits = _prefill_logits(ref)

    fresh = GlmMoeDsaForCausalLM(_make_cfg()).to(torch.float32).eval()
    GlmMoeDsaForCausalLM.load_weights(fresh, _to_hf_state_dict(ref, fp8=False))
    assert torch.allclose(ref_logits, _prefill_logits(fresh), atol=1e-5)


def test_per_expert_fp8_load_recovers_model() -> None:
    """Per-expert layout + block-FP8: load dequantizes + renames, recovering a
    model whose logits track the BF16 reference within FP8 quantization error."""
    torch.manual_seed(0)
    ref = GlmMoeDsaForCausalLM(_make_cfg()).to(torch.float32).eval()
    ref_logits = _prefill_logits(ref)

    fresh = GlmMoeDsaForCausalLM(_make_cfg()).to(torch.float32).eval()
    GlmMoeDsaForCausalLM.load_weights(fresh, _to_hf_state_dict(ref, fp8=True))
    fp8_logits = _prefill_logits(fresh)

    assert torch.all(torch.isfinite(fp8_logits))
    cs = float(
        torch.nn.functional.cosine_similarity(ref_logits.flatten(), fp8_logits.flatten(), dim=0)
    )
    assert cs > 0.97, f"FP8-loaded logits drifted too far from BF16: cos={cs:.4f}"
