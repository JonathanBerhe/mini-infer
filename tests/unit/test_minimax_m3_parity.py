"""MiniMax-M3 owned model: full-model bit-parity vs the HF reference.

The 428B checkpoint is out of reach for CPU CI, so this uses a tiny-random config
exercising the distinctive bits: dense (0-2) vs MSA+MoE (3-4) layers, per-head
Gemma QK-norm, full RoPE over head_dim, the block indexer + additive mask (sized
so top-k actually drops a block), and the swigluoai MoE. Identical weights are
loaded into both models (via load_weights), so any divergence is a math bug, not
a loading one. Skips cleanly when transformers < 5.12 (no minimax_m3_vl).

Beyond one-shot prefill parity, the decode tests pin the serving path: greedy
decode steps through the PagedKVCache (including the sparse layers' index_k
stream) and batched ragged decode through one shared cache, both token-equal
with HF's incremental decode.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.models import REGISTRY
from mini_infer.models.minimax_m3 import MiniMaxM3Config, MiniMaxM3ForCausalLM


def _make_cache(
    model: MiniMaxM3ForCausalLM, *, num_blocks: int = 32, num_slots: int = 1
) -> PagedKVCache:
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    for _ in range(num_slots):
        cache.add_request_slot()
    return cache


def test_registry_has_minimax_m3() -> None:
    assert REGISTRY.lookup("MiniMaxM3SparseForConditionalGeneration") is MiniMaxM3ForCausalLM


def _tiny_hf_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )

    # 3 dense + 2 sparse layers. index_block_size=4 with total_q=12 -> 3 blocks;
    # topk_blocks=2 + local -> at least one block is dropped (MSA is not a no-op).
    return MiniMaxM3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,  # MoE per-expert
        dense_intermediate_size=128,
        shared_intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=8,
        num_experts_per_tok=4,
        routed_scaling_factor=2.0,
        rms_norm_eps=1e-6,
        # Deployment-shaped rope config: FLAT rope_theta + partial_rotary_factor
        # (no explicit rope_parameters), exactly like the real config.json. HF
        # standardizes the flat factor into rope_parameters and runs PARTIAL
        # rope over the first rotary_dim dims; an explicit rope_parameters dict
        # without the factor silently degenerates HF to full rope and masks the
        # divergence (which is how the real-model gate caught it).
        rope_theta=5_000_000.0,
        partial_rotary_factor=0.5,
        rotary_dim=8,
        hidden_act="swigluoai",
        tie_word_embeddings=False,
        index_n_heads=2,
        index_head_dim=16,
        index_block_size=4,
        index_topk_blocks=2,
        index_local_blocks=1,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 2,
        layer_types=["full_attention"] * 3 + ["minimax_m3_sparse"] * 2,
    )


def test_full_model_parity_vs_hf() -> None:
    """Load the HF state_dict through load_weights, then full-model logit parity."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLForCausalLM as HFModel,
    )

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_config()
    hf_cfg._attn_implementation = "eager"
    hf_model = HFModel(hf_cfg).to(torch.float32).eval()

    my_cfg = MiniMaxM3Config.from_hf(hf_cfg)
    my_model = MiniMaxM3ForCausalLM(my_cfg).to(torch.float32).eval()
    MiniMaxM3ForCausalLM.load_weights(my_model, hf_model.state_dict())

    total_q = 12
    input_ids = torch.randint(0, my_cfg.vocab_size, (1, total_q), dtype=torch.long)
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

    assert hf_logits.shape == my_logits.shape, f"{hf_logits.shape} vs {my_logits.shape}"
    cs = float(
        torch.nn.functional.cosine_similarity(hf_logits.flatten(), my_logits.flatten(), dim=0)
    )
    assert cs > 0.999, f"full-model logit parity failed: cos_sim={cs:.6f}"
    assert torch.equal(hf_logits.argmax(dim=-1), my_logits.argmax(dim=-1))
    assert torch.allclose(hf_logits, my_logits, atol=1e-3), (
        f"max_abs_diff={(hf_logits - my_logits).abs().max().item():.6f}"
    )


def _build_hf_and_mine() -> tuple[Any, MiniMaxM3ForCausalLM]:
    """A tiny HF model + a weight-synced mini-infer model (shared by decode tests)."""
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLForCausalLM as HFModel,
    )

    hf_cfg = _tiny_hf_config()
    hf_cfg._attn_implementation = "eager"
    hf_model = HFModel(hf_cfg).to(torch.float32).eval()
    my_model = MiniMaxM3ForCausalLM(MiniMaxM3Config.from_hf(hf_cfg)).to(torch.float32).eval()
    MiniMaxM3ForCausalLM.load_weights(my_model, hf_model.state_dict())
    return hf_model, my_model


def _hf_greedy(hf_model: Any, prompt: list[int], n_new: int) -> list[int]:
    """HF incremental greedy decode (its own cache, indexer keys included)."""
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


def _mine_batched_generate(
    model: MiniMaxM3ForCausalLM, prompts: list[list[int]], n_new: int
) -> list[list[int]]:
    """Greedy-generate all prompts together through one shared PagedKVCache.

    Prefill is one ragged packed forward; each decode step is one packed forward
    of B tokens (one per request) at their own positions. Mirrors how the
    continuous-batching scheduler drives the model.
    """
    batch = len(prompts)
    cache = _make_cache(model, num_slots=batch)
    gen = [list(p) for p in prompts]
    cur_len = [len(p) for p in prompts]

    packed = [tok for p in prompts for tok in p]
    cu = [0]
    pos: list[int] = []
    for p in prompts:
        cu.append(cu[-1] + len(p))
        pos.extend(range(len(p)))

    with torch.inference_mode():
        logits = model(
            input_ids=torch.tensor([packed], dtype=torch.long),
            position_ids=torch.tensor([pos], dtype=torch.long),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        )
        nxt = [int(logits[0, cu[b + 1] - 1].argmax()) for b in range(batch)]
        for b in range(batch):
            gen[b].append(nxt[b])

        for _ in range(n_new - 1):
            logits = model(
                input_ids=torch.tensor([nxt], dtype=torch.long),
                position_ids=torch.tensor([[cur_len[b] for b in range(batch)]], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor(list(range(batch + 1)), dtype=torch.int32),
            )
            nxt = [int(logits[0, b].argmax()) for b in range(batch)]
            for b in range(batch):
                gen[b].append(nxt[b])
                cur_len[b] += 1
    return gen


def test_greedy_decode_parity_vs_hf() -> None:
    """Multi-step greedy decode matches HF token-for-token.

    HF decodes incrementally with its own cache (index keys included); mini-infer
    decodes through the PagedKVCache with the sparse layers' index_k stream. The
    context grows past topk_blocks * index_block_size, so decode-time block
    selection actually prunes (the mask is not a causal no-op).
    """
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()

    prompt = [3, 1, 4, 1, 5]
    n_new = 8  # context reaches 13 tokens -> 4 blocks of 4 > topk 2 + local

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


def test_greedy_decode_kernel_path_matches_hf() -> None:
    """Greedy decode with the paged decode path enabled matches HF tokens.

    On CPU the dispatcher runs the pure-torch block-sparse reference, so this
    pins the model wiring end to end: prefill takes the materialized oracle,
    every decode step takes `select_cached` + paged reads of only the selected
    blocks (plus the dense layers' full-history paged path). Tokens must equal
    HF exactly, same as the oracle path.
    """
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()
    my_model.set_decode_kernel(True)

    prompt = [3, 1, 4, 1, 5]
    n_new = 8

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


def test_batched_decode_kernel_path_matches_hf() -> None:
    """Batched ragged decode with the paged decode path on matches HF per request."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()
    my_model.set_decode_kernel(True)

    prompts = [[3, 1, 4, 1, 5], [2, 7, 1, 8]]
    n_new = 6

    hf_gen = [_hf_greedy(hf_model, p, n_new) for p in prompts]
    my_gen = _mine_batched_generate(my_model, prompts, n_new)

    assert my_gen == hf_gen, f"HF {hf_gen} vs ours {my_gen}"


def test_batched_decode_matches_hf() -> None:
    """Continuous-batching shape: two ragged-length prompts decoded together.

    Exercises per-request cu_seqlens in the indexer and SDPA, the index_k stream
    per cache slot, and block allocation across slots. Each request's tokens must
    equal HF run on that prompt alone.
    """
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()

    prompts = [[3, 1, 4, 1, 5], [2, 7, 1, 8]]  # different lengths
    n_new = 6

    hf_gen = [_hf_greedy(hf_model, p, n_new) for p in prompts]
    my_gen = _mine_batched_generate(my_model, prompts, n_new)

    assert my_gen == hf_gen, f"HF {hf_gen} vs ours {my_gen}"


def _to_disk_layout(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite the HF in-memory state_dict into the documented 428B disk layout.

    Per the spec's `model.safetensors.index.json` map: everything under a
    `language_model.` prefix, the MoE block under `block_sparse_moe.` with
    per-expert `experts.E.{w1,w3,w2}` (split from the stacked 3D tensors),
    the router bias at the block level, shared expert as separate
    `{gate,up,down}_proj`, and the indexer as `index_{q,k}_{proj,norm}`.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        if key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key[: -len(".experts.gate_up_proj")]
            inter = tensor.shape[1] // 2
            for j in range(tensor.shape[0]):
                base = f"language_model.{prefix.replace('.mlp', '.block_sparse_moe')}"
                out[f"{base}.experts.{j}.w1.weight"] = tensor[j, :inter]
                out[f"{base}.experts.{j}.w3.weight"] = tensor[j, inter:]
            continue
        if key.endswith(".mlp.experts.down_proj"):
            prefix = key[: -len(".experts.down_proj")]
            for j in range(tensor.shape[0]):
                base = f"language_model.{prefix.replace('.mlp', '.block_sparse_moe')}"
                out[f"{base}.experts.{j}.w2.weight"] = tensor[j]
            continue
        if key.endswith(".mlp.shared_experts.gate_up_proj.weight"):
            prefix = key[: -len(".gate_up_proj.weight")]
            half = tensor.shape[0] // 2
            base = f"language_model.{prefix.replace('.mlp.', '.block_sparse_moe.')}"
            out[f"{base}.gate_proj.weight"] = tensor[:half]
            out[f"{base}.up_proj.weight"] = tensor[half:]
            continue
        new_key = key
        if ".mlp.gate.e_score_correction_bias" in new_key:
            new_key = new_key.replace(
                ".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias"
            )
        elif ".mlp.gate.weight" in new_key:
            new_key = new_key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        elif ".mlp.shared_experts." in new_key:
            new_key = new_key.replace(".mlp.shared_experts.", ".block_sparse_moe.shared_experts.")
        new_key = new_key.replace(".self_attn.indexer.q_proj.", ".self_attn.index_q_proj.")
        new_key = new_key.replace(".self_attn.indexer.k_proj.", ".self_attn.index_k_proj.")
        new_key = new_key.replace(".self_attn.indexer.q_norm.", ".self_attn.index_q_norm.")
        new_key = new_key.replace(".self_attn.indexer.k_norm.", ".self_attn.index_k_norm.")
        out[f"language_model.{new_key}"] = tensor
    return out


def test_disk_layout_load_matches_in_memory_load() -> None:
    """The documented on-disk 428B layout loads to the same model as HF's
    in-memory state_dict (`block_sparse_moe.` MoE, per-expert w1/w3/w2, block-
    level router bias, index_* names, language_model. prefix). This is the
    contract `load_model` relies on when serving from safetensors."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    torch.manual_seed(0)
    hf_model, my_model = _build_hf_and_mine()

    disk_model = (
        MiniMaxM3ForCausalLM(MiniMaxM3Config.from_hf(_tiny_hf_config())).to(torch.float32).eval()
    )
    MiniMaxM3ForCausalLM.load_weights(disk_model, _to_disk_layout(hf_model.state_dict()))

    total_q = 12
    input_ids = torch.randint(0, my_model.cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu = torch.tensor([0, total_q], dtype=torch.int32)
    with torch.inference_mode():
        ref = my_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=_make_cache(my_model),
            cu_seqlens_q=cu,
        )
        got = disk_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=_make_cache(disk_model),
            cu_seqlens_q=cu,
        )
    assert torch.equal(ref, got)


def _quant_block_fp8(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-FP8 e4m3 quantization (ceil blocks), the staged-checkpoint format.

    Mirrors the GLM fixture generator; exact inverse of
    `dequantize_block_fp8_to_bf16_partial` up to e4m3 rounding.
    """
    import math

    out, inn = w.shape
    scale = torch.zeros(math.ceil(out / block), math.ceil(inn / block), dtype=torch.float32)
    q = torch.zeros_like(w, dtype=torch.float8_e4m3fn)
    for bi in range(scale.shape[0]):
        for bj in range(scale.shape[1]):
            blk = w[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block]
            amax = float(blk.abs().max())
            s = amax / 448.0 if amax > 0 else 1.0
            scale[bi, bj] = s
            q[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block] = (blk / s).to(
                torch.float8_e4m3fn
            )
    return q, scale


def _fp8_disk_layout(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Disk layout with routed experts pre-quantized to block-FP8 (staging format)."""
    disk = _to_disk_layout(hf_state_dict)
    out: dict[str, torch.Tensor] = {}
    expert_re = __import__("re").compile(r"\.block_sparse_moe\.experts\.\d+\.(w1|w2|w3)\.weight$")
    for key, tensor in disk.items():
        if expert_re.search(key):
            q, scale = _quant_block_fp8(tensor.float())
            out[key] = q
            out[key + "_scale_inv"] = scale
        else:
            out[key] = tensor
    return out


def test_fp8_resident_load_recovers_model() -> None:
    """The pre-quantized staged layout + expert_dtype='fp8' recovers the model.

    Routed experts stay e4m3-resident (`Fp8Expert` buffers); logits must track
    the bf16 reference within FP8 quantization error.
    """
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    torch.manual_seed(0)
    hf_model, ref_model = _build_hf_and_mine()

    hf_cfg = _tiny_hf_config()
    hf_cfg.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    fp8_cfg = MiniMaxM3Config.from_hf(hf_cfg)
    assert fp8_cfg.expert_dtype == "fp8"
    fp8_model = MiniMaxM3ForCausalLM(fp8_cfg).to(torch.float32).eval()
    MiniMaxM3ForCausalLM.load_weights(fp8_model, _fp8_disk_layout(hf_model.state_dict()))

    total_q = 12
    input_ids = torch.randint(0, fp8_cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu = torch.tensor([0, total_q], dtype=torch.int32)
    with torch.inference_mode():
        ref = ref_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=_make_cache(ref_model),
            cu_seqlens_q=cu,
        )
        got = fp8_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=_make_cache(fp8_model),
            cu_seqlens_q=cu,
        )
    assert torch.all(torch.isfinite(got))
    cs = float(torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0))
    assert cs > 0.97, f"FP8-resident logits drifted too far from bf16: cos={cs:.4f}"


def test_streaming_load_matches_full_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Shard-by-shard streaming load equals the one-shot full-dict load exactly."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    import json

    from safetensors.torch import save_file

    torch.manual_seed(0)
    hf_model, ref_model = _build_hf_and_mine()
    disk = {k: v.contiguous() for k, v in _to_disk_layout(hf_model.state_dict()).items()}

    # Write the disk layout as 3 shards + manifest (keys round-robin over shards).
    keys = sorted(disk.keys())
    shard_names = [f"model-{i + 1:05d}-of-00003.safetensors" for i in range(3)]
    weight_map = {k: shard_names[i % 3] for i, k in enumerate(keys)}
    for shard in shard_names:
        save_file({k: disk[k] for k in keys if weight_map[k] == shard}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    streamed = MiniMaxM3ForCausalLM(MiniMaxM3Config.from_hf(_tiny_hf_config()))
    streamed = streamed.to(torch.float32).eval()
    MiniMaxM3ForCausalLM.load_weights_streaming(
        streamed, str(tmp_path), device="cpu", dtype=torch.float32
    )

    total_q = 10
    input_ids = torch.randint(0, ref_model.cfg.vocab_size, (1, total_q), dtype=torch.long)
    position_ids = torch.arange(total_q, dtype=torch.long).unsqueeze(0)
    cu = torch.tensor([0, total_q], dtype=torch.int32)
    with torch.inference_mode():
        ref = ref_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=_make_cache(ref_model),
            cu_seqlens_q=cu,
        )
        got = streamed(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=_make_cache(streamed),
            cu_seqlens_q=cu,
        )
    assert torch.equal(ref, got)


def test_prefix_hit_matches_full_prefill() -> None:
    """A prefix-hit request (cached prefix + suffix prefill) yields the same
    final-token logits as a fresh full-compute prefill of the prompt.

    The M3-specific stake: the sparse layers' index_k stream must ride the
    shared prefix blocks, and the suffix's block selection must score against
    the REUSED cached index keys (the indexer never re-sees the prefix's
    hidden states). The prompt spans 3 index blocks with topk 2, so selection
    over the cached region actually decides the mask.
    """
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    from mini_infer.cache.prefix_cache import PrefixCache

    torch.manual_seed(0)
    _, model = _build_hf_and_mine()
    pool = BlockPool(
        num_blocks=32,
        block_size=4,
        num_layers=model.cfg.num_hidden_layers,
        num_kv_heads=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
        layer_streams=model.per_layer_streams(),
        attention_backend="torch",
        prefix_cache=PrefixCache(block_size=4),
    )
    cache = PagedKVCache(pool)
    prompt = [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23]  # 3 full blocks
    plen = len(prompt)

    # Request A: full prefill (cache empty, no hit), publishes blocks.
    a_idx = cache.add_request_slot(prompt_token_ids=prompt)
    assert cache.seq_lens_list()[a_idx] == 0
    with torch.inference_mode():
        logits_a = model(
            input_ids=torch.tensor([prompt], dtype=torch.long),
            position_ids=torch.arange(plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, plen], dtype=torch.int32),
        )
    last_a = logits_a[0, plen - 1]
    cache.remove_request(a_idx)

    # Request B: same prompt, prefix hit. Prefill only the uncached suffix.
    b_idx = cache.add_request_slot(prompt_token_ids=prompt)
    cached = cache.seq_lens_list()[b_idx]
    assert cached > 0, "expected a prefix hit to pre-populate cached tokens"
    suffix = prompt[cached:]
    with torch.inference_mode():
        logits_b = model(
            input_ids=torch.tensor([suffix], dtype=torch.long),
            position_ids=torch.arange(cached, plen, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, len(suffix)], dtype=torch.int32),
        )
    last_b = logits_b[0, -1]

    assert torch.allclose(last_a, last_b, atol=1e-4), (
        f"prefix-hit last-token logits diverged: max_abs_diff="
        f"{(last_a - last_b).abs().max().item():.6f}"
    )


def test_from_hf_rejects_bare_rotary_dim() -> None:
    """rotary_dim without partial_rotary_factor is ambiguous (HF would run full
    rope, honoring the field would run partial); from_hf must refuse."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    hf_cfg = _tiny_hf_config()
    # Strip the factor everywhere HF may carry it, leaving only rotary_dim.
    hf_cfg.partial_rotary_factor = None
    if isinstance(getattr(hf_cfg, "rope_parameters", None), dict):
        hf_cfg.rope_parameters.pop("partial_rotary_factor", None)
    hf_cfg.rotary_dim = 8  # != head_dim 16

    with pytest.raises(ValueError, match="rotary_dim"):
        MiniMaxM3Config.from_hf(hf_cfg)


def test_remap_rejects_scaleless_fp8_weight() -> None:
    """An e4m3 weight with no co-located weight_scale_inv must fail loudly,
    not be copied scale-free into a bf16 param."""
    pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    from mini_infer.models.minimax_m3 import _remap_m3_state

    cfg = MiniMaxM3Config.from_hf(_tiny_hf_config())
    orphan = {
        "model.layers.3.mlp.experts.0.w1.weight": torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
    }
    with pytest.raises(ValueError, match="weight_scale_inv"):
        _remap_m3_state(cfg, orphan)
