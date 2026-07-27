"""Real-checkpoint parity gate for the DSpark drafter port (ADR-027, Stage B).

The CPU tests in `tests/unit/test_dspark_drafter_parity.py` check our drafter
against re-transcribed reference formulas on micro-configs with random
weights. That catches math bugs but shares an author with the code under test.
This script closes that gap by running the ACTUAL `DeepSpec` reference
implementation and our port side by side, on the real released weights
(`deepseek-ai/dspark_qwen3_4b_block7` + `Qwen/Qwen3-4B`), and diffing them.

Three checks, in dependency order:

1. **Tap parity.** Our `Qwen3ForCausalLM.forward(tap_layers=...)` hidden states
   vs HF's `output_hidden_states[layer_id + 1]`, for the drafter's five tapped
   layers. Confirms the tap indexes the same thing `deepspec`'s
   `extract_context_feature` does, and that our Qwen3 forward matches HF's on
   the real 4B checkpoint (not just the tiny golden configs).
2. **Drafter parity.** Reference drafter vs our drafter, fed the SAME injected
   context tensor, comparing per-position base logits, Markov-corrected logits,
   confidence logits, and sampled tokens. Feeding one context isolates drafter
   math from any target-forward difference measured in check 1.
3. **Multi-round drafter parity.** Same as 2 but across several accept/reject
   rounds driven by real verification, exercising the draft cache's
   accumulate-context / discard-block-K/V behavior and the anchor convention.

Then it writes `tests/fixtures/dspark/qwen3_4b_block7_greedy.json`: a record
of what the REFERENCE produced (draft tokens, confidence logits, per-round
acceptance) together with its provenance (weights, transformers version,
dtype). Replaying it still needs the real 4B target plus the drafter, so a
test over it is `requires_model`-marked like the TurboQuant integration
tests, not part of the cheap CPU suite. Its value is pinning the reference's
behavior at a known version, so a later transformers bump that shifts these
numbers is caught as a diff rather than silently absorbed.

Cost note: one short L4 run. Weights (~8 GB target + ~2.8 GB drafter) dominate
the wall clock; the compute itself is seconds.

Run with:
    uv run modal run scripts/modal_dspark_parity.py
    DSPARK_GPU=A10 uv run modal run scripts/modal_dspark_parity.py
"""

import json
import os
from pathlib import Path
from typing import Any

import modal

_GPU = os.environ.get("DSPARK_GPU", "L4")
_TARGET = "Qwen/Qwen3-4B"
_DRAFTER = "deepseek-ai/dspark_qwen3_4b_block7"
_DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec.git"

app = modal.App("mini-infer-dspark-parity")

# HF cache lives in a Volume so a re-run doesn't re-download ~11 GB of weights.
hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        # DeepSpec pins transformers==5.10.2; mini-infer's dev group pins
        # 5.14.x (bumped for Inkling). Both bumps in that range have already
        # forced numerics realignment in other ported families, so this run
        # deliberately uses OUR pin and reports whether the reference still
        # agrees. A divergence here is itself the finding: it would justify
        # the isolated-venv fixture generation the plan holds in reserve.
        "transformers>=5.14,<5.15",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
        "accelerate>=0.30",
    )
    .run_commands(f"git clone --depth 1 {_DEEPSPEC_REPO} /deepspec")
    .env({"HF_HOME": "/hf-cache", "PYTHONPATH": "/deepspec"})
    .add_local_python_source("mini_infer")
)

_PROMPT = "Explain why speculative decoding speeds up language model inference."


def _summarize(name: str, a: Any, b: Any) -> dict[str, Any]:
    """Difference between two tensors, with metrics that mean something in bf16.

    The first version of this reported `max_rel_diff` (elementwise, denominator
    clamped at 1e-6) and `allclose(atol=1e-3, rtol=1e-3)`. Both were useless
    here and are kept out deliberately:

    - Elementwise max-rel explodes on tensors containing near-zero entries
      (it read 1e3 to 1e5 on logits whose agreement was otherwise fine), so it
      measured the smallest denominator, not the disagreement.
    - `rtol=1e-3` is TIGHTER than bf16's own relative resolution
      (2**-8 ~= 3.9e-3), so `close=False` was guaranteed by construction for
      any bf16 tensor, including a byte-perfect implementation.

    What replaces them: `rel_l2` (‖a-b‖ / ‖b‖, a whole-tensor relative error
    that near-zero entries can't blow up), `ref_absmax` (so a max_abs_diff can
    be read against the scale it occurred at), and `ulp_ratio`
    (`max_abs_diff / (ref_absmax * 2**-8)`), which answers the actual question:
    is the worst element off by about one bf16 rounding step (~1.0, drift) or
    by many (a real divergence)?
    """
    import torch

    a32, b32 = a.detach().float(), b.detach().float()
    diff = (a32 - b32).abs()
    ref_absmax = float(b32.abs().max())
    max_abs = float(diff.max())
    bf16_ulp = ref_absmax * 2**-8
    return {
        "name": name,
        "shape": list(a32.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(diff.mean()),
        "ref_absmax": ref_absmax,
        "rel_l2": float(diff.norm() / b32.norm().clamp_min(1e-12)),
        "ulp_ratio": (max_abs / bf16_ulp) if bf16_ulp > 0 else 0.0,
        "exact": bool(torch.equal(a32, b32)),
    }


@app.function(gpu=_GPU, image=image, volumes={"/hf-cache": hf_cache}, timeout=60 * 45)
def parity() -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_infer.cache.block_pool import BlockPool
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.engine.dspark import Qwen3DSparkConfig, Qwen3DSparkDrafter
    from mini_infer.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    device = torch.device("cuda")
    dtype = torch.bfloat16
    results: dict[str, Any] = {"gpu": _GPU, "target": _TARGET, "drafter": _DRAFTER}

    import transformers

    results["transformers_version"] = transformers.__version__
    print(f"transformers {transformers.__version__}", flush=True)

    # ---- weights -----------------------------------------------------------
    target_dir = snapshot_download(_TARGET)
    drafter_dir = snapshot_download(_DRAFTER)
    print("weights ready", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(_TARGET)
    prompt_ids = tokenizer.encode(_PROMPT)
    n_prompt = len(prompt_ids)
    print(f"prompt tokens: {n_prompt}", flush=True)

    drafter_cfg_json = json.loads((Path(drafter_dir) / "config.json").read_text())
    cfg = Qwen3DSparkConfig.from_hf(drafter_cfg_json)
    block_size = cfg.block_size
    tap_layers = frozenset(cfg.target_layer_ids)

    # ---- reference target (HF) --------------------------------------------
    hf_target = (
        AutoModelForCausalLM.from_pretrained(_TARGET, dtype=dtype, attn_implementation="sdpa")
        .to(device)
        .eval()
    )
    ids = torch.tensor([prompt_ids], device=device)
    with torch.inference_mode():
        hf_out = hf_target(ids, output_hidden_states=True, use_cache=True)
    # `deepspec/modeling/dspark/common.py::extract_context_feature`: index
    # `layer_id + 1` because HF's tuple puts the embedding output at 0.
    ref_context = torch.cat([hf_out.hidden_states[i + 1] for i in cfg.target_layer_ids], dim=-1)
    print(f"ref context: {tuple(ref_context.shape)}", flush=True)

    # ---- check 1: our Qwen3 tap vs HF hidden states ------------------------
    target_cfg_json = json.loads((Path(target_dir) / "config.json").read_text())
    ours_target = Qwen3ForCausalLM(Qwen3Config.from_hf(_AsAttr(target_cfg_json)))
    target_sd: dict[str, torch.Tensor] = {}
    for shard in sorted(Path(target_dir).glob("*.safetensors")):
        target_sd.update(load_file(str(shard)))
    Qwen3ForCausalLM.load_weights(ours_target, target_sd)
    ours_target = ours_target.to(device=device, dtype=dtype).eval()
    del target_sd

    pool = BlockPool(
        num_blocks=512,
        block_size=16,
        num_layers=ours_target.cfg.num_hidden_layers,
        num_kv_heads=ours_target.cfg.num_key_value_heads,
        head_dim=ours_target.cfg.head_dim,
        dtype=dtype,
        device=str(device),
        attention_backend="torch",
    )
    cache = PagedKVCache(pool)
    cache.add_request_slot()
    sink: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        ours_logits = ours_target(
            input_ids=torch.tensor([prompt_ids], device=device),
            position_ids=torch.arange(n_prompt, device=device).unsqueeze(0),
            past_key_values=cache,
            cu_seqlens_q=torch.tensor([0, n_prompt], dtype=torch.int32, device=device),
            tap_layers=tap_layers,
            hidden_state_sink=sink,
        )
    tap_checks = [
        _summarize(f"tap_layer_{i}", sink[i], hf_out.hidden_states[i + 1])
        for i in cfg.target_layer_ids
    ]
    tap_checks.append(_summarize("target_logits", ours_logits, hf_out.logits))
    results["check1_tap_parity"] = tap_checks
    for c in tap_checks:
        print(
            f"  {c['name']}: max_abs={c['max_abs_diff']:.3e} "
            f"ulp={c['ulp_ratio']:.2f} rel_l2={c['rel_l2']:.2e}",
            flush=True,
        )

    # ---- both drafters, same weights --------------------------------------
    drafter_sd = load_file(str(Path(drafter_dir) / "model.safetensors"))
    ours_drafter = Qwen3DSparkDrafter(cfg)
    Qwen3DSparkDrafter.load_weights(ours_drafter, drafter_sd)
    ours_drafter = ours_drafter.to(device=device, dtype=dtype).eval()

    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    ref_drafter = (
        Qwen3DSparkModel.from_pretrained(drafter_dir, dtype=dtype, attn_implementation="sdpa")
        .to(device)
        .eval()
    )
    print("both drafters loaded", flush=True)

    # ---- check 2: single round, identical injected context -----------------
    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block

    def _draft_inputs(anchor_token: int) -> torch.Tensor:
        di = torch.full((1, block_size), int(cfg.mask_token_id), dtype=torch.long, device=device)
        di[:, 0] = anchor_token
        return di

    anchor = int(hf_out.logits[0, -1].argmax())
    draft_input_ids = _draft_inputs(anchor)
    # Absolute, global position ids; `start` is the committed length. The
    # drafter slices [cache_len : start + block_size] out of this, which spans
    # BOTH the injected context and the block (ADR-027 point 4).
    position_ids = torch.arange(n_prompt + 64, device=device).unsqueeze(0)
    start = n_prompt

    from transformers import DynamicCache

    ref_cache = DynamicCache()
    with torch.inference_mode():
        ref_block_hidden = forward_dspark_draft_block(
            ref_drafter,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=ref_cache,
            target_hidden_states=ref_context,
            start=start,
            block_size=block_size,
        )
        ref_base = ref_drafter.compute_logits(ref_block_hidden[:, :block_size, :])
        ref_tokens, ref_corrected = ref_drafter.sample_draft_tokens(
            ref_base,
            first_prev_token_ids=draft_input_ids[:, 0],
            temperature=0.0,
            hidden_states=ref_block_hidden[:, :block_size, :],
        )
        ref_prev = torch.cat([draft_input_ids[:, :1], ref_tokens[:, :-1]], dim=1)
        ref_conf = ref_drafter.predict_confidence_step(
            ref_block_hidden[:, :block_size, :], prev_token_ids=ref_prev
        )

    from mini_infer.engine.dspark.draft_cache import DSparkDraftCache

    ours_cache = DSparkDraftCache(cfg.num_hidden_layers)
    with torch.inference_mode():
        ours_block_hidden = ours_drafter.forward_backbone(
            noise_embedding=ours_drafter.embed_tokens(draft_input_ids),
            target_hidden_states=ref_context,
            position_ids=position_ids[:, ours_cache.get_seq_length() : start + block_size],
            past_key_values=ours_cache,
        )
        ours_cache.truncate_to(start)
        ours_base = ours_drafter.compute_logits(ours_block_hidden[:, :block_size, :])
        ours_tokens, ours_corrected = ours_drafter.sample_draft_tokens(
            ours_base, first_prev_token_ids=draft_input_ids[:, 0], temperature=0.0
        )
        ours_prev = torch.cat([draft_input_ids[:, :1], ours_tokens[:, :-1]], dim=1)
        ours_conf = ours_drafter.predict_confidence_step(
            ours_block_hidden[:, :block_size, :], prev_token_ids=ours_prev
        )

    assert ref_conf is not None and ours_conf is not None
    round1 = [
        _summarize("block_hidden", ours_block_hidden, ref_block_hidden),
        _summarize("base_logits", ours_base, ref_base),
        _summarize("corrected_logits", ours_corrected, ref_corrected),
        _summarize("confidence_logits", ours_conf, ref_conf),
    ]
    results["check2_single_round"] = round1
    results["check2_tokens_equal"] = bool(torch.equal(ours_tokens, ref_tokens))
    results["check2_ref_tokens"] = ref_tokens[0].tolist()
    results["check2_ours_tokens"] = ours_tokens[0].tolist()
    for c in round1:
        print(
            f"  {c['name']}: max_abs={c['max_abs_diff']:.3e} "
            f"ulp={c['ulp_ratio']:.2f} rel_l2={c['rel_l2']:.2e}",
            flush=True,
        )
    print(f"  tokens equal: {results['check2_tokens_equal']}", flush=True)
    print(f"  ref tokens:  {results['check2_ref_tokens']}", flush=True)
    print(f"  ours tokens: {results['check2_ours_tokens']}", flush=True)

    # ---- check 3: multi-round with real verification -----------------------
    # Greedy verification against the target, mirroring `deepspec`'s loop:
    # accept while the target's argmax agrees, then commit the bonus/corrected
    # token. Both drafters step in lockstep from identical state each round.
    rounds: list[dict[str, Any]] = []
    output_ids = torch.tensor([[*prompt_ids, anchor]], device=device)
    cur_start = n_prompt
    context = ref_context
    ref_cache = DynamicCache()
    ours_cache = DSparkDraftCache(cfg.num_hidden_layers)
    target_cache = DynamicCache()
    with torch.inference_mode():
        # Re-prime the target cache over the prompt so verification is incremental.
        _ = hf_target(
            torch.tensor([prompt_ids], device=device), past_key_values=target_cache, use_cache=True
        )

        for rnd in range(4):
            di = _draft_inputs(int(output_ids[0, cur_start]))
            ref_bh = forward_dspark_draft_block(
                ref_drafter,
                draft_input_ids=di,
                position_ids=position_ids,
                past_key_values_draft=ref_cache,
                target_hidden_states=context,
                start=cur_start,
                block_size=block_size,
            )
            r_base = ref_drafter.compute_logits(ref_bh[:, :block_size, :])
            r_tok, _ = ref_drafter.sample_draft_tokens(
                r_base,
                first_prev_token_ids=di[:, 0],
                temperature=0.0,
                hidden_states=ref_bh[:, :block_size, :],
            )

            o_bh = ours_drafter.forward_backbone(
                noise_embedding=ours_drafter.embed_tokens(di),
                target_hidden_states=context,
                position_ids=position_ids[:, ours_cache.get_seq_length() : cur_start + block_size],
                past_key_values=ours_cache,
            )
            ours_cache.truncate_to(cur_start)
            o_base = ours_drafter.compute_logits(o_bh[:, :block_size, :])
            o_tok, _ = ours_drafter.sample_draft_tokens(
                o_base, first_prev_token_ids=di[:, 0], temperature=0.0
            )

            # Verify the proposal on the target: feed [anchor, d_0..d_{g-1}].
            verify_ids = torch.cat([di[:, :1], r_tok], dim=1)
            v_out = hf_target(
                verify_ids,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            v_argmax = v_out.logits[0].argmax(dim=-1)
            accepted = 0
            for i in range(block_size):
                if int(v_argmax[i]) == int(r_tok[0, i]):
                    accepted += 1
                else:
                    break
            bonus = int(v_argmax[accepted])

            rounds.append(
                {
                    "round": rnd,
                    "start": cur_start,
                    "accepted": accepted,
                    "tokens_equal": bool(torch.equal(o_tok, r_tok)),
                    "base_logits": _summarize(f"r{rnd}_base", o_base, r_base),
                    "ref_tokens": r_tok[0].tolist(),
                    "ours_tokens": o_tok[0].tolist(),
                }
            )
            print(
                f"  round {rnd}: accepted={accepted}/{block_size} "
                f"tokens_equal={rounds[-1]['tokens_equal']} "
                f"max_abs={rounds[-1]['base_logits']['max_abs_diff']:.3e}",
                flush=True,
            )

            committed = [int(t) for t in r_tok[0, :accepted]] + [bonus]
            output_ids = torch.cat(
                [output_ids[:, : cur_start + 1], torch.tensor([committed], device=device)], dim=1
            )
            # Next round's context: only the just-verified window (deepspec's
            # `_update` reassigns, it does not append).
            verified = torch.cat([v_out.hidden_states[i + 1] for i in cfg.target_layer_ids], dim=-1)
            context = verified[:, : accepted + 1, :]
            cur_start += accepted + 1
            # Target cache: keep only the accepted prefix + bonus.
            target_cache.crop(cur_start)

    results["check3_rounds"] = rounds
    results["check3_all_tokens_equal"] = all(r["tokens_equal"] for r in rounds)

    # ---- fixture -----------------------------------------------------------
    results["fixture"] = {
        "target": _TARGET,
        "drafter": _DRAFTER,
        "transformers_version": transformers.__version__,
        "dtype": "bfloat16",
        "prompt": _PROMPT,
        "prompt_token_ids": prompt_ids,
        "block_size": block_size,
        "anchor_token_id": anchor,
        "ref_draft_token_ids": ref_tokens[0].tolist(),
        "ref_confidence_logits": [float(x) for x in ref_conf[0].tolist()],
        "ref_base_logits_argmax": ref_base[0].argmax(dim=-1).tolist(),
        "rounds": [
            {
                "round": r["round"],
                "start": r["start"],
                "accepted": r["accepted"],
                "ref_tokens": r["ref_tokens"],
            }
            for r in rounds
        ],
    }
    return results


class _AsAttr:
    """Adapt a parsed `config.json` dict to the attribute access `from_hf` expects."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.__dict__.update(d)


@app.local_entrypoint()
def main() -> None:
    results = parity.remote()

    out_dir = Path("docs/benchmarks/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dspark-parity-raw.json").write_text(json.dumps(results, indent=2))

    fixture_dir = Path("tests/fixtures/dspark")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "qwen3_4b_block7_greedy.json").write_text(
        json.dumps(results["fixture"], indent=2)
    )

    print("\n=== DSpark real-checkpoint parity ===")
    print(f"transformers: {results['transformers_version']}")
    print("\ncheck 1: our Qwen3 tap vs HF hidden states")
    for c in results["check1_tap_parity"]:
        print(
            f"  {c['name']:<20} max_abs={c['max_abs_diff']:.3e}  "
            f"ulp={c['ulp_ratio']:.2f} rel_l2={c['rel_l2']:.2e}"
        )
    print("\ncheck 2: reference drafter vs ours (same injected context)")
    for c in results["check2_single_round"]:
        print(
            f"  {c['name']:<20} max_abs={c['max_abs_diff']:.3e}  "
            f"ulp={c['ulp_ratio']:.2f} rel_l2={c['rel_l2']:.2e}"
        )
    print(f"  draft tokens identical: {results['check2_tokens_equal']}")
    print("\ncheck 3: multi-round with real verification")
    for r in results["check3_rounds"]:
        print(
            f"  round {r['round']}: accepted={r['accepted']}  "
            f"tokens_equal={r['tokens_equal']}  "
            f"max_abs={r['base_logits']['max_abs_diff']:.3e}"
        )
    print(f"  all rounds identical: {results['check3_all_tokens_equal']}")
    print("\nwrote tests/fixtures/dspark/qwen3_4b_block7_greedy.json")
