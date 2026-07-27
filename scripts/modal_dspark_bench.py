"""Stage C: what the DSpark drafter actually buys, measured (ADR-027 / plan Stage C).

Reports accepted length (tau) rather than wall-clock as the headline. ADR-011
learned this the hard way: at 7B on an H100, speculative decoding's wall-clock
win washed out to 1.00x aggregate even though acceptance was healthy, because
the win depends on the target/draft cost ratio and on how memory-bound decode
is. Tau is a property of the drafter and the workload, not of the GPU, so it
is the number that can be compared against the paper's Table 1 and that stays
meaningful on whatever hardware is cheap that week.

What it measures:

- **tau, per dataset.** Mean committed tokens per verification round (accepted
  draft tokens plus the bonus), using `deepspec`'s own definition so the
  numbers line up with Table 1: ~3.49 chat / ~5.57 math / ~5.12 code for
  Qwen3-4B at block 7.
- **Per-position survival.** How deep into the block acceptance holds up. The
  paper's argument for the parallel backbone is that position 1 is stronger
  than an autoregressive drafter's, and its claim for the Markov head is that
  the tail does not decay the way DFlash's does.
- **Confidence calibration.** ECE and AUROC of the raw confidence head against
  observed acceptance. The released checkpoints ship no Sequential Temperature
  Scaling, so this measures the uncalibrated head the paper reports at
  ECE 3-8%.
- **Threshold sweep.** Acceptance rate against tokens verified per round, the
  trade the scheduler exists to make.

Baseline: target-alone greedy over the identical prompts, which also serves as
a correctness check, since greedy spec-decode must reproduce it exactly.

Run with:
    uv run modal run scripts/modal_dspark_bench.py
    DSPARK_GPU=A10 DSPARK_SAMPLES=40 uv run modal run scripts/modal_dspark_bench.py
"""

import json
import os
from pathlib import Path
from typing import Any

import modal

_GPU = os.environ.get("DSPARK_GPU", "L4")
_SAMPLES = int(os.environ.get("DSPARK_SAMPLES", "20"))
_MAX_NEW = int(os.environ.get("DSPARK_MAX_NEW", "128"))
_TARGET = "Qwen/Qwen3-4B"
_DRAFTER = "deepseek-ai/dspark_qwen3_4b_block7"
_DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec.git"

# One per domain the paper breaks out, so our tau is comparable per-domain
# rather than averaged into a single number that hides the spread.
_DATASETS = {"gsm8k": "math", "humaneval": "code", "mt-bench": "chat"}
# Swept on the chat set only: it is where the paper reports the widest
# acceptance movement (45.7% -> 95.7%) and where truncation should matter most,
# since chat is the low-tau domain.
_SWEEP_THRESHOLDS = [0.3, 0.5, 0.7]
# The baseline costs ~3x the speculative run and only serves the correctness
# check, so it is spot-checked on a subset rather than every prompt.
_BASELINE_LIMIT = int(os.environ.get("DSPARK_BASELINE_LIMIT", "15"))
# Longer-generation probe for the truncation hypothesis.
_PROBE_MAX_NEW = int(os.environ.get("DSPARK_PROBE_MAX_NEW", "384"))
_PROBE_SAMPLES = int(os.environ.get("DSPARK_PROBE_SAMPLES", "15"))

app = modal.App("mini-infer-dspark-bench")
hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "transformers>=5.14,<5.15",
        "safetensors>=0.4",
        "huggingface_hub>=0.20",
        "accelerate>=0.30",
    )
    .run_commands(f"git clone --depth 1 {_DEEPSPEC_REPO} /deepspec")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
)


def expected_calibration_error(
    observations: list[tuple[float, bool]], *, num_bins: int = 20
) -> float:
    """Equal-width-bin ECE: mean |confidence - accuracy| weighted by bin count.

    `num_bins=20` matches `deepspec/eval/dspark/confidence_head.py`'s
    `CONFIDENCE_NUM_BINS`, so the number is comparable to what their recorder
    reports.
    """
    if not observations:
        return 0.0
    total = len(observations)
    ece = 0.0
    for b in range(num_bins):
        lo, hi = b / num_bins, (b + 1) / num_bins
        # Last bin closes on the right so p == 1.0 is not dropped.
        bucket = [
            (p, hit) for p, hit in observations if (lo <= p < hi) or (b == num_bins - 1 and p == hi)
        ]
        if not bucket:
            continue
        mean_conf = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, hit in bucket if hit) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_conf - accuracy)
    return ece


def auroc(observations: list[tuple[float, bool]]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), tie-corrected. None if one class is absent."""
    pos = [p for p, hit in observations if hit]
    neg = [p for p, hit in observations if not hit]
    if not pos or not neg:
        return None
    ordered = sorted(observations, key=lambda t: t[0])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based, averaged over the tie group
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[idx] for idx, (_, hit) in enumerate(ordered) if hit)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@app.function(gpu=_GPU, image=image, volumes={"/hf-cache": hf_cache}, timeout=60 * 90)
def bench(
    samples: int = _SAMPLES,
    max_new: int = _MAX_NEW,
    baseline_limit_arg: int = _BASELINE_LIMIT,
    probe_max_new: int = _PROBE_MAX_NEW,
    probe_samples: int = _PROBE_SAMPLES,
) -> dict[str, Any]:
    """Args are passed explicitly, NOT read from the module globals.

    The globals are populated from env vars at import time, and the remote
    container re-imports this module without the caller's environment, so a
    `DSPARK_SAMPLES=2` in front of `modal run` silently had no effect on the
    remote side and the full-size job ran instead. Threading the values
    through the call keeps local and remote in agreement.

    There is deliberately no fp32 arm here. A real fp32 control needs its own
    `ModelRunner`, because the `BlockPool`'s storage dtype is fixed at
    construction: converting the model alone would push fp32 K/V into a bf16
    pool and quietly measure nothing. Two 4B targets plus two drafters do not
    fit on a 24 GB card, and the precision question is already settled on CPU
    against a real Qwen3 across the all-rejected, partial-accept, and
    full-accept paths (see the benchmark writeup).
    """
    import time

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from mini_infer.engine.dspark import (
        DSparkSpeculativeRunner,
        Qwen3DSparkConfig,
        Qwen3DSparkDrafter,
    )
    from mini_infer.engine.model_runner import ModelRunner

    results: dict[str, Any] = {
        "gpu": _GPU,
        "target": _TARGET,
        "drafter": _DRAFTER,
        "samples_per_dataset": samples,
        "max_new_tokens": max_new,
    }

    target = ModelRunner.from_pretrained(
        _TARGET, dtype=torch.bfloat16, attention_backend="torch", num_blocks=2048
    )
    drafter_dir = snapshot_download(_DRAFTER)
    cfg = Qwen3DSparkConfig.from_hf(json.loads((Path(drafter_dir) / "config.json").read_text()))
    drafter = Qwen3DSparkDrafter(cfg)
    Qwen3DSparkDrafter.load_weights(
        drafter, load_file(str(Path(drafter_dir) / "model.safetensors"))
    )
    drafter = drafter.to(device=target.device, dtype=torch.bfloat16).eval()
    block_size = cfg.block_size
    print(f"loaded; block_size={block_size}", flush=True)

    def load_prompts(name: str) -> list[str]:
        rows = []
        with open(f"/deepspec/eval_datasets/{name}.jsonl") as fh:
            for line in fh:
                if len(rows) >= samples:
                    break
                obj = json.loads(line)
                # The bundled sets are not uniform; take the first
                # string-valued field among the usual prompt keys.
                for key in ("turns", "prompt", "question", "text", "instruction"):
                    val = obj.get(key)
                    if isinstance(val, list) and val and isinstance(val[0], str):
                        rows.append(val[0])
                        break
                    if isinstance(val, str) and val:
                        rows.append(val)
                        break
        return rows

    baseline_cache: dict[tuple[tuple[int, ...], int], tuple[list[int], float]] = {}

    def target_alone(prompt_ids: list[int], max_tokens: int) -> tuple[list[int], float]:
        # The baseline is a property of (prompt, length) alone, so the
        # threshold sweep re-derived an identical answer for every threshold.
        # Caching it turns the sweep's dominant cost into a lookup; the cached
        # timing is still the real measured one from the first computation.
        key = (tuple(prompt_ids), max_tokens)
        if key in baseline_cache:
            return baseline_cache[key]
        cache = PagedKVCache(target.block_pool)
        cache.add_request_slot()
        try:
            t0 = time.perf_counter()
            logits = target.forward_step_packed(cache, prompt_ids, [0, len(prompt_ids)], [0])
            token = int(logits[0, -1].argmax())
            out = [token]
            pos = len(prompt_ids)
            eos = target.tokenizer.eos_token_id
            while len(out) < max_tokens and token != eos:
                logits = target.forward_step_packed(cache, [token], [0, 1], [pos])
                token = int(logits[0, -1].argmax())
                out.append(token)
                pos += 1
            result = (out[:max_tokens], time.perf_counter() - t0)
            baseline_cache[key] = result
            return result
        finally:
            cache.free()

    def run_config(
        name: str,
        prompts: list[str],
        threshold: float,
        *,
        templated: bool,
        max_tokens: int,
        baseline_limit: int,
    ) -> dict[str, Any]:
        """`baseline_limit` caps how many prompts also run target-alone.

        The baseline costs one target forward per token, several times what
        the speculative run costs, and it only serves the correctness check.
        tau is measured on every prompt; the baseline is spot-checked on the
        first `baseline_limit` of them.
        """
        spec = DSparkSpeculativeRunner(target, drafter, confidence_threshold=threshold)
        tau_num = tau_den = 0
        offered_total = accepted_total = 0
        offered_at = [0] * block_size
        survived_at = [0] * block_size
        observations: list[tuple[float, bool]] = []
        spec_time = base_time = 0.0
        mismatches = 0
        baseline_checked = 0
        divergence_indices: list[int] = []
        divergence_fracs: list[float] = []
        target_forwards = 0
        tokens_out = 0

        for prompt_idx, text in enumerate(prompts):
            if templated:
                prompt_ids = target.tokenizer.encode_chat(text)[:512]
            else:
                prompt_ids = target.tokenizer.encode(text)[:512]
            if not prompt_ids:
                continue
            t0 = time.perf_counter()
            got, st = spec.run_greedy(prompt_ids, max_tokens)
            spec_time += time.perf_counter() - t0

            tau_num += sum(st.acceptance_lengths)
            tau_den += len(st.acceptance_lengths)
            offered_total += sum(st.proposal_lengths)
            accepted_total += sum(st.accepted_draft_lengths)
            target_forwards += st.n_target_forwards
            tokens_out += len(got)
            observations.extend(st.confidence_observations)
            for offered, accepted in zip(
                st.proposal_lengths, st.accepted_draft_lengths, strict=True
            ):
                for p in range(block_size):
                    if offered > p:
                        offered_at[p] += 1
                    if accepted > p:
                        survived_at[p] += 1

            if prompt_idx >= baseline_limit:
                continue
            baseline_checked += 1
            expected, bt = target_alone(prompt_ids, max_tokens)
            base_time += bt
            if got != expected:
                mismatches += 1
                div = next(
                    (i for i, (a, b) in enumerate(zip(got, expected, strict=False)) if a != b),
                    min(len(got), len(expected)),
                )
                # Where the FIRST token differs, as a fraction of the
                # generation. One flipped token makes every later token differ
                # too (different context), so a binary mismatch flag cannot
                # distinguish "one late tie-break flipped" from "wrong from the
                # start". Divergences clustered late are the signature of
                # accumulated rounding; divergence at index 0 would mean the
                # very first verify disagreed, i.e. a logic bug.
                divergence_indices.append(div)
                divergence_fracs.append(div / max(1, len(expected)))

        return {
            "dataset": name,
            "threshold": threshold,
            "templated": templated,
            "max_new_tokens": max_tokens,
            "baseline_checked": baseline_checked,
            "num_prompts": len(prompts),
            "tau": (tau_num / tau_den) if tau_den else 0.0,
            "draft_tokens_per_proposal": (offered_total / tau_den) if tau_den else 0.0,
            # Fraction of OFFERED draft tokens the target kept: the number the
            # paper's threshold sweep moves.
            "acceptance_rate": (accepted_total / offered_total) if offered_total else 0.0,
            "survival_by_position": [
                (survived_at[p] / offered_at[p]) if offered_at[p] else None
                for p in range(block_size)
            ],
            "ece": expected_calibration_error(observations),
            "auroc": auroc(observations),
            "n_confidence_obs": len(observations),
            "spec_seconds": spec_time,
            "baseline_seconds": base_time,
            "wallclock_speedup": (base_time / spec_time) if spec_time else 0.0,
            "target_forwards": target_forwards,
            "tokens_generated": tokens_out,
            "greedy_mismatches": mismatches,
            "divergence_indices": divergence_indices,
            "mean_divergence_frac": (
                sum(divergence_fracs) / len(divergence_fracs) if divergence_fracs else None
            ),
            "min_divergence_index": min(divergence_indices) if divergence_indices else None,
        }

    baseline_limit = max(1, min(samples, baseline_limit_arg))

    # Two arms over the same prompts. The first run fed raw dataset text; the
    # drafter was trained on target-regenerated responses in the model's chat
    # format, so raw text is off-distribution for it and was the leading
    # suspect for our tau falling short of the paper. Running both isolates
    # that one variable instead of changing scale and formatting together.
    per_dataset = []
    for templated in (False, True):
        arm = "templated" if templated else "raw"
        for name, domain in _DATASETS.items():
            prompts = load_prompts(name)
            row = run_config(
                name,
                prompts,
                0.0,
                templated=templated,
                max_tokens=max_new,
                baseline_limit=baseline_limit,
            )
            row["domain"] = domain
            row["arm"] = arm
            per_dataset.append(row)
            print(
                f"  [{arm}] {name} ({domain}): tau={row['tau']:.2f} "
                f"accept={row['acceptance_rate']:.1%} ece={row['ece']:.3f} "
                f"mismatch={row['greedy_mismatches']}/{row['baseline_checked']}",
                flush=True,
            )
    results["per_dataset"] = per_dataset

    # Sweep on the templated chat set: the arm we expect to ship, and the
    # domain where the paper reports the widest acceptance movement.
    chat_prompts = load_prompts("mt-bench")
    sweep = [next(r for r in per_dataset if r["dataset"] == "mt-bench" and r["arm"] == "templated")]
    for th in _SWEEP_THRESHOLDS:
        row = run_config(
            "mt-bench",
            chat_prompts,
            th,
            templated=True,
            max_tokens=max_new,
            baseline_limit=baseline_limit,
        )
        sweep.append(row)
        print(
            f"  [sweep] threshold={th}: tau={row['tau']:.2f} "
            f"offered={row['draft_tokens_per_proposal']:.2f} "
            f"accept={row['acceptance_rate']:.1%}",
            flush=True,
        )
    results["threshold_sweep"] = sweep

    # Truncation probe. Cutting at `max_new` should bias the long-answer
    # domains down hardest, because math and code answers get formulaic
    # exactly in their later tokens, which is where acceptance is highest.
    # Same prompts, longer budget, one dataset: if tau climbs, the cap was
    # part of the gap.
    probe_n = max(1, min(samples, probe_samples))
    probe_prompts = load_prompts("gsm8k")[:probe_n]
    probe = run_config(
        "gsm8k",
        probe_prompts,
        0.0,
        templated=True,
        max_tokens=_PROBE_MAX_NEW,
        baseline_limit=0,
    )
    probe["arm"] = "templated"
    results["length_probe"] = probe
    print(
        f"  [probe] gsm8k @ {probe_max_new} tokens, n={probe_n}: tau={probe['tau']:.2f}",
        flush=True,
    )

    return results


@app.local_entrypoint()
def main() -> None:
    # Explicit args: the remote container re-imports this module without
    # the caller's env, so the globals cannot be relied on there.
    results = bench.remote(
        samples=_SAMPLES,
        max_new=_MAX_NEW,
        baseline_limit_arg=_BASELINE_LIMIT,
        probe_max_new=_PROBE_MAX_NEW,
        probe_samples=_PROBE_SAMPLES,
    )
    out = Path("docs/benchmarks/data")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dspark-stage-c.json").write_text(json.dumps(results, indent=2))

    print("\n=== accepted length by domain: raw vs chat-templated prompts ===")
    print(
        f"{'dataset':<12}{'domain':<7}{'arm':<11}{'tau':>7}{'accept':>9}{'ECE':>8}{'AUROC':>8}{'div':>6}"
    )
    for r in results["per_dataset"]:
        au = f"{r['auroc']:.3f}" if r["auroc"] is not None else "n/a"
        div = f"{r['greedy_mismatches']}/{r['baseline_checked']}"
        print(
            f"{r['dataset']:<12}{r['domain']:<7}{r['arm']:<11}{r['tau']:>7.2f}"
            f"{r['acceptance_rate']:>8.1%}{r['ece']:>8.3f}{au:>8}{div:>6}"
        )

    raw = {r["dataset"]: r for r in results["per_dataset"] if r["arm"] == "raw"}
    tpl = {r["dataset"]: r for r in results["per_dataset"] if r["arm"] == "templated"}
    print("\n  chat-template effect on tau:")
    for name in raw:
        a, b = raw[name]["tau"], tpl[name]["tau"]
        delta = f"{(b / a - 1) * 100:+.0f}%" if a else "n/a"
        print(f"    {name:<12}{a:>6.2f} -> {b:>5.2f}  ({delta})")

    print("\n=== confidence threshold sweep (mt-bench, templated) ===")
    print(f"{'threshold':>10}{'tau':>7}{'offered/round':>15}{'accept':>9}")
    for r in results["threshold_sweep"]:
        print(
            f"{r['threshold']:>10.2f}{r['tau']:>7.2f}"
            f"{r['draft_tokens_per_proposal']:>15.2f}{r['acceptance_rate']:>8.1%}"
        )

    print("\n=== per-position survival (templated, threshold off) ===")
    for r in results["per_dataset"]:
        if r["arm"] != "templated":
            continue
        cells = " ".join(
            f"{v:.2f}" if v is not None else "  -  " for v in r["survival_by_position"]
        )
        print(f"  {r['dataset']:<12}{cells}")

    probe = results.get("length_probe")
    if probe:
        base = tpl.get(probe["dataset"])
        print("\n=== generation-length probe (gsm8k, templated) ===")
        if base:
            print(f"  {base['max_new_tokens']:>4} tokens: tau={base['tau']:.2f}")
        print(f"  {probe['max_new_tokens']:>4} tokens: tau={probe['tau']:.2f}")

    print("\nwrote docs/benchmarks/data/dspark-stage-c.json")
