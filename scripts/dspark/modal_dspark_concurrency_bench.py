"""DSpark speculative decoding under concurrency, inside ContinuousScheduler.

`modal_dspark_bench.py` measures one request at a time. Here several requests share each step.
The benchmark reports acceptance, the speedup over plain decoding per concurrency level, and
where the step time goes. It also records the forward cost per query-token width, the capacity
curve that a batch-global confidence scheduler needs. `mini_infer.bench.dspark_sweep.run_point`
measures each point in-process, as a closed loop. See ADR-031.

Per dataset and concurrency level it runs:

- the plain arm, the throughput baseline.
- the DSpark arm at each confidence threshold, with tau, per-position survival, and token
  agreement with the plain arm. bf16 matmuls of a different shape flip near-ties, so agreement
  is statistical.
- a timed pass at a few levels. Its CUDA syncs at every phase boundary slow the step, so
  throughput comes from the untimed points only.

The run raises when a rendered chat template does not encode to the `encode_chat` ids, or
when a request in a DSpark point does not speculate.

Each point goes to the `mini-infer-dspark-results` volume as it finishes. A rerun with the same
`run_tag` skips those points, so a dropped client loses no finished point. Use `--detach`.

Run with:
    uv run modal run --detach scripts/dspark/modal_dspark_concurrency_bench.py
    DSPARK_GPU=A10 uv run modal run --detach scripts/dspark/modal_dspark_concurrency_bench.py \\
        --samples 8 --concurrencies 1,4 --run-tag smoke
"""

import json
import os
from pathlib import Path
from typing import Any

import modal

# Modal fixes the GPU when the decorator runs, so only the GPU comes from the environment.
_GPU = os.environ.get("DSPARK_GPU", "L4")
_TARGET = "Qwen/Qwen3-4B"
_DRAFTER = "deepseek-ai/dspark_qwen3_4b_block7"
_DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec.git"
# Pinned, because the prompts are part of the workload.
_DEEPSPEC_COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"
_DATASETS = {"gsm8k": "math", "mt-bench": "chat"}

app = modal.App("mini-infer-dspark-concurrency-bench")
hf_cache = modal.Volume.from_name("mini-infer-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("mini-infer-dspark-results", create_if_missing=True)

# The same packages as the batch-1 benchmark, the only stack that ran DSpark on a GPU.
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
    .run_commands(
        f"git clone {_DEEPSPEC_REPO} /deepspec",
        f"git -C /deepspec checkout {_DEEPSPEC_COMMIT}",
    )
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("mini_infer")
)


def _load_prompts(name: str, limit: int) -> list[str]:
    """The first `limit` prompts of one bundled DeepSpec dataset (first turn only)."""
    rows: list[str] = []
    with open(f"/deepspec/eval_datasets/{name}.jsonl") as fh:
        for line in fh:
            if len(rows) >= limit:
                break
            obj = json.loads(line)
            # The bundled datasets use different prompt keys.
            for key in ("turns", "prompt", "question", "text", "instruction"):
                val = obj.get(key)
                if isinstance(val, list) and val and isinstance(val[0], str):
                    rows.append(val[0])
                    break
                if isinstance(val, str) and val:
                    rows.append(val)
                    break
    return rows


@app.function(
    gpu=_GPU,
    image=image,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
    # A cost cap, not an expected duration: a hung container bills for the whole timeout.
    timeout=60 * 90,
)
def bench(
    datasets: list[str],
    samples: int,
    max_tokens: int,
    concurrencies: list[int],
    thresholds: list[float],
    timed_concurrencies: list[int],
    num_blocks: int,
    run_tag: str,
) -> dict[str, Any]:
    """Run the sweep on the GPU and return the metadata and all points.

    Settings are arguments: Modal imports this module again without the caller's environment.
    """
    import time

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from mini_infer.bench.dspark_sweep import run_point, run_sweep
    from mini_infer.engine.dspark import DSparkSpec, Qwen3DSparkConfig, Qwen3DSparkDrafter
    from mini_infer.engine.model_runner import ModelRunner

    out_dir = Path("/results/concurrency") / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name()
    t_load = time.perf_counter()
    runner = ModelRunner.from_pretrained(
        _TARGET, dtype=torch.bfloat16, attention_backend="torch", num_blocks=num_blocks
    )
    drafter_dir = Path(snapshot_download(_DRAFTER))
    cfg = Qwen3DSparkConfig.from_hf(json.loads((drafter_dir / "config.json").read_text()))
    drafter = Qwen3DSparkDrafter(cfg)
    Qwen3DSparkDrafter.load_weights(drafter, load_file(str(drafter_dir / "model.safetensors")))
    drafter = drafter.to(device=runner.device, dtype=torch.bfloat16).eval()
    print(f"loaded on {gpu_name} in {time.perf_counter() - t_load:.0f}s", flush=True)

    meta: dict[str, Any] = {
        "gpu": gpu_name,
        "target": _TARGET,
        "drafter": _DRAFTER,
        "deepspec_commit": _DEEPSPEC_COMMIT,
        "block_size": cfg.block_size,
        "attention_backend": "torch",
        "dtype": "bfloat16",
        "num_blocks": num_blocks,
        "max_tokens": max_tokens,
        "samples_per_dataset": samples,
        "prompt_format": "chat template, enable_thinking=False",
    }
    spec_limit = DSparkSpec(drafter).max_spec_prompt_tokens

    def prompts_for(name: str) -> list[str]:
        tok = runner.tokenizer
        rendered: list[str] = []
        for text in _load_prompts(name, samples):
            prompt = tok.render_chat(text, enable_thinking=False)
            ids = tok.encode(prompt)
            # Admission runs plain `encode` on this text, so it must give the `encode_chat` ids.
            if ids != tok.encode_chat(text, enable_thinking=False):
                raise AssertionError(f"rendered chat template does not round-trip ({name})")
            # A prompt over `spec_limit` runs the plain path, so leave it out.
            if len(ids) <= spec_limit:
                rendered.append(prompt)
        if len(rendered) < samples:
            print(f"{name}: kept {len(rendered)} of {samples} prompts under {spec_limit} tokens")
        return rendered

    # Warm up, so that the cold-start cost does not go into the first measured point.
    warm = prompts_for(datasets[0])[:2]
    run_point(runner, warm, concurrency=2, max_tokens=16, timeout_s=600.0)
    run_point(
        runner, warm, concurrency=2, max_tokens=16, dspark=DSparkSpec(drafter), timeout_s=600.0
    )

    points = run_sweep(
        runner,
        drafter,
        {name: prompts_for(name) for name in datasets},
        out_dir=out_dir,
        max_tokens=max_tokens,
        concurrencies=concurrencies,
        thresholds=thresholds,
        timed_concurrencies=timed_concurrencies,
        sync=torch.cuda.synchronize,
        on_checkpoint=results_vol.commit,
        max_spec_prompt_tokens=spec_limit,
        # The slowest point (plain, concurrency 1) takes a few minutes.
        # A longer point has stalled, so stop it before it wastes more GPU time.
        timeout_s=900.0,
        identity=json.dumps(meta, sort_keys=True),
    )
    for key, point in points.items():
        line = f"{key}: {point['tokens_per_s']:.1f} tok/s"
        if point.get("tau") is not None:
            line += f", tau {point['tau']:.2f}"
        if point.get("steady_speedup_vs_plain") is not None:
            line += f", steady x{point['steady_speedup_vs_plain']:.2f}"
        line += f", {point['mean_in_flight']:.1f} in flight"
        print(line, flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta))
    results_vol.commit()
    return {"meta": meta, "points": points}


@app.local_entrypoint()
def main(
    datasets: str = ",".join(_DATASETS),
    samples: int = 32,
    max_tokens: int = 128,
    concurrencies: str = "1,2,4,8,16",
    thresholds: str = "0.0,0.5",
    timed_concurrencies: str = "1,4,16",
    num_blocks: int = 2560,
    run_tag: str = "main",
) -> None:
    result = bench.remote(
        datasets=[d for d in datasets.split(",") if d],
        samples=samples,
        max_tokens=max_tokens,
        concurrencies=[int(c) for c in concurrencies.split(",")],
        thresholds=[float(t) for t in thresholds.split(",")],
        timed_concurrencies=[int(c) for c in timed_concurrencies.split(",") if c],
        num_blocks=num_blocks,
        run_tag=run_tag,
    )
    out = Path("docs/benchmarks/data") / f"dspark-concurrency-{run_tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out} ({len(result['points'])} points)")
