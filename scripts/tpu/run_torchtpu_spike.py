#!/usr/bin/env python
"""Run the mini-infer engine on a TPU through TorchTPU and check parity.

TorchTPU is the PyTorch backend that adds a "tpu" device. This script runs
four checks and prints each result when the check ends:

1. eager: ModelRunner, PagedKVCache and the torch attention path run a model
   on the TPU. The TPU replays the greedy tokens of a CPU run, and the logits
   of each step must match the CPU (cosine > 0.99, the ADR-023 bar).
2. defer_and_fuse: the same replay in the fused eager mode of TorchTPU.
3. compile: the same replay under torch.compile(backend="tpu"). TorchTPU
   compiles static shapes only, so the step times show the recompile cost.
4. pallas: the Pallas paged decode and mixed kernels, called on torch tensors
   through torch.tpu.pallas.jax_op, against the NumPy references of
   run_tpu_pallas_kernels.py.

On a TPU VM or a Colab TPU runtime (see scripts/tpu/colab/README.md):

    pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
    PYTHONPATH=src python scripts/tpu/run_torchtpu_spike.py

With --dry-run, every check runs on the CPU (Pallas in interpret mode). Use it
to find script bugs before a TPU session. Its numbers say nothing about a TPU.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_tpu_pallas_kernels as kernels  # NumPy references and data makers

from mini_infer.engine.model_runner import ModelRunner

PROMPT = (
    "Paged attention stores the key and value cache of each request in fixed-size "
    "blocks, so the server can pack many requests of different lengths into one "
    "memory pool. Explain in three sentences why this raises throughput."
)


def _replay(runner, prompt_ids, steps, tokens=None):
    """Prefill, then decode greedily (or feed `tokens`). Return logits, tokens, ms per step."""
    cache, logits = runner.prefill(prompt_ids)
    all_logits, chosen = [], []
    start = time.perf_counter()
    for i in range(steps):
        all_logits.append(logits.float().cpu())  # the copy also waits for the device
        chosen.append(int(all_logits[-1].argmax()) if tokens is None else tokens[i])
        if i + 1 < steps:
            cache, logits = runner.decode(cache, chosen[-1])
    ms_per_step = (time.perf_counter() - start) * 1e3 / max(steps - 1, 1)
    cache.free()
    return all_logits, chosen, ms_per_step


def _compare(name, runner, reference, prompt_ids, mode=contextlib.nullcontext):
    """Replay the reference tokens twice (the first pass compiles). Print parity and speed."""
    ref_logits, ref_tokens, _ = reference
    with mode():
        _, _, first_ms = _replay(runner, prompt_ids, len(ref_tokens), ref_tokens)
        got_logits, _, ms = _replay(runner, prompt_ids, len(ref_tokens), ref_tokens)
    pairs = list(zip(got_logits, ref_logits, ref_tokens, strict=True))
    cosines = [kernels._cosine(g.numpy(), r.numpy()) for g, r, _ in pairs]
    same_top1 = sum(int(g.argmax()) == t for g, _, t in pairs)
    ok = min(cosines) > 0.99
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name:15s} min cosine={min(cosines):.5f}  "
        f"top-1 {same_top1}/{len(ref_tokens)}  first pass {first_ms:.1f} ms/step, "
        f"second pass {ms:.1f} ms/step",
        flush=True,
    )
    return ok


def _interpret_jax_op(name, fn):
    """CPU stand-in for torch.tpu.pallas.jax_op: torch -> NumPy -> JAX -> torch."""
    import jax.numpy as jnp

    def call(*tensors):
        return torch.from_numpy(np.array(fn(*(jnp.asarray(t.numpy()) for t in tensors))))

    return call


def _check_pallas(device, on_tpu, sync):
    from mini_infer.backends.tpu.pallas_paged_attention import (
        pallas_paged_attention,
        pallas_paged_mixed_attention,
    )

    jax_op = torch.tpu.pallas.jax_op if on_tpu else _interpret_jax_op
    decode = jax_op(
        "mini_infer::paged_decode", functools.partial(pallas_paged_attention, interpret=not on_tpu)
    )
    mixed = jax_op(
        "mini_infer::paged_mixed",
        functools.partial(pallas_paged_mixed_attention, interpret=not on_tpu),
    )
    shape = {"head_dim": 128, "page_size": 16, "max_pages": 8}
    scale = 1.0 / 128**0.5
    all_ok = True
    for label, (nh, nkv) in {"MHA 8/8": (8, 8), "GQA 8/2": (8, 2)}.items():
        data = kernels._make_paged(8, nh, nkv, seed=1, **shape)
        cases = [(f"decode {label}", decode, data, kernels._paged_reference(*data, scale))]
        data = kernels._make_mixed([1, 8, 1, 4, 2, 1, 6, 1], nh, nkv, seed=3, **shape)
        cases.append((f"mixed {label}", mixed, data, kernels._mixed_reference(*data, scale)))
        for name, op, arrays, ref in cases:
            args = [torch.from_numpy(a).to(device) for a in arrays]
            out = op(*args)  # the first call compiles
            sync()
            start = time.perf_counter()
            for _ in range(10):
                out = op(*args)
            sync()
            ms = (time.perf_counter() - start) * 1e2
            all_ok &= kernels._report(name, out.cpu().numpy(), ref, ms)
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=16, help="greedy tokens to compare")
    parser.add_argument("--compile-steps", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="run every check on the CPU")
    args = parser.parse_args()

    on_tpu = hasattr(torch, "tpu") and not args.dry_run
    if not on_tpu and not args.dry_run:
        print("No TorchTPU device. Install torch_tpu on a TPU host, or pass --dry-run.")
        return 2
    device = "tpu" if on_tpu else "cpu"
    sync = torch.accelerator.synchronize if on_tpu else lambda: None
    for package in ("torch", "torch_tpu", "jax"):
        try:
            print(f"{package} {version(package)}")
        except PackageNotFoundError:
            print(f"{package} not installed")
    print("device:", torch.tpu.get_device_properties().name if on_tpu else "cpu (DRY RUN)")

    cpu = ModelRunner.from_pretrained(args.model, device="cpu", attention_backend="torch")
    prompt_ids = cpu.tokenizer.encode(PROMPT)
    reference = _replay(cpu, prompt_ids, args.steps)
    short_reference = (*(r[: args.compile_steps] for r in reference[:2]), None)
    print(f"CPU reference: {len(prompt_ids)} prompt tokens, {args.steps} greedy steps")
    target = ModelRunner.from_pretrained(
        args.model, device=device, dtype=torch.bfloat16, attention_backend="torch"
    )

    def compiled():
        target._model = torch.compile(target._model, backend="tpu" if on_tpu else "eager")
        # TorchTPU sets dynamic=False, so each KV length compiles its own graph.
        torch._dynamo.config.recompile_limit = 64
        ok = _compare("compile", target, short_reference, prompt_ids)
        print("  graphs compiled:", torch._dynamo.utils.counters["stats"]["unique_graphs"])
        return ok

    def fused():
        mode = functools.partial(torch.tpu.set_eager_mode, torch.tpu.EagerMode.DEFER_AND_FUSE)
        return _compare("defer_and_fuse", target, reference, prompt_ids, mode)

    checks = {"eager": lambda: _compare("eager", target, reference, prompt_ids)}
    if on_tpu:  # the fused eager mode exists only in TorchTPU
        checks["defer_and_fuse"] = fused
    checks["compile"] = compiled
    checks["pallas"] = lambda: _check_pallas(device, on_tpu, sync)
    results = {}
    for name, check in checks.items():
        print(f"{name}:", flush=True)
        try:
            results[name] = bool(check())
        except Exception:  # one broken check must not hide the others
            traceback.print_exc()
            results[name] = False
    print("summary:", "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
