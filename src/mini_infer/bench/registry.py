"""The technique registry: the single source of truth for what gets benched.

Kept separate from `harness.py` (the model-agnostic core) and shared by both
the local entry point (`scripts/bench_all.py`) and the Modal GPU wrapper
(`scripts/modal_bench_all.py`), so CPU and GPU runs bench the exact same set of
techniques against the exact same workload.

The heavy imports (`ModelRunner` and the schedulers) live inside
`build_registry` so importing this module stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mini_infer.bench.harness import (
    BenchEnv,
    BenchResult,
    Technique,
    Workload,
    make_scheduler_technique,
)

DEFAULT_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "A common greeting in English is",
    "The chemical symbol for water is",
    "An example of a fruit is",
    "The first president of the United States was",
    "Python is a programming language created by",
    "The opposite of hot is",
]


def _pending_run(workload: Workload, env: BenchEnv) -> list[BenchResult]:
    """Placeholder for techniques registered but not yet wired.

    Never invoked: `pending` techniques are short-circuited by
    `Technique.applicability` before `run` is reached.
    """
    raise RuntimeError("pending technique should not be run")


def build_registry() -> list[Technique]:
    """The full technique surface, in display order.

    Scheduler-driven configs reuse `make_scheduler_technique`: one model load,
    swept across concurrency levels. CUDA-only configs carry `requires_cuda` so
    they skip cleanly on a CPU box; `lossless=False` exempts lossy quant from
    the parity check; `pending` configs are registered for visibility until
    their driver is wired.
    """
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
    from mini_infer.workers import PDScheduler

    def runner_with(
        *, override_backend: str | None = None, **kwargs: Any
    ) -> Callable[[Workload], Any]:
        def _build(workload: Workload) -> Any:
            # Techniques that don't pin a backend use the workload's; the
            # FlashInfer KV-quant techniques pass `override_backend`.
            backend = (
                override_backend if override_backend is not None else workload.attention_backend
            )
            return ModelRunner.from_pretrained(
                workload.model,
                device=workload.device,
                attention_backend=backend,
                **kwargs,
            )

        return _build

    def continuous(runner: Any) -> Any:
        return ContinuousScheduler(runner)

    return [
        # --- CPU-runnable, scheduler-driven ---
        make_scheduler_technique("baseline", runner_with(), continuous),
        make_scheduler_technique(
            "int8_w8a16",
            runner_with(quant="int8"),
            continuous,
            note="Triton W8A16 on CUDA; dequant fallback on CPU",
            lossless=False,
        ),
        make_scheduler_technique("prefix_cache", runner_with(prefix_cache=True), continuous),
        make_scheduler_technique(
            "pd_serial",
            runner_with(),
            lambda runner: PDScheduler(runner, mode="serial"),
        ),
        make_scheduler_technique(
            "pd_parallel",
            runner_with(),
            lambda runner: PDScheduler(runner, mode="parallel"),
        ),
        # --- CUDA-only: skip-with-reason on CPU, run on a GPU host ---
        make_scheduler_technique(
            "attn_flashinfer",
            runner_with(override_backend="flashinfer"),
            continuous,
            requires_cuda=True,
            note="FlashInfer attention backend",
        ),
        make_scheduler_technique(
            "kv_fp8",
            runner_with(override_backend="flashinfer", kv_quant="fp8"),
            continuous,
            requires_cuda=True,
            note="FlashInfer FP8 KV cache",
            lossless=False,
        ),
        make_scheduler_technique(
            "kv_nvfp4",
            runner_with(override_backend="flashinfer", kv_quant="nvfp4"),
            continuous,
            requires_cuda=True,
            note="FlashInfer NVFP4 KV cache (Blackwell)",
            lossless=False,
        ),
        # --- registered for visibility, driver wiring is a follow-up ---
        Technique(
            name="spec_decode",
            run=_pending_run,
            pending="draft+target SpeculativeRunner wiring (different driver than the schedulers)",
        ),
        Technique(
            name="tensor_parallel",
            run=_pending_run,
            pending="needs a multi-process launch (world_size>1), not this single-process harness",
        ),
    ]
