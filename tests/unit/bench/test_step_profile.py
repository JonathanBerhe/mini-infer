"""Tests for the shared step profile (`mini_infer.bench.step_profile`).

Three layers:

1. The trace reduction on a hand-made Chrome trace with known numbers.
2. `label_modules` on the smallest model of every family that the unit tests
   build, with the labels, the categories, the wrapped decode methods and the
   restore checked per family. The test id names the family, so the run shows
   which families were checked.
3. The scheduler-step profile of the bench harness on the micro Qwen3 runner
   that the DSpark tests use, on the CPU.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from mini_infer.bench import (
    BenchEnv,
    Workload,
    format_step_profiles,
    make_scheduler_technique,
    profile_scheduler_steps,
    run_suite,
)
from mini_infer.bench.step_profile import (
    DEFAULT_LABELLED_METHODS,
    STEP_LABEL,
    categorize,
    format_step_summary,
    label_modules,
    module_labels,
    profile_steps,
    summarize_step_trace,
)

_CPU_ENV = BenchEnv(device="cpu", cuda_available=False, cuda_device_count=0)


# ---------- 1. the reduction on a hand-made trace ----------


def _event(
    name: str, cat: str, ts: float, dur: float, *, tid: int = 1, correlation: int | None = None
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "name": name,
        "cat": cat,
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": 1,
        "tid": tid,
        "args": {},
    }
    if correlation is not None:
        event["args"]["correlation"] = correlation
    return event


def _hand_made_trace() -> list[dict[str, Any]]:
    """Two steps on host thread 1, their GPU work on device thread 7 (microseconds)."""
    return [
        # Step 0: an attention matmul, an MLP copy with a host sync, the sampling.
        _event(STEP_LABEL, "user_annotation", 0, 100),
        _event("layers.*.self_attn", "user_annotation", 10, 30),
        _event("aten::mm", "cpu_op", 12, 10),
        _event("cudaLaunchKernel", "cuda_runtime", 14, 2, correlation=1),
        _event("void nvjet_mm<float>(float*)", "kernel", 50, 20, tid=7, correlation=1),
        _event("layers.*.mlp", "user_annotation", 45, 30),
        _event("aten::copy_", "cpu_op", 46, 4),
        _event("cudaMemcpyAsync", "cuda_runtime", 47, 1, correlation=2),
        _event("Memcpy HtoD (Pageable -> Device)", "gpu_memcpy", 60, 5, tid=7, correlation=2),
        _event("aten::item", "cpu_op", 54, 12),
        _event("cudaStreamSynchronize", "cuda_runtime", 55, 10, correlation=3),
        _event("sampling", "user_annotation", 80, 10),
        _event("aten::argmax", "cpu_op", 81, 4),
        _event("cudaLaunchKernel", "cuda_runtime", 82, 1, correlation=4),
        _event("argmax_kernel", "kernel", 95, 3, tid=7, correlation=4),
        # Step 1: an attention matmul and a norm multiply.
        _event(STEP_LABEL, "user_annotation", 200, 80),
        _event("layers.*.self_attn", "user_annotation", 210, 20),
        _event("aten::mm", "cpu_op", 212, 5),
        _event("cudaLaunchKernel", "cuda_runtime", 213, 1, correlation=5),
        _event("void nvjet_mm<float>(float*)", "kernel", 240, 10, tid=7, correlation=5),
        _event("norm", "user_annotation", 240, 10),
        _event("aten::mul", "cpu_op", 241, 2),
        _event("cudaLaunchKernel", "cuda_runtime", 242, 1, correlation=6),
        _event(
            "vectorized_elementwise_kernel<4, mul>(int)", "kernel", 260, 4, tid=7, correlation=6
        ),
        # A kernel whose launch is not in the trace, and a metadata event.
        _event("orphan_kernel", "kernel", 300, 5, tid=7, correlation=99),
        {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "python"}},
    ]


def test_summarize_hand_made_trace_per_step_and_by_part() -> None:
    summary = summarize_step_trace(_hand_made_trace())

    first, second = summary["steps"]
    assert first == {
        "wall_ms": pytest.approx(0.100),
        "gpu_busy_ms": pytest.approx(0.023),  # [50, 70] and [60, 65] overlap, plus [95, 98]
        "kernels": 2,
        "memcpys": 1,
        "memsets": 0,
        "launches": 2,
        "launch_ms": pytest.approx(0.003),
        "syncs": 1,
        "sync_ms": pytest.approx(0.010),
    }
    assert second == {
        "wall_ms": pytest.approx(0.080),
        "gpu_busy_ms": pytest.approx(0.014),
        "kernels": 2,
        "memcpys": 0,
        "memsets": 0,
        "launches": 2,
        "launch_ms": pytest.approx(0.002),
        "syncs": 0,
        "sync_ms": 0.0,
    }

    categories = {row["key"]: row for row in summary["categories"]}
    assert [row["key"] for row in summary["categories"]] == [
        "attention",
        "MLP",
        "norms",
        "sampling",
    ]
    assert categories["attention"] == {
        "key": "attention",
        "gpu_ms": pytest.approx(0.015),
        "kernels": 1.0,
        "host_ms": pytest.approx(0.025),
    }
    assert categories["MLP"] == {
        "key": "MLP",
        "gpu_ms": pytest.approx(0.0025),
        "kernels": 0.5,
        "host_ms": pytest.approx(0.015),
    }
    assert categories["norms"]["gpu_ms"] == pytest.approx(0.002)
    assert categories["norms"]["host_ms"] == pytest.approx(0.005)
    assert categories["sampling"]["gpu_ms"] == pytest.approx(0.0015)
    assert categories["sampling"]["host_ms"] == pytest.approx(0.005)

    labels = {row["key"]: row for row in summary["labels"]}
    assert labels[("layers.*.self_attn", "attention")]["gpu_ms"] == pytest.approx(0.015)
    assert labels[("layers.*.self_attn", "attention")]["kernels"] == 1.0
    assert labels[("layers.*.mlp", "MLP")]["kernels"] == 0.5
    assert labels[("norm", "norms")]["gpu_ms"] == pytest.approx(0.002)
    assert labels[("sampling", "sampling")]["gpu_ms"] == pytest.approx(0.0015)

    kernels = {row["key"]: row for row in summary["kernels"]}
    assert kernels[("nvjet_mm", "aten::mm")] == {
        "key": ("nvjet_mm", "aten::mm"),
        "gpu_ms": pytest.approx(0.015),
        "kernels": 1.0,
        "category": "attention",
    }
    assert kernels[("Memcpy HtoD", "aten::copy_")]["category"] == "MLP"
    assert kernels[("vectorized_elementwise_kernel", "aten::mul")]["category"] == "norms"
    assert kernels[("argmax_kernel", "aten::argmax")]["kernels"] == 0.5

    host_labels = {row["key"]: row for row in summary["host_labels"]}
    assert host_labels["layers.*.self_attn"]["host_ms"] == pytest.approx(0.025)
    assert host_labels["layers.*.mlp"]["category"] == "MLP"
    assert summary["sync_sites"] == [("layers.*.mlp (aten::item)", 0.5)]
    assert summary["unattributed_device_events"] == 1


def test_step_self_category_takes_the_time_between_module_calls() -> None:
    plain = summarize_step_trace(_hand_made_trace())
    assert "scheduler" not in {row["key"] for row in plain["categories"]}

    summary = summarize_step_trace(_hand_made_trace(), step_self_category="scheduler")
    categories = {row["key"]: row for row in summary["categories"]}
    # Step 0 spends 30 us outside its ranges, step 1 spends 50 us: 40 us per step.
    assert categories["scheduler"] == {
        "key": "scheduler",
        "gpu_ms": 0.0,
        "kernels": 0.0,
        "host_ms": pytest.approx(0.040),
    }
    assert categories["attention"]["host_ms"] == pytest.approx(0.025)


def test_summarize_rejects_a_trace_without_steps() -> None:
    with pytest.raises(ValueError, match="no 'step' ranges"):
        summarize_step_trace([_event("layers.*.mlp", "user_annotation", 0, 1)])


def test_format_step_summary_gpu_and_cpu_shapes() -> None:
    summary = summarize_step_trace(_hand_made_trace())
    text = format_step_summary(summary)
    assert "GPU busy" in text and "top kernels by GPU time" in text
    assert "attention" in text and "nvjet_mm" in text
    assert "unattributed GPU events: 1" in text

    cpu_only = format_step_summary(summary, gpu=False)
    assert "GPU busy" not in cpu_only
    assert "host ms by part of the model" in cpu_only
    assert "top profiler ranges by host time" in cpu_only

    for index, row in enumerate(summary["steps"]):
        row.update(phase="prefill" if index == 0 else "decode", batch=1, tokens=6 - 5 * index)
    with_phase = format_step_summary(summary)
    assert "phase" in with_phase and "prefill" in with_phase and "decode" in with_phase


def test_categorize_reads_the_class_tag_and_the_wrapped_method() -> None:
    assert categorize("layers.*.self_attn[_KimiDeltaAttention].q_conv1d", "") == "linear attention"
    assert categorize("layers.*.self_attn[_KimiMlaAttention].q_a_proj", "") == "attention"
    assert (
        categorize("backbone.layers.*.mixer[_NemotronMamba2Mixer].norm", "") == "linear attention"
    )
    assert categorize("backbone.layers.*.mixer[_NemotronMoE]", "") == "MoE routed experts"
    assert categorize("layers.*.mlp[GlmMoeFFN]", "") == "MoE other"
    assert categorize("layers.*.mlp[SwiGLU].gate_proj", "") == "MLP"
    assert categorize("layers.*.block_sparse_moe:_loop_experts", "") == "MoE routed experts"
    assert categorize("layers.*.hc_attn:hc_pre", "") == "hyper-connections"
    assert categorize("layers.*.post_attention_layernorm", "") == "norms"
    assert categorize("layers.*.self_attn.q_norm", "") == "attention"
    assert categorize("layers.*.self_attention_res_proj", "") == "attention"
    assert categorize("layers.*.attn_sconv", "") == "attention"
    assert categorize(None, "ncclDevKernel_AllReduce") == "NCCL collectives"
    assert categorize("layers.*", "") == "other"


# ---------- 2. every model family ----------


def _tiny_llama_cfg() -> Any:
    """The tiny Llama shape of the distributed loader test."""
    from tests.unit.distributed.test_distributed_loader import _tiny_llama_state_dict

    return _tiny_llama_state_dict()[0].cfg


def _tiny_shape_fields() -> dict[str, Any]:
    """The tiny Llama shape as keyword arguments that each Llama-shape config takes."""
    fields = dataclasses.asdict(_tiny_llama_cfg())
    del fields["rope_scaling"]  # the llama3 rescale is a Llama-only field
    return fields


def _llama() -> torch.nn.Module:
    from tests.unit.distributed.test_distributed_loader import _tiny_llama_state_dict

    return _tiny_llama_state_dict()[0]


def _mistral() -> torch.nn.Module:
    from mini_infer.models.mistral.mistral import MistralConfig, MistralForCausalLM

    return MistralForCausalLM(MistralConfig(**_tiny_shape_fields()))


def _mixtral() -> torch.nn.Module:
    """The tiny Llama shape with the expert counts of the Mixtral MoE test."""
    from mini_infer.models.mistral.mixtral import MixtralConfig, MixtralForCausalLM

    cfg = MixtralConfig(**_tiny_shape_fields(), num_local_experts=4, num_experts_per_tok=2)
    return MixtralForCausalLM(cfg)


def _qwen2() -> torch.nn.Module:
    from mini_infer.models.qwen.qwen2 import Qwen2Config, Qwen2ForCausalLM

    return Qwen2ForCausalLM(Qwen2Config(**_tiny_shape_fields()))


def _qwen3() -> torch.nn.Module:
    from mini_infer.models.qwen.qwen3 import Qwen3ForCausalLM
    from tests.unit._dspark_doubles import target_config

    return Qwen3ForCausalLM(target_config())


def _qwen3_5() -> torch.nn.Module:
    from tests.unit.engine.test_cuda_graph_decode import _model

    return _model()


def _gemma3() -> torch.nn.Module:
    """Gemma 3 has no synthetic config in the unit tests; its tests load the 1B checkpoint."""
    from mini_infer.engine.model_runner import ModelRunner

    return ModelRunner.from_pretrained("unsloth/gemma-3-1b-it", device="cpu")._model


def _gemma4() -> torch.nn.Module:
    from mini_infer.models.gemma.gemma4 import Gemma4ForCausalLM
    from tests.unit.models.gemma.test_models_gemma4 import _make_cfg

    return Gemma4ForCausalLM(_make_cfg())


def _deepseek_v2() -> torch.nn.Module:
    from mini_infer.models.deepseek_v2.deepseek_v2 import DeepseekV2ForCausalLM
    from tests.unit.models.deepseek_v2.test_models_deepseek_v2 import _make_cfg

    return DeepseekV2ForCausalLM(_make_cfg())


def _deepseek_v4() -> torch.nn.Module:
    from mini_infer.models.deepseek_v4.deepseek_v4 import DeepseekV4ForCausalLM
    from tests.unit.models.deepseek_v4.test_models_deepseek_v4 import _make_hc_hybrid_config

    return DeepseekV4ForCausalLM(_make_hc_hybrid_config(use_moe_ffn=True))


def _deepseek_v4_swa() -> torch.nn.Module:
    from mini_infer.models.deepseek_v4.deepseek_v4 import DeepseekV4ForCausalLM
    from tests.unit.models.deepseek_v4.test_models_deepseek_v4 import _make_hybrid_config

    return DeepseekV4ForCausalLM(_make_hybrid_config(seq_len_friendly_compress_ratios=(0, 4, 8, 4)))


def _glm() -> torch.nn.Module:
    from mini_infer.models.glm.glm_moe_dsa import GlmMoeDsaForCausalLM
    from tests.unit.models.glm.test_models_glm_moe_dsa import _make_cfg

    return GlmMoeDsaForCausalLM(_make_cfg())


def _inkling() -> torch.nn.Module:
    from mini_infer.models.inkling.inkling import InklingConfig, InklingForCausalLM
    from tests.unit.models.inkling.test_inkling_parity import _tiny_hf_config

    return InklingForCausalLM(InklingConfig.from_hf(_tiny_hf_config()))


def _kimi_linear() -> torch.nn.Module:
    from tests.unit.models.kimi_linear.test_models_kimi_linear import _tiny_model

    return _tiny_model()


def _kimi_k3() -> torch.nn.Module:
    from tests.unit.models.kimi_linear.test_models_kimi_k3 import _tiny_model

    return _tiny_model()


def _minimax_m3() -> torch.nn.Module:
    from mini_infer.models.minimax_m3.minimax_m3 import MiniMaxM3Config, MiniMaxM3ForCausalLM
    from tests.unit.models.minimax_m3.test_minimax_m3_parity import _tiny_hf_config

    return MiniMaxM3ForCausalLM(MiniMaxM3Config.from_hf(_tiny_hf_config()))


def _nemotron_h() -> torch.nn.Module:
    from tests.unit.models.nemotron_h.test_models_nemotron_h import _model

    return _model()


_DENSE = {"attention", "MLP", "norms", "embedding", "lm_head", "RoPE tables"}
_MOE = {"MoE gate", "MoE routed experts", "MoE shared expert"}
_STATE_CACHE_METHODS = {
    "forward_prefill_with_cache",
    "forward_decode_with_cache",
    "forward_decode_with_cache_ragged",
}


@dataclasses.dataclass(frozen=True)
class _Family:
    build: Callable[[], torch.nn.Module]
    # Categories that the family's labels must reach.
    categories: frozenset[str]
    # Decode methods outside `Module.__call__` that must get a wrapper.
    methods: frozenset[str]


def _family(
    build: Callable[[], torch.nn.Module], categories: set[str], methods: set[str] = frozenset()
) -> _Family:
    return _Family(build, frozenset(categories), frozenset(methods))


FAMILIES: dict[str, _Family] = {
    "llama": _family(_llama, _DENSE),
    "mistral": _family(_mistral, _DENSE),
    "mixtral": _family(
        _mixtral,
        (_DENSE - {"MLP"}) | {"MoE gate", "MoE routed experts", "MoE other"},
        {"_loop_experts"},
    ),
    "qwen2": _family(_qwen2, _DENSE),
    "qwen3": _family(_qwen3, _DENSE),
    "qwen3_5": _family(
        _qwen3_5,
        (_DENSE - {"RoPE tables"}) | {"linear attention"},
        _STATE_CACHE_METHODS | {"decode_step_static"},
    ),
    "gemma3": _family(_gemma3, _DENSE),
    "gemma4": _family(_gemma4, _DENSE),
    "deepseek_v2": _family(_deepseek_v2, _DENSE | _MOE | {"MoE other"}, {"_loop_experts"}),
    "deepseek_v4": _family(
        _deepseek_v4,
        (_DENSE - {"MLP"}) | _MOE | {"hyper-connections"},
        _STATE_CACHE_METHODS
        | {
            "decode_step_static",
            "forward_decode",
            "forward_decode_ragged",
            "forward_decode_step",
            "forward_decode_step_ragged",
            "hc_pre",
            "hc_post",
            "_loop_experts",
        },
    ),
    "deepseek_v4_swa": _family(
        _deepseek_v4_swa,
        _DENSE,
        _STATE_CACHE_METHODS
        | {
            "decode_step_static",
            "forward_decode",
            "forward_decode_ragged",
            "forward_decode_step",
            "forward_decode_step_ragged",
        },
    ),
    "glm": _family(_glm, _DENSE | _MOE | {"MoE other"}, {"_loop_experts", "forward_cached"}),
    "inkling": _family(_inkling, (_DENSE - {"RoPE tables"}) | _MOE | {"MoE other"}),
    "kimi_linear": _family(
        _kimi_linear,
        (_DENSE - {"RoPE tables"}) | _MOE | {"MoE other", "linear attention"},
        _STATE_CACHE_METHODS | {"_loop_experts"},
    ),
    "kimi_k3": _family(
        _kimi_k3,
        (_DENSE - {"RoPE tables"}) | _MOE | {"MoE other", "linear attention"},
        _STATE_CACHE_METHODS | {"_loop_experts"},
    ),
    "minimax_m3": _family(
        _minimax_m3, _DENSE | _MOE | {"MoE other"}, {"_loop_experts", "forward_cached"}
    ),
    "nemotron_h": _family(
        _nemotron_h,
        (_DENSE - {"MLP", "RoPE tables"}) | _MOE | {"linear attention"},
        _STATE_CACHE_METHODS | {"decode_step_static"},
    ),
}

_FAMILY_PARAMS = [
    pytest.param(name, marks=[pytest.mark.requires_model] if name == "gemma3" else [], id=name)
    for name in FAMILIES
]


def _build(name: str) -> torch.nn.Module:
    try:
        return FAMILIES[name].build()
    except ImportError as exc:  # a transformers without this family's config
        pytest.skip(f"{name}: cannot build the tiny model on this machine ({exc})")


@pytest.mark.parametrize("family", _FAMILY_PARAMS)
def test_label_modules_covers_every_family(family: str) -> None:
    """Every module gets a label, every leaf a part, the decode methods a wrapper."""
    from mini_infer.models.blocks.moe.mixtral_moe import PackedSwigluExperts

    spec = FAMILIES[family]
    model = _build(family).eval()
    modules = dict(model.named_modules())
    labels = module_labels(model)
    assert len(labels) == len(modules), "every module has a label"

    leaves = {
        label
        for name, module in modules.items()
        if not list(module.children())
        for label in [labels[module]]
    }
    uncategorized = sorted(label for label in leaves if categorize(label, "") == "other")
    assert uncategorized == [], f"{family}: leaf modules without a part: {uncategorized}"
    categories = {categorize(label, "") for label in labels.values()}
    assert spec.categories <= categories, f"{family}: missing {spec.categories - categories}"

    hooks_before = (
        len(torch.nn.modules.module._global_forward_pre_hooks),
        len(torch.nn.modules.module._global_forward_hooks),
    )
    original_run = PackedSwigluExperts.run
    restore = label_modules(model)
    try:
        wrapped = {
            method
            for module in modules.values()
            for method in DEFAULT_LABELLED_METHODS
            if method in module.__dict__
        }
        assert wrapped == set(spec.methods), f"{family}: wrapped {sorted(wrapped)}"
        assert PackedSwigluExperts.run is not original_run
        # The hooks name a real module call: the lm_head under the profiler.
        lm_head = modules["lm_head"]
        x = torch.zeros(1, 1, lm_head.weight.shape[1], dtype=lm_head.weight.dtype)
        with (
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof,
            torch.no_grad(),
            torch.profiler.record_function(STEP_LABEL),
        ):
            lm_head(x)
    finally:
        restore()
    names = {event.name for event in prof.events()}
    assert "lm_head" in names

    assert not any(
        method in module.__dict__
        for module in modules.values()
        for method in DEFAULT_LABELLED_METHODS
    ), f"{family}: a wrapper survived restore()"
    assert PackedSwigluExperts.run is original_run
    assert hooks_before == (
        len(torch.nn.modules.module._global_forward_pre_hooks),
        len(torch.nn.modules.module._global_forward_hooks),
    )
    print(
        f"step profile family check: {family} ran, {len(labels)} modules, "
        f"{len(categories - {'other'})} parts, wrapped {sorted(wrapped)}"
    )


# ---------- 3. the bench path on the CPU ----------


def test_profile_scheduler_steps_on_the_micro_runner(tmp_path: Path) -> None:
    from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
    from tests.unit._dspark_doubles import micro_setup

    runner, _ = micro_setup(seed=3)
    scheduler = ContinuousScheduler(runner)
    trace = tmp_path / "trace.json"
    summary = profile_scheduler_steps(
        runner, scheduler, ["hello world", "abc"], max_tokens=4, trace_path=str(trace)
    )

    steps = summary["steps"]
    assert len(steps) >= 2
    assert steps[0]["phase"] == "prefill" and steps[0]["tokens"] > steps[0]["batch"]
    assert "decode" in {row["phase"] for row in steps}
    assert all(row["wall_ms"] > 0 for row in steps)
    assert all(row["kernels"] == 0 and row["launches"] == 0 for row in steps), "no GPU here"
    categories = {row["key"]: row for row in summary["categories"]}
    for part in ("attention", "MLP", "norms", "lm_head", "embedding", "runner glue", "scheduler"):
        assert categories[part]["host_ms"] > 0, part
    assert summary["step_unit"] == "ContinuousScheduler._step"
    assert trace.exists()
    assert "forward_step_packed" not in runner.__dict__
    assert "_step" not in scheduler.__dict__ and "_engine_loop" not in scheduler.__dict__
    print(format_step_summary(summary))


def test_profile_scheduler_steps_needs_a_step_method() -> None:
    from tests.unit._dspark_doubles import micro_setup

    runner, _ = micro_setup(seed=6)
    with pytest.raises(TypeError, match="has no _step"):
        profile_scheduler_steps(runner, object(), ["abc"], max_tokens=1)


def test_make_scheduler_technique_attaches_the_profile(tmp_path: Path) -> None:
    from mini_infer.scheduler.continuous_scheduler import ContinuousScheduler
    from tests.unit._dspark_doubles import micro_setup

    workload = Workload(
        model="micro",
        prompts=["hello", "abc"],
        concurrency_levels=[1, 2],
        max_tokens=3,
        profile=True,
        profile_trace_dir=str(tmp_path),
    )
    technique = make_scheduler_technique(
        "micro", lambda workload: micro_setup(seed=4)[0], ContinuousScheduler
    )
    results, skipped = run_suite(workload, [technique], env=_CPU_ENV)

    assert skipped == []
    assert [result.concurrency for result in results] == [1, 2]
    assert results[0].step_profile is not None and results[0].step_profile["steps"]
    assert results[1].step_profile is None
    assert (tmp_path / "micro-c1.json").exists()
    text = format_step_profiles(results)
    assert text.startswith("=== step profile: micro at concurrency 1 ===")
    assert "step unit: ContinuousScheduler._step" in text
    assert "host ms by part of the model" in text
    assert format_step_profiles(results[1:]) == ""

    class _NoStep:
        """A scheduler without `_step`: the profile records why it did not run."""

        def __init__(self, runner: Any) -> None:
            self._inner = ContinuousScheduler(runner)

        def start(self) -> None:
            self._inner.start()

        def submit(self, request: Any) -> Any:
            return self._inner.submit(request)

        def stop(self) -> None:
            self._inner.stop()

    plain = make_scheduler_technique("no_step", lambda workload: micro_setup(seed=4)[0], _NoStep)
    results, skipped = run_suite(workload, [plain], env=_CPU_ENV)
    assert skipped == []
    assert results[0].step_profile == {
        "skipped": "_NoStep has no _step; the step profile covers "
        "ContinuousScheduler-shaped schedulers"
    }
    assert format_step_profiles(results) == (
        "=== step profile: no_step at concurrency 1 ===\n"
        "skipped: _NoStep has no _step; the step profile covers "
        "ContinuousScheduler-shaped schedulers"
    )


def test_profile_steps_on_the_micro_model() -> None:
    """The single-thread entry point of the V4 script, on the CPU."""
    from mini_infer.cache.paged_kv_cache import PagedKVCache
    from tests.unit._dspark_doubles import micro_setup

    runner, _ = micro_setup(seed=5)
    cache = PagedKVCache(runner.block_pool)
    cache.add_request_slot()
    runner.forward_step_packed(cache, [1, 2, 3], [0, 3], [0])
    position = 3

    def step() -> None:
        nonlocal position
        runner.forward_step_packed(cache, [4], [0, 1], [position])
        position += 1

    summary = profile_steps(step, steps=2, model=runner._model, step_self_category="loop")
    assert len(summary["steps"]) == 2
    categories = {row["key"]: row for row in summary["categories"]}
    for part in ("attention", "MLP", "norms", "lm_head", "embedding", "loop"):
        assert categories[part]["host_ms"] > 0, part
    assert categories["attention"]["gpu_ms"] == 0.0
    assert summary["labels"] == [] and summary["kernels"] == []
    assert {row["key"] for row in summary["host_labels"]} >= {"layers.*.self_attn", "lm_head"}
