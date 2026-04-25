# GPU access setup

> **Status: M1 Pro / MPS for Phase 0 + Phase 1; cloud CUDA from Phase 2 onward.** No paid GPU until then.

The plan is two-stage:

1. **Phase 0 + Phase 1**: develop on Apple Silicon with PyTorch's MPS backend (`device="mps"`). Free, fast iteration, no cloud setup.
2. **Phase 2 onward**: provision a cloud CUDA instance when the next item on the roadmap needs CUDA-only kernels (PagedAttention, bitsandbytes quantization, NCCL tensor parallelism, or speculative decoding benchmarks). Provider pick is a Phase-2 entry condition, not a Phase-0 blocker.

This file documents both stages. Stage 2 is intentionally a stub for now; fill it in when a provider is picked.

---

## Stage 1: Local development on Apple Silicon (MPS)

PyTorch's MPS backend covers everything needed for Phase 1 (skeleton, golden tests, end-to-end smoke).

### What works on MPS

- Loading `Qwen/Qwen2.5-0.5B-Instruct` via HF Transformers (`device_map="mps"` or explicit `.to("mps")`).
- Token generation with greedy / temperature / top-k / top-p sampling.
- Naive scheduler (single-request loop).
- Contiguous KV cache (the simplest Phase 1 layout).
- FastAPI server with SSE streaming (no GPU at the API layer).
- Golden tests at `temperature=0`. Run mini-infer and the HF reference both on MPS so any MPS-specific numerical drift cancels in the comparison.

### What does NOT work on MPS

These are the techniques that gate the Phase-2 transition:

- **PagedAttention kernels**: vLLM's CUDA kernels have no Metal port. The block-manager bookkeeping can be implemented on MPS, but the actual kernel speedup needs CUDA.
- **Quantization via `bitsandbytes`**: CUDA-only.
- **Tensor parallelism via NCCL**: CUDA-only; also moot on M1 (single GPU).
- **Fused FlashAttention through `F.scaled_dot_product_attention`**: MPS dispatches to slower math or memory-efficient kernels; no fused FA.
- **Custom Triton kernels** (Stretch Goal D): Triton's Metal backend is minimal.

### Local smoke test

From the project root:

```bash
uv run python -c "import torch; print('mps available:', torch.backends.mps.is_available()); print('mps built:', torch.backends.mps.is_built())"
```

Expected output: both `True` on M1 Pro / Apple Silicon with PyTorch 2.4+. Verified locally on 2026-04-25 with torch 2.11.

### Design implication

`ModelRunner` (Phase 1) accepts a `device: str` argument with `auto` as default. Resolution order: MPS if available, else CUDA, else CPU. Hardware-specific code is confined to clearly marked modules per CLAUDE.md ("don't write code that only works on NVIDIA GPUs"). The orchestration layer (scheduler, API, KV cache bookkeeping) stays device-agnostic.

---

## Stage 2: Cloud CUDA on Modal

**Provider chosen: Modal** (per-second billing, no idle cost, serverless containers). Picked 2026-04-25 because the workload is intermittent: short runs to validate CUDA-specific code paths, then occasional benchmarks. We avoid paying for an idle hourly instance.

### Cost discipline (read this before running anything)

Every `modal run` bills the account. There is no free tier. See the project's local memory note `feedback_modal_costs_money.md` for the rule we've adopted: state GPU + duration + cost estimate before each invocation, and get explicit approval. Smoke tests are not benchmarks; benchmarks need their own cost budget.

### One-time setup

1. Create the Modal account (done).
2. `uv add --dev modal` (already in `pyproject.toml`).
3. `uv run modal token new` to authenticate the CLI against the account.

### Smoke test

`scripts/modal_smoke.py` spins up an A10 24GB container, installs the project's runtime deps, mounts the `mini_infer` package, loads `Qwen/Qwen2.5-0.5B-Instruct`, runs greedy generation, asserts the output contains "Paris", and exits. Run with:

```bash
uv run modal run scripts/modal_smoke.py
```

Expected cost: roughly $0.02 once the image is cached. First run includes ~2 minutes of image build (also billed but cheaper); the built image is reused on subsequent runs.

The smoke verified on 2026-04-25:

```
OK | torch=2.11.0+cu130 | gpu=NVIDIA A10 | runner.device=cuda |
output=' Paris. It is the largest city in'
```

CUDA path produces the same tokens as the CPU/fp32 reference path, confirming the device-aware design works without code changes.

### Image / dep notes

- Modal's `debian_slim(python_version="3.11")` + `pip_install("torch>=2.4", ...)` resolves the CUDA torch wheel from PyPI on Linux x86_64 (cu130 currently). No special index URLs needed.
- mini-infer is mounted via `add_local_python_source("mini_infer")` rather than installed editable, since the smoke doesn't need a build step.
- HF model downloads happen inside the container's ephemeral filesystem; each cold run re-downloads Qwen. **For repeated runs**, add a `modal.Volume` keyed on the HF cache directory so the model survives between invocations. This is a Phase 2 task when we start running benchmarks.

### Recommended GPU tiers

| Workload | GPU | Approx cost |
|---|---|---|
| CUDA smoke / correctness checks | A10 24GB | ~$1.10/hr equivalent |
| Phase 2 functional dev + small benchmarks | A100 40GB | ~$1.30/hr equivalent |
| Phase 3 / Phase 4 publishable benchmarks | H100 80GB | ~$3.60/hr equivalent |

For the 0.5B starter model, an A10 is plenty. Step up to A100 when measuring throughput meaningfully or running quantized variants.

### Open follow-ups

- Add a `modal.Volume` for the HF cache so benchmark runs don't re-download model weights every cold start.
- A `scripts/modal_benchmarks.py` skeleton that runs a measurable workload (e.g., throughput/TTFT for the Phase 2 PagedAttention vs the Phase 1 baseline). Designed to require explicit confirmation before each run.
