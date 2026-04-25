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

## Stage 2: Cloud CUDA (Phase 2 onward)

Provision when the next item on the roadmap is one of:

- PagedAttention benchmarks (need real CUDA kernels).
- Quantization with bitsandbytes.
- Tensor parallelism (also requires multi-GPU).
- Speculative decoding throughput benchmarks (Phase 3).

### Candidate providers

No recommendation yet. Pick based on usage shape.

| Provider | Pricing model | Idle cost | Best fit |
|---|---|---|---|
| **Modal** | Per-second, serverless | None | Intermittent benchmarks; cold-start tolerable |
| **Lambda Labs** | Per-hour, dedicated instance | Yes (instance keeps running) | Long interactive sessions |
| **RunPod** | Per-hour, can stop/start; spot pricing available | Only while running | Middle ground; spot is very cheap when preemption is tolerable |

Approximate H100 80GB pricing (verify before signing up):

- Modal: ~$3.60/hr equivalent on-demand.
- Lambda Labs: ~$1.10 to $2.50/hr.
- RunPod: ~$2 to $3/hr on-demand; spot can drop below $1/hr.

For Phase 2 starting work (a 0.5B model with PagedAttention, then quantized variants), an A100 40GB or even an L4 / 4090 is plenty and significantly cheaper. Step up to H100 / H200 only if Phase 3 brings larger models or multi-GPU TP.

### Provisioning checklist

Fill in provider-specific steps when one is picked.

- [ ] Account and billing; credit limit configured to cap runaway spend.
- [ ] GPU type chosen (A100 40-80GB is the default target; smaller is fine for Phase 1's 0.5B model if you stretch local-only further).
- [ ] CUDA-compatible OS image (Ubuntu 22.04 + CUDA 12.x is the safe baseline).
- [ ] On the instance: clone the repo, run `uv sync`.
- [ ] Smoke test: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"` returns `True` and the expected device.
- [ ] Document repeatable access (SSH config block, `modal` CLI commands, RunPod template ID) in this file, replacing the stub.
- [ ] Optional: a script that runs the integration suite on the cloud instance from the laptop.

### Open questions to resolve before picking

- Expected weekly GPU usage (hours)? Drives the Modal vs reserved-instance cost math.
- Need persistent disk between sessions (cached HF models, intermediate state)? Lambda + RunPod handle this naturally; Modal needs a `modal.Volume`.
- Comfort with vendor-specific Python integration (Modal) vs vanilla SSH (Lambda)? Affects how invasive the provider choice is on the codebase.
