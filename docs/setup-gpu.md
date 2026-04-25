# GPU access setup

> **Status: Decision deferred.** Pick a provider before Phase 1 Week 5; `model_runner.py` cannot be exercised end-to-end without one. CI's CPU-only checks do not validate model loading or kernel correctness.

## Why we need GPU access

Phases 1 onward require a real GPU for:

- Running the Hugging Face reference model and our `model_runner` side-by-side for golden tests (the `.gitkeep`'d `tests/golden/` and `tests/integration/` directories will fill in here).
- Verifying KV cache correctness, then Phase 2 PagedAttention behavior.
- Throughput benchmarks under `tests/benchmarks/`. CLAUDE.md is explicit: unmeasured improvements do not count, so we need a stable GPU to take numbers on.

Phase 1's starter model is `Qwen/Qwen2.5-0.5B-Instruct`, which fits on essentially any modern GPU with >= 8 GB of VRAM. Picking a more powerful card (H100, A100) is a future-proofing choice for Phase 2 and 3 workloads.

## Candidate providers

No recommendation yet. Facts only; pick based on your usage shape.

| Provider | Pricing model | Idle cost | Best fit | Friction |
|---|---|---|---|---|
| **Modal** | Per-second, serverless | None | Intermittent benchmarks, cold-start tolerable | Decorator-based Python integration; need to learn `modal` CLI and deployment model |
| **Lambda Labs** | Per-hour, dedicated instance | Yes (instance keeps running) | Long interactive sessions, batched work | SSH into a Linux box; familiar but you pay while you think |
| **RunPod** | Per-hour, can stop/start; spot pricing available | Only while running | Middle ground; spot can be very cheap if you tolerate preemption | Slightly more setup than Lambda; API + CLI well documented |

Approximate H100 80GB pricing (subject to change, verify before signing up):

- Modal: ~$0.001/sec (~$3.60/hr equivalent) on-demand.
- Lambda Labs: ~$1.10-$2.50/hr depending on availability.
- RunPod: ~$2-$3/hr on-demand; spot can drop below $1/hr.

For Phase 1 (a single 0.5B model, occasional runs), an A100 40GB or even a smaller card (L4, 4090) is plenty and will be cheaper.

## Provisioning checklist (provider-agnostic)

When a provider is picked, fill in the provider-specific steps and remove this stub.

- [ ] Account created, billing set up, credit limit configured to avoid runaway spend.
- [ ] GPU type chosen (default target: A100 80GB or H100 80GB; smaller acceptable for Phase 1).
- [ ] CUDA-compatible OS image (Ubuntu 22.04 + CUDA 12.x is the safe baseline as of 2026-04).
- [ ] On the instance: `git clone` the repo and run `uv sync`.
- [ ] Smoke test: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"` returns `True` and the expected device name.
- [ ] Document repeatable access: SSH config block, or `modal` CLI commands, or RunPod template ID. Goes in this file, replacing the stub.
- [ ] Optional but useful: a `Makefile` or `scripts/gpu_run.sh` so `make gpu-test` (or equivalent) runs the integration suite remotely.

## Open questions to resolve before picking

- Expected weekly GPU usage (hours per week)? Drives Modal vs reserved-instance cost math.
- Need persistent disk between sessions (cached HF models, intermediate state)? Lambda + RunPod handle this naturally; Modal needs a `modal.Volume`.
- Comfort with a vendor-specific Python integration (Modal) vs vanilla SSH (Lambda)? Affects how invasive the provider choice is on the codebase.
