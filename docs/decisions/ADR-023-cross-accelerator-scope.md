# ADR-023: Cross-accelerator scope (NVIDIA GPU + Google TPU + AWS Trainium)

Date: 2026-07-04
Status: Accepted

## Context

mini-infer shipped a CUDA + CPU/MPS engine, with TPU, Trainium, AMD, and
Intel listed as explicit non-goals (roadmap-2026.md, CLAUDE.md). That was
the right call while the mission was read narrowly as "port architectures
and validate them." But the mission also states mini-infer is "a complete
inventory of the production techniques those architectures depend on," and
that inventory is incomplete on a single accelerator.

Real frontier inference does not run on one kind of silicon. The largest
deployments span NVIDIA GPU, Google TPU, and AWS Trainium, and several core
primitives require materially different kernels per accelerator because the
compilers and memory models differ. Paged attention is the clearest case:
GPU-style paged KV uses non-contiguous gather/scatter that fights a TPU's
static-shape compiler, which is why Google's Ragged Paged Attention
(arXiv 2604.15464) re-derives it in Pallas/Mosaic rather than porting the
GPU kernel. A CUDA-only inventory therefore omits how these techniques
actually map onto the hardware the field runs on.

## Decision

Expand scope to a cross-accelerator engine targeting NVIDIA GPU
(CUDA/Triton), Google TPU (JAX/Pallas), and AWS Trainium (NKI), with
CPU/MPS as the reference oracle. AMD ROCm and Intel Gaudi remain out of
scope.

Rules that keep this from diluting the project's identity:

1. **Backend isolation.** Backend-specific code lives in clearly-marked
   per-backend modules and never leaks into the scheduler, cache, API, or
   orchestration layers. A new architecture must run on the reference path
   without touching backend code.
2. **Vendor path first.** Use the vendor toolchain before hand-writing
   anything (XLA on TPU, the Neuron compiler on Trainium, cuBLAS and
   FlashInfer on CUDA). Hand-write a kernel only where the vendor library
   lacks the math.
3. **Same parity bar everywhere.** Every hand-written kernel validates
   against the reference path within tolerance (cosine similarity > 0.99),
   regardless of accelerator. A cross-accelerator port is held to the same
   bar as a same-accelerator kernel.
4. **Scoped per primitive.** This is not a promise to port every
   architecture to every backend at once. Cross-accelerator work is picked
   one primitive at a time, by inventory value.

First work item: paged attention in JAX/Pallas for TPU, anchored on Ragged
Paged Attention.

## Alternatives Considered

1. **Status quo (CUDA + CPU/MPS only).** Rejected: leaves the technique
   inventory incomplete, since it omits how core primitives map onto TPU
   and Trainium, the accelerators frontier inference actually runs on.
2. **Scoped portability track** (TPU/Trainium as parity-validated
   exploration only, core identity stays CUDA). Rejected in favor of
   treating cross-accelerator correctness as a first-class pillar of the
   positioning, not a side experiment.
3. **Full production multi-backend including AMD ROCm and Intel Gaudi.**
   Rejected: those are not in the frontier accelerator mix this project
   targets, and supporting them is multi-person-year work with no
   correspondence to the technique-inventory mission.

## Consequences

Positive:

- The technique inventory now covers how primitives map across the real
  frontier accelerator mix, not just CUDA.
- Adds concrete new inventory items: paged attention on TPU (Pallas), and a
  path to Trainium (NKI) kernels.
- The parity discipline extends cleanly; cross-accelerator ports are
  validated exactly like same-accelerator kernels.

Costs and risks:

- Two new toolchains to learn and maintain (JAX/Pallas/Mosaic for TPU,
  Neuron SDK/NKI for Trainium), each less mature than CUDA.
- TPU and Trainium require accelerator access. Free-tier TPU (Colab,
  Kaggle) covers small-scale development; larger runs need paid cloud
  accelerators.
- The parity-validation surface multiplies across backends.
- Risk of diluting the readable, single-contributor, correctness-first
  focus if breadth outruns depth. Mitigated by per-primitive scoping
  (rule 4) and strict per-backend module isolation (rule 1).
