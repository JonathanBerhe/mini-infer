# ADR-023: Cross-accelerator scope (NVIDIA GPU + Google TPU + AWS Trainium)

Date: 2026-07-04
Status: Accepted

## Context

mini-infer first targeted only CUDA and CPU/MPS. Its mission is a complete
inventory of production inference techniques. Large deployments also run
on Google TPU and AWS Trainium, where core primitives need different
kernels. For example, Google's Ragged Paged Attention (arXiv 2604.15464)
re-derives paged attention in Pallas/Mosaic for the static-shape TPU
compiler.

## Decision

Target NVIDIA GPU (CUDA/Triton), Google TPU (JAX/Pallas) and AWS Trainium
(NKI), with CPU/MPS as the reference oracle. AMD ROCm and Intel Gaudi stay
out of scope. Rules:

1. Backend isolation. Backend-specific code lives in per-backend modules,
   never in the scheduler, cache, API or orchestration layers. A new
   architecture must run on the reference path with no backend code.
2. Vendor path first. Use XLA on TPU, the Neuron compiler on Trainium, and
   cuBLAS and FlashInfer on CUDA. Hand-write a kernel only where the vendor
   library lacks the math.
3. Same parity bar everywhere. Every hand-written kernel must match the
   reference path (cosine similarity > 0.99) on every accelerator.
4. Scoped per primitive. Port one primitive at a time, by inventory value.

The first item is Ragged Paged Attention in JAX/Pallas for TPU.

## Alternatives Considered

1. CUDA + CPU/MPS only. Rejected, because the inventory omits TPU and
   Trainium.
2. TPU and Trainium as side experiments. Rejected, because
   cross-accelerator correctness is a core goal.
3. Also AMD ROCm and Intel Gaudi. Rejected, because they are outside the
   target mix.

## Consequences

- The inventory gains TPU paged attention and a path to Trainium kernels.
- Two new toolchains (JAX/Pallas/Mosaic, Neuron SDK/NKI) are less mature
  than CUDA. Larger runs need paid cloud access.
- Breadth can outrun depth. Rules 4 and 1 limit this risk.
