# Benchmarks

Each file in this folder is one dated report: the workload, the hardware, the
commit, the results and the command that reproduces them. The time to the
first token and the tokens per second of each supported model are in
[2026-11-05-model-metrics.md](2026-11-05-model-metrics.md), and the
[README](../../README.md#performance) shows their summary.

## Optimization results

These results measure one optimization each, against the path without it.
Each row links to the report or ADR that measured it.

| Workload | Model | GPU | Result |
|---|---|---|---|
| [Serving, 32 requests](2026-11-05-b300-paged-decode.md) | Qwen2.5-7B | B200, B300 | About 4,500 tokens per second |
| [Serving, 8 requests](../decisions/ADR-053-paged-prefill-cuda-graph.md) | Qwen2.5-7B | H100 | 1,251 tokens per second, 3x the eager path. 1,833 with int8 weights |
| [Prefill, 32k-token prompt](../decisions/ADR-054-gdn-scalar-gate-prefill.md) | Qwen3.8-27B | B200 | 12.6x faster: 74.1 s to 5.9 s |
| [Decode step, 1 request](2026-11-05-gpu-support-gate.md) | Qwen3.8-27B | B300 | 5.6x faster with a CUDA graph and a fused kernel: 80.2 ms to 14.2 ms |
| [Decode, 16 requests](../decisions/ADR-046-v4-tp-decode-host-syncs.md) | DeepSeek-V4-Flash | 2x B200 | 399 tokens per second |
| [Speculative decoding](2026-10-19-dspark-concurrency.md) | Qwen3-4B with the DSpark drafter | L4 | Decode 2.5x (chat) to 4.5x (math) faster |
| [FP8 MoE experts](../decisions/ADR-038-gluon-fp8-grouped-hopper.md) | GLM-5.3 | H100 | 1.6x faster with Gluon kernels. As fast as bf16 with half the expert memory |
| [Largest model](2026-07-03-minimax-m3-428b-gate.md) | MiniMax-M3, 428B parameters | 4x H200 | Runs with FP8 experts, 121 GB per GPU |
| [Decode step, 32k-token request](2026-11-06-state-cache-flashinfer.md) | Qwen3.8-27B | B200 | 3.1x faster with FlashInfer: 46.8 ms to 15.2 ms |
| [Decode step, 2k request in a 64k buffer](2026-11-06-state-cache-flashinfer.md) | Qwen3.8-27B | B200 | 5.3x faster with FlashInfer: 79.0 ms to 14.9 ms |
| [Decode step, 32k-token request](2026-11-06-state-cache-flashinfer.md) | Nemotron-H 30B | B200 | 1.5x faster with FlashInfer: 11.6 ms to 7.7 ms |
