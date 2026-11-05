# GPU support matrix

Date: 2026-11-05

This page records which model family runs on which datacenter GPU, on which
path, and the last GPU run that shows it. The four GPUs are the H100 and the
H200 (Hopper, compute capability 9.0) and the B200 and the B300 (Blackwell,
10.0 and 10.3). Benchmark numbers stay in their reports under
[benchmarks](benchmarks/) and [decisions](decisions/). This page only links to
them.

## How to read the matrix

Each cell has one of these states:

| State | Meaning |
|---|---|
| Fast | A recorded run on this GPU used the fastest path that the code has for this GPU. |
| Fast, old stack | As Fast, but the run used an image from before the CUDA 13.2 stack (torch 2.10 or older). |
| Runs | The model ran end to end on this GPU and gave the expected text, but the run did not use every part of the fast path (for example no CUDA graph). |
| Fallback | The model runs, but on a slower path. The cell names the path. |
| Partial | Only one layer or one kernel ran on this GPU, not the model. |
| No | The model cannot run on this GPU with the current code or image. |
| Unchecked | No recorded run on this GPU. |

The architecture tests in `src/mini_infer/device.py` read the major version only.
So the H100 and the H200 run the same code, and the B200 and the B300 run the
same code. A pass on one GPU of a pair predicts the other GPU, but only a run
proves the image, the memory fit and the speed.

The main stack is the image of the serving scripts: base
`nvidia/cuda:13.2.1-devel-ubuntu22.04` (or `debian_slim` where no nvcc is
needed), `torch==2.14.1+cu132`, `triton==3.8.0`, `flashinfer-python==0.6.10`
and `transformers==5.14.0`. It has sm_100 kernels, it meets the CUDA 13.1+
rule of the Modal B300, and it has the Triton release of the Gluon MoE
kernels.

## Matrix

| Family and model | Fast path | H100 | H200 | B200 | B300 |
|---|---|---|---|---|---|
| Qwen2 (Qwen2.5-7B), paged | FlashInfer, decode and prefill CUDA graphs, fused norm, RoPE and SiLU kernels, int8 W8A16 with per-arch tiles | Fast, old stack (2026-11-04, [ADR-053](decisions/ADR-053-paged-prefill-cuda-graph.md)) | Fast (2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) | Fast (2026-11-05, PR #79) | Fast (2026-11-05, PR #79, within 1% of the B200) |
| Llama, Mistral, Qwen3, paged | Same code as Qwen2 | Unchecked (the 2-GPU TP smoke of 2026-05-15 loads Qwen2.5-7B) | Runs: Mistral 7B and Qwen3 8B with FlashInfer, no graph (2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)). Llama 3.1 8B loads and generates, but the port ignores its `llama3` RoPE scaling | Unchecked | Unchecked |
| Mixtral, paged MoE | As Qwen2, plus grouped bf16 expert matmuls (sm_90+) | Partial (one layer, 2026-10-21) | Unchecked | Partial (one layer, 2026-10-21) | Unchecked |
| Gemma 3, paged | FlashInfer with sliding windows | Unchecked | Fast (1B, 2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)). The 4B, 12B and 27B checkpoints do not load | Unchecked | Unchecked |
| Gemma 4 31B, paged | Torch attention (head_dim 512 is above the FlashInfer limit) | Unchecked | Unchecked | Fallback by design, old stack (2026-05-06) | Unchecked |
| DeepSeek-V2, paged MLA | Torch attention (required), grouped bf16 experts | Unchecked | Fast (Lite, 2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) | Unchecked | Unchecked |
| DeepSeek-V4-Flash, own cache, 2 GPUs | FP4 experts (W4A8 on sm_100, Gluon or Triton FP4), HC Sinkhorn kernel, fused partial RoPE and norm, decode CUDA graph | Partial (kernels and one layer) | Unchecked | Fast, old stack (2x B200, 2026-10-25, [ADR-046](decisions/ADR-046-v4-tp-decode-host-syncs.md)) | Unchecked |
| GLM-5.2 and GLM-5.3, paged MLA with DSA | Torch attention (required), grouped FP8 experts (Gluon or Triton) | Partial (one layer, 2026-10-21) | Unchecked | Partial (one layer, 2026-10-21) | Partial (tiny config equal to transformers 5.14, 2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) |
| Inkling, paged | Torch attention (required) | Unchecked | Unchecked | Fallback (tiny weights, 2026-10-19): per-expert MoE loop | Unchecked |
| Kimi Linear and Kimi K3, own cache | None on GPU: KDA in PyTorch, per-expert MoE loop, no CUDA graph | Unchecked | Unchecked | Runs (48B, eager, 2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) with a tokenizer and a loader of the smoke script. The library now has both. The library path is not yet run on a GPU | Unchecked |
| MiniMax-M3 428B, paged, 4 GPUs | Torch attention with the MSA block mask (required), grouped FP8 experts. The MSA decode kernel stays off (0.98x) | No: four 80 GB GPUs are too small | Fast, old stack (4x H200, 2026-07-03, [report](benchmarks/2026-07-03-minimax-m3-428b-gate.md)) | Partial (MSA decode kernel only, 2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) | Unchecked |
| Nemotron-H 30B-A3B, own cache | Grouped bf16 experts, decode CUDA graph. The Mamba-2 scan is PyTorch | Fast (2026-10-21, [report](benchmarks/2026-10-21-moe-grouped-gemm.md)) | Fast (2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) | Fast (2026-10-21) | Unchecked |
| Qwen3.8 27B, own cache | Chunkwise GDN prefill, fused GDN decode kernel, decode CUDA graph | Fast, old stack (2026-11-04, [ADR-054](decisions/ADR-054-gdn-scalar-gate-prefill.md)) | Unchecked | Fast, old stack (2026-11-04) | Fast (2026-11-05, [gate](benchmarks/2026-11-05-gpu-support-gate.md)) |
| DSpark drafter on Qwen3-4B | Torch attention | Unchecked (L4 only, main stack passed on 2026-11-05) | Unchecked | Unchecked | Unchecked |

Notes on the matrix:

- On 2026-11-05 the H100 ran the GPU unit tests on the main stack: 116
  passed and 1 failed. The failure (an fp32 Nemotron-H graph test) is fixed.
  The Nemotron-H run on the H200 passed that test with the fix.
- The H200 ran the Qwen2.5-7B bench, six paged models and Nemotron-H on
  2026-11-05.
- The B300 ran 25 of 26 GPU test files on the main stack on 2026-11-05
  (PR #77), the Qwen2.5-7B bench of PR #79, Qwen3.8 27B and the GLM tiny
  config.
- On Hopper, DeepSeek-V4 keeps bf16 activations for its FP4 experts, because
  Hopper has no FP4 tensor cores. That is the fastest Hopper path, not a
  fallback.
- The GLM tiny-config smokes of 2026-06-29 compared against transformers
  5.6.2, whose DSA indexer RoPE is not the one of the current port.
- The paged CUDA graphs need the FlashInfer backend and full-attention
  layers only. So Gemma 3 (sliding-window layers) and every model that needs
  the torch backend decode eagerly.

## Images after this change

| Script | Stack | GPUs |
|---|---|---|
| `scripts/smoke/modal_gpu_tests.py` | Main stack with FlashInfer. `MINI_INFER_TEST_FLASH_ATTN=1` selects the old torch 2.5.1 image with flash-attn 2.8.3. | All four (the flash-attn image: Hopper only) |
| `scripts/smoke/modal_smoke.py` | Main stack with FlashInfer, one image for every GPU. State-cache models run through `StateCacheGenerator.from_pretrained`, as the API server loads them. | All four |
| `scripts/bench/modal_openloop_bench.py` | Main stack with FlashInfer. The server takes `MINI_INFER_ATTENTION_BACKEND`. | All four |
| `scripts/minimax_m3/modal_m3_generate.py`, `modal_msa_kernel_bench.py` | Main stack, `debian_slim` (torch 2.6.0 had no sm_100 kernels) | All four (M3 on H100 needs more than 4 GPUs) |
| `scripts/glm/modal_glm_smoke.py`, `modal_glm_tp_smoke.py` | Main stack, `debian_slim`, transformers 5.14 as the HF reference | All four |
| `scripts/dspark/*.py` | Main stack, `debian_slim` (torch 2.9.1 had no CUDA 13.1+ build) | All four |
| `scripts/deepseek_v4/bench_hc_sinkhorn.py` | Main stack, `debian_slim` | All four |
| `scripts/deepseek_v4/modal_hc_kernel_probe.py` | Kept on torch 2.5.1: it reproduces a Triton 3.1 crash (ADR-018) | Hopper and older |
| `scripts/bench/modal_packed_bench.py` | Kept per arch: Blackwell on the main stack, other GPUs on torch 2.5.1 with flash-attn, because this bench measures the `flash_attn` backend | All four |

No flash-attn wheel exists for torch 2.14 and CUDA 13. So on the main stack
the fast attention path of the paged models is FlashInfer on every GPU.

## Slower paths and where they show

Several paths fall back to a slower one without an error. This change makes
the most common ones visible:

- `effective_attention_backend()` in `src/mini_infer/cache/packed_attention.py`
  gives the backend that really runs. A `flash_attn` request without
  flash-attn runs the torch reference path, and a `flashinfer` request without
  FlashInfer runs flash-attn or torch.
- `ModelRunner.from_pretrained` logs a warning when the requested backend
  cannot run on a CUDA device.
- The paged CUDA graphs refuse a FlashInfer pool when the process cannot
  import FlashInfer. Before, they looked only at the requested backend.
- `mini_infer.fast_paths.fast_path_report()` gives one line with the GPU, the
  compute capability, the torch, Triton, FlashInfer and flash-attn versions,
  `gluon=True/False` and `attention=requested->effective`. The GPU test
  runner, the smoke, the open-loop bench, the packed bench, the main bench
  and the M3, GLM, DSpark and HC scripts print it or its parts at the top of
  their report.

These fallbacks are still silent:

- Without `torch._grouped_mm` (before sm_90), the MoE layers loop over the
  experts.
- Without Triton, the int8 W8A16 layers dequantize the whole weight, and the
  fused norm, RoPE, GDN, HC Sinkhorn and MSA kernels run their PyTorch code.
- The Gluon MoE kernels skip small batches below a mean-rows threshold. That
  is a tuning choice, not a fallback.

## Gaps that need code, not images

- The API server builds the paged runner without the CUDA graphs, and with
  the `flash_attn` backend unless `MINI_INFER_ATTENTION_BACKEND` is set. On
  the main stack the default therefore runs the torch reference path.
- `modal_paged_smoke.py`, `modal_concurrent_smoke.py`, `modal_pd_smoke.py`,
  `modal_llama_tp_smoke.py`, `modal_bench_paged.py` and
  `modal_concurrent_bench.py` request the default backend on an image without
  flash-attn or FlashInfer. They test correctness on the torch path.
- Kimi Linear and Kimi K3 have no GPU kernel for KDA, no grouped MoE path and
  no CUDA graph. Inkling has no grouped MoE path.
- The Llama port ignores the `llama3` RoPE scaling of Llama 3.1, 3.2 and
  3.3. Short prompts change a little, and prompts past 8k tokens are wrong.
- The Gemma 3 port loads `Gemma3ForCausalLM` (1B) only. The 4B, 12B and 27B
  checkpoints are `Gemma3ForConditionalGeneration`.
- DeepSeek-V4 with bf16 experts sends fp32 activations to
  `functional.grouped_mm`, whose fp32 fallback copies offsets to the host.
  So an fp32 decode graph of V4 meets the same host copy as Nemotron-H.
  Production runs bf16.

## How to fill a cell

Pick the GPU with the environment variable of the script, for example:

```bash
MINI_INFER_TEST_GPU=B300 uv run modal run scripts/smoke/modal_gpu_tests.py
```

```bash
MINI_INFER_BENCH_GPU=H200 MINI_INFER_BENCH_MODEL=Qwen/Qwen3-8B uv run modal run scripts/smoke/modal_smoke.py
```

Use `H100!` for a run that must land on an H100, because Modal can place an
`H100` request on an H200. Then update the cell with the date, the stack and
a link to the report.
