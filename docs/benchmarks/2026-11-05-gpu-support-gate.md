# Gate runs for the GPU support matrix

Date: 2026-11-05
Hardware: one NVIDIA H100 80GB HBM3, H200, B200 or B300 SXM6 AC per run, and
one L4, on Modal
Software in every run: CUDA 13.2, torch 2.14.1+cu132, Triton 3.8.0,
transformers 5.14.0, and FlashInfer 0.6.10 where the image has it
Engine: mini-infer at `b7ad0aa` (branch `feat/gpu-support-matrix`) with the
changes of this report
Matrix: [gpu-support.md](../gpu-support.md)

## Question

After the move to the CUDA 13.2 stack, does each model family run on the
H100, H200, B200 and B300, and on which path? The matrix had one main-stack
result per GPU pair at most. These runs give each family one Hopper result
and one Blackwell result, and they test the images that this change moved.

## Results

| Run | GPU | Script | Result |
|---|---|---|---|
| GPU unit tests | H100 | `scripts/smoke/modal_gpu_tests.py` | 116 passed, 1 failed, 45 skipped. The failure is fixed, see below |
| GLM tiny config | B300 | `scripts/glm/modal_glm_smoke.py` | Pass. fp32 logits equal to transformers 5.14 |
| MiniMax-M3 MSA kernel | B200 | `scripts/minimax_m3/modal_msa_kernel_bench.py` | Kernel equal to the torch path. 0.75x end to end, so it stays off |
| DSpark parity | L4 | `scripts/dspark/modal_dspark_parity.py` | Pass. Draft tokens equal to the reference in every round |
| Qwen2.5-7B bench | H200 | `scripts/bench/modal_paged_graph_bench.py` | Pass, fast path |
| Qwen3.8 27B | B300 | `scripts/qwen/modal_qwen38_smoke.py --cuda-graph` | Pass, fast path |
| Nemotron-H 30B | H200 | `scripts/nemotron_h/modal_nemotron_h_smoke.py` | Stopped at the fp32 unit test, before the model loaded. Needs a rerun with the fix |
| Llama 3.1 8B Instruct | H200 | `scripts/smoke/modal_smoke.py` | Runs, but see the RoPE note |
| Mistral 7B Instruct v0.3 | H200 | `scripts/smoke/modal_smoke.py` | Pass, FlashInfer |
| Qwen3 8B | H200 | `scripts/smoke/modal_smoke.py` | Pass, FlashInfer |
| Gemma 3 1B | H200 | `scripts/smoke/modal_smoke.py` | Pass, FlashInfer |
| DeepSeek-V2-Lite-Chat | H200 | `scripts/smoke/modal_smoke.py` | Pass, torch attention (required) |
| Kimi Linear 48B | B200 | `scripts/smoke/modal_smoke.py` | Pass, eager decode |

### GPU unit tests on the H100

The runner pinned a real H100 (`H100!`). Its report line was:

```
gpu=NVIDIA H100 80GB HBM3 cc=9.0 torch=2.14.1+cu132 triton=3.8.0 flashinfer=0.6.10 flash_attn=none gluon=True attention=flash_attn->torch
```

The skips are the six TPU files (no JAX in the image), 38 Gluon and NVFP4
tests that need an sm_100 GPU, and the flash-attn test. The image has no
flash-attn, so the default `flash_attn` request runs the torch path, and the
line shows it.

One test failed: `test_cuda_graph_serves_the_same_tokens_as_eager[float32]`
of Nemotron-H. The bf16 case of the same test passed. In torch 2.14,
`torch._grouped_mm` has CUTLASS kernels for bf16 only. Other dtypes take a
per-group fallback that copies the group offsets to the host, and a CUDA
graph capture cannot do that copy. The Nemotron-H run on the H200 stopped at
the same test, before it loaded the model. `supports_grouped_experts` now
takes the activation dtype, and an fp32 Nemotron-H model keeps its gathered
expert path, which needs no host sync. Production on a GPU runs bf16 and did
not change.

### GLM tiny config on the B300

The tiny-config smoke compares the port with `GlmMoeDsaForCausalLM` of
transformers 5.14 in fp32. The prefill logits had a cosine similarity of
1.000000 and a maximum absolute difference of 0.0. The 11 greedy tokens were
equal. In bf16 the prefill and decode logits were finite. GLM uses the torch
attention path on every GPU.

### MiniMax-M3 MSA decode kernel on the B200

The parity checks passed at four shapes: cosine similarity 1.000000 and a
maximum absolute difference of 1.22e-4 or less. The sparsity probe, which
writes noise outside the selected blocks, gave bit-identical output.

| Context | Batch | Kernel (us) | Torch (us) | Kernel speedup |
|---:|---:|---:|---:|---:|
| 1024 | 1 | 974 | 790 | 0.81x |
| 4096 | 1 | 930 | 990 | 1.07x |
| 16384 | 1 | 937 | 2393 | 2.56x |
| 65536 | 1 | 932 | 7465 | 8.01x |
| 65536 | 8 | 6554 | 55596 | 8.48x |

The kernel takes about 930 us at every context, so it wins only past about
4k tokens. On the synthetic M3-shaped model at a 32k context, the kernel gave
49.4 tokens per second against 65.8 for the torch path (0.75x). The greedy
tokens differed from decode step 2 on this random-weight model. The kernel
stays off by default, as the 2026-07-03 report decided on 4x H200.

### Qwen2.5-7B on the H200

The paged path with FlashInfer and the decode and prefill CUDA graphs, 32 new
tokens, median of three sweeps. The H200 has 132 SMs.

| Weights | 1 request | 8 requests | 32 requests | GPU ms per decode step at 8 |
|---|---:|---:|---:|---:|
| bf16, tokens per second | 208 | 1589 | 3046 | 4.61 |
| int8, tokens per second | 254 | 2174 | 3275 | 3.17 |

int8 is lossy. Against bf16, 1 of 1, 5 of 8 and 12 of 32 requests decoded
the same 32 tokens. The B200 gives 2221 (bf16) and 2401 (int8) tokens per
second at 8 requests with the same software
([2026-11-05-b300-paged-decode.md](2026-11-05-b300-paged-decode.md)).

### Qwen3.8 27B on the B300

The run first executed the CUDA graph and fused Gated DeltaNet unit tests on
the B300: 18 passed. Then it compared the port with transformers on the real
checkpoint (`Qwen/Qwen3.8-27B` at `1d4bf0f`):

- Prefill logits: cosine similarity 0.999956 to 0.999981 on four prompts,
  with the same last top-1 token.
- Greedy decode, 32 tokens: three prompts equal to the reference. On the
  fourth prompt the tokens are equal for 14 tokens, then a bf16 near-tie
  flips one token.
- Serving through `StateCacheContinuousScheduler`: 51.2 tokens per second at
  batch 1 with the decode graph, 18.4 without it (prefill included).

| Decode step, batch 1 | Eager (ms) | CUDA graph (ms) |
|---|---:|---:|
| Fused GDN step off | 80.2 | 19.3 |
| Fused GDN step on | 51.8 | 14.2 |

At batch 4 the fused step with the graph takes 15.3 ms. The graph gives the
same tokens as eager in every case.

### Paged models on the H200

`modal_smoke.py` loads each model on the paged path with the FlashInfer
backend, generates 8 greedy tokens after "The capital of France is", and
requires "Paris" in the output. It runs no CUDA graph.

- Mistral 7B Instruct v0.3: "Paris, but the capital of the French", with
  `attention=flashinfer`.
- Qwen3 8B: " Paris. The capital of Italy is Rome", with
  `attention=flashinfer`.
- Gemma 3 1B Instruct (the ungated mirror `unsloth/gemma-3-1b-it`): " Paris.
  The largest city in France", with `attention=flashinfer` and its sliding
  window layers.
- DeepSeek-V2-Lite-Chat: " Paris. The official language is", with
  `attention=torch`, which its MLA attention needs.
- Llama 3.1 8B Instruct (the ungated mirror
  `unsloth/Meta-Llama-3.1-8B-Instruct`, same weights): the model loaded and
  generated " a city of romance, art, fashion". The Llama port reads
  `rope_theta` only and ignores the `llama3` RoPE scaling of Llama 3.1. So
  this output does not show that the port is correct.

The Gemma 3 port loads `Gemma3ForCausalLM`, the 1B text model. The 4B, 12B
and 27B checkpoints are `Gemma3ForConditionalGeneration` and do not load.
The Hugging Face account of these runs has no access to the gated Meta and
Google repositories, so the runs used ungated mirrors.

### DSpark on the L4

The DSpark parity script ran on an L4, the GPU of its earlier runs, to test
the image move from torch 2.9.1 to the main stack. The Qwen3-4B tap of the
port agrees with the transformers hidden states (relative L2 error 2.9e-3 or
less at the five tap layers). The drafter gives the same draft tokens as the
reference drafter, and four rounds with real verification give the same
tokens. The script writes a test fixture and a data file. Their tokens did
not change, so the committed files stay as they are.

### Kimi Linear 48B on the B200

This was the first run of Kimi Linear with real weights. The output was
" Paris, and the capital of Germany is". Two gaps needed changes in the
smoke script:

- The checkpoint ships its tokenizer only as hub code (`tokenization_kimi.py`
  with `tiktoken`), and `Tokenizer.from_pretrained` runs no hub code. The
  smoke loads that tokenizer with `trust_remote_code` at a pinned revision,
  in the container only.
- `load_model` builds the model on the host and then loads the state dict
  onto the GPU, so the GPU holds two copies of the weights at its peak. For
  98 GB of bf16 weights that does not fit on one B200. The smoke builds the
  model in bf16 on the GPU and copies the weights in from host memory.

Kimi Linear has no decode CUDA graph and no grouped MoE path, so it decoded
eagerly with the per-expert loop.
