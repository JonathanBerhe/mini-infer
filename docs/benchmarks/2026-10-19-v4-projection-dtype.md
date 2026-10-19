# DeepSeek-V4 compressor projections: fp32 output against bf16 output

Date: 2026-10-19
Hardware: 2x NVIDIA B200, tensor parallel over 2 ranks, bf16 weights, `torch` 2.10.0+cu128
Model: `deepseek-ai/DeepSeek-V4-Flash` (43 layers: 21 CSA, 20 HCA, 2 SWA)
Prompt: 8,192 tokens of technical prose from this repository
(`docs/architectures/deepseek-v4.md`, then `deepseek-v2-mla.md`, cut at 8,192
tokens), followed by 32 greedy tokens
Engine: mini-infer at `ecd31c6`, with the reference's FP8 and FP4 round trips
on (`--simulate-act-quant --simulate-fp8-gemm-inputs`) and the expert clamp
at `swiglu_limit`
Upstream code compared: vLLM `main` at `b965ee5`, `vllm/models/deepseek_v4/attention.py`
Script: `scripts/modal_v4_projection_dtype_probe.py`
Data: `docs/benchmarks/data/v4-projection-dtype-probe-fixed-rope.json`
(main run), plus two earlier runs (see "Earlier runs")

## Question

vLLM issue [#53459](https://github.com/vllm-project/vllm/issues/53459) asks
whether two GEMMs of DeepSeek-V4 can emit bf16 instead of fp32. In
`_run_parallel_input_projections`, vLLM runs the main attention compressor
(`compressor_kv_score`) and the Lightning Indexer's compressor
(`indexer_compressor_kv_score`) with `torch.mm(..., out_dtype=torch.float32)`.

The issue calls this an fp32 accumulate. A bf16 GEMM accumulates in fp32 in
both cases. The output dtype decides one thing only: whether each projected kv
and score entry is rounded to bf16 once, before the compression softmax pools
it. This report measures that rounding, its effect on the model's predictions,
and the time that bf16 output saves.

## Result

- bf16 output does not change the quality of the model's predictions. On the
  8,192-token prompt, the negative log-likelihood (NLL) of the real next token
  rises by 0.0007 nats per token. Computing the same fp32 GEMM in a different
  reduction order raises it by 0.0005. Perplexity goes from 4.417 to 4.420
  and 4.419, and top-1 accuracy stays at 65.4 to 65.5 percent.
- End to end, bf16 output moves the outputs as far as a reordered fp32 GEMM
  does, and not further. The argmax next token agrees at 96.0 percent of the
  prompt positions in both comparisons. The same setting twice gives
  identical outputs, so the forward is deterministic.
- Per layer, the rounding does change some choices. A small difference
  before an FP8 or FP4 rounding can move a value into the next quantization
  bin. Compressed entries move by 1.0 percent (main) and 3.0 percent
  (indexer). On every CSA layer, 72 percent of the queries change at least
  one of their up to 512 selected blocks, on average 2.5 blocks. These are
  near-ties, and the end-to-end numbers show that they do not cost quality.
- On B200, bf16 output makes each of these GEMMs 5 to 8 percent faster at
  8,192 tokens. That is 2.2 to 5.3 µs per GEMM, or 0.21 ms per 8,192-token
  forward over all 41 compressed layers.
- vLLM runs these GEMMs on side streams, in parallel with the larger
  `fused_wqa_wkv` GEMM, for batches of up to 1,024 tokens. That limit is the
  default of `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`. In those batches the
  saving is hidden.
- The change also needs new warmup specializations. vLLM warms its
  `save_partial_states` Triton kernel only for fp32 kv and score inputs. With
  bf16 inputs, a new kernel compiles while the server runs.
- So bf16 output is safe for accuracy, and its gain is small. It saves at
  most 0.2 ms per 8,192-token forward on B200, and nothing in batches that use
  the side streams.

## How it is measured

The probe measures three things:

1. Isolated drift. During one prefill with fp32 output, every compressor and
   every indexer runs a second time on the same input. The second run uses
   bf16 output and a copy of the state. The difference is the effect of the
   rounding at that layer alone, without drift from earlier layers.
2. End-to-end drift. Two full prefills of the same prompt run, one per
   setting, and each one decodes 32 greedy tokens. The comparison covers the
   logits at every position and the generated tokens. It also covers every
   attention block's output at 380 positions: every 64th token and the last
   256. It also scores the real next token at every prompt position, as an
   NLL and a top-1 accuracy.
3. GEMM time. `torch.mm` runs with and without `out_dtype=torch.float32` on
   random bf16 inputs at the real shapes (8,192 tokens, K = 4,096). Each
   variant gets 10 warmup and 50 timed calls, with TF32 off.

The main run compares three pairs of settings on one model load:

- fp32 output against bf16 output, the question itself.
- fp32 output against fp32 output, which shows whether the forward is
  deterministic.
- `torch.mm(..., out_dtype=torch.float32)` against an fp32 GEMM over upcast
  inputs. The math is the same and only the reduction order differs, so this
  pair is the noise floor.

Both ranks produced the same tokens and the same argmax flips. The fp32
setting ran the same `torch.mm(..., out_dtype=torch.float32)` call as vLLM.
Peak memory was 110 GB per GPU.

## End-to-end drift

Logits and tokens over the 8,192 prompt positions:

| pair | argmax agreement | mean KL at 380 positions | NLL change | top-1 accuracy | greedy tokens kept |
|---|---:|---:|---:|---:|---:|
| fp32 twice | 100% | 0 | 0 | 65.41% / 65.41% | 32 of 32 |
| reordered fp32 GEMM | 96.0% | 0.011 nats | +0.0005 nats | 65.41% / 65.51% | 23 of 32 |
| bf16 output | 96.0% | 0.012 nats | +0.0007 nats | 65.41% / 65.51% | 23 of 32 |

The NLL change is the mean over 8,191 scored positions. The probe also reports
a standard error from blocks of 256 positions (0.0016 and 0.0019 nats). That
error assumes that blocks are independent, but a change early in the prompt
reaches every later block, so it understates the real uncertainty. Both
changes are far below it anyway.

Relative RMS change of the attention-block output between the two runs of a
pair:

| layer | type | fp32 twice | reordered fp32 GEMM | bf16 output |
|---:|---|---:|---:|---:|
| 2 | CSA | 0 | 0.6% | 1.5% |
| 5 | HCA | 0 | 5.8% | 5.9% |
| 9 | HCA | 0 | 7.6% | 7.6% |
| 14 | CSA | 0 | 14.9% | 14.1% |
| 22 | CSA | 0 | 19.2% | 20.1% |
| 34 | CSA | 0 | 19.6% | 18.8% |
| 42 | CSA | 0 | 4.8% | 4.9% |

At the first CSA layer, bf16 output moves the output more than the reordered
GEMM does. From layer 5 on, the two are the same size. The model makes
discrete choices (FP8 and FP4 bins, top-k selection, expert routing), so any
small change spreads through the layers. A different GEMM kernel also
reorders the reduction, and vLLM is not batch invariant by default (see its
batch-invariance work in
[#27433](https://github.com/vllm-project/vllm/issues/27433)).

The probe also records the NLL of the fp32 run per block of 256 positions.
It stays between 0.74 and 2.53 nats over the whole prompt, with no rise
toward the end. So quality does not fall with distance.

## Isolated drift per layer

Change of the compressed entries when only that layer's projection output is
rounded to bf16:

| projection | layers | relative RMS change |
|---|---:|---:|
| CSA main compressor | 21 | 1.02% |
| CSA indexer compressor | 21 | 2.96% |
| HCA compressor | 20 | 0.47% |

Indexer top-k selection, 8,192 queries per CSA layer, up to 512 blocks each,
as the mean over the 21 CSA layers:

| measure | value |
|---|---:|
| selected blocks that stay selected | 99.503% |
| queries with the same selected set | 27.6% |
| changed blocks per query | 2.54 |

## GEMM time on B200

| GEMM (kv and score fused) | N | fp32 output | bf16 output | saved |
|---|---:|---:|---:|---:|
| CSA main compressor | 2,048 | 95.1 µs | 89.8 µs | 5.3 µs (5.6%) |
| HCA compressor | 1,024 | 49.1 µs | 46.5 µs | 2.5 µs (5.2%) |
| CSA indexer compressor | 512 | 27.8 µs | 25.6 µs | 2.2 µs (8.0%) |

All three GEMMs run at 1.2 to 1.5 PFLOP/s, so they are bound by compute, not
by the output write. Over 21 CSA and 20 HCA layers, bf16 output saves 0.21 ms
per 8,192-token forward. The largest error against an fp32 reference GEMM is
4.9e-5 with fp32 output and 1.6e-2 with bf16 output. At decode batch sizes
the output is small. With 64 tokens, the CSA GEMM writes 0.5 MB of fp32
output and reads 16 MB of weights. Decode batches were not timed. The GEMM
times come from the first run (`v4-projection-dtype-probe.json`).

## Earlier runs and the RoPE fix

Two earlier runs used a version of the port with a RoPE bug. It rotated every
layer with `rope_theta` (10000) and YaRN. The reference rotates compressed
layers with `compress_rope_theta` (160000) and YaRN, and sliding-window layers
with `rope_theta` and no YaRN. With the bug, the fp32 setting scored a
perplexity of 247 and a top-1 accuracy of 21.5 percent on this prompt. The
fix (commit `009555e`) brings these to 4.4 and 65.4 percent. A test now
compares each layer's table with the reference.

The earlier runs answered the same question on that weaker model:

- Run 1 (`v4-projection-dtype-probe.json`) had no FP8 or FP4 round trips. It
  recorded the GEMM times above, an isolated change of 0.3 percent in the
  compressed entries, and 78.9 percent argmax agreement with bf16 output.
- Run 2 (`v4-projection-dtype-probe-qat.json`) had the round trips on. bf16
  output and the reordered GEMM gave 73.7 and 74.9 percent argmax agreement,
  with NLL changes of +0.19 and +0.10 nats. A model with flat predictions has
  many near-ties, so it drifts much more.

Both runs agree with the main run on the direction: bf16 output drifts about
as far as a reordered fp32 GEMM.

## Limits

- The run used one prompt of 8,192 tokens, one model (V4-Flash), and one GPU
  type (B200). vLLM itself was not run.
- The port simulates the reference's FP8 and FP4 round trips in PyTorch.
  vLLM's kernels quantize with their own layouts (for example MXFP4 for the
  indexer), so its exact bin edges differ.
- The GEMM times use random inputs at one batch size.
