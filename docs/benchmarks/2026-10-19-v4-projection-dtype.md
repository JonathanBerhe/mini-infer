# DeepSeek-V4 compressor projections: fp32 output against bf16 output

Date: 2026-10-19
Hardware: 2x NVIDIA B200, tensor parallel over 2 ranks, bf16 weights, `torch` 2.10.0+cu128
Model: `deepseek-ai/DeepSeek-V4-Flash` (43 layers: 21 CSA, 20 HCA, 2 SWA)
Prompt: 8,192 tokens of technical prose from this repository
(`docs/architectures/deepseek-v4.md`, then `deepseek-v2-mla.md`, cut at 8,192
tokens), followed by 32 greedy tokens
Engine: run 1 used the first version of the probe (one comparison, no
reference quantization). Run 2 ran at `ee8d465`, with the reference's FP8 and
FP4 round trips on (`--simulate-act-quant --simulate-fp8-gemm-inputs`) and the
expert clamp at `swiglu_limit`.
Upstream code compared: vLLM `main` at `b965ee5`, `vllm/models/deepseek_v4/attention.py`
Script: `scripts/modal_v4_projection_dtype_probe.py`
Data: `docs/benchmarks/data/v4-projection-dtype-probe.json` (run 1),
`docs/benchmarks/data/v4-projection-dtype-probe-qat.json` (run 2)

## Question

vLLM issue [#53459](https://github.com/vllm-project/vllm/issues/53459) asks
whether two GEMMs of DeepSeek-V4 can emit bf16 instead of fp32. In
`_run_parallel_input_projections`, vLLM runs the main attention compressor
(`compressor_kv_score`) and the Lightning Indexer's compressor
(`indexer_compressor_kv_score`) with `torch.mm(..., out_dtype=torch.float32)`.

The issue calls this an fp32 accumulate. A bf16 GEMM accumulates in fp32 in
both cases. The output dtype decides one thing only: whether each projected kv
and score entry is rounded to bf16 once, before the compression softmax pools
it. This report measures that rounding and the time that bf16 output saves.

## Result

- On B200, bf16 output makes each of these GEMMs 5 to 8 percent faster at
  8,192 tokens. That is 2.2 to 5.3 µs per GEMM, or 0.21 ms per 8,192-token
  forward over all 41 compressed layers.
- vLLM runs these GEMMs on side streams, in parallel with the larger
  `fused_wqa_wkv` GEMM, for batches of up to 1,024 tokens. That limit is the
  default of `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`. In those batches the
  saving is hidden.
- Without the reference's FP8 and FP4 round trips (run 1), the rounding moves
  each compressed entry by about 0.3 percent, one bf16 rounding step. The
  indexer selects up to 512 blocks per query. On every CSA layer, the
  rounding changes at least one of them for 40 to 51 percent of the queries.
- With the round trips (run 2), the rounding changes more, not less. A small
  difference before an FP8 or FP4 rounding can move a value into the next
  quantization bin, which is a full quantization step. Compressed entries then
  move by 1.0 percent (main) and 3.0 percent (indexer), and 73 percent of the
  queries change at least one selected block.
- End to end, bf16 output drifts as far as a reordered fp32 GEMM. The same
  setting twice gives identical outputs, so the forward is deterministic. An
  fp32 GEMM over upcast inputs changes values by about 1e-7, and it moves the
  outputs as far as bf16 output does. The argmax agreement over the prompt is
  74.9 and 73.7 percent, and the attention outputs drift by the same amount at
  every layer.
- So the end-to-end drift comes from the model's discrete choices: FP8 and FP4
  bins, top-k selection, and expert routing. Any small change triggers it. A
  different GEMM kernel also reorders the reduction, and vLLM is not batch
  invariant by default (see its batch-invariance work in
  [#27433](https://github.com/vllm-project/vllm/issues/27433)).
- Next-token quality does not separate the two options. With bf16 output, the
  negative log-likelihood (NLL) of the real next token rises by 0.19 nats per
  token. With the reordered GEMM, it rises by 0.10. A change early in the
  prompt carries into every later position, so the one prompt gives two
  samples, not a measured gap.
- Keeping fp32 is still the better choice for vLLM, for the speed reason
  alone. The change saves at most 0.2 ms per 8,192-token forward on B200, and
  nothing in batches that use the side streams. It also needs new warmup
  specializations. vLLM warms its `save_partial_states` Triton kernel only for
  fp32 kv and score inputs. With bf16 inputs, a new kernel compiles while the
  server runs.

## Open issue: absolute quality

Run 2 measured the quality of the fp32 setting on the prompt itself. The
model predicts the next prompt token with a perplexity of 247 and a top-1
accuracy of 21.5 percent. That is far below
what a model of this size usually reaches on technical prose. Run 1 did not
record these numbers. Until the cause is known, quality numbers from this port
at 8,192 tokens are not trustworthy. The drift comparisons above are relative
and stay valid, but a weak model has more near-ties, so it can also drift more.

The probe now records the NLL of each run per block of 256 positions. That
curve shows whether the quality falls along the prompt. A fall along the
prompt points at the long-context path.

## How it is measured

The probe measures three things:

1. Isolated drift. During one prefill with fp32 output, every compressor and
   every indexer runs a second time on the same input. The second run uses
   bf16 output and a copy of the state. The difference is the effect of the
   rounding at that layer alone, without drift from earlier layers.
2. End-to-end drift. Two full prefills of the same prompt run, one per
   setting, and each one decodes 32 greedy tokens. The comparison covers the
   logits at every position and the generated tokens. It also covers every
   attention block's output at 380 positions: every 64th token and the last 256.
   It also scores the real next token at every prompt position, as an NLL and
   a top-1 accuracy (run 2 only).
3. GEMM time. `torch.mm` runs with and without `out_dtype=torch.float32` on
   random bf16 inputs at the real shapes (8,192 tokens, K = 4,096). Each
   variant gets 10 warmup and 50 timed calls, with TF32 off.

Run 2 compares three pairs of settings on one model load:

- fp32 output against bf16 output, the question itself.
- fp32 output against fp32 output, which shows whether the forward is
  deterministic.
- `torch.mm(..., out_dtype=torch.float32)` against an fp32 GEMM over upcast
  inputs. The math is the same and only the reduction order differs, so this
  pair is the noise floor.

In both runs, both ranks produced the same tokens and the same argmax flips.
The fp32 setting ran the same `torch.mm(..., out_dtype=torch.float32)` call as
vLLM. Peak memory was 110 GB per GPU.

## Isolated drift per layer

Change of the compressed entries when only that layer's projection output is
rounded to bf16:

| projection | layers | run 1, relative RMS change | run 2, relative RMS change |
|---|---:|---:|---:|
| CSA main compressor | 21 | 0.31% | 1.01% |
| CSA indexer compressor | 21 | 0.30% | 2.97% |
| HCA compressor | 20 | 0.16% | 0.30% |

Indexer top-k selection, 8,192 queries per CSA layer, up to 512 blocks each,
as the mean over the 21 CSA layers:

| measure | run 1 | run 2 |
|---|---:|---:|
| selected blocks that stay selected | 99.890% | 99.338% |
| queries with the same selected set | 54.8% | 26.8% |
| changed blocks per query | 0.56 | 3.39 |

The share of changed blocks is small. The share of queries with any change is
large, because each query selects up to 512 blocks. One near-tie at the edge
of the top-k is enough.

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
output and reads 16 MB of weights. Decode batches were not timed.

## End-to-end drift

Relative RMS change of the attention-block output between the two runs of a
pair, run 2:

| layer | type | fp32 twice | reordered fp32 GEMM | bf16 output |
|---:|---|---:|---:|---:|
| 2 | CSA | 0 | 0.6% | 1.5% |
| 5 | HCA | 0 | 7.5% | 7.9% |
| 9 | HCA | 0 | 13.8% | 14.1% |
| 14 | CSA | 0 | 29.2% | 29.2% |
| 22 | CSA | 0 | 28.5% | 28.9% |
| 34 | CSA | 0 | 30.4% | 30.8% |
| 42 | CSA | 0 | 10.1% | 10.8% |

At the first CSA layer, bf16 output moves the output more than the reordered
GEMM does. By layer 5 the two are the same size.

Logits and tokens over the 8,192 prompt positions:

| pair | argmax agreement | mean KL at 380 positions | NLL change | top-1 accuracy | greedy tokens kept |
|---|---:|---:|---:|---:|---:|
| run 2, fp32 twice | 100% | 0 | 0 | 21.5% / 21.5% | 32 of 32 |
| run 2, reordered fp32 GEMM | 74.9% | 0.226 nats | +0.10 nats | 21.5% / 20.8% | 32 of 32 |
| run 2, bf16 output | 73.7% | 0.234 nats | +0.19 nats | 21.5% / 20.0% | 3 of 32 |
| run 1, bf16 output | 78.9% | 0.145 nats | not recorded | not recorded | 18 of 32 |

The NLL change is the mean over 8,191 scored positions. The probe also reports
a standard error from blocks of 256 positions (0.016 and 0.017 nats). That
error assumes that blocks are independent, but a change early in the prompt
reaches every later block, so it understates the real uncertainty.

## Limits

- The run used one prompt of 8,192 tokens, one model (V4-Flash), and one GPU
  type (B200). vLLM itself was not run.
- The absolute quality of the port on this prompt is unexplained (see "Open
  issue: absolute quality").
- The GEMM times use random inputs at one batch size.
