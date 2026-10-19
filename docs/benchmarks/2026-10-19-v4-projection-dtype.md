# DeepSeek-V4 compressor projections: fp32 output against bf16 output

Date: 2026-10-19
Hardware: 2x NVIDIA B200, tensor parallel over 2 ranks, bf16 weights, `torch` 2.10.0+cu128
Model: `deepseek-ai/DeepSeek-V4-Flash` (43 layers: 21 CSA, 20 HCA, 2 SWA)
Prompt: 8,192 tokens of technical prose from this repository
(`docs/architectures/deepseek-v4.md`, then `deepseek-v2-mla.md`, cut at 8,192
tokens), followed by 32 greedy tokens
Engine: mini-infer at `3a846fe` with the projection-dtype probe of this change
Upstream code compared: vLLM `main` at `b965ee5`, `vllm/models/deepseek_v4/attention.py`
Script: `scripts/modal_v4_projection_dtype_probe.py`
Data: `docs/benchmarks/data/v4-projection-dtype-probe.json`

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
- In our port, the rounding moves each compressed entry by about 0.3 percent,
  which is one bf16 rounding step. The indexer selects up to 512 blocks per
  query. On every CSA layer, the rounding changes at least one of them for 40
  to 51 percent of the queries. On average 0.56 blocks change per query.
  These changes are near-ties at the edge of the top-k.
- vLLM quantizes the same entries right after the compression. The indexer
  keys go to FP8 or MXFP4, and the main entries go to FP8 on the default
  `fp8_ds_mla` cache. The indexer queries are FP8 or MXFP4 as well.
  DeepSeek's reference applies the same FP8 and FP4 round-trips. FP8 e4m3 has
  a relative step of 2^-3 and bf16 has 2^-7. So the quantization that follows
  is 16 times coarser than the rounding in question. Only a configuration
  with a bf16 cache for the main attention keeps the rounding as a new error.
- End to end, our port gives different outputs under the two policies. The
  argmax next token agrees at 78.9 percent of the prompt positions, and the
  greedy continuations agree for 18 of 32 tokens. This run had no control
  for noise and no quality metric, so it does not show that either output is
  worse. See "Open question".
- Keeping fp32 is the better choice. The change saves at most about 0.2 ms
  per 8,192-token forward on B200, and nothing in batches that use the side
  streams. It also needs new warmup specializations. vLLM warms its
  `save_partial_states` Triton kernel only for fp32 kv and score inputs. With
  bf16 inputs, a new kernel compiles while the server runs.

## How it is measured

The probe measures three things:

1. Isolated drift. During one prefill with fp32 output, every compressor and
   every indexer runs a second time on the same input. The second run uses
   bf16 output and a copy of the state. The difference is the effect of the
   rounding at that layer alone, without drift from earlier layers.
2. End-to-end drift. Two full prefills of the same prompt run, one per
   policy, and each one decodes 32 greedy tokens. The comparison covers the
   logits at every position and the generated tokens. It also covers every
   attention block's output at 380 positions: every 64th token and the last 256.
3. GEMM time. `torch.mm` runs with and without `out_dtype=torch.float32` on
   random bf16 inputs at the real shapes (8,192 tokens, K = 4,096). Each
   variant gets 10 warmup and 50 timed calls, with TF32 off.

Both ranks produced the same tokens and the same argmax flips, so the
tensor-parallel run is consistent. The fp32 policy ran the same
`torch.mm(..., out_dtype=torch.float32)` call as vLLM. Peak memory was 110 GB
per GPU.

## Isolated drift per layer

Change of the compressed entries when only that layer's projection output is
rounded to bf16:

| projection | layers | relative RMS change, mean | relative RMS change, max | lowest cosine |
|---|---:|---:|---:|---:|
| CSA main compressor | 21 | 0.31% | 0.37% | 0.999993 |
| CSA indexer compressor | 21 | 0.30% | 0.36% | 0.999994 |
| HCA compressor | 20 | 0.16% | 0.24% | 0.999997 |

Indexer top-k selection, 8,192 queries per CSA layer, up to 512 blocks each:

| measure | lowest layer | highest layer | mean over layers |
|---|---:|---:|---:|
| selected blocks that stay selected | 99.866% | 99.909% | 99.890% |
| queries with the same selected set | 48.7% | 60.4% | 54.8% |
| changed blocks per query | | | 0.56 |

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
the output is small: 64 tokens give 0.5 MB of fp32 output, against 16 MB of
weights that the CSA GEMM reads. Decode batches were not timed.

## End-to-end drift in our port

Change of the attention-block output between the two runs:

| layer | type | relative RMS change | cosine |
|---:|---|---:|---:|
| 2 | CSA | 0.5% | 0.99988 |
| 3 | HCA | 0.6% | 0.99988 |
| 5 | HCA | 2.9% | 0.99947 |
| 9 | HCA | 6.9% | 0.99749 |
| 14 | CSA | 19.1% | 0.98156 |
| 22 | CSA | 23.9% | 0.97119 |
| 34 | CSA | 29.3% | 0.95719 |
| 42 | CSA | 8.7% | 0.99613 |

The change grows with depth. Each CSA layer adds new near-tie flips to its
top-k, and discrete choices (top-k selection, MoE routing) carry a small
change into larger ones. At the logits, the argmax agrees at 6,462 of 8,192
positions. Among the flips, the largest top-1 to top-2 gap in the fp32 run is
3.6 logits. The mean KL divergence at the 380 sampled positions is 0.145 nats.
The greedy continuations agree for their first 18 of 32 tokens.

## Open question

The end-to-end numbers above have no reference point. The probe now has two
controls and a quality metric, which did not run on the real model yet:

1. fp32 against fp32 shows whether the forward is deterministic.
2. `torch.mm(..., out_dtype=torch.float32)` against an fp32 GEMM over upcast
   inputs shows how far a change of about 1e-7 spreads in this model. The
   math is the same and only the rounding order differs.
3. The probe scores the real next token at every prompt position under both
   policies, as a negative log-likelihood and a top-1 accuracy. This shows
   whether bf16 output is worse or only different.

To run them, pass `--pairs fp32_vs_native,fp32_vs_fp32,fp32_vs_fp32_upcast
--skip-gemm-bench` to the script. The result does not change the conclusion
for vLLM, because the time that bf16 output saves is too small to justify a
change.

## Limits

- Our port does not apply the FP8 and FP4 round-trips that DeepSeek's
  reference and vLLM apply to the compressed entries and the indexer queries.
  So the isolated drift measures the rounding without that quantization. In
  vLLM's default path, a 16 times coarser quantization follows the same
  entries.
- The run used one prompt of 8,192 tokens, one model (V4-Flash), and one GPU
  type (B200). vLLM itself was not run.
- The GEMM times use random inputs at one batch size.
