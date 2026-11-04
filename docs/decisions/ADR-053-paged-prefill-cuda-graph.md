# ADR-053: Replay prefill and mixed steps from CUDA graphs

Date: 2026-11-04
Status: Accepted

## Context

[ADR-050](ADR-050-paged-decode-cuda-graph.md) replays a decode step from a
CUDA graph. Every other packed step ran eagerly: the step that admits a
prompt, and the step that carries a prefill chunk next to the decoders. Such
a step of Qwen2.5-7B launches about 330 kernels for 4 to 6 ms of GPU work, so
the host set its time. At 51 new tokens it took 12.9 ms on an H100 and 5.9 ms
on a B200.

## Decision

1. `StaticDecodeStep` becomes `StaticStep(rows, slots)`. It holds a write
   index per row and query boundaries per step, so a request can add any
   number of tokens. `PagedKVCache.begin_static_step` does the host work.
   The rows after the new tokens are padding: one in each padding slot while
   they last, and the rest in the last slot. Padding rows write into a
   scratch block and attend to it. A decode step is the case of one row per
   slot.
2. `PagedPrefillGraph` captures a step for each bucket of new tokens (16 to
   512). A bucket of `rows` tokens has `min(max_requests + 1, rows)` slots,
   because a FlashInfer graph plan bounds the longest request by
   `rows - slots + 1`. A step takes the smallest bucket that holds its tokens
   and its requests. The buckets share one CUDA memory pool.
3. `ModelRunner.forward_step_packed` sends a decode step to the decode graph
   and any other step that fits a bucket to the prefill graph. A step with
   hidden-state taps or device-side ids runs eagerly. `prefill_graph=True` on
   `from_pretrained` turns the graph on. The default stays off.
4. A static step publishes its filled prompt blocks to the prefix cache when
   it ends, unless the step failed.
5. The FlashInfer workspace grows from 128 to 256 MiB. A graph plan sizes its
   split-KV buffers for its largest grid, and on a B200 the 512-token bucket
   needs 136 MB.

## Alternatives Considered

- One graph per layer piece, with eager attention between the pieces. It
  needs no fixed attention plan, but it splits the model forward and costs
  an eager attention call per layer.
- One graph per pair of token count and request count. That is more
  captures and more memory. Padding slots cost a few scratch pages instead.
- Plan the attention for the actual step lengths. A graph plan must keep the
  tile size of its capture, so the plan covers the whole bucket.

## Consequences

Measured on Qwen2.5-7B with the FlashInfer backend, decode graph on, in one
container per GPU.

| Prefill step, median wall ms | H100 eager | H100 graph | B200 eager | B200 graph |
|---|---:|---:|---:|---:|
| 1 prompt, 16 tokens | 12.86 | 6.46 | 5.62 | 3.57 |
| 8 prompts, 51 tokens | 12.86 | 7.25 | 5.90 | 3.97 |
| 1 prompt, 500 tokens | 17.06 | 13.72 | 8.86 | 8.80 |

- A prefill step launches 2 times instead of about 330. At concurrency 8
  the H100 goes from 1203 to 1251 tokens per second in bf16 and from 1669 to
  1833 in int8. The B200 goes from 2109 to 2168 and from 2217 to 2347.
- The attention of a graph sums in another order than the eager attention.
  The B200 tokens and the H100 int8 tokens were identical. Two of eight H100
  bf16 requests at concurrency 8 diverged after 19 and 20 tokens.
- The capture takes about 0.3 s and 0.6 GiB per GPU. A step with more
  requests than slots, or more tokens than the largest bucket, runs eagerly.
