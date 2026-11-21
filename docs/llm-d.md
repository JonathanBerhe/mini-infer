# Run mini-infer behind the llm-d router

The [llm-d](https://github.com/llm-d/llm-d) router sends each request to the server that can serve it best. It reads two things from each server. Prometheus gauges tell it the load. KV-cache events tell it which prompt prefixes each server holds, so that it can send a request to the server that already has the start of its prompt.

mini-infer gives both in the format of vLLM. A router that is configured for vLLM reads a mini-infer server with no change. [ADR-060](decisions/ADR-060-vllm-router-contract.md) gives the reasons for this design.

## Start the server

The `kv-events` extra installs `pyzmq` and `msgpack`. The dev group of `uv sync` also installs them.

| Variable | Example | Effect |
|---|---|---|
| `MINI_INFER_SERVED_MODEL_NAME` | `qwen-7b` | The name that clients send in the `model` field. `/v1/models`, the metric labels and the event topic use it. The default is `MINI_INFER_MODEL`. |
| `MINI_INFER_PREFIX_CACHE` | `1` | Gives the server a prefix cache. A request then reuses the KV blocks of an earlier prompt with the same start. |
| `MINI_INFER_KV_EVENTS_ENDPOINT` | `tcp://*:5557` | Publishes KV-cache events on this ZMQ endpoint. It also turns on the prefix cache, because the events report its blocks. |
| `MINI_INFER_KV_EVENTS_TOPIC` | `kv@10.0.0.7:8000@qwen-7b` | The ZMQ topic. The default is `kv@<served model name>`. |

The endpoint follows the rule of vLLM. An address with `*`, `::`, `ipc://` or `inproc://` binds a socket. Any other address connects to a socket that the router binds.

This command starts a server that the router can subscribe to:

```bash
MINI_INFER_MODEL=Qwen/Qwen2.5-7B-Instruct MINI_INFER_HOST=0.0.0.0 MINI_INFER_KV_EVENTS_ENDPOINT='tcp://*:5557' uv run --extra kv-events python -m mini_infer.api.server
```

## Endpoints

| Endpoint | API key | Content |
|---|---|---|
| `GET /health` | No | 200 while the engine thread runs, 503 after it stops. |
| `GET /metrics` | No | `vllm:num_requests_waiting`, `vllm:num_requests_running`, `vllm:kv_cache_usage_perc` and `vllm:cache_config_info`. |
| `GET /v1/models` | Yes | The served model name. |
| `POST /v1/completions` | Yes | The completions API. |

The KV-cache usage is a fraction from 0 to 1. Cached blocks that no request holds count as free, because the pool can take them back at once. vLLM counts them the same way.

## Configure the router

Use the vLLM engine type, which is the default. Set the router block size to `MINI_INFER_BLOCK_SIZE`, which is 16 by default.

The router has two ways to receive events:

- Per pod (the default). The router connects to `tcp://<pod IP>:5557` on each server. Bind `tcp://*:5557` on the server and keep the default topic.
- One shared socket. The router binds one socket for all servers. Set the server endpoint to the router address, and set the topic to `kv@<pod address>@<model name>`.

This block of the router configuration enables the per-pod mode. It comes from `test/perf/config/router-configs/precise-prefix.yaml` in llm-d-router:

```yaml
- type: precise-prefix-cache-producer
  parameters:
    tokenProcessorConfig:
      blockSizeTokens: 16
    kvEventsConfig:
      topicFilter: "kv@"
      discoverPods: true
      podDiscoveryConfig:
        socketPort: 5557
```

## Token IDs must match

The router computes its block keys from token IDs. It tokenizes each prompt itself, with a vLLM render server in the llm-d deployments. mini-infer tokenizes a `/v1/completions` prompt with the Hugging Face tokenizer of `MINI_INFER_MODEL`, with its special tokens. Use the same model on both sides. If the token IDs differ, the router finds no prefix and routes by load only.

## Limits

- Only the paged path with `ContinuousScheduler` publishes events. A state-cache model or `MINI_INFER_USE_PD=1` refuses `MINI_INFER_KV_EVENTS_ENDPOINT` at startup. For these servers, `/metrics` returns no samples.
- The prefix cache holds prompt blocks only. In a multi-turn chat, the next prompt matches the blocks of the previous prompt, but not the blocks of the previous answer.
- The server has no replay or snapshot socket. If the network drops a message, the router index stays wrong for those blocks. In the per-pod mode the router finds a server restart from the sequence number. The first message after startup is `AllBlocksCleared`, which also clears old entries.
- The server has no `/v1/chat/completions` endpoint and no LoRA adapters.

## What the tests cover

The unit tests run on the CPU:

- `tests/unit/cache/test_prefix_cache_events.py`: the events of the prefix cache, the parent chain and the evictions.
- `tests/unit/api/test_kv_events_publisher.py`: the wire format, and a real ZMQ round trip.
- `tests/unit/api/test_metrics_endpoints.py`: `/health`, `/metrics`, `/v1/models` and the load numbers.

On 2026-11-21 we also ran the router's own Go code against mini-infer, at llm-d-router commit `733d0e4`:

- The router's vLLM adapter decoded `BlockStored`, `BlockRemoved` and `AllBlocksCleared` from mini-infer. It kept the parent hash of each stored block.
- A live server with Qwen2.5-0.5B-Instruct on an M1 Mac published to the router's event pool in the per-pod mode, with its in-memory index and a block size of 16. Two prompts of 80 and 71 tokens shared their first 32 tokens. The index mapped all 5 full blocks of the first prompt and all 4 full blocks of the second prompt to the server.

These runs did not cover a full llm-d deployment in Kubernetes or the routing decisions of the router.
