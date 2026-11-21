# ADR-060: Give routers the metrics and KV-cache events of vLLM

Date: 2026-11-21
Status: Accepted

## Context

A KV-cache-aware router, such as the [llm-d](https://github.com/llm-d/llm-d)
router, sends each request to the server that already holds the longest
part of its prompt. It also scores servers by their load. It reads two
things from each server:

- Prometheus gauges for the queued requests, the running requests and the
  KV-cache use (the llm-d `docs/plugin-metric-protocol.md`).
- KV-cache events over ZMQ: a block entered the cache, a block left it, or
  the cache is empty. The router builds an index of the blocks that each
  server holds from these events.

The mini-infer HTTP server had neither. It also ran without a prefix cache.

## Decision

1. `/metrics` returns the gauges with vLLM's names and meaning:
   `vllm:num_requests_waiting`, `vllm:num_requests_running`,
   `vllm:kv_cache_usage_perc` (a fraction, with unheld cached blocks counted
   as free) and `vllm:cache_config_info`. `ContinuousScheduler.load_stats()`
   gives the values without a lock.
2. `PrefixCache` calls an optional listener with a `BlockStored` event when a
   new block enters it and a `BlockRemoved` event when it evicts one. Each
   stored block carries its tokens and the hash of its parent block, because
   the router drops a block whose parent it does not know.
3. `ZmqKVEventPublisher` sends the events in vLLM's wire format: frames
   `[topic, 8-byte sequence, msgpack batch]`, with each event as a tagged
   array in vLLM's field order. A background thread sends them, so the
   engine thread only puts them on a queue.
4. `MINI_INFER_KV_EVENTS_ENDPOINT` turns on the events and the prefix cache.
   `/health` and `/v1/models` complete the endpoints that routers and
   probes expect.

## Alternatives Considered

- Metrics with `mini_infer:` names. The llm-d router accepts other names
  through configuration, but every deployment then needs that change. The
  vLLM names work with the router defaults.
- A mini-infer adapter in the router. The llm-d engine-adapter registry
  allows one, but the project can decline code that has no production
  users. The vLLM format needs no router change.
- `prometheus_client` for `/metrics`. Four gauges in the text format need
  no new dependency.
- One `BlockStored` event per request, with all its new blocks. The cache
  publishes one block at a time, and the router accepts one-block events.

## Consequences

The llm-d router's own Go code, at llm-d-router commit `733d0e4`, decoded
the three event types. A live server with Qwen2.5-0.5B-Instruct on an M1 Mac
fed the router's event pool and in-memory index. The index mapped every
full block of two prompts with a shared prefix to the server
([docs/llm-d.md](../llm-d.md)).

The work has these costs and limits:

- `pyzmq` and `msgpack` are new optional dependencies (the `kv-events`
  extra).
- Only the paged path with `ContinuousScheduler` supports the events. The
  state-cache and prefill/decode servers refuse them at startup.
- The publisher has no replay or snapshot socket, so a lost message leaves
  the router index wrong for those blocks until they change.
- The metric names say `vllm`, but the server is not vLLM. This is a
  compatibility contract, not a claim about the engine.
