"""`/health`, `/metrics`, `/v1/models` and the scheduler load they report.

The endpoint tests set a stub scheduler on the app and skip the lifespan, so
they load no model.
"""

import queue
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from mini_infer.api.metrics import render_metrics
from mini_infer.api.server import app
from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.engine.sampler import SamplingParams
from mini_infer.scheduler import ContinuousScheduler, Request, RequestState, SchedulerLoad
from mini_infer.scheduler.request_state import RunningRequest

MakeClient = Callable[..., TestClient]

LOAD = SchedulerLoad(
    num_waiting=3, num_running=2, kv_cache_usage=0.25, block_size=16, num_blocks=1024
)


class _StubScheduler:
    def __init__(self, *, running: bool, load: SchedulerLoad | None) -> None:
        self.is_running = running
        if load is not None:
            self.load_stats = lambda: load


@pytest.fixture
def make_client() -> Iterator[MakeClient]:
    """Return a function that serves the app with a stub scheduler."""
    saved = dict(app.state._state)

    def _make(*, running: bool = True, load: SchedulerLoad | None = LOAD) -> TestClient:
        app.state.scheduler = _StubScheduler(running=running, load=load)
        app.state.served_model_name = "served/model"
        return TestClient(app)

    yield _make
    app.state._state.clear()
    app.state._state.update(saved)


def test_health_is_200_while_the_engine_runs(make_client: MakeClient) -> None:
    response = make_client(running=True).get("/health")
    assert response.status_code == 200


def test_health_is_503_after_the_engine_stops(make_client: MakeClient) -> None:
    response = make_client(running=False).get("/health")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "engine_unavailable"


def test_metrics_endpoint_serves_vllm_gauges(make_client: MakeClient) -> None:
    response = make_client().get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    lines = response.text.splitlines()
    assert 'vllm:num_requests_waiting{model_name="served/model"} 3.0' in lines
    assert 'vllm:num_requests_running{model_name="served/model"} 2.0' in lines
    assert 'vllm:kv_cache_usage_perc{model_name="served/model"} 0.25' in lines
    assert 'vllm:cache_config_info{block_size="16",num_gpu_blocks="1024"} 1.0' in lines
    assert "# TYPE vllm:kv_cache_usage_perc gauge" in lines


def test_metrics_without_load_stats_has_no_samples(make_client: MakeClient) -> None:
    response = make_client(load=None).get("/metrics")
    assert response.status_code == 200
    assert response.text == ""


def test_models_lists_the_served_name(make_client: MakeClient) -> None:
    response = make_client().get("/v1/models")
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == ["served/model"]


def test_label_values_are_escaped() -> None:
    text = render_metrics(LOAD, 'a"b\\c\nd')
    assert 'model_name="a\\"b\\\\c\\nd"' in text


def _pool(num_blocks: int, *, prefix_cache: PrefixCache | None = None) -> BlockPool:
    return BlockPool(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
        prefix_cache=prefix_cache,
    )


def _running_request(state: RequestState) -> RunningRequest:
    request = RunningRequest(
        request=Request(prompt="p", sampling_params=SamplingParams(), max_tokens=1),
        output_queue=queue.Queue(maxsize=4),
    )
    request.state = state
    return request


def test_load_stats_counts_queues_and_blocks() -> None:
    pool = _pool(num_blocks=8)
    scheduler = ContinuousScheduler(SimpleNamespace(block_pool=pool))  # type: ignore[arg-type]
    for _ in range(3):
        pool.allocate()
    scheduler._waiting.put(_running_request(RequestState.WAITING))
    scheduler._running.extend(
        [_running_request(RequestState.DECODING), _running_request(RequestState.DONE)]
    )

    assert scheduler.load_stats() == SchedulerLoad(
        num_waiting=1, num_running=1, kv_cache_usage=3 / 8, block_size=4, num_blocks=8
    )


def test_load_stats_counts_unheld_cached_blocks_as_free() -> None:
    prefix_cache = PrefixCache(block_size=4)
    pool = _pool(num_blocks=4, prefix_cache=prefix_cache)
    scheduler = ContinuousScheduler(SimpleNamespace(block_pool=pool))  # type: ignore[arg-type]
    held, cached = pool.allocate(), pool.allocate()
    tokens = (1, 2, 3, 4)
    prefix_cache.publish(PrefixCache.hash_block(None, tokens), tokens, cached)
    pool.free(cached)  # refcount 0: still cached, but the pool can take it back

    assert held != cached
    assert scheduler.load_stats().kv_cache_usage == pytest.approx(1 / 4)
