"""Prometheus metrics for the `/metrics` endpoint.

The metric names are vLLM's, because routers read vLLM servers with these names
by default. The llm-d router uses the queue gauges, the KV-cache gauge and
`vllm:cache_config_info` (see its `docs/plugin-metric-protocol.md`), so a router
configured for vLLM can score mini-infer servers with no change. The text uses
the Prometheus text exposition format, version 0.0.4.
"""

from mini_infer.scheduler.load import SchedulerLoad

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _label_value(value: str) -> str:
    """Escape a label value as the exposition format requires."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _gauge(name: str, help_text: str, labels: dict[str, str], value: float) -> list[str]:
    label_text = ",".join(f'{key}="{_label_value(val)}"' for key, val in labels.items())
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
        f"{name}{{{label_text}}} {float(value)!r}",
    ]


def render_metrics(load: SchedulerLoad | None, model_name: str) -> str:
    """Return the metrics text. With no load snapshot, the text holds no samples.

    `model_name` is the name that clients send in the `model` field, as in
    vLLM's `model_name` label.
    """
    if load is None:
        return ""
    model = {"model_name": model_name}
    lines = [
        *_gauge(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed.",
            model,
            load.num_waiting,
        ),
        *_gauge(
            "vllm:num_requests_running",
            "Number of requests in model execution batches.",
            model,
            load.num_running,
        ),
        *_gauge(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage.",
            model,
            load.kv_cache_usage,
        ),
        *_gauge(
            "vllm:cache_config_info",
            "Information of the KV-cache configuration.",
            {"block_size": str(load.block_size), "num_gpu_blocks": str(load.num_blocks)},
            1,
        ),
    ]
    return "\n".join(lines) + "\n"
