"""The contract between PrefillWorker and DecodeWorker.

After prefill, the prefill worker has accumulated per-layer per-stream KV
tensors for the request. The decode worker needs:

  1. Those KV tensors materialized into its own paged cache.
  2. The first sampled token (sampled from the last prefill logit). This
     is yielded as the request's first output token and fed into the
     decode loop's first step.
  3. Sampling params + max-tokens + EOS so the decode loop knows when to
     stop.

`KVHandoff` is the serializable bundle that carries all of that.

The dataclass holds the KV tensors by reference (single-process pass).
The per-stream layout (rather than the legacy K/V pair) supports every
model family in the registry uniformly: standard models have streams
`("k", "v")`; MLA has `("kv_latent", "k_rope")`; V4 carries its own
named set. The decode worker pairs each handoff stream with the
matching pool stream by name.
"""

from __future__ import annotations

import dataclasses

import torch

from mini_infer.engine.sampler import SamplingParams


@dataclasses.dataclass
class KVHandoff:
    """KV state + first token + decode parameters for a request crossing PD.

    Attributes:
        request_id: opaque identifier (used for logging / multi-request
            routing in later slices).
        kv_streams_per_layer: outer list indexed by `layer_idx`; inner
            dict maps `stream_name` (matches `BlockPool.stream_names(l)`)
            to a packed tensor of shape `(prefill_len, num_kv_heads_s,
            head_dim_s)`. Tensors are on the prefill worker's device; the
            decode worker is responsible for moving them to its own device.
        prefill_len: number of prompt tokens encoded in the handoff. This
            is the absolute position where decode resumes; equivalently,
            the number of K/V positions already written per layer/stream.
        first_sampled_token_id: token sampled from the last prefill logit.
            The decode worker yields this as the first output token and
            feeds it into the first decode step.
        sampling_params: greedy / temperature / top-k / top-p. Same
            instance the request was admitted with.
        max_tokens: total max output length (including the first sampled
            token). Decode stops when emitted tokens reach this many.
        eos_token_id: stop token; decode stops when emitted. `None`
            disables EOS-based termination (only max_tokens applies).
    """

    request_id: str
    kv_streams_per_layer: list[dict[str, torch.Tensor]]
    prefill_len: int
    first_sampled_token_id: int
    sampling_params: SamplingParams
    max_tokens: int
    eos_token_id: int | None = None

    @property
    def num_layers(self) -> int:
        return len(self.kv_streams_per_layer)

    def stream_names_for_layer(self, layer_idx: int) -> list[str]:
        """Stream names present in this layer's handoff entry (debug aid)."""
        return list(self.kv_streams_per_layer[layer_idx].keys())
