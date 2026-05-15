"""KV-handoff transport over `torch.distributed`.

For point-to-point KV transfer between a prefill rank and a decode rank
that share a `torch.distributed` process group (NCCL on CUDA, gloo on CPU).

Wire format
-----------

For each handoff the sender emits, in order, a fixed-shape header tensor
followed by every KV stream's tensor:

  1. Header (7x int64): `[prefill_len, first_sampled_token_id,
     eos_token_id_or_minus_one, max_tokens, temperature_micros,
     top_k, top_p_micros]`. Temperature and top_p are encoded as fixed-
     point integers (multiplied by 1e6) to keep the header dtype uniform.
  2. KV streams, in `pool` order: for each layer (0..num_layers-1), for
     each stream in `sorted(pool.stream_names(layer))`, send the packed
     tensor of shape `(prefill_len, num_kv_heads_s, head_dim_s)`.

The receiver pre-allocates each stream tensor from the known pool
topology + the header-provided `prefill_len`. Both sides must have an
identical `BlockPool` (same model config, same dtype, same device).

Stream names are NOT transmitted: both sides know them from the model.
This makes the wire format compact (~no metadata overhead per stream)
and lets `dist.send/recv` operate on plain tensors only (NCCL doesn't
ship Python objects without an extra serialization step).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from mini_infer.cache.block_pool import BlockPool
from mini_infer.engine.sampler import SamplingParams
from mini_infer.workers.kv_handoff import KVHandoff

# Header packing (7 int64 fields). If the layout ever changes, both sides
# need to be redeployed together; bumping `_HEADER_LEN` here is the gate.
_HEADER_LEN = 7
_FIXED_POINT_SCALE = 1_000_000  # micros


def _encode_sampling_params(params: SamplingParams) -> tuple[int, int, int]:
    """Encode (temperature, top_k, top_p) as ints for the header tensor."""
    return (
        round(params.temperature * _FIXED_POINT_SCALE),
        params.top_k,
        round(params.top_p * _FIXED_POINT_SCALE),
    )


def _decode_sampling_params(t_micros: int, top_k: int, p_micros: int) -> SamplingParams:
    return SamplingParams(
        temperature=t_micros / _FIXED_POINT_SCALE,
        top_k=top_k,
        top_p=p_micros / _FIXED_POINT_SCALE,
    )


def send_handoff(
    handoff: KVHandoff, dst_rank: int, *, group: dist.ProcessGroup | None = None
) -> None:
    """Send a `KVHandoff` to `dst_rank`. Must be matched by `recv_handoff` there.

    Caller invariants:
      - `torch.distributed` is initialised and `dst_rank != dist.get_rank()`.
      - The receiver knows the model topology (BlockPool stream layout)
        identical to the sender's.
    """
    if not dist.is_initialized():
        raise RuntimeError("send_handoff called without torch.distributed initialised")
    if dst_rank == dist.get_rank():
        raise ValueError(f"send_handoff: dst_rank {dst_rank} == current rank")

    t_micros, top_k, p_micros = _encode_sampling_params(handoff.sampling_params)
    header = torch.tensor(
        [
            handoff.prefill_len,
            handoff.first_sampled_token_id,
            handoff.eos_token_id if handoff.eos_token_id is not None else -1,
            handoff.max_tokens,
            t_micros,
            top_k,
            p_micros,
        ],
        dtype=torch.int64,
    )
    dist.send(header, dst=dst_rank, group=group)

    for layer_streams in handoff.kv_streams_per_layer:
        # Sorted-name order: receiver iterates in the same order.
        for stream_name in sorted(layer_streams.keys()):
            stream_tensor = layer_streams[stream_name].contiguous()
            dist.send(stream_tensor, dst=dst_rank, group=group)


def recv_handoff(
    src_rank: int,
    pool: BlockPool,
    *,
    request_id: str = "",
    group: dist.ProcessGroup | None = None,
) -> KVHandoff:
    """Receive a `KVHandoff` from `src_rank` into freshly-allocated tensors.

    Tensors are allocated on the local `pool`'s device with `pool.dtype`.
    The caller is responsible for handing the result to a `DecodeWorker`
    (or whatever consumes it).
    """
    if not dist.is_initialized():
        raise RuntimeError("recv_handoff called without torch.distributed initialised")
    if src_rank == dist.get_rank():
        raise ValueError(f"recv_handoff: src_rank {src_rank} == current rank")

    header = torch.zeros(_HEADER_LEN, dtype=torch.int64)
    dist.recv(header, src=src_rank, group=group)
    prefill_len = int(header[0].item())
    first_sampled = int(header[1].item())
    eos_raw = int(header[2].item())
    eos_opt = None if eos_raw == -1 else eos_raw
    max_tokens = int(header[3].item())
    sampling_params = _decode_sampling_params(
        int(header[4].item()), int(header[5].item()), int(header[6].item())
    )

    dtype = pool.dtype
    kv_streams_per_layer: list[dict[str, torch.Tensor]] = []
    for layer_idx in range(pool.num_layers):
        layer_streams: dict[str, torch.Tensor] = {}
        for stream_name in sorted(pool.stream_names(layer_idx)):
            spec = pool.stream_spec(layer_idx, stream_name)
            stream_storage = pool.storage_for_stream(layer_idx, stream_name)
            stream_tensor = torch.empty(
                (prefill_len, spec.num_kv_heads, spec.head_dim),
                dtype=dtype,
                device=stream_storage.device,
            )
            dist.recv(stream_tensor, src=src_rank, group=group)
            layer_streams[stream_name] = stream_tensor
        kv_streams_per_layer.append(layer_streams)

    return KVHandoff(
        request_id=request_id,
        kv_streams_per_layer=kv_streams_per_layer,
        prefill_len=prefill_len,
        first_sampled_token_id=first_sampled,
        sampling_params=sampling_params,
        max_tokens=max_tokens,
        eos_token_id=eos_opt,
    )
