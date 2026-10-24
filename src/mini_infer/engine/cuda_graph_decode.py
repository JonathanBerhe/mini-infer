"""Replay one ragged decode step of a state-cache model from a CUDA graph.

A decode step of the Qwen hybrid runs thousands of small kernels, and every
kernel costs the host a launch. At batch 1 the host cannot launch them as
fast as the GPU runs them, so the step is launch-bound, far below the time
it takes to read the weights. A CUDA graph records the whole step once and
then replays it with a single launch.

The model opts in with `SUPPORTS_DECODE_CUDA_GRAPH = True` and provides
`decode_step_static(input_id, positions, state_cache)`. That step must have
no host sync and no data-dependent shape, and it must update the cache only
in place: a replay reads and writes the exact tensors the graph saw at
capture. A model whose step meets this only on some devices or settings
also provides `can_capture_decode(device) -> bool`.

Warmup and capture run the step for real, so they write into every row of
the cache. Capture before any request is admitted; the scheduler's admission
copies a full row over whatever the capture left there.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.device import is_cuda_device

# Steps run eagerly on a side stream before capture, so that lazy
# initialization (cuBLAS handles, kernel selection, allocator pools) happens
# outside the graph.
_WARMUP_STEPS = 2


def supports_decode_cuda_graph(model: Any, device: torch.device | str) -> bool:
    """Whether `model` can decode through `DecodeCudaGraph` on `device`."""
    if not (is_cuda_device(device) and getattr(model, "SUPPORTS_DECODE_CUDA_GRAPH", False)):
        return False
    can_capture = getattr(model, "can_capture_decode", None)
    return can_capture is None or bool(can_capture(device))


class DecodeCudaGraph:
    """One captured `decode_step_static` over a fixed batch and a fixed cache."""

    def __init__(
        self,
        model: Any,
        state_cache: Any,
        *,
        batch_size: int,
        device: torch.device | str,
    ) -> None:
        if not supports_decode_cuda_graph(model, device):
            raise ValueError(
                f"{type(model).__name__} on {device} cannot decode through a CUDA graph"
            )
        self._input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self._positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        main_stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        side_stream.wait_stream(main_stream)
        with torch.inference_mode(), torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_STEPS):
                model.decode_step_static(self._input_ids, self._positions, state_cache)
        main_stream.wait_stream(side_stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self._graph):
            self._logits: torch.Tensor = model.decode_step_static(
                self._input_ids, self._positions, state_cache
            )

    def replay(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run the step for `(B, 1)` token ids and `(B,)` positions.

        The inputs may live on the host; they are copied into the graph's
        static buffers. The returned logits are the graph's own output
        buffer, which the next replay overwrites.
        """
        self._input_ids.copy_(input_ids)
        self._positions.copy_(positions)
        self._graph.replay()
        return self._logits
