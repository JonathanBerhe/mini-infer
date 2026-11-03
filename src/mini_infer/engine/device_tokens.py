"""Input ids of a decode step that stay on the device.

A scheduler that reads every sampled token to the host waits for the GPU at
the end of each step, then builds the next step while the GPU idles. With
`DeviceTokens` the next step takes its ids from the tensor that the last
step sampled, so the host can launch it first and read the tokens later.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from mini_infer.device import is_cuda_device


@dataclasses.dataclass(frozen=True)
class DeviceTokens:
    """One new token per slot, read from `source` on the device.

    `source` holds the ids that the previous step sampled, one per row of that
    step. Slot `b` of the coming step takes row `rows[b]`. A slot can go
    missing between the two steps (its request finished), so the rows need
    not be `0, 1, 2, ...`.
    """

    source: torch.Tensor
    rows: Sequence[int]

    def __post_init__(self) -> None:
        if self.source.dim() != 1:
            raise ValueError(
                f"source must be a 1-D id tensor, got shape {tuple(self.source.shape)}"
            )
        count = self.source.shape[0]
        bad = [row for row in self.rows if not 0 <= row < count]
        if bad:
            raise ValueError(f"rows {bad} fall outside the {count} sampled ids")

    def __len__(self) -> int:
        return len(self.rows)

    def gather(self) -> torch.Tensor:
        """The `(len(rows),)` ids in slot order, on the device of `source`, without a host wait.

        Rows `0, 1, ..., n-1` are the leading rows of `source` as they are: a
        view, no kernel. That is every step on which no request finished, and
        the steps where only the last requests of the batch did. Any other
        order gathers with an index that travels from pinned memory with a
        non-blocking copy on CUDA, so this call enqueues work and returns.
        """
        count = len(self.rows)
        if all(row == slot for slot, row in enumerate(self.rows)):
            return self.source[:count]
        device = self.source.device
        on_cuda = is_cuda_device(device)
        index = torch.tensor(list(self.rows), dtype=torch.long, pin_memory=on_cuda)
        return self.source.index_select(0, index.to(device, non_blocking=on_cuda))
