"""Vocab-parallel embedding for tensor-parallel inference.

Sharding strategy
-----------------
Each rank owns a contiguous slice of the vocabulary's embedding table.
For a token whose ID falls in this rank's slice, we look it up locally;
for any other token we emit a zero vector. The all-reduce sum then
recovers the correct embedding (zeros from non-owners + the real vector
from the owner sums to the real vector).

This is the Megatron pattern; it costs one all-reduce per embedding
lookup, but the alternative (broadcasting the full table to every rank
or sending IDs around) is worse for vocabulary sizes in the 100k-1M
range that V4 / Llama 3 / Gemma all use.

Single-device behaviour
-----------------------
At `world_size=1` this reduces to a plain `nn.Embedding`: the entire
vocab lives on the single rank, no masking is needed, no collective
runs. Existing single-device tests stay bit-identical.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from mini_infer.distributed.comm import all_reduce_sum
from mini_infer.distributed.group import get_rank, get_world_size
from mini_infer.distributed.linear import _split_size


class VocabParallelEmbedding(nn.Embedding):
    """Embedding sharded along the vocabulary axis.

    Subclasses `nn.Embedding` so callers that walk modules looking for
    `isinstance(m, nn.Embedding)` (any future quantizer, weight tying
    detection, etc.) treat us as the plain embedding we are at
    `world_size=1`.

    Args:
        vocab_size: full vocabulary size (sharded across ranks).
        hidden_size: embedding dimension (replicated; per-rank stores the
            full hidden width).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        world_size = get_world_size()
        vocab_per_rank = _split_size(vocab_size, world_size, "vocab_size")
        # `nn.Embedding.__init__` builds a `weight` of shape
        # `[vocab_per_rank, hidden_size]` and initialises it with
        # `N(0, 1)`. We then overwrite `num_embeddings` to the FULL vocab
        # size so loader code sees the logical (un-sharded) value.
        super().__init__(vocab_per_rank, hidden_size, dtype=dtype)
        self.num_embeddings = vocab_size  # logical, not per-rank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.world_size = world_size
        self.rank = get_rank()
        self.vocab_per_rank = vocab_per_rank
        self.vocab_start = self.rank * vocab_per_rank
        self.vocab_end = self.vocab_start + vocab_per_rank

    def load_full_weight(
        self,
        full_weight: torch.Tensor,
        *,
        target_device: torch.device | str | None = None,
    ) -> None:
        """Slice the rank's vocab range out of the full embedding table."""
        if full_weight.shape != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"full_weight shape {tuple(full_weight.shape)} does not match expected "
                f"({self.vocab_size}, {self.hidden_size})"
            )
        if self.weight.is_meta:
            sliced = full_weight[self.vocab_start : self.vocab_end].contiguous()
            if target_device is not None:
                sliced = sliced.to(device=target_device)
            self.weight = nn.Parameter(sliced, requires_grad=False)
        else:
            sliced = (
                full_weight[self.vocab_start : self.vocab_end].to(self.weight.dtype).contiguous()
            )
            with torch.no_grad():
                self.weight.copy_(sliced)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up `input_ids` against this rank's vocab slice; sum-reduce.

        Tokens outside this rank's slice are masked to zero before the
        local lookup; the all-reduce sums in the real vector from
        whichever rank owns each token.
        """
        if self.world_size == 1:
            # Fast path: the entire vocab is on this rank, no masking needed.
            # Bit-identical to `nn.Embedding(vocab_size, hidden_size)(input_ids)`.
            return functional.embedding(input_ids, self.weight)

        # Build a mask of "is this token in my slice", then shift IDs into
        # local coordinates `[0, vocab_per_rank)` and zero out the rest.
        # Out-of-slice IDs are clamped to 0; the mask zeros their lookup
        # so the reduction sees zeros from non-owners.
        in_slice = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp(min=0, max=self.vocab_per_rank - 1)
        # `clamp` keeps everything in bounds; the mask zeroes non-owners
        # afterwards.
        local_embeddings = functional.embedding(local_ids, self.weight)
        # `in_slice` has the same shape as `input_ids`; broadcast against
        # the trailing hidden dim.
        local_embeddings = local_embeddings * in_slice.unsqueeze(-1).to(local_embeddings.dtype)
        return all_reduce_sum(local_embeddings)
