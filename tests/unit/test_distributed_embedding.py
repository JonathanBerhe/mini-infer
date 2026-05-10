"""Unit tests for `VocabParallelEmbedding`.

Same contract as the linear tests:
  1. world_size=1 ≡ plain `nn.Embedding(vocab_size, hidden_size)`.
  2. world_size=2 under gloo: every rank's output (after the all-reduce
     in the layer) matches the single-device reference.

Note: world_size=2 lookups exercise the masking path (tokens outside a
rank's vocab slice should produce zero locally and be filled in by the
peer rank's contribution after all-reduce). We deliberately pick token
IDs that span both halves of the vocab so both ranks are exercised.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional

from mini_infer.distributed.embedding import VocabParallelEmbedding
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

# ----------------------------- world_size=1 -----------------------------


def test_vocab_parallel_embedding_world_size_1_matches_nn_embedding() -> None:
    torch.manual_seed(0)
    vocab_size, hidden_size = 64, 16
    full_table = torch.randn(vocab_size, hidden_size)

    reference = nn.Embedding(vocab_size, hidden_size)
    with torch.no_grad():
        reference.weight.copy_(full_table)
    tp = VocabParallelEmbedding(vocab_size, hidden_size)
    tp.load_full_weight(full_table)

    input_ids = torch.tensor([[0, 1, 2, 63, 32, 17]], dtype=torch.long)
    expected = reference(input_ids)
    actual = tp(input_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vocab_parallel_embedding_load_full_weight_rejects_wrong_shape() -> None:
    layer = VocabParallelEmbedding(64, 16)
    with pytest.raises(ValueError, match="full_weight shape"):
        layer.load_full_weight(torch.zeros(32, 16))


# ----------------------------- world_size=2 -----------------------------


def _vocab_parallel_worker(
    rank: int,
    world_size: int,
    full_table: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    vocab_size, hidden_size = full_table.shape
    layer = VocabParallelEmbedding(vocab_size, hidden_size)
    layer.load_full_weight(full_table)
    return layer(input_ids).detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_vocab_parallel_embedding_world_size_2_matches_reference() -> None:
    """All-reduce sums each token's contribution from its owner rank;
    the resulting tensor matches a plain embedding lookup with the full table."""
    torch.manual_seed(0)
    vocab_size, hidden_size = 64, 8  # 64 / 2 = 32 per rank
    full_table = torch.randn(vocab_size, hidden_size)
    # Mix tokens from both halves so the masking path is exercised on
    # every rank.
    input_ids = torch.tensor([[0, 31, 32, 63], [10, 50, 5, 40]], dtype=torch.long)

    reference_output = functional.embedding(input_ids, full_table)

    per_rank_outputs = run_multi_process(
        2,
        _vocab_parallel_worker,
        full_table,
        input_ids,
    )
    assert len(per_rank_outputs) == 2
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda m, r=rank: f"rank {r} embedding mismatch: {m}",
        )


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_vocab_parallel_embedding_world_size_2_handles_boundary_tokens() -> None:
    """Tokens at vocab_start and vocab_end-1 are the trickiest masking cases.

    A boundary off-by-one in the slice arithmetic would produce wrong
    embeddings here even when interior tokens look correct.
    """
    torch.manual_seed(1)
    vocab_size, hidden_size = 32, 4
    full_table = torch.randn(vocab_size, hidden_size)
    # Tokens at the exact slice boundaries: 0, 15 (last on rank 0), 16
    # (first on rank 1), 31 (last overall).
    input_ids = torch.tensor([[0, 15, 16, 31]], dtype=torch.long)

    reference_output = functional.embedding(input_ids, full_table)

    per_rank_outputs = run_multi_process(
        2,
        _vocab_parallel_worker,
        full_table,
        input_ids,
    )
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda m, r=rank: f"rank {r} boundary token mismatch: {m}",
        )
