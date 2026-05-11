"""Tests for `load_state_dict_with_tp`.

Pattern:
  1. Build a tiny owned model (Llama-shape, ws=1).
  2. Snapshot its `state_dict()` — this is what HF weights would look like.
  3. Build a fresh model.
  4. Call `load_state_dict_with_tp(fresh, state_dict)`.
  5. Confirm every parameter ends up bit-identical to the snapshot.

The multi-process counterpart (ws=2) confirms that the same state_dict
loaded onto two ranks produces a model that runs forward to the same
output as the single-rank reference.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.distributed.loader import load_state_dict_with_tp
from mini_infer.models.llama import LlamaConfig, LlamaForCausalLM
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process


def _tiny_llama_state_dict() -> tuple[LlamaForCausalLM, dict[str, torch.Tensor]]:
    """Build a tiny Llama, snapshot its state_dict, return both."""

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg)
    return model, {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_load_state_dict_with_tp_world_size_1_matches_state_dict_load() -> None:
    """At ws=1, our loader produces bit-identical weights to `load_state_dict`."""
    from mini_infer.models.llama import LlamaConfig, LlamaForCausalLM

    _, source_state_dict = _tiny_llama_state_dict()

    torch.manual_seed(1)  # different seed so the fresh model has different init
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    fresh_model = LlamaForCausalLM(cfg)
    missing, unexpected = load_state_dict_with_tp(fresh_model, source_state_dict)

    assert missing == set(), f"unexpected missing: {missing}"
    assert unexpected == set(), f"unexpected unexpected: {unexpected}"
    # Every parameter on the fresh model should now equal the source.
    for name, param in fresh_model.named_parameters():
        torch.testing.assert_close(
            param.detach(), source_state_dict[name], rtol=0, atol=0
        )


def _llama_tp_loader_worker(
    rank: int,
    world_size: int,
    source_state_dict: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Build Llama at this rank, load the state_dict via TP loader, return embedding output.

    We compare the EMBEDDING-only output (rather than the full forward) so
    the test stays decoupled from PagedKVCache setup — Phase 4's contract
    is per-rank loading, not full-model parity (which Phase 2/3 already
    cover at the per-module level).
    """
    from mini_infer.models.llama import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).eval()
    missing, unexpected = load_state_dict_with_tp(model, source_state_dict)
    assert missing == set(), f"rank {rank} missing: {missing}"
    assert unexpected == set(), f"rank {rank} unexpected: {unexpected}"

    # Run through embed + lm_head — exercises both VocabParallelEmbedding
    # (under TP this path is non-trivial: mask + all-reduce) and the
    # ColumnParallelLinear lm_head with `gather_output=True`.
    with torch.no_grad():
        embedded = model.model.embed_tokens(input_ids)  # (B, T, hidden)
        normed = model.model.norm(embedded)
        logits = model.lm_head(normed)  # (B, T, vocab) — gathered from all ranks
    return logits.detach().cpu()


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_load_state_dict_with_tp_world_size_2_matches_single_device() -> None:
    """ws=2 with the same state_dict produces the same embed -> norm -> lm_head
    output as the single-device reference loaded with `state_dict()`."""
    from mini_infer.models.llama import LlamaConfig, LlamaForCausalLM

    _, source_state_dict = _tiny_llama_state_dict()
    input_ids = torch.tensor([[1, 5, 10, 31, 0]], dtype=torch.long)

    # Reference: build a fresh single-device model with the same state_dict.
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    reference = LlamaForCausalLM(cfg).eval()
    load_state_dict_with_tp(reference, source_state_dict)
    with torch.no_grad():
        ref_embed = reference.model.embed_tokens(input_ids)
        ref_normed = reference.model.norm(ref_embed)
        ref_logits = reference.lm_head(ref_normed)

    per_rank_outputs = run_multi_process(
        2,
        _llama_tp_loader_worker,
        source_state_dict,
        input_ids,
    )
    for rank, output in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            output,
            ref_logits.detach().cpu(),
            rtol=1e-4,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} embed->lm_head mismatch: {m}",
        )
