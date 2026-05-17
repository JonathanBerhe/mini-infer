"""Multi-process TP parity tests for the attention modules.

These tests target the *projection layer* of each attention type (Q/K/V
input, output projection). The downstream computation (RoPE,
softmax/SDPA, KV cache update, indexer top-k) operates on a per-head /
per-batch basis and is mathematically unchanged under TP — sharded heads
each compute the same kernel they always did. So the failure modes that
only TP introduces are weight-slicing and the column-then-row pairing,
which is what we exercise here.

Pattern:
  1. Build the attention module at world_size=1 with random weights —
     this is the reference.
  2. Spawn 2 ranks under gloo. Each rank constructs the same module
     class (its TP-aware linears self-shard) and loads the *same* full
     weight from the parent via `load_full_weight`.
  3. Feed the same input through. Compare the column-parallel projection
     outputs (each rank emits a slice that should concatenate to the
     reference) and the row-parallel projection outputs (each rank emits
     the full reduced output).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from tests.unit._distributed_test_utils import is_multi_process_available, run_multi_process

# ----------------------------- world_size=1 contract -----------------------------


def test_gqa_block_constructs_at_world_size_1() -> None:
    """At world_size=1 the TP-aware linears in GQA construct without
    raising and have parameter shapes equal to a plain `nn.Linear`. This
    is the contract that keeps existing single-device GQA tests green."""
    from mini_infer.models.blocks import GroupedQueryAttention

    hidden = 64
    num_q_heads = 8
    num_kv_heads = 4
    head_dim = 16

    block = GroupedQueryAttention(
        hidden_size=hidden,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qkv_bias=False,
        layer_idx=0,
    )

    assert isinstance(block.q_proj, ColumnParallelLinear)
    assert isinstance(block.k_proj, ColumnParallelLinear)
    assert isinstance(block.v_proj, ColumnParallelLinear)
    assert isinstance(block.o_proj, RowParallelLinear)
    # At world_size=1 the per-rank slices are the entire matrix.
    assert block.q_proj.weight.shape == (num_q_heads * head_dim, hidden)
    assert block.k_proj.weight.shape == (num_kv_heads * head_dim, hidden)
    assert block.v_proj.weight.shape == (num_kv_heads * head_dim, hidden)
    assert block.o_proj.weight.shape == (hidden, num_q_heads * head_dim)


def test_gqa_block_world_size_1_matches_nn_linear_projections() -> None:
    """A GQA's q_proj/k_proj/v_proj/o_proj at world_size=1 produce the
    same outputs as plain `nn.Linear` modules with the same weights.
    Bit-identical: this is what makes the existing 393 single-device
    tests indifferent to whether TP-aware linears are used."""
    from mini_infer.models.blocks import GroupedQueryAttention

    torch.manual_seed(0)
    hidden = 32
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8

    block = GroupedQueryAttention(
        hidden_size=hidden,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qkv_bias=False,
        layer_idx=0,
    )

    # Snapshot the random weights into plain `nn.Linear`s and confirm
    # both modules produce the same output for the same input.
    q_ref = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
    k_ref = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
    v_ref = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
    o_ref = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
    with torch.no_grad():
        q_ref.weight.copy_(block.q_proj.weight)
        k_ref.weight.copy_(block.k_proj.weight)
        v_ref.weight.copy_(block.v_proj.weight)
        o_ref.weight.copy_(block.o_proj.weight)

    x = torch.randn(2, 5, hidden)
    torch.testing.assert_close(block.q_proj(x), q_ref(x), rtol=0, atol=0)
    torch.testing.assert_close(block.k_proj(x), k_ref(x), rtol=0, atol=0)
    torch.testing.assert_close(block.v_proj(x), v_ref(x), rtol=0, atol=0)
    # Row-parallel input is sharded; at ws=1 sharded == full.
    o_input = torch.randn(2, 5, num_q_heads * head_dim)
    torch.testing.assert_close(block.o_proj(o_input), o_ref(o_input), rtol=0, atol=0)


# ----------------------------- world_size=2 parity -----------------------------


def _gqa_projection_worker(
    rank: int,
    world_size: int,
    hidden: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    full_q_weight: torch.Tensor,
    full_k_weight: torch.Tensor,
    full_v_weight: torch.Tensor,
    full_o_weight: torch.Tensor,
    x: torch.Tensor,
    o_input_full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the four GQA projections at the current rank and return outputs.

    The caller will compare these against the single-device reference:
      - q/k/v outputs are *sliced* along the last dim (column-parallel).
      - o output is the *full* hidden state (row-parallel = all-reduced).
    """
    from mini_infer.models.blocks import GroupedQueryAttention

    block = GroupedQueryAttention(
        hidden_size=hidden,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qkv_bias=False,
        layer_idx=0,
    )
    block.q_proj.load_full_weight(full_q_weight)
    block.k_proj.load_full_weight(full_k_weight)
    block.v_proj.load_full_weight(full_v_weight)
    block.o_proj.load_full_weight(full_o_weight)

    # The o_proj input is column-sharded (consumes the upstream Q's
    # sliced output). Pre-shard the test input here so the row-parallel
    # path sees what it would in a real attention forward.
    o_per_rank = (num_q_heads * head_dim) // world_size
    o_start = rank * o_per_rank
    o_input_local = o_input_full[..., o_start : o_start + o_per_rank].contiguous()

    return {
        "q": block.q_proj(x).detach().cpu(),
        "k": block.k_proj(x).detach().cpu(),
        "v": block.v_proj(x).detach().cpu(),
        "o": block.o_proj(o_input_local).detach().cpu(),
    }


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_gqa_projection_world_size_2_matches_reference() -> None:
    """GQA's column-parallel Q/K/V slices concatenate to the reference;
    the row-parallel O output equals the reference after all-reduce."""
    torch.manual_seed(0)
    hidden = 32
    num_q_heads = 4  # 4 / 2 = 2 heads per rank
    num_kv_heads = 2  # 2 / 2 = 1 head per rank
    head_dim = 8
    batch_size = 2
    seqlen = 3

    full_q_weight = torch.randn(num_q_heads * head_dim, hidden)
    full_k_weight = torch.randn(num_kv_heads * head_dim, hidden)
    full_v_weight = torch.randn(num_kv_heads * head_dim, hidden)
    full_o_weight = torch.randn(hidden, num_q_heads * head_dim)
    x = torch.randn(batch_size, seqlen, hidden)
    o_input_full = torch.randn(batch_size, seqlen, num_q_heads * head_dim)

    # Reference outputs computed directly with `nn.functional.linear`.
    expected_q = nn.functional.linear(x, full_q_weight)
    expected_k = nn.functional.linear(x, full_k_weight)
    expected_v = nn.functional.linear(x, full_v_weight)
    expected_o = nn.functional.linear(o_input_full, full_o_weight)

    per_rank_outputs = run_multi_process(
        2,
        _gqa_projection_worker,
        hidden,
        num_q_heads,
        num_kv_heads,
        head_dim,
        full_q_weight,
        full_k_weight,
        full_v_weight,
        full_o_weight,
        x,
        o_input_full,
    )
    assert len(per_rank_outputs) == 2

    # Column-parallel: per-rank slices concatenate (along last dim) to
    # the reference. Row-parallel: every rank holds the full output.
    concat_q = torch.cat([out["q"] for out in per_rank_outputs], dim=-1)
    concat_k = torch.cat([out["k"] for out in per_rank_outputs], dim=-1)
    concat_v = torch.cat([out["v"] for out in per_rank_outputs], dim=-1)
    torch.testing.assert_close(concat_q, expected_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(concat_k, expected_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(concat_v, expected_v, rtol=1e-5, atol=1e-5)
    for rank, out in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            out["o"],
            expected_o,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} o_proj mismatch: {m}",
        )


# ----------------------------- MLA -----------------------------


def test_mla_block_world_size_1_construction() -> None:
    """At world_size=1 MLA's `q_b_proj`, `kv_b_proj`, `o_proj` are TP-aware
    but degrade to plain `nn.Linear` semantics. The `q_a_proj` and
    `kv_a_proj_with_mqa` stay as plain `nn.Linear` — they're replicated
    under TP."""
    from mini_infer.models.blocks.mla import MLAAttention

    block = MLAAttention(
        hidden_size=32,
        num_heads=4,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=12,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_idx=0,
    )
    assert isinstance(block.q_b_proj, ColumnParallelLinear)
    assert isinstance(block.kv_b_proj, ColumnParallelLinear)
    assert isinstance(block.o_proj, RowParallelLinear)
    # Replicated inputs — plain nn.Linear, not TP-aware.
    assert isinstance(block.q_a_proj, nn.Linear)
    assert not isinstance(block.q_a_proj, ColumnParallelLinear)
    assert isinstance(block.kv_a_proj_with_mqa, nn.Linear)
    assert not isinstance(block.kv_a_proj_with_mqa, ColumnParallelLinear)


def _mla_projection_worker(
    rank: int,
    world_size: int,
    hidden: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int,
    full_q_a_weight: torch.Tensor,
    full_q_b_weight: torch.Tensor,
    full_kv_a_weight: torch.Tensor,
    full_kv_b_weight: torch.Tensor,
    full_o_weight: torch.Tensor,
    x: torch.Tensor,
    q_latent_for_b: torch.Tensor,
    kv_latent_for_b: torch.Tensor,
    o_input_full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run MLA's projections at this rank.

    Each projection is fed a *fixed* input rather than chained through
    the upstream layernorms. The layernorms have learnable weights that
    are randomly init'd per process, so a chained test would compare
    different random parameters. Decoupling input from upstream means
    we test only what TP changes — the linear math — and the result is
    bit-stable across processes.
    """
    from mini_infer.models.blocks.mla import MLAAttention

    block = MLAAttention(
        hidden_size=hidden,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_idx=0,
    )
    with torch.no_grad():
        block.q_a_proj.weight.copy_(full_q_a_weight)
        block.kv_a_proj_with_mqa.weight.copy_(full_kv_a_weight)
    block.q_b_proj.load_full_weight(full_q_b_weight)
    block.kv_b_proj.load_full_weight(full_kv_b_weight)
    block.o_proj.load_full_weight(full_o_weight)

    o_per_rank = (num_heads * v_head_dim) // world_size
    o_start = rank * o_per_rank
    o_input_local = o_input_full[..., o_start : o_start + o_per_rank].contiguous()

    return {
        "q_a": block.q_a_proj(x).detach().cpu(),  # replicated; same on every rank
        "q_b": block.q_b_proj(q_latent_for_b).detach().cpu(),  # col-par
        "kv_a": block.kv_a_proj_with_mqa(x).detach().cpu(),  # replicated
        "kv_b": block.kv_b_proj(kv_latent_for_b).detach().cpu(),  # col-par
        "o": block.o_proj(o_input_local).detach().cpu(),  # row-par
    }


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_mla_projection_world_size_2_matches_reference() -> None:
    """MLA: replicated layers produce identical output on every rank;
    column-parallel layers produce slices that concatenate to the
    reference; row-parallel `o_proj` produces the all-reduced full output."""
    torch.manual_seed(0)
    hidden = 32
    num_heads = 4  # 4 / 2 = 2 heads per rank
    kv_lora_rank = 16
    qk_nope_head_dim = 8
    qk_rope_head_dim = 4
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 8
    q_lora_rank = 12

    full_q_a_weight = torch.randn(q_lora_rank, hidden)
    full_q_b_weight = torch.randn(num_heads * qk_head_dim, q_lora_rank)
    full_kv_a_weight = torch.randn(kv_lora_rank + qk_rope_head_dim, hidden)
    full_kv_b_weight = torch.randn(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    full_o_weight = torch.randn(hidden, num_heads * v_head_dim)
    x = torch.randn(1, 4, hidden)
    q_latent_for_b = torch.randn(1, 4, q_lora_rank)
    kv_latent_for_b = torch.randn(1, 4, kv_lora_rank)
    o_input_full = torch.randn(1, 4, num_heads * v_head_dim)

    expected_q_a = nn.functional.linear(x, full_q_a_weight)
    expected_q_b = nn.functional.linear(q_latent_for_b, full_q_b_weight)
    expected_kv_a = nn.functional.linear(x, full_kv_a_weight)
    expected_kv_b = nn.functional.linear(kv_latent_for_b, full_kv_b_weight)
    expected_o = nn.functional.linear(o_input_full, full_o_weight)

    per_rank_outputs = run_multi_process(
        2,
        _mla_projection_worker,
        hidden,
        num_heads,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        full_q_a_weight,
        full_q_b_weight,
        full_kv_a_weight,
        full_kv_b_weight,
        full_o_weight,
        x,
        q_latent_for_b,
        kv_latent_for_b,
        o_input_full,
    )
    assert len(per_rank_outputs) == 2

    for rank, out in enumerate(per_rank_outputs):
        # Replicated layers: every rank gets the same output.
        torch.testing.assert_close(
            out["q_a"],
            expected_q_a,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} q_a (replicated) mismatch: {m}",
        )
        torch.testing.assert_close(
            out["kv_a"],
            expected_kv_a,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} kv_a (replicated) mismatch: {m}",
        )
        # Row-parallel: every rank gets the all-reduced full output.
        torch.testing.assert_close(
            out["o"],
            expected_o,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, r=rank: f"rank {r} o_proj (row-par) mismatch: {m}",
        )

    # Column-parallel: per-rank slices concatenate to the reference.
    concat_q_b = torch.cat([out["q_b"] for out in per_rank_outputs], dim=-1)
    concat_kv_b = torch.cat([out["kv_b"] for out in per_rank_outputs], dim=-1)
    torch.testing.assert_close(concat_q_b, expected_q_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(concat_kv_b, expected_kv_b, rtol=1e-5, atol=1e-5)


# ----------------------------- HCA / CSA -----------------------------


def test_hca_block_world_size_1_construction() -> None:
    """At world_size=1 HCA's `q_b_proj` is TP-aware and `q_a_proj` /
    `swa_kv_proj` / `compressor` stay replicated. `sink` is per-head sliced
    (no-op at ws=1) and `grouped_output.wo_b` is row-parallel."""
    from mini_infer.models.blocks.hca import HCAAttention
    from mini_infer.models.blocks.v4 import AttentionSink, GroupedOutputProjection

    block = HCAAttention(
        hidden_size=32,
        num_heads=4,
        q_lora_rank=16,
        kv_head_dim=8,
        rope_head_dim=4,
        num_groups=2,
        o_lora_rank=16,
        window_size=8,
        compression_ratio=4,
    )
    assert isinstance(block.q_b_proj, ColumnParallelLinear)
    assert isinstance(block.swa_kv_proj, nn.Linear)
    assert not isinstance(block.swa_kv_proj, ColumnParallelLinear)
    assert isinstance(block.sink, AttentionSink)
    assert isinstance(block.grouped_output, GroupedOutputProjection)
    assert isinstance(block.grouped_output.wo_b, RowParallelLinear)
    # At ws=1, num_heads_local == num_heads.
    assert block.num_heads_local == 4


def test_csa_block_world_size_1_construction() -> None:
    """Same checks for CSA, plus the indexer's TP-aware sub-modules."""
    from mini_infer.models.blocks.csa import CSAAttention
    from mini_infer.models.blocks.v4 import LightningIndexer

    block = CSAAttention(
        hidden_size=32,
        num_heads=4,
        q_lora_rank=16,
        kv_head_dim=8,
        rope_head_dim=4,
        num_groups=2,
        o_lora_rank=16,
        window_size=8,
        compression_ratio=4,
        index_num_heads=2,
        index_head_dim=8,
        index_top_k=2,
    )
    assert isinstance(block.q_b_proj, ColumnParallelLinear)
    assert isinstance(block.indexer, LightningIndexer)
    assert isinstance(block.indexer.wq_b, ColumnParallelLinear)
    assert isinstance(block.indexer.weights_proj, ColumnParallelLinear)
    assert isinstance(block.grouped_output.wo_b, RowParallelLinear)
    # At ws=1 the indexer's local head count equals the global one.
    assert block.indexer.num_heads_local == 2
    assert block.num_heads_local == 4


def _hca_projection_worker(
    rank: int,
    world_size: int,
    hidden: int,
    num_heads: int,
    q_lora_rank: int,
    kv_head_dim: int,
    rope_head_dim: int,
    num_groups: int,
    o_lora_rank: int,
    window_size: int,
    compression_ratio: int,
    full_q_b_weight: torch.Tensor,
    full_sink_logits: torch.Tensor,
    full_wo_a: torch.Tensor,
    full_wo_b_weight: torch.Tensor,
    q_lora_latent_for_b: torch.Tensor,
    attn_out_full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run HCA's TP-relevant pieces and return outputs per rank.

    Tests:
      - `q_b_proj` (column-parallel by head): per-rank slices concat.
      - `sink` (per-head logits, sharded): per-rank logits concat.
      - `grouped_output` (sharded by group, row-parallel `wo_b`): every
        rank sees the all-reduced hidden state.
    """
    from mini_infer.models.blocks.hca import HCAAttention

    block = HCAAttention(
        hidden_size=hidden,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_head_dim=kv_head_dim,
        rope_head_dim=rope_head_dim,
        num_groups=num_groups,
        o_lora_rank=o_lora_rank,
        window_size=window_size,
        compression_ratio=compression_ratio,
    )
    block.q_b_proj.load_full_weight(full_q_b_weight)
    block.sink.load_full_logits(full_sink_logits)
    block.grouped_output.load_full_wo_a(full_wo_a)
    block.grouped_output.wo_b.load_full_weight(full_wo_b_weight)

    # Slice attn_out along the head axis so each rank receives the
    # contiguous block of `num_heads_local` heads it owns.
    num_heads_local = num_heads // world_size
    head_start = rank * num_heads_local
    attn_out_local = attn_out_full[:, :, head_start : head_start + num_heads_local, :].contiguous()

    return {
        "q_b": block.q_b_proj(q_lora_latent_for_b).detach().cpu(),  # col-par
        "sink_logits": block.sink.sink_logits.detach().cpu(),  # sharded by head
        "grouped_out": block.grouped_output(attn_out_local).detach().cpu(),
    }


@pytest.mark.skipif(
    not is_multi_process_available(),
    reason="multi-process gloo not available in this environment",
)
def test_hca_projection_world_size_2_matches_reference() -> None:
    """HCA: q_b_proj slices concatenate; sink logits concatenate; grouped
    output is the all-reduced full hidden state on every rank."""
    torch.manual_seed(0)
    hidden = 32
    num_heads = 4
    q_lora_rank = 16
    kv_head_dim = 8
    rope_head_dim = 4
    num_groups = 2  # 2 groups / 2 ranks = 1 group per rank
    o_lora_rank = 16
    window_size = 8
    compression_ratio = 4

    full_q_b_weight = torch.randn(num_heads * kv_head_dim, q_lora_rank)
    full_sink_logits = torch.randn(num_heads, dtype=torch.float32)
    full_wo_a = torch.randn(num_groups * o_lora_rank, (num_heads // num_groups) * kv_head_dim)
    full_wo_b_weight = torch.randn(hidden, num_groups * o_lora_rank)
    q_lora_latent_for_b = torch.randn(1, 4, q_lora_rank)
    attn_out_full = torch.randn(1, 4, num_heads, kv_head_dim)

    expected_q_b = nn.functional.linear(q_lora_latent_for_b, full_q_b_weight)

    # Reference grouped-output via the un-sharded formula.
    heads_per_group = num_heads // num_groups
    per_group_in = attn_out_full.view(1, 4, num_groups, heads_per_group * kv_head_dim)
    wo_a_grouped = full_wo_a.view(num_groups, o_lora_rank, -1)
    grouped_partial = torch.einsum("bsgd,grd->bsgr", per_group_in, wo_a_grouped).flatten(2)
    expected_grouped_out = nn.functional.linear(grouped_partial, full_wo_b_weight)

    per_rank_outputs = run_multi_process(
        2,
        _hca_projection_worker,
        hidden,
        num_heads,
        q_lora_rank,
        kv_head_dim,
        rope_head_dim,
        num_groups,
        o_lora_rank,
        window_size,
        compression_ratio,
        full_q_b_weight,
        full_sink_logits,
        full_wo_a,
        full_wo_b_weight,
        q_lora_latent_for_b,
        attn_out_full,
    )
    assert len(per_rank_outputs) == 2

    # Column-parallel: q_b slices concatenate; sink_logits concatenate.
    concat_q_b = torch.cat([out["q_b"] for out in per_rank_outputs], dim=-1)
    concat_sink = torch.cat([out["sink_logits"] for out in per_rank_outputs], dim=0)
    torch.testing.assert_close(concat_q_b, expected_q_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(concat_sink, full_sink_logits, rtol=0, atol=0)

    # grouped_output: every rank sees the all-reduced full hidden state.
    for rank, out in enumerate(per_rank_outputs):
        torch.testing.assert_close(
            out["grouped_out"],
            expected_grouped_out,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda m, r=rank: f"rank {r} grouped_output mismatch: {m}",
        )
