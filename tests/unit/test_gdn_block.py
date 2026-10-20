"""Gated DeltaNet block math, pinned against the transformers reference.

Two things are established here. First, that `blocks/gdn.py`'s small helpers
(the per-head decay gate and the SiLU-gated output norm) reproduce their
reference counterparts exactly, including the dtype ordering that only shows
up in bf16. Second, and the load-bearing claim of the whole port, that the
KDA delta rule in `blocks/kda.py` computes the Gated DeltaNet recurrence when
its per-channel decay is fed a broadcast per-head scalar. If that second
claim were false this family would need its own copy of the recurrence.

The reference functions are imported from the pinned transformers install
rather than vendored, because the checkpoint's own architecture
(`Qwen3_5ForConditionalGeneration`) is implemented there.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from mini_infer.models.blocks.gdn import broadcast_gate, gated_rmsnorm_silu, gdn_gate
from mini_infer.models.blocks.kda import (
    causal_conv1d_prefill,
    causal_conv1d_step,
    kda_chunkwise,
    kda_recurrent,
    l2norm,
)

transformers_qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

hf_chunk = transformers_qwen.torch_chunk_gated_delta_rule
hf_recurrent = transformers_qwen.torch_recurrent_gated_delta_rule
HFGatedNorm = transformers_qwen.Qwen3_5RMSNormGated


def _inputs(batch, seq_len, heads, dim_k, dim_v, *, seed=0):
    """Random delta-rule inputs with a per-HEAD scalar decay, as GDN produces."""
    torch.manual_seed(seed)
    return (
        torch.randn(batch, seq_len, heads, dim_k),
        torch.randn(batch, seq_len, heads, dim_k),
        torch.randn(batch, seq_len, heads, dim_v),
        -torch.rand(batch, seq_len, heads) * 0.5,  # log-decay, <= 0
        torch.rand(batch, seq_len, heads),
    )


def _ours(q, k, v, gate, beta, dim_k, initial_state=None, *, chunked):
    """Run our shared recurrence the way the GDN layer calls it."""
    q_scaled = l2norm(q) * (dim_k**-0.5)
    gate_channels = broadcast_gate(gate, dim_k)
    runner = kda_chunkwise if chunked else kda_recurrent
    return runner(q_scaled, l2norm(k), v, gate_channels, beta, initial_state=initial_state)


# Head counts are deliberately uneven: the 27B broadcasts 16 key heads up to
# 48 value heads, so the kernel always sees a value-head count here.
@pytest.mark.parametrize(
    ("batch", "seq_len", "heads", "dim_k", "dim_v"),
    [
        (2, 16, 6, 16, 16),  # square, 3x-ratio head count
        (2, 16, 6, 16, 24),  # non-square head dims
        (2, 80, 6, 16, 16),  # crosses the 64-token chunk boundary
        (2, 37, 6, 16, 16),  # not a chunk multiple, exercises padding
        (1, 130, 4, 32, 32),  # multiple chunks
    ],
)
def test_broadcast_gate_reproduces_reference_delta_rule(batch, seq_len, heads, dim_k, dim_v):
    q, k, v, gate, beta = _inputs(batch, seq_len, heads, dim_k, dim_v)
    for chunked, reference in ((False, hf_recurrent), (True, hf_chunk)):
        ref_out, ref_state = reference(
            q.clone(),
            k.clone(),
            v.clone(),
            gate.clone(),
            beta.clone(),
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out, state = _ours(q, k, v, gate, beta, dim_k, chunked=chunked)
        assert torch.allclose(out, ref_out, atol=1e-5), f"chunked={chunked}"
        assert torch.allclose(state, ref_state, atol=1e-5), f"chunked={chunked}"


def test_broadcast_gate_matches_reference_with_initial_state():
    batch, seq_len, heads, dim_k, dim_v = 2, 24, 6, 16, 16
    q, k, v, gate, beta = _inputs(batch, seq_len, heads, dim_k, dim_v, seed=3)
    initial = torch.randn(batch, heads, dim_k, dim_v) * 0.1
    for chunked, reference in ((False, hf_recurrent), (True, hf_chunk)):
        ref_out, ref_state = reference(
            q.clone(),
            k.clone(),
            v.clone(),
            gate.clone(),
            beta.clone(),
            initial_state=initial.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out, state = _ours(q, k, v, gate, beta, dim_k, initial.clone(), chunked=chunked)
        assert torch.allclose(out, ref_out, atol=1e-5), f"chunked={chunked}"
        assert torch.allclose(state, ref_state, atol=1e-5), f"chunked={chunked}"


def test_chunkwise_state_feeds_recurrent_decode():
    """Prefill with the chunked form, then continue one token at a time."""
    batch, heads, dim_k, dim_v = 1, 6, 16, 16
    q, k, v, gate, beta = _inputs(batch, 20, heads, dim_k, dim_v, seed=5)
    _, prefill_state = _ours(
        q[:, :16], k[:, :16], v[:, :16], gate[:, :16], beta[:, :16], dim_k, chunked=True
    )
    state = prefill_state
    outs = []
    for t in range(16, 20):
        out, state = _ours(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            gate[:, t : t + 1],
            beta[:, t : t + 1],
            dim_k,
            state,
            chunked=False,
        )
        outs.append(out)
    split = torch.cat(outs, dim=1)
    whole, _ = _ours(q, k, v, gate, beta, dim_k, chunked=True)
    assert torch.allclose(split, whole[:, 16:], atol=1e-5)


def test_fused_conv_prefill_matches_sequential_steps():
    """The fused q/k/v convolution: a span equals the same tokens stepped."""
    batch, channels, kernel = 2, 12, 4
    torch.manual_seed(7)
    x = torch.randn(batch, 9, channels)
    weight = torch.randn(channels, kernel) * 0.3
    span, span_state = causal_conv1d_prefill(x, weight)
    state = torch.zeros(batch, channels, kernel)
    steps = []
    for t in range(x.shape[1]):
        y, state = causal_conv1d_step(x[:, t], weight, state)
        steps.append(y)
    assert torch.allclose(span, torch.stack(steps, dim=1), atol=1e-5)
    assert torch.allclose(span_state, state, atol=1e-5)


def test_fused_conv_first_chunk_shorter_than_kernel():
    """A continuation whose first chunk is shorter than the kernel still lines up."""
    batch, channels, kernel = 1, 8, 4
    torch.manual_seed(11)
    x = torch.randn(batch, 10, channels)
    weight = torch.randn(channels, kernel) * 0.3
    whole, _ = causal_conv1d_prefill(x, weight)
    head, state = causal_conv1d_prefill(x[:, :2], weight)  # shorter than kernel
    tail, _ = causal_conv1d_prefill(x[:, 2:], weight, state)
    assert torch.allclose(torch.cat([head, tail], dim=1), whole, atol=1e-5)


def test_gdn_gate_matches_reference_expression():
    torch.manual_seed(13)
    batch, seq_len, heads = 2, 9, 48
    a_raw = torch.randn(batch, seq_len, heads)
    a_log = torch.rand(heads) * 2
    dt_bias = torch.randn(heads)
    expected = -a_log.float().exp() * functional.softplus(a_raw.float() + dt_bias)
    gate = gdn_gate(a_raw, a_log, dt_bias)
    assert torch.allclose(gate, expected, atol=1e-6)
    # exp(g) has to be a decay, never a growth factor
    assert bool((gate <= 0).all())


def test_broadcast_gate_is_constant_across_key_channels():
    torch.manual_seed(17)
    gate = -torch.rand(2, 5, 6)
    wide = broadcast_gate(gate, 16)
    assert wide.shape == (2, 5, 6, 16)
    assert bool((wide[..., 0:1] == wide).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_output_norm_matches_reference(dtype):
    """Pins the SiLU gate and the affine-in-input-dtype ordering.

    The ordering is invisible in fp32 and observable in bf16, which is why
    both run.
    """
    torch.manual_seed(19)
    width = 128
    reference = HFGatedNorm(width, eps=1e-6).to(dtype)
    with torch.no_grad():
        reference.weight.uniform_(0.5, 1.5)
    x = torch.randn(64, width, dtype=dtype)
    gate = torch.randn(64, width, dtype=dtype)
    with torch.no_grad():
        expected = reference(x, gate)
    got = gated_rmsnorm_silu(x, gate, reference.weight, 1e-6)
    assert got.dtype == dtype
    assert torch.equal(got, expected)
