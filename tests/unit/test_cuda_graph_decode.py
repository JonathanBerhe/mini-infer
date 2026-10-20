"""CUDA-graph decode for state-cache models (`DecodeCudaGraph`).

The CPU tests pin the contract: which model and device may capture, and that
the capturable step equals the checked ragged step. The CUDA tests serve the
same prompts with and without the graph and require identical tokens, since
a replay runs the same kernels as the eager step. They skip without CUDA and
run in `scripts/modal_qwen38_bench.py`.
"""

from __future__ import annotations

import pytest
import torch

from mini_infer.engine.cuda_graph_decode import supports_decode_cuda_graph
from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.qwen3_5 import Qwen35Config, Qwen35ForConditionalGeneration
from mini_infer.scheduler import Request, StateCacheContinuousScheduler

VOCAB = 128
PROMPTS = ["short", "a considerably longer prompt than the other one", "middling prompt", "x"]


def _model(
    device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Qwen35ForConditionalGeneration:
    """A tiny 3:1 hybrid with the 27B's 3x Gated DeltaNet head ratio."""
    cfg = Qwen35Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = Qwen35ForConditionalGeneration(cfg).eval()
    # Zero-init norms and gates make a degenerate model; spread the weights
    # so that wrong decode state changes the greedy tokens.
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.3, 0.3)
    return model.to(device=device, dtype=dtype)


class _FakeTokenizer:
    """Deterministic stand-in: prompt -> ids derived from the text; ids -> text."""

    def encode(self, text: str) -> list[int]:
        return [(ord(c) * 7 + i) % VOCAB for i, c in enumerate(text)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return None


def _serve(model: Qwen35ForConditionalGeneration, *, cuda_graph: bool) -> list[list[int]]:
    scheduler = StateCacheContinuousScheduler(
        StateCacheGenerator(model, _FakeTokenizer()),  # type: ignore[arg-type]
        max_batch_size=2,
        max_seq_len=96,
        cuda_graph=cuda_graph,
    )
    scheduler.start()
    try:
        handles = [
            scheduler.submit(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=12))
            for p in PROMPTS
        ]
        return [handle.wait(timeout=120).tokens for handle in handles]
    finally:
        scheduler.stop()


def test_support_needs_cuda_and_the_model_opt_in() -> None:
    model = _model()
    assert supports_decode_cuda_graph(model, "cuda")
    assert supports_decode_cuda_graph(model, torch.device("cuda", 0))
    assert not supports_decode_cuda_graph(model, "cpu")
    assert not supports_decode_cuda_graph(model, "mps")
    assert not supports_decode_cuda_graph(object(), "cuda")


def test_scheduler_rejects_cuda_graph_where_unsupported() -> None:
    generator = StateCacheGenerator(_model(), _FakeTokenizer())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cuda_graph=True"):
        StateCacheContinuousScheduler(generator, cuda_graph=True)


def test_static_decode_step_matches_checked_ragged_step() -> None:
    """The capturable step is the ragged step minus its host-side checks."""
    model = _model()
    prompts = [torch.randint(0, VOCAB, (1, 9)), torch.randint(0, VOCAB, (1, 4))]
    caches = []
    for _ in range(2):
        batched = model.build_state_cache(max_seq_len=64, batch_size=2)
        for row, prompt in enumerate(prompts):
            single = model.build_state_cache(max_seq_len=64, batch_size=1)
            with torch.inference_mode():
                model.forward_prefill_with_cache(prompt, state_cache=single)
            batched.copy_row_from(single, src_row=0, dst_row=row)
        caches.append(batched)
    tokens = torch.tensor([[3], [5]])
    positions = torch.tensor([9, 4])
    with torch.inference_mode():
        checked = model.forward_decode_with_cache_ragged(
            tokens, positions=positions, state_cache=caches[0]
        )
        static = model.decode_step_static(tokens, positions, caches[1])
    assert torch.equal(checked, static)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graph_replay_serves_the_same_tokens_as_eager(dtype: torch.dtype) -> None:
    """Admissions between replays must reach the graph through the shared cache."""
    model = _model("cuda", dtype)
    eager = _serve(model, cuda_graph=False)
    graphed = _serve(model, cuda_graph=True)
    assert graphed == eager
    assert all(len(tokens) == 12 for tokens in graphed)
