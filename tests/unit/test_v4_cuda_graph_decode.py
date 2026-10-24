"""CUDA-graph decode for DeepSeek-V4 (`decode_step_static` under `DecodeCudaGraph`).

The CPU tests pin the contract: the ragged step is the static step plus host
checks, the step writes compressed blocks only for the rows that flush, the
cache reports its capacity, and capture support follows the MoE path. The
token streams of the static step against the per-request scalar decode are
pinned in `test_v4_ragged_decode.py`. The CUDA tests run the static step with
syncs turned into errors, and serve the same prompts with and without the
graph, which must give identical tokens. They skip without CUDA.
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from mini_infer.cache.state_cache import StateCache, StateLayerSpec
from mini_infer.engine.cuda_graph_decode import supports_decode_cuda_graph
from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.blocks import hash_routed_moe_ffn
from mini_infer.models.blocks.fp4_expert import FP4Expert
from mini_infer.models.blocks.hyper_connections import HCHeadReduction, HyperConnections
from mini_infer.models.blocks.v4.compressor import write_flushed_blocks
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from mini_infer.scheduler import Request, StateCacheContinuousScheduler
from tests.unit.test_fp4_grouped import fill_fp4

VOCAB = 128
PROMPTS = ["short", "a considerably longer prompt than the other one", "middling prompt", "x"]
MoE = Literal["none", "bf16", "fp4"]


def _model(
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    *,
    moe: MoE = "none",
    simulate_fp8_gemm_inputs: bool = False,
) -> DeepseekV4ForCausalLM:
    """SWA, CSA, HCA and CSA layers with Hyper-Connections, and MoE FFNs if asked."""
    use_moe = moe != "none"
    cfg = DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=128 if simulate_fp8_gemm_inputs else 64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=128 if simulate_fp8_gemm_inputs else 32,
        kv_head_dim=72 if simulate_fp8_gemm_inputs else 32,
        rope_head_dim=8,
        o_num_groups=2,
        o_lora_rank=64 if simulate_fp8_gemm_inputs else 32,
        window_size=8,
        compress_ratios=(0, 4, 8, 4),
        index_num_heads=2,
        index_head_dim=32 if simulate_fp8_gemm_inputs else 16,
        index_top_k=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_moe_ffn=use_moe,
        moe_intermediate_size=(128 if simulate_fp8_gemm_inputs else 32) if use_moe else 0,
        num_routed_experts=4 if use_moe else 0,
        num_activated_experts=2 if use_moe else 0,
        num_hash_routed_layers=2 if use_moe else 0,
        n_shared_experts=1 if use_moe else 0,
        expert_dtype="fp4" if moe == "fp4" else "bf16",
        moe_swiglu_limit=10.0 if use_moe else 0.0,
        use_hyper_connections=True,
        hc_mult=4,
        simulate_act_quant=simulate_fp8_gemm_inputs,
        simulate_fp8_gemm_inputs=simulate_fp8_gemm_inputs,
    )
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg).eval()
    generator = torch.Generator().manual_seed(1)
    for module in model.modules():
        if isinstance(module, FP4Expert):
            fill_fp4(module, generator)
    model = model.to(device=device, dtype=dtype)
    # `from_checkpoint` keeps these in fp32: the Hyper-Connections mix in fp32,
    # and the FP4 scales are fp32 in the checkpoint.
    for module in model.modules():
        if isinstance(module, HyperConnections | HCHeadReduction | FP4Expert):
            module.float()
    return model


class _FakeTokenizer:
    """Deterministic stand-in: prompt -> ids derived from the text; ids -> text."""

    def encode(self, text: str) -> list[int]:
        return [(ord(c) * 7 + i) % VOCAB for i, c in enumerate(text)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return None


def _serve(model: DeepseekV4ForCausalLM, *, cuda_graph: bool) -> list[list[int]]:
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


def _prefilled_cache(
    model: DeepseekV4ForCausalLM, prompts: list[torch.Tensor], device: str = "cpu"
) -> StateCache:
    """A batched cache whose row `b` holds the prefilled state of `prompts[b]`."""
    batched = model.build_state_cache(max_seq_len=64, batch_size=len(prompts), device=device)
    for row, prompt in enumerate(prompts):
        single = model.build_state_cache(max_seq_len=64, batch_size=1, device=device)
        with torch.inference_mode():
            model.forward_prefill_with_cache(prompt.to(device), state_cache=single)
        batched.copy_row_from(single, src_row=0, dst_row=row)
    return batched


def test_support_needs_cuda_and_a_capturable_moe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    dense = _model()
    assert not supports_decode_cuda_graph(dense, "cpu")
    assert dense.can_capture_decode("cpu")
    # On the CPU the MoE blocks keep the per-expert loop, which syncs.
    assert not _model(moe="bf16").can_capture_decode("cpu")
    monkeypatch.setattr(hash_routed_moe_ffn, "supports_grouped_experts", lambda device: True)
    assert _model(moe="bf16").can_capture_decode("cpu")
    # The FP8 GEMM-input round trip keeps the loop on every device.
    assert not _model(moe="bf16", simulate_fp8_gemm_inputs=True).can_capture_decode("cpu")


def test_ragged_step_is_the_static_step_after_checks() -> None:
    model = _model()
    prompts = [torch.randint(0, VOCAB, (1, 11)), torch.randint(0, VOCAB, (1, 4))]
    checked_cache = _prefilled_cache(model, prompts)
    static_cache = _prefilled_cache(model, prompts)
    tokens = torch.tensor([[3], [5]])
    positions = torch.tensor([11, 4])  # row 0 closes a CSA block, row 1 does not
    with torch.inference_mode():
        checked = model.forward_decode_with_cache_ragged(
            tokens, positions=positions, state_cache=checked_cache
        )
        static = model.decode_step_static(tokens, positions, static_cache)
    assert torch.equal(checked, static)
    for layer_idx in range(model.cfg.num_hidden_layers):
        assert torch.equal(
            checked_cache.layer(layer_idx).compressed_kv,
            static_cache.layer(layer_idx).compressed_kv,
        )


def test_ragged_step_rejects_a_position_past_the_cache() -> None:
    model = _model()
    cache = model.build_state_cache(max_seq_len=16, batch_size=2)
    assert cache.max_seq_len == 16
    with torch.inference_mode(), pytest.raises(ValueError, match="max_seq_len 16"):
        model.forward_decode_with_cache_ragged(
            torch.tensor([[1], [2]]), positions=torch.tensor([3, 16]), state_cache=cache
        )


def test_cache_capacity_is_its_shortest_compressed_history() -> None:
    # Ratios 4 and 8 round 30 tokens up to 8 and 4 blocks: 32 tokens each.
    assert _model().build_state_cache(max_seq_len=30).max_seq_len == 32
    window_only = StateCache(
        [StateLayerSpec(kv_head_dim=8, compression_ratio=0, n_win=4, max_n_compressed=1)]
    )
    assert window_only.max_seq_len is None
    window_only.check_capacity(10**6)


def test_flushed_blocks_land_only_in_flushing_rows() -> None:
    history = torch.arange(3 * 4 * 2, dtype=torch.float32).view(3, 4, 2)
    before = history.clone()
    compressed = torch.full((3, 1, 2), -1.0)
    flush = torch.tensor([True, False, True])
    # Row 2 sits past the history; its slot clamps to the last one.
    write_flushed_blocks(history, compressed, flush, torch.tensor([7, 5, 23]), 4)
    expected = before.clone()
    expected[0, 1] = -1.0
    expected[2, 3] = -1.0
    assert torch.equal(history, expected)


def _capturable_on_this_gpu(moe: MoE) -> DeepseekV4ForCausalLM:
    model = _model("cuda", torch.bfloat16, moe=moe)
    if not supports_decode_cuda_graph(model, "cuda"):
        pytest.skip(f"MoE experts ({moe}) keep the per-expert loop on this GPU")
    return model


@pytest.mark.requires_cuda
@pytest.mark.parametrize("moe", ["none", "bf16", "fp4"])
def test_static_step_has_no_host_sync(moe: MoE) -> None:
    """Every row and flush pattern: CSA blocks close at 3, 7, 11; HCA ones at 7."""
    model = _capturable_on_this_gpu(moe)
    prompts = [torch.randint(0, VOCAB, (1, 2)), torch.randint(0, VOCAB, (1, 5))]
    cache = _prefilled_cache(model, prompts, device="cuda")
    tokens = torch.tensor([[3], [5]], device="cuda")
    positions = torch.tensor([2, 5], device="cuda")
    with torch.inference_mode():
        model.decode_step_static(tokens, positions, cache)  # lazy setup may sync
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(8):
                positions = positions + 1
                logits = model.decode_step_static(tokens, positions, cache)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(logits).all()


@pytest.mark.requires_cuda
@pytest.mark.parametrize("moe", ["none", "bf16", "fp4"])
def test_graph_replay_serves_the_same_tokens_as_eager(moe: MoE) -> None:
    """Admissions between replays must reach the graph through the shared cache."""
    model = _capturable_on_this_gpu(moe)
    eager = _serve(model, cuda_graph=False)
    graphed = _serve(model, cuda_graph=True)
    assert graphed == eager
    assert all(len(tokens) == 12 for tokens in graphed)
