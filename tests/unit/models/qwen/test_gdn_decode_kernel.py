"""Fused Gated DeltaNet decode step (`qwen/gdn_decode_kernel.py`) vs the eager layer.

The reference is `_QwenGatedDeltaNet.forward` on its eager path; the
candidate is `gdn_decode_step` followed by the same `out_proj`. Both start
from the same prefilled state and run several decode steps, so the conv tail
and the recurrent state updates are checked as well as the output.

Runs on CUDA, or on the CPU under Triton's interpreter
(`TRITON_INTERPRET=1`), which is how the kernels are debugged without a GPU.
Skips otherwise.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("triton")

from mini_infer.models.qwen import gdn_decode_kernel
from mini_infer.models.qwen.qwen3_5 import Qwen35Config, _QwenGatedDeltaNet
from mini_infer.models.qwen.qwen3_5_state_cache import QwenGdnLayerState


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("needs CUDA, or TRITON_INTERPRET=1 on the CPU")


def _config(head_dim: int) -> Qwen35Config:
    """One linear layer's worth of config with the 27B's 3x head ratio."""
    return Qwen35Config(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
        max_position_embeddings=64,
    )


def _layer(cfg: Qwen35Config, device: str, dtype: torch.dtype) -> _QwenGatedDeltaNet:
    torch.manual_seed(0)
    layer = _QwenGatedDeltaNet(cfg)
    with torch.no_grad():
        for param in layer.parameters():
            param.uniform_(-0.5, 0.5)
        # Gate inputs past torch's softplus threshold and far below zero.
        layer.dt_bias[0] = 25.0
        layer.dt_bias[1] = -30.0
    return layer.to(device=device, dtype=dtype).eval()


def _prefilled_state(
    layer: _QwenGatedDeltaNet, batch: int, device: str, dtype: torch.dtype
) -> QwenGdnLayerState:
    cfg = layer.cfg
    state = QwenGdnLayerState(
        recurrent_state=torch.zeros(
            batch,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            device=device,
        ),
        conv=torch.zeros(
            batch, cfg.linear_conv_channels, cfg.linear_conv_kernel_dim, device=device, dtype=dtype
        ),
    )
    prompt = torch.randn(batch, 5, cfg.hidden_size, device=device).to(dtype)
    with torch.inference_mode():
        layer(prompt, state, torch.zeros(batch, dtype=torch.long, device=device), None)
    return state


def _clone(state: QwenGdnLayerState) -> QwenGdnLayerState:
    return QwenGdnLayerState(recurrent_state=state.recurrent_state.clone(), conv=state.conv.clone())


def _fused_step(
    layer: _QwenGatedDeltaNet, hidden: torch.Tensor, state: QwenGdnLayerState
) -> torch.Tensor:
    cfg = layer.cfg
    gated = gdn_decode_kernel.gdn_decode_step(
        layer.in_proj_qkv(hidden)[:, 0],
        state.conv,
        layer.conv1d.weight,
        layer.in_proj_a(hidden)[:, 0],
        layer.in_proj_b(hidden)[:, 0],
        layer.A_log,
        layer.dt_bias,
        state.recurrent_state,
        layer.in_proj_z(hidden)[:, 0],
        layer.norm.weight,
        num_k_heads=cfg.linear_num_key_heads,
        head_k_dim=cfg.linear_key_head_dim,
        head_v_dim=cfg.linear_value_head_dim,
        rms_eps=cfg.rms_norm_eps,
    )
    out: torch.Tensor = layer.out_proj(gated.view(hidden.shape[0], 1, -1))
    return out


@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_step_matches_eager_layer(
    head_dim: int, batch: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _device()
    # The eager reference must not route through the kernel on CUDA.
    monkeypatch.setattr(gdn_decode_kernel, "_KERNEL_DISABLED_FOR_BENCH", True)
    layer = _layer(_config(head_dim), device, dtype)
    eager_state = _prefilled_state(layer, batch, device, dtype)
    fused_state = _clone(eager_state)
    row_starts = torch.zeros(batch, dtype=torch.long, device=device)

    for _ in range(3):
        hidden = torch.randn(batch, 1, layer.cfg.hidden_size, device=device).to(dtype)
        with torch.inference_mode():
            expected = layer(hidden, eager_state, row_starts, None)
            got = _fused_step(layer, hidden, fused_state)
        if dtype == torch.float32:
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(
                fused_state.recurrent_state, eager_state.recurrent_state, rtol=1e-5, atol=1e-5
            )
        else:
            cosine = torch.nn.functional.cosine_similarity(
                got.double().flatten(), expected.double().flatten(), dim=0
            )
            assert float(cosine) > 0.999
            torch.testing.assert_close(
                fused_state.recurrent_state, eager_state.recurrent_state, rtol=2e-2, atol=2e-2
            )
        # The conv tail holds raw inputs, so it must match exactly.
        assert torch.equal(fused_state.conv, eager_state.conv)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_decode_with_fused_step_matches_eager(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whole-model ragged decode, kernel on vs off, from the same prefilled cache."""
    from mini_infer.models.qwen import qwen3_5

    device = _device()
    cfg = _config(32)
    cfg.num_hidden_layers = 4
    cfg.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    cfg.vocab_size = 64
    torch.manual_seed(0)
    model = qwen3_5.Qwen35ForConditionalGeneration(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.3, 0.3)
    model = model.to(device=device, dtype=dtype).eval()

    caches = []
    for _ in range(2):
        cache = model.build_state_cache(max_seq_len=32, batch_size=2, device=device, dtype=dtype)
        for row, length in enumerate((7, 3)):
            single = model.build_state_cache(
                max_seq_len=32, batch_size=1, device=device, dtype=dtype
            )
            prompt = torch.arange(length, device=device).view(1, length) % cfg.vocab_size
            with torch.inference_mode():
                model.forward_prefill_with_cache(prompt, state_cache=single)
            cache.copy_row_from(single, src_row=0, dst_row=row)
        caches.append(cache)

    positions = torch.tensor([7, 3], device=device)
    tokens = torch.tensor([[5], [9]], device=device)
    for step in range(3):
        outputs = []
        for use_kernel, cache in zip((False, True), caches, strict=True):
            monkeypatch.setattr(qwen3_5, "supports_gdn_decode_kernel", lambda _d, k=use_kernel: k)
            with torch.inference_mode():
                outputs.append(
                    model.forward_decode_with_cache_ragged(
                        tokens, positions=positions + step, state_cache=cache
                    )
                )
        expected, got = outputs
        if dtype == torch.float32:
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
        else:
            cosine = torch.nn.functional.cosine_similarity(
                got.double().flatten(), expected.double().flatten(), dim=0
            )
            assert float(cosine) > 0.999


def test_fused_step_rejects_a_non_power_of_two_head() -> None:
    device = _device()
    layer = _layer(_config(32), device, torch.float32)
    state = _prefilled_state(layer, 1, device, torch.float32)
    hidden = torch.randn(1, 1, layer.cfg.hidden_size, device=device)
    with pytest.raises(ValueError, match="power of two"):
        gdn_decode_kernel.gdn_decode_step(
            layer.in_proj_qkv(hidden)[:, 0],
            state.conv,
            layer.conv1d.weight,
            layer.in_proj_a(hidden)[:, 0],
            layer.in_proj_b(hidden)[:, 0],
            layer.A_log,
            layer.dt_bias,
            state.recurrent_state,
            layer.in_proj_z(hidden)[:, 0],
            layer.norm.weight,
            num_k_heads=2,
            head_k_dim=24,
            head_v_dim=32,
            rms_eps=1e-6,
        )
