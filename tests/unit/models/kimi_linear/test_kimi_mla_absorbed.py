"""The absorbed MLA decode of the Kimi family against the decompressed form.

A decode step folds `kv_b_proj` into the query and the output instead of
rebuilding the keys and values of every history token
(`_KimiMlaAttention._attend_absorbed`). These tests pin it to the
decompressed form, which the reference parity tests pin to the checkpoints'
own code, for the three MLA variants: Kimi Linear (dense, full-rank Q),
Kimi K3 (low-rank Q, output gate) and Kimi K3.1 (a DSA indexer whose top-k
selection runs). They also pin the state layout: each row holds the latent
after `kv_a_layernorm`.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from mini_infer.models.kimi_linear.kimi_k3 import KimiK3Config, KimiK3ForConditionalGeneration
from mini_infer.models.kimi_linear.kimi_linear import (
    KimiLinearConfig,
    KimiLinearForCausalLM,
    _KimiMlaAttention,
)
from mini_infer.models.kimi_linear.state_cache import KimiMlaLayerState, KimiStateCache
from mini_infer.quant.int8 import Int8Linear
from tests.unit.models.kimi_linear.test_models_kimi_k3 import (
    _tiny_hf_namespace as _k3_namespace,
)
from tests.unit.models.kimi_linear.test_models_kimi_k31 import _k31_model
from tests.unit.models.kimi_linear.test_models_kimi_linear import (
    _tiny_hf_namespace as _kimi_linear_namespace,
)

# The tiny configs of all three variants have one MLA layer, the last of four.
_MLA_LAYER = 3


def _kimi_linear() -> KimiLinearForCausalLM:
    torch.manual_seed(0)
    return KimiLinearForCausalLM(KimiLinearConfig.from_hf(_kimi_linear_namespace())).float().eval()


def _k3() -> KimiLinearForCausalLM:
    torch.manual_seed(1)
    return KimiK3ForConditionalGeneration(KimiK3Config.from_hf(_k3_namespace())).float().eval()


def _k31() -> KimiLinearForCausalLM:
    # Top-k 3: every history longer than 3 tokens runs the selection.
    return _k31_model(seed=2)


_VARIANTS: dict[str, Callable[[], KimiLinearForCausalLM]] = {
    "kimi_linear": _kimi_linear,
    "k3": _k3,
    "k31_dsa": _k31,
}


def _mla(model: KimiLinearForCausalLM) -> _KimiMlaAttention:
    attn = model.model.layers[_MLA_LAYER].self_attn
    assert isinstance(attn, _KimiMlaAttention)
    return attn


def _random_state(attn: _KimiMlaAttention, batch: int, capacity: int) -> KimiMlaLayerState:
    width = attn.kv_lora_rank + attn.qk_rope_head_dim
    index_k = None
    if attn.indexer is not None:
        index_k = torch.randn(batch, capacity, attn.indexer.head_dim)
    return KimiMlaLayerState(kv=torch.randn(batch, capacity, width), index_k=index_k)


def _copy(state: KimiMlaLayerState) -> KimiMlaLayerState:
    index_k = state.index_k.clone() if state.index_k is not None else None
    return KimiMlaLayerState(kv=state.kv.clone(), index_k=index_k)


_FORMS = {
    True: _KimiMlaAttention._attend_absorbed,
    False: _KimiMlaAttention._attend_decompressed,
}


def _count_forms(monkeypatch: pytest.MonkeyPatch, absorbed: bool) -> dict[bool, int]:
    """Select the decode form and count the calls of each form."""
    monkeypatch.setattr(_KimiMlaAttention, "absorbed_decode", absorbed)
    calls = {True: 0, False: 0}
    for form, method in _FORMS.items():

        def counted(
            self: _KimiMlaAttention,
            *args: torch.Tensor,
            _form: bool = form,
            _method: Callable[..., torch.Tensor] = method,
        ) -> torch.Tensor:
            calls[_form] += 1
            return _method(self, *args)

        monkeypatch.setattr(_KimiMlaAttention, _FORMS[form].__name__, counted)
    return calls


def _decode(
    attn: _KimiMlaAttention,
    x: torch.Tensor,
    state: KimiMlaLayerState,
    starts: torch.Tensor,
    *,
    absorbed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor:
    """One decode step of `attn` in the chosen form, which must be the form that runs."""
    calls = _count_forms(monkeypatch, absorbed)
    with torch.inference_mode():
        out: torch.Tensor = attn(x, state, starts)
    assert calls == {absorbed: 1, not absorbed: 0}
    return out


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_absorbed_decode_equals_the_decompressed_form(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ragged decode step over random history rows: rows of 1, 6 and 20
    tokens in a buffer of capacity 24 give the same output in both forms, and
    write the same new row."""
    attn = _mla(_VARIANTS[variant]())
    torch.manual_seed(3)
    state = _random_state(attn, batch=3, capacity=24)
    x = torch.randn(3, 1, attn.kv_a_proj_with_mqa.in_features)
    starts = torch.tensor([0, 5, 19])
    absorbed_state, decompressed_state = _copy(state), _copy(state)
    absorbed = _decode(attn, x, absorbed_state, starts, absorbed=True, monkeypatch=monkeypatch)
    decompressed = _decode(
        attn, x, decompressed_state, starts, absorbed=False, monkeypatch=monkeypatch
    )
    torch.testing.assert_close(absorbed, decompressed, rtol=1e-5, atol=1e-5)
    assert torch.equal(absorbed_state.kv, decompressed_state.kv)
    # The two forms must not agree by accident: the history changes the output.
    state.kv[1, :5] = torch.randn(5, state.kv.shape[-1])
    changed = _decode(attn, x, _copy(state), starts, absorbed=True, monkeypatch=monkeypatch)
    assert not torch.allclose(changed[1], absorbed[1], atol=1e-3)
    torch.testing.assert_close(changed[0], absorbed[0], rtol=0, atol=0)


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_rows_past_each_rows_length_do_not_change_its_output(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slot that held a longer request keeps stale rows past the new
    request's length. The absorbed decode reads the buffer up to the longest
    row of the batch and must mask those stale rows out of the shorter rows."""
    attn = _mla(_VARIANTS[variant]())
    torch.manual_seed(4)
    clean = _random_state(attn, batch=2, capacity=24)
    stale = _copy(clean)
    stale.kv[1, 7:] = 1e3 * torch.randn(17, clean.kv.shape[-1])
    if stale.index_k is not None:
        stale.index_k[1, 7:] = 1e3 * torch.randn(17, stale.index_k.shape[-1])
    x = torch.randn(2, 1, attn.kv_a_proj_with_mqa.in_features)
    starts = torch.tensor([20, 6])
    want = _decode(attn, x, clean, starts, absorbed=True, monkeypatch=monkeypatch)
    got = _decode(attn, x, stale, starts, absorbed=True, monkeypatch=monkeypatch)
    torch.testing.assert_close(got[1], want[1], rtol=0, atol=0)


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_model_decode_is_the_same_in_both_forms(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefill two prompts of different lengths into one batched cache, then
    run four ragged decode steps: the logits of both forms agree at every step.
    The K3.1 rows grow past top-k 3, so its selection runs at every step."""
    model = _VARIANTS[variant]()
    torch.manual_seed(5)
    prompts = [torch.randint(0, 96, (1, 9)), torch.randint(0, 96, (1, 4))]
    steps = torch.randint(0, 96, (4, 2, 1))

    def run(absorbed: bool) -> list[torch.Tensor]:
        calls = _count_forms(monkeypatch, absorbed)
        batched = model.build_state_cache(max_seq_len=32, batch_size=2)
        with torch.inference_mode():
            for row, prompt in enumerate(prompts):
                single: KimiStateCache = model.build_state_cache(max_seq_len=32)
                model.forward_prefill_with_cache(prompt, state_cache=single)
                batched.copy_row_from(single, src_row=0, dst_row=row)
            # A prefill is a span: it runs the decompressed form in both runs.
            assert calls == {True: 0, False: 2}
            positions = torch.tensor([p.shape[1] for p in prompts])
            out = []
            for tokens in steps:
                out.append(
                    model.forward_decode_with_cache_ragged(
                        tokens, positions=positions, state_cache=batched
                    )
                )
                positions = positions + 1
        # One MLA layer, one call per decode step.
        if absorbed:
            assert calls == {True: len(steps), False: 2}
        else:
            assert calls == {True: 0, False: 2 + len(steps)}
        return out

    for absorbed, decompressed in zip(run(True), run(False), strict=True):
        torch.testing.assert_close(absorbed, decompressed, rtol=1e-4, atol=1e-4)


def test_the_state_keeps_the_normalized_latent() -> None:
    """Each state row is `[kv_a_layernorm(latent), k_rot]` of `kv_a_proj_with_mqa(x)`."""
    attn = _mla(_kimi_linear())
    torch.manual_seed(6)
    x = torch.randn(1, 5, attn.kv_a_proj_with_mqa.in_features)
    state = KimiMlaLayerState(kv=torch.zeros(1, 8, attn.kv_lora_rank + attn.qk_rope_head_dim))
    with torch.inference_mode():
        attn(x, state, torch.zeros(1, dtype=torch.long))
        latent, k_rot = torch.split(
            attn.kv_a_proj_with_mqa(x), [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
        )
        want = torch.cat([attn.kv_a_layernorm(latent), k_rot], dim=-1)
    assert torch.equal(state.kv[:, :5], want)
    assert torch.equal(state.kv[:, 5:], torch.zeros_like(state.kv[:, 5:]))


@pytest.mark.parametrize("absorbed", [True, False])
def test_bf16_decode_stays_within_the_bf16_tolerance_of_fp32(
    absorbed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In bf16 the two forms round at other places. Each one stays within the
    bf16 tolerance of the fp32 decompressed form; on the CPU their errors
    measured about the same size (a ratio of 0.8 to 1.4 over eight seeds)."""
    attn = _mla(_kimi_linear())
    torch.manual_seed(7)
    state = _random_state(attn, batch=2, capacity=64)
    x = torch.randn(2, 1, attn.kv_a_proj_with_mqa.in_features)
    starts = torch.tensor([60, 31])
    want = _decode(attn, x, _copy(state), starts, absorbed=False, monkeypatch=monkeypatch)
    attn_bf16 = attn.to(torch.bfloat16)
    state_bf16 = KimiMlaLayerState(kv=state.kv.to(torch.bfloat16))
    got = _decode(
        attn_bf16,
        x.to(torch.bfloat16),
        state_bf16,
        starts,
        absorbed=absorbed,
        monkeypatch=monkeypatch,
    )
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, rtol=1.6e-2, atol=1e-2)


def test_an_int8_kv_b_proj_keeps_the_decompressed_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """The absorbed form reads the `kv_b_proj` weight directly, so a weight
    that is not floating point (an `Int8Linear`) must keep the decompressed form."""
    attn = _mla(_kimi_linear())
    attn.kv_b_proj = Int8Linear.from_float(attn.kv_b_proj)
    torch.manual_seed(8)
    state = _random_state(attn, batch=1, capacity=8)
    x = torch.randn(1, 1, attn.kv_a_proj_with_mqa.in_features)
    calls = _count_forms(monkeypatch, absorbed=True)
    with torch.inference_mode():
        attn(x, state, torch.tensor([4]))
    assert calls == {True: 0, False: 1}
