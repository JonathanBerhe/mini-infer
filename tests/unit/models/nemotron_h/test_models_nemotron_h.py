"""Nemotron-H: real config parsing, loading, MoE paths, state cache and serving.

Synthetic configs throughout, so there is no model download. Parity against
transformers lives in `test_nemotron_h_parity.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models import REGISTRY
from mini_infer.models.nemotron_h.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from mini_infer.models.nemotron_h.state_cache import (
    NemotronAttentionLayerState,
    NemotronMambaLayerState,
)
from mini_infer.scheduler import Request, StateCacheContinuousScheduler

FIXTURE = Path(__file__).parents[3] / "fixtures" / "nemotron_h" / "lightning_30b_a3b_config.json"
VOCAB = 64


def _config(**overrides: object) -> NemotronHConfig:
    raw: dict[str, object] = {
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "layers_block_type": ["mamba", "moe", "attention", "mamba", "moe"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "ssm_state_size": 8,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 4,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "moe_shared_expert_intermediate_size": 24,
        "routed_scaling_factor": 2.5,
        "max_position_embeddings": 128,
    }
    raw.update(overrides)
    return NemotronHConfig.from_hf(SimpleNamespace(**raw))


def _model(device: str = "cpu", dtype: torch.dtype = torch.float32) -> NemotronHForCausalLM:
    torch.manual_seed(0)
    model = NemotronHForCausalLM(_config())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("dt_bias"):
                param.uniform_(0.5, 1.5)
            elif not name.endswith("A_log"):
                param.uniform_(-0.3, 0.3)
    return model.to(device=device, dtype=dtype).eval()


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [(ord(ch) * 7 + i) % VOCAB for i, ch in enumerate(text)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return None


def test_real_lightning_config_parses() -> None:
    cfg = NemotronHConfig.from_hf(SimpleNamespace(**json.loads(FIXTURE.read_text())))
    assert cfg.num_hidden_layers == 52
    assert cfg.layer_types.count("mamba") == 23
    assert cfg.layer_types.count("moe") == 23
    assert cfg.layer_types.count("attention") == 6
    assert cfg.mamba_intermediate_size == 4096
    assert cfg.mamba_conv_channels == 6144
    assert (cfg.n_routed_experts, cfg.num_experts_per_tok) == (128, 6)
    assert cfg.moe_latent_size is None
    assert cfg.time_step_limit == (0.0, float("inf"))


def test_hybrid_override_pattern_is_understood() -> None:
    cfg = _config(layers_block_type=None, hybrid_override_pattern="M-*E")
    assert cfg.layer_types == ["mamba", "mlp", "attention", "moe"]


def test_unsupported_config_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="relu2"):
        _config(mlp_hidden_act="silu")
    with pytest.raises(ValueError, match="layer types"):
        _config(layers_block_type=["mamba", "conv"])


def test_registry_and_serving_markers() -> None:
    assert REGISTRY.lookup("NemotronHForCausalLM") is NemotronHForCausalLM
    assert NemotronHForCausalLM.USES_STATE_CACHE is True
    assert NemotronHForCausalLM.SUPPORTS_DECODE_CUDA_GRAPH is True


def test_state_cache_holds_state_only_for_mamba_and_attention() -> None:
    model = _model()
    cache = model.build_state_cache(max_seq_len=16, batch_size=3)
    kinds = [type(cache.layer(i)) for i in range(cache.num_layers)]
    assert kinds[0] is NemotronMambaLayerState
    assert kinds[1] is type(None)
    assert kinds[2] is NemotronAttentionLayerState
    mamba = cache.layer(0)
    assert isinstance(mamba, NemotronMambaLayerState)
    assert mamba.ssm.dtype == torch.float32
    assert tuple(mamba.ssm.shape) == (3, 4, 8, 8)
    assert tuple(mamba.conv.shape) == (3, model.cfg.mamba_conv_channels, 4)
    assert cache.max_seq_len == 16
    with pytest.raises(ValueError, match="max_seq_len"):
        cache.check_capacity(17)


def test_gathered_and_looped_moe_paths_agree() -> None:
    model = _model()
    moe = model.backbone.layers[1].mixer
    tokens = torch.randn(40, 32)
    indices, weights = moe.gate(tokens)
    with torch.no_grad():
        gathered = moe._gathered(tokens, indices, weights)
        looped = moe._looped(tokens, indices, weights)
    torch.testing.assert_close(gathered, looped, rtol=1e-5, atol=1e-6)


def test_static_decode_step_matches_checked_ragged_step() -> None:
    model = _model()
    caches = []
    for _ in range(2):
        batched = model.build_state_cache(max_seq_len=32, batch_size=2)
        for row, length in enumerate((9, 3)):
            single = model.build_state_cache(max_seq_len=32, batch_size=1)
            with torch.inference_mode():
                model.forward_prefill_with_cache(
                    torch.arange(length).view(1, length) % VOCAB, state_cache=single
                )
            batched.copy_row_from(single, src_row=0, dst_row=row)
        caches.append(batched)
    tokens, positions = torch.tensor([[3], [5]]), torch.tensor([9, 3])
    with torch.inference_mode():
        checked = model.forward_decode_with_cache_ragged(
            tokens, positions=positions, state_cache=caches[0]
        )
        static = model.decode_step_static(tokens, positions, caches[1])
    assert torch.equal(checked, static)


def test_from_checkpoint_matches_in_memory_load(tmp_path: Path) -> None:
    """Shard-by-shard load of a two-shard checkpoint equals `load_weights`."""
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    hf_config = transformers.NemotronHConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        layers_block_type=["mamba", "moe", "attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=8,
        n_groups=2,
        chunk_size=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=24,
        num_nextn_predict_layers=0,
    )
    hf_config.architectures = ["NemotronHForCausalLM"]
    hf_config.save_pretrained(tmp_path)
    torch.manual_seed(1)
    source = NemotronHForCausalLM(NemotronHConfig.from_hf(hf_config)).eval()
    state: dict[str, torch.Tensor] = {}
    for name, tensor in source.state_dict().items():
        if ".experts_" in name:
            base, kind = name.rsplit(".experts_", 1)
            for expert in range(tensor.shape[0]):
                state[f"{base}.experts.{expert}.{kind}_proj.weight"] = tensor[expert].clone()
        elif name.endswith("conv1d.weight"):
            state[name] = tensor.unsqueeze(1).clone()
        else:
            state[name] = tensor.clone()
    keys = sorted(state)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {key: shards[i % 2] for i, key in enumerate(keys)}
    for shard in shards:
        save_file(
            {key: state[key].contiguous() for key in keys if weight_map[key] == shard},
            str(tmp_path / shard),
        )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    loaded = NemotronHForCausalLM.from_checkpoint(str(tmp_path), device="cpu", dtype=torch.float32)
    ids = torch.randint(0, VOCAB, (1, 7))
    with torch.no_grad():
        assert torch.equal(loaded(ids), source(ids))
    gate_bias = loaded.backbone.layers[1].mixer.gate.e_score_correction_bias
    assert gate_bias.dtype == torch.float32


def _serve(model: NemotronHForCausalLM, *, cuda_graph: bool = False) -> list[list[int]]:
    scheduler = StateCacheContinuousScheduler(
        StateCacheGenerator(model, _FakeTokenizer()),  # type: ignore[arg-type]
        max_batch_size=2,
        max_seq_len=64,
        cuda_graph=cuda_graph,
    )
    scheduler.start()
    try:
        handles = [
            scheduler.submit(Request(prompt=p, sampling_params=SamplingParams(), max_tokens=10))
            for p in ("short", "a longer prompt than the first one", "mid prompt", "x")
        ]
        return [handle.wait(timeout=120).tokens for handle in handles]
    finally:
        scheduler.stop()


def test_scheduler_serves_concurrent_requests() -> None:
    """Different prompt lengths in flight together, moved between cache rows."""
    model = _model()
    results = _serve(model)
    assert all(len(tokens) == 10 for tokens in results)
    # One request at a time must give the same greedy tokens (fp32).
    alone = []
    for prompt in ("short", "a longer prompt than the first one", "mid prompt", "x"):
        generator = StateCacheGenerator(model, _FakeTokenizer())  # type: ignore[arg-type]
        ids = generator.tokenizer.encode(prompt)
        with torch.inference_mode():
            alone.append(generator.generate_ids(ids, max_new_tokens=10, eos_token_id=None))
    assert results == alone


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_graph_serves_the_same_tokens_as_eager(dtype: torch.dtype) -> None:
    model = _model("cuda", dtype)
    assert _serve(model, cuda_graph=True) == _serve(model, cuda_graph=False)
