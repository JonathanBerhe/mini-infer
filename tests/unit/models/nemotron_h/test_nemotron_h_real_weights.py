"""Real-weight smoke for Nemotron-H on `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.

The 4B is the smallest `NemotronHForCausalLM` checkpoint: 21 Mamba-2, 17
dense squared-ReLU MLP and 4 attention layers, with no MoE (the 30B-A3B GPU
gate in `scripts/nemotron_h/modal_nemotron_h_smoke.py` covers the MoE). It loads through
`StateCacheGenerator.from_pretrained`, which streams the single 8 GB file in
1 GiB batches, and decodes through the state-cache scheduler.

The download is 8 GB, so the tests are marked `large_model` and skip unless
the run passes `--run-large-models`:

    uv run pytest tests/unit/models/nemotron_h/test_nemotron_h_real_weights.py --run-large-models -v
"""

from __future__ import annotations

import pytest

from mini_infer.engine.sampler import SamplingParams
from mini_infer.engine.state_cache_generator import StateCacheGenerator
from mini_infer.models.nemotron_h.nemotron_h import NemotronHForCausalLM
from mini_infer.scheduler import Request, StateCacheContinuousScheduler

pytestmark = [pytest.mark.requires_model, pytest.mark.large_model]

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"


@pytest.fixture(scope="module")
def generator() -> StateCacheGenerator:
    from huggingface_hub import snapshot_download

    # No `*.py`: the checkpoint ships remote code, but transformers' own
    # `nemotron_h` config class parses it, so none of it runs.
    local_dir = snapshot_download(
        MODEL_NAME, revision=REVISION, allow_patterns=["*.safetensors", "*.json", "*.jinja"]
    )
    return StateCacheGenerator.from_pretrained(local_dir)


def test_nemotron_nano_4b_loads_through_registry(generator: StateCacheGenerator) -> None:
    model = generator.model
    assert isinstance(model, NemotronHForCausalLM)
    kinds = model.cfg.layer_types
    assert (kinds.count("mamba"), kinds.count("mlp"), kinds.count("attention")) == (21, 17, 4)
    assert kinds.count("moe") == 0


def test_nemotron_nano_4b_decodes_paris_for_france_prompt(generator: StateCacheGenerator) -> None:
    scheduler = StateCacheContinuousScheduler(generator, max_batch_size=1, max_seq_len=64)
    scheduler.start()
    try:
        result = scheduler.run(
            Request(
                prompt="The capital of France is",
                sampling_params=SamplingParams(),
                max_tokens=12,
            )
        )
    finally:
        scheduler.stop()
    assert "Paris" in result.text, f"unexpected output: {result.text!r}"
    assert result.finish_reason in {"stop", "length"}
