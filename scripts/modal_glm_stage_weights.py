"""Stage the GLM-5.2-FP8 weights into a Modal Volume (CPU-only, no GPU billing).

Running the real 753B checkpoint is gated on a multi-GPU node, but the ~755 GB
download should NOT happen while expensive GPUs sit idle. This script downloads
the checkpoint into a persistent Modal Volume on a CPU-only function; a later
GPU run mounts the same Volume and loads from local disk (seconds, not a
re-download).

COST NOTES (read before running):
  - This function is CPU-only (no `gpu=`), so it bills CPU + egress, not GPUs.
    A full download is roughly tens of minutes to a couple of hours.
  - The Volume then STORES ~755 GB persistently, which bills per GB-month until
    you delete it. After you're done with the GPU run, free it with:
        uv run modal volume delete glm-5-2-fp8-weights
  - There is no point running this until you intend to fund the GPU run; the
    weights are only useful paired with a multi-GPU node + the FP8-resident
    serving path (not yet built; see ADR-021).

Run with:
    uv run modal run scripts/modal_glm_stage_weights.py
"""

import modal

_REPO = "zai-org/GLM-5.2-FP8"
_VOLUME_NAME = "glm-5-2-fp8-weights"
_MOUNT = "/weights"

app = modal.App("mini-infer-glm-stage")
weights_volume = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.20",
    "hf-transfer>=0.1.6",  # parallel, fast LFS downloads
)


@app.function(
    image=image,
    volumes={_MOUNT: weights_volume},
    timeout=24 * 3600,  # large checkpoint; allow a long, resumable download
    cpu=8.0,
)
def stage() -> dict:
    """Download the FP8 checkpoint into the Volume (resumable; commits on finish)."""
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    target = f"{_MOUNT}/{_REPO.split('/')[-1]}"
    # allow_patterns keeps it to weights + the files the loader needs; skip the
    # original-precision BF16 shards if any are mirrored here.
    path = snapshot_download(
        repo_id=_REPO,
        local_dir=target,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
    )
    weights_volume.commit()

    total = 0
    for root, _dirs, files in os.walk(target):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return {"path": path, "staged_gb": round(total / 1e9, 1)}


@app.local_entrypoint()
def main() -> None:
    info = stage.remote()
    print(f"Staged {_REPO} -> Volume {_VOLUME_NAME!r} at {info['path']}")
    print(f"Size on volume: {info['staged_gb']} GB")
    print(
        f"Remember to `modal volume delete {_VOLUME_NAME}` when done (storage bills per GB-month)."
    )
