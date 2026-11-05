"""The Kimi MLA decode forms and the decode-step split on one H100.

Runs `scripts/kimi_linear/profile_kimi_decode.py` in bf16 on the GPU:

1. `mla`: one MLA layer at the Kimi-Linear-48B widths, the decompressed and
   the absorbed decode, against the history length, the buffer capacity and
   a ragged batch, with the largest difference between the two outputs.
2. `split`: the decode step of a 4-layer slice of the 48B shape with all 256
   experts, for each MLA form, at 1,024 and 8,192 history tokens. The trace
   gives the GPU time, the kernels and the host syncs by part of the model.

No checkpoint is read: the weights are random, at the real widths. Each case
prints its line when it finishes, so a timeout keeps the cases that ran.

Run from the repository root:

    uv run modal run scripts/kimi_linear/modal_kimi_mla_decode.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
_PROFILE = _HERE.parent / "profile_kimi_decode.py"

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.14.1+cu132", extra_index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("triton==3.8.0", "transformers==5.14.0", "safetensors>=0.4")
    .add_local_file(str(_PROFILE), "/root/profile_kimi_decode.py")
    .add_local_python_source("mini_infer")
)
app = modal.App("mini-infer-kimi-mla-decode", image=image)

_RUNS = [
    ["mla", "--mla-path", "absorbed", "--steps", "32"],
    *(
        [
            "split",
            "--mla-path",
            path,
            "--experts",
            "256",
            "--prompt",
            str(history),
            "--max-seq-len",
            str(2 * history),
        ]
        for history in (1024, 8192)
        for path in ("decompressed", "absorbed")
    ),
]


@app.function(gpu="H100!", timeout=20 * 60)
def measure() -> None:
    for args in _RUNS:
        print(f"\n=== profile_kimi_decode.py {' '.join(args)}", flush=True)
        subprocess.run(["python", "-u", "/root/profile_kimi_decode.py", *args], check=True)


@app.local_entrypoint()
def main() -> None:
    measure.remote()
