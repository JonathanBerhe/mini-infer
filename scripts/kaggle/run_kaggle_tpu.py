"""Kaggle 'script' kernel: run the mini-infer Pallas TPU kernels on a real TPU.

This is the code_file for the Kaggle kernel defined in kernel-metadata.json. It
is the closest Kaggle offers to `modal run`: push it from a terminal, it runs on
a Kaggle TPU VM, and you poll for status and pull the log.

    pip install kaggle                       # once; API token at ~/.kaggle/kaggle.json
    # edit kernel-metadata.json: set "id" to "<your-username>/mini-infer-tpu-pallas"
    kaggle kernels push   -p scripts/kaggle
    kaggle kernels status  <your-username>/mini-infer-tpu-pallas    # poll to complete
    kaggle kernels output  <your-username>/mini-infer-tpu-pallas -p out/   # fetch log

Prerequisites (see README.md): the tpu-pallas-backend branch pushed to GitHub,
and TPU + internet enabled (kernel-metadata.json sets enable_tpu / enable_internet).
Kaggle's TPU image ships JAX + libtpu, so this does NOT reinstall jax (doing so
can break the preconfigured TPU runtime).
"""

import os
import subprocess
import sys

# If the repository is private, either make it public, push the source as a
# Kaggle dataset and mount it, or use an https://<TOKEN>@github.com/... URL here.
# Do NOT commit a token into this file.
REPO_URL = "https://github.com/JonathanBerhe/mini-infer.git"
BRANCH = "tpu-pallas-backend"
DEST = "/kaggle/working/mini-infer"


def _sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    import jax

    print("jax", jax.__version__, "devices:", jax.devices(), flush=True)
    if not any(getattr(d, "platform", "") == "tpu" for d in jax.devices()):
        print(
            "ERROR: no TPU device visible. Set the accelerator to TPU "
            "(kernel-metadata.json enable_tpu, or the Kaggle notebook UI) and re-run. "
            "This kernel refuses to fall back to CPU so a green run really means a TPU run.",
            flush=True,
        )
        return 1

    if not os.path.isdir(DEST):
        _sh("git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, DEST)
    # The tpu backend submodules import guard on jax only and mini_infer/__init__
    # is empty, so adding src to the path is all the runner needs (no pip install).
    sys.path.insert(0, os.path.join(DEST, "src"))
    sys.path.insert(0, os.path.join(DEST, "scripts"))
    os.chdir(DEST)

    import run_tpu_pallas_kernels as runner

    # runner.main() auto-detects the TPU and runs every kernel with interpret=False,
    # checking each against a NumPy reference and printing cosine + ms/call.
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
