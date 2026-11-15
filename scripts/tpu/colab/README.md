# Run the TPU Pallas kernels on a free Colab TPU

Colab needs only a Google account: no signup form, no approval queue, no credit
card, and no image-based identity verification. That makes it the low-friction
way to get the on-TPU confirmation that interpret mode cannot give (real Mosaic
lowering, tiling, and the scalar-prefetch gather).

## One click

Open the notebook straight from the public branch:

https://colab.research.google.com/github/JonathanBerhe/mini-infer/blob/main/scripts/tpu/colab/run_tpu.ipynb

Then:
1. **Runtime -> Change runtime type -> TPU**.
2. **Runtime -> Run all**.

The single cell clones the public `main` branch and runs every kernel (dense,
paged decode, paged prefill, mixed prefill/decode; MHA + GQA) with
`interpret=False`, checking each against a NumPy reference. A real TPU run
prints `devices: [TpuDevice(...)]` and ends with `ALL PASS`, and it refuses to
fall back to CPU, so a green run genuinely ran on the TPU.

The `BRANCH` variable at the top of the notebook cell picks what is validated:
`main` by default; set it to a feature branch (and open the notebook from that
branch) while one is under validation.

Note: normally the whole validation is a single run with no installs. The cell
first probes the environment with a one-line Pallas kernel; only on a broken
`libtpu` does it install the matched `jax[tpu]` pair and restart the runtime
once, because the bad `libtpu` is already loaded in the process and a restart
is the only way to swap it. "Broken" means either Colab's preinstalled
`libtpu` is older than its `jaxlib` (the skew rejects compiled kernels with
"Unsupported version: expected <= 7 but got 8", an environment problem, not a
kernel bug) or the TPU VM's `libtpu` failed to load at all (jax sees no TPU
despite the TPU runtime). Run the cell again after the reconnect and it
proceeds. On a non-TPU runtime it installs nothing and just
tells you to switch the runtime type. Colab free-tier TPU availability is
dynamic; retry later if none is free. If Colab keeps refusing to align, use
the GCP Cloud TPU path below: a fresh VM installs a matched `jax[tpu]` and
avoids the skew entirely.

## TorchTPU: run the engine on the TPU

TorchTPU is the PyTorch backend that adds a `tpu` device.
`scripts/tpu/run_torchtpu_spike.py` uses it to run the engine itself
(`ModelRunner`, `PagedKVCache`, torch attention) on the TPU. It compares each
decode step with a CPU run, then calls the Pallas kernels on torch tensors
through `torch.tpu.pallas.jax_op`.

A TPU accepts one process only, and the notebook above gives the TPU to JAX.
Thus, use a new runtime for this run, not the notebook above.

1. Open a new Colab notebook and set **Runtime -> Change runtime type -> TPU**.
2. Paste this cell and run it. Set `BRANCH` to the branch under test.

```python
BRANCH = "feat/torchtpu-pytorch-evaluation-a3611a"
!pip uninstall -q -y torch_xla  # the old PyTorch/XLA backend also drives the TPU
!pip install -q torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
!git clone -q --depth 1 --branch {BRANCH} https://github.com/JonathanBerhe/mini-infer.git
%cd mini-infer
!PYTHONPATH=src python scripts/tpu/run_torchtpu_spike.py
```

The script prints one line for each check when the check ends, and a summary
line at the end. A failed check prints its traceback and the other checks
continue. On a laptop, `--dry-run` runs the same checks on the CPU:

```bash
PYTHONPATH=src uv run --extra tpu python scripts/tpu/run_torchtpu_spike.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct
```

## Scriptable alternative: Google Cloud TPU (paid, gcloud, closest to `modal run`)

Needs a GCP account with billing (a card, no Persona). A `v5litepod-1` for a few
minutes costs cents. Zones and accelerator types vary by quota.

```bash
ZONE=us-central2-b
gcloud compute tpus tpu-vm create mini-infer-tpu \
  --zone=$ZONE --accelerator-type=v5litepod-1 --version=tpu-ubuntu2204-base
gcloud compute tpus tpu-vm ssh mini-infer-tpu --zone=$ZONE --command="\
  pip install -q 'jax[tpu]' numpy && \
  git clone --depth 1 -b main https://github.com/JonathanBerhe/mini-infer.git && \
  cd mini-infer && PYTHONPATH=src python scripts/tpu/run_tpu_pallas_kernels.py"
gcloud compute tpus tpu-vm delete mini-infer-tpu --zone=$ZONE --quiet   # stop billing
```

## Free alternative: TPU Research Cloud (application, no card)

TRC grants free TPUs (~30 days) for research; apply with an email at
https://sites.research.google/trc/ . Once granted, use the same `gcloud` flow
above in your TRC project. No image-based identity verification.
