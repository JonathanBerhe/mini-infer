# Run the TPU Pallas kernels on a free Colab TPU

Colab needs only a Google account: no signup form, no approval queue, no credit
card, and no image-based identity verification. That makes it the low-friction
way to get the on-TPU confirmation that interpret mode cannot give (real Mosaic
lowering, tiling, and the scalar-prefetch gather).

## One click

Open the notebook straight from the public branch:

https://colab.research.google.com/github/JonathanBerhe/mini-infer/blob/main/scripts/colab/run_tpu.ipynb

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

Note: on the first run the cell aligns Colab's `libtpu` with its `jaxlib` and
restarts the runtime once, then you run the cell again. Colab's preinstalled
`libtpu` can be older than `jaxlib` and reject the compiled kernel with
"Unsupported version: expected <= 7 but got 8". That is an environment version
skew, not a kernel bug: the kernel compiles to valid Mosaic; the runtime just
cannot load what this jaxlib emitted. Colab free-tier TPU availability is also
dynamic (typically TPU v2); retry later if none is free. If Colab keeps refusing
to align, use the GCP Cloud TPU path below: a fresh VM installs a matched
`jax[tpu]` and avoids the skew entirely.

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
  cd mini-infer && PYTHONPATH=src python scripts/run_tpu_pallas_kernels.py"
gcloud compute tpus tpu-vm delete mini-infer-tpu --zone=$ZONE --quiet   # stop billing
```

## Free alternative: TPU Research Cloud (application, no card)

TRC grants free TPUs (~30 days) for research; apply with an email at
https://sites.research.google/trc/ . Once granted, use the same `gcloud` flow
above in your TRC project. No image-based identity verification.
