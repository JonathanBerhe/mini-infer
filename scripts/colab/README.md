# Run the TPU Pallas kernels on a free Colab TPU

Colab needs only a Google account: no signup form, no approval queue, no credit
card, and no image-based identity verification. That makes it the low-friction
way to get the on-TPU confirmation that interpret mode cannot give (real Mosaic
lowering, tiling, and the scalar-prefetch gather).

## One click

Open the notebook straight from the public branch:

https://colab.research.google.com/github/JonathanBerhe/mini-infer/blob/tpu-pallas-backend/scripts/colab/run_tpu.ipynb

Then:
1. **Runtime -> Change runtime type -> TPU**.
2. **Runtime -> Run all**.

The single cell clones the public `tpu-pallas-backend` branch and runs every
kernel (dense, paged decode, paged prefill; MHA + GQA) with `interpret=False`,
checking each against a NumPy reference. A real TPU run prints
`devices: [TpuDevice(...)]` and ends with `ALL PASS`, and it refuses to fall back
to CPU, so a green run genuinely ran on the TPU.

Note: Colab free-tier TPU availability is dynamic; if none is free, retry later
or use Colab Pro. Colab's free TPU is typically TPU v2, which is plenty for this
correctness confirmation.

## Scriptable alternative: Google Cloud TPU (paid, gcloud, closest to `modal run`)

Needs a GCP account with billing (a card, no Persona). A `v5litepod-1` for a few
minutes costs cents. Zones and accelerator types vary by quota.

```bash
ZONE=us-central2-b
gcloud compute tpus tpu-vm create mini-infer-tpu \
  --zone=$ZONE --accelerator-type=v5litepod-1 --version=tpu-ubuntu2204-base
gcloud compute tpus tpu-vm ssh mini-infer-tpu --zone=$ZONE --command="\
  pip install -q 'jax[tpu]' numpy && \
  git clone --depth 1 -b tpu-pallas-backend https://github.com/JonathanBerhe/mini-infer.git && \
  cd mini-infer && PYTHONPATH=src python scripts/run_tpu_pallas_kernels.py"
gcloud compute tpus tpu-vm delete mini-infer-tpu --zone=$ZONE --quiet   # stop billing
```

## Free alternative: TPU Research Cloud (application, no card)

TRC grants free TPUs (~30 days) for research; apply with an email at
https://sites.research.google/trc/ . Once granted, use the same `gcloud` flow
above in your TRC project. No image-based identity verification.
