# Run the TPU Pallas kernels on a real TPU (Kaggle)

> Note: Kaggle now gates accelerator use behind image-based identity
> verification (Persona). If you would rather not do that, use the Colab path in
> `scripts/colab/` instead (Google account only, no verification), or GCP Cloud
> TPU / TPU Research Cloud. This Kaggle bundle is kept for anyone fine with the
> verification.

Everything in the TPU backend (ADR-023, ADR-024) is validated in Pallas
interpret mode on CPU. This bundle runs the same kernels on an actual TPU, which
is the one thing interpret mode cannot confirm (real Mosaic lowering, tiling, and
the scalar-prefetch gather). Kaggle TPU is free (quota-limited), so it fits the
project's budget better than Cloud TPU.

It is a batch flow, not a live one: you push the kernel, poll for status, then
fetch the log. That is the closest Kaggle has to `modal run`.

## Prerequisites

1. **Push the branch to GitHub** so the kernel can clone it:
   ```bash
   git push -u origin tpu-pallas-backend
   ```
   If the repo is private, either make it public, use an
   `https://<TOKEN>@github.com/...` URL in `run_kaggle_tpu.py` (do not commit the
   token), or push the source as a Kaggle dataset and mount it.
2. **Install the CLI and add a token:** `pip install kaggle`, then put your
   `kaggle.json` API token at `~/.kaggle/kaggle.json` (chmod 600). The Kaggle
   account must be phone-verified to use accelerators.
3. **Set your username** in `kernel-metadata.json`: change `id` to
   `"<your-username>/mini-infer-tpu-pallas-kernels"`.

## Run

```bash
kaggle kernels push   -p scripts/kaggle
kaggle kernels status  <your-username>/mini-infer-tpu-pallas-kernels          # poll until "complete"
kaggle kernels output  <your-username>/mini-infer-tpu-pallas-kernels -p out/  # fetch the log
```

The log should end with `ALL PASS` and, per kernel, a line like
`[PASS] prefill GQA q_len=8 (8q/2kv)   cosine=0.99999...  max_abs_err=...  X.XXX ms/call`,
covering dense (causal and non-causal), paged decode (MHA and GQA), and paged
prefill / multi-query (MHA and GQA). The kernel refuses to fall back to CPU, so a
green run genuinely means it ran on the TPU.

## Caveats

- **TPU via metadata often falls back to CPU.** `enable_tpu: true` is a valid
  field (it is in `kaggle kernels init`), but a CLI push commonly still gets a
  CPU worker (observed here: `jax devices: [CpuDevice]`). When that happens the
  kernel logs `ERROR: no TPU device visible` and exits non-zero by design, so an
  ERROR with that message means provisioning fell back, not that the kernels are
  wrong. The reliable fix is the UI: open the kernel, set the accelerator to
  "TPU VM v3-8", and Save & Run All. The account must be phone-verified with TPU
  quota. `machine_shape` in the metadata is the other lever if you script it.
- **Internet must stay enabled** (`enable_internet: true`) so the kernel can
  clone the branch.
- **JAX is preinstalled** on Kaggle's TPU image; the script deliberately does not
  reinstall it, which would risk breaking the preconfigured TPU runtime.
