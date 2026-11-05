"""One report line that names the fast paths this process can take.

Several GPU paths fall back to a slower one without an error: the Gluon MoE
kernels need Triton 3.8 and a Hopper or datacenter Blackwell GPU, the
FlashInfer backend needs `flashinfer-python`, and the `flash_attn` backend
needs `flash-attn`. Without them the engine still runs, so a benchmark can
time the slow path and nobody sees it. Modal scripts print this line at the
top of their report, so that a fallback shows next to the numbers.
"""

from __future__ import annotations

from importlib import metadata

import torch

from mini_infer.cache.packed_attention import effective_attention_backend
from mini_infer.device import is_datacenter_blackwell_device, is_hopper_device


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "none"


def fast_paths(attention_backend: str = "flash_attn") -> dict[str, str]:
    """The GPU, the library versions and the fast path of each feature.

    `attention_backend` is the backend that the run requests. The `attention`
    entry is the backend that runs, as `requested->effective` when the two
    differ. `gluon` is "True" only when the Gluon MoE kernels can run on the
    current GPU with the installed Triton.
    """
    from mini_infer.models.blocks.moe.fp8_grouped_gluon import _GLUON_AVAILABLE

    on_gpu = torch.cuda.is_available()
    device = "cuda" if on_gpu else "cpu"
    if on_gpu:
        major, minor = torch.cuda.get_device_capability()
        gpu, cc = torch.cuda.get_device_name(), f"{major}.{minor}"
    else:
        gpu, cc = "none", "none"
    gluon_arch = on_gpu and (is_hopper_device() or is_datacenter_blackwell_device())
    effective = effective_attention_backend(attention_backend, device)
    attention = effective if effective == attention_backend else f"{attention_backend}->{effective}"
    return {
        "gpu": gpu,
        "cc": cc,
        "torch": torch.__version__,
        "triton": _version("triton"),
        "flashinfer": _version("flashinfer-python"),
        "flash_attn": _version("flash-attn"),
        "gluon": str(bool(_GLUON_AVAILABLE and gluon_arch)),
        "attention": attention,
    }


def fast_path_report(attention_backend: str = "flash_attn") -> str:
    """`fast_paths` as one `key=value` line for the header of a run report."""
    return " ".join(f"{key}={value}" for key, value in fast_paths(attention_backend).items())
