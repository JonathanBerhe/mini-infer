"""Hardware-detection helpers used across the engine.

Centralizing these avoids the slow bleed of `device.type == "cuda"` and
`x.is_cuda` checks across kernel wrappers, dispatchers, and quant
modules. Each kernel module still owns its own `supports_X_kernel(device)`
predicate (which checks library availability + bench toggles + the
device), but the device piece itself comes from here.
"""

from __future__ import annotations

import torch


def is_cuda_device(device: torch.device | str) -> bool:
    """Whether a torch device is a CUDA device.

    Accepts both string ("cuda", "cuda:0", "cpu", ...) and `torch.device`
    inputs so call sites don't have to normalize first.
    """
    if isinstance(device, str):
        return device == "cuda" or device.startswith("cuda:")
    return device.type == "cuda"


def require_cuda_device(device: torch.device | str, what: str) -> None:
    """Raise `RuntimeError` if `device` is not CUDA.

    `what` is a short noun phrase naming the caller (e.g. "fused W8A16
    kernel"); it gets formatted into the error message for diagnostics.
    """
    if not is_cuda_device(device):
        raise RuntimeError(f"{what} requires CUDA; got device={device}")


def _cuda_major(device: torch.device | str | None) -> int | None:
    """Compute-capability major of a CUDA device, or `None` if it is not one."""
    if not torch.cuda.is_available():
        return None
    if device is not None and not is_cuda_device(device):
        return None
    if device is None:
        idx: int | str = torch.cuda.current_device()
    elif isinstance(device, str):
        idx = device
    else:
        idx = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(idx)
    return major


def is_blackwell_device(device: torch.device | str | None = None) -> bool:
    """Whether the current (or specified) CUDA device is Blackwell (SM_100+).

    Pass a specific device to query that one; pass `None` (default) for
    the current device. Compute capability >= (10, 0) is the Blackwell
    family (B100, B200, B300, RTX 50-series, ...). H100 is (9, 0); A100
    is (8, 0).
    """
    major = _cuda_major(device)
    return major is not None and major >= 10


def is_datacenter_blackwell_device(device: torch.device | str | None = None) -> bool:
    """Whether the current (or specified) CUDA device is SM_10x (B200, B300).

    Same device argument as `is_blackwell_device`. These are the Blackwell
    GPUs with `tcgen05` and tensor memory; the RTX 50-series (12, 0) has
    neither.
    """
    return _cuda_major(device) == 10


def is_hopper_device(device: torch.device | str | None = None) -> bool:
    """Whether the current (or specified) CUDA device is Hopper (SM_90).

    Same device argument as `is_blackwell_device`. H100 / H200 are (9, 0).
    For kernels that use Hopper-only instructions (`wgmma`).
    """
    return _cuda_major(device) == 9


def is_hopper_or_newer_device(device: torch.device | str | None = None) -> bool:
    """Whether the current (or specified) CUDA device is Hopper or newer (SM_90+).

    Same device argument as `is_blackwell_device`. H100 / H200 are (9, 0);
    every Blackwell part also qualifies.
    """
    major = _cuda_major(device)
    return major is not None and major >= 9
