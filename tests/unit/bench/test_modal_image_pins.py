"""Every Modal image installs exact torch, Triton and FlashInfer versions.

A lower bound such as `torch>=2.6` installs the newest release whenever Modal
rebuilds the image, so two GPU runs a few hours apart can time different
kernels. Benchmark numbers from separate runs are comparable only when the
images pin these three packages exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

# A quoted requirement for one of the three packages: a pip operator, then a
# version number. A log line such as f"torch={torch.__version__}" is no match.
_SPEC = re.compile(
    r"""["'](torch|triton|flashinfer-python)\s*((?:==|>=|<=|~=|!=|>|<)\s*\d[^"']*)["']"""
)


def _image_specs() -> list[tuple[str, str, str]]:
    specs = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "modal.Image" not in text:
            continue
        for match in _SPEC.finditer(text):
            specs.append((str(path.relative_to(SCRIPTS)), match.group(1), match.group(2)))
    return specs


def test_the_scan_finds_the_modal_images() -> None:
    files = {path for path, _, _ in _image_specs()}
    assert len(files) >= 20, f"found only {sorted(files)}"


def test_every_modal_image_pins_exact_versions() -> None:
    loose = [
        f"{path}: {name}{spec}" for path, name, spec in _image_specs() if not spec.startswith("==")
    ]
    assert loose == [], "pin these with ==:\n" + "\n".join(loose)


def test_every_modal_image_installs_the_same_flashinfer() -> None:
    versions = {spec for _, name, spec in _image_specs() if name == "flashinfer-python"}
    assert len(versions) == 1, f"FlashInfer versions differ between images: {sorted(versions)}"
