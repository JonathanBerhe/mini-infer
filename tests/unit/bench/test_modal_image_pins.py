"""Every Modal image installs exact torch, Triton and FlashInfer versions.

A lower bound such as `torch>=2.6` installs the newest release whenever Modal
rebuilds the image, so two GPU runs a few hours apart can time different
kernels. Benchmark numbers from separate runs are comparable only when the
images pin these three packages exactly.

The Gluon MoE kernels run only with one Triton release. If the main images
install another one, the MoE blocks fall back to the Triton kernels and no
run reports it, so the main images must pin the Triton of the Gluon kernels.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
GLUON_GATE = ROOT / "src" / "mini_infer" / "models" / "blocks" / "moe" / "fp8_grouped_gluon.py"

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


def _gluon_triton() -> str:
    """The Triton release of the Gluon kernels, as "major.minor"."""
    text = GLUON_GATE.read_text(encoding="utf-8")
    match = re.search(r"_GLUON_AVAILABLE = .*==\s*\((\d+),\s*(\d+)\)", text)
    assert match, f"no Triton version check in {GLUON_GATE}"
    return f"{match.group(1)}.{match.group(2)}"


def test_the_main_images_install_the_triton_of_the_gluon_kernels() -> None:
    # The main images are the files that install the most common torch. A
    # file can hold more than one image, so it must pin that Triton once.
    torch_files: dict[str, set[str]] = {}
    triton_files: dict[str, set[str]] = {}
    for path, name, spec in _image_specs():
        if name == "torch":
            torch_files.setdefault(path, set()).add(spec.split("+")[0])
        elif name == "triton":
            triton_files.setdefault(path, set()).add(spec)
    counts = Counter(spec for specs in torch_files.values() for spec in specs)
    main_torch = counts.most_common(1)[0][0]
    want = f"=={_gluon_triton()}"
    wrong = [
        f"{path}: torch{main_torch}, triton {sorted(triton_files.get(path, set())) or 'unpinned'}"
        for path, specs in sorted(torch_files.items())
        if main_torch in specs
        and not any(t == want or t.startswith(want + ".") for t in triton_files.get(path, set()))
    ]
    assert wrong == [], f"pin triton{want}.x, the Triton of the Gluon kernels:\n" + "\n".join(wrong)
