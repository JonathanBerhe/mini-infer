"""Vendor the Latent Sparse Attention reference (pure PyTorch) for the parity tests.

Why vendor: `tests/unit/models/qwen/test_lsa_parity.py` compares our bucket
builder and key selection (`mini_infer.models.qwen.lsa`) against the authors'
own plain-PyTorch oracle, `lsa.core.reference`. The revision is PINNED so the
oracle cannot drift under us.

Only four source files are copied, at their package paths, under
`third_party/lsa_reference/lsa/`. The real `lsa/__init__.py` and
`lsa/core/__init__.py` pull in Triton and CuTe DSL kernels, so this script
writes empty ones instead: importing `lsa.core.reference` then runs nothing
but the copied files. The repository is Apache-2.0, and its LICENSE is copied
beside the sources.

Idempotent: skips the download if the target directory already holds the
expected files. Re-run with `--force` to refresh.

Run with:
    uv run python scripts/qwen/clone_lsa_reference.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "Prefrontal-AI/latent-sparse-attention"
# main as of 2026-11-02; bump deliberately, then re-run the parity suite.
REVISION = "906be195c1052aaa74f95852ad1917ce1926ba64"
# Repository path -> path under the target directory.
EXPECTED_FILES = {
    "LICENSE": "LICENSE",
    "src/lsa/budget.py": "lsa/budget.py",
    "src/lsa/core/reference.py": "lsa/core/reference.py",
    "src/lsa/core/shape.py": "lsa/core/shape.py",
    "src/lsa/core/ward.py": "lsa/core/ward.py",
}
EMPTY_PACKAGE_FILES = ("lsa/__init__.py", "lsa/core/__init__.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--target",
        default="third_party/lsa_reference",
        help="Local directory to vendor the reference into (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if target already populated.",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    wanted = [*EXPECTED_FILES.values(), *EMPTY_PACKAGE_FILES]
    if all((target / name).exists() for name in wanted) and not args.force:
        print(f"Reference already vendored at {target}; pass --force to refresh.")
        return 0

    print(f"Cloning {REPO}@{REVISION[:12]} -> {target}")
    for repo_path, local_path in EXPECTED_FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{repo_path}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        dest = target / local_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        print(f"  {dest}")
    for name in EMPTY_PACKAGE_FILES:
        (target / name).write_text("")

    print()
    print(f"Done. The parity tests import `lsa.core.reference` from {target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
