"""Vendor Moonshot's Kimi Linear reference modeling code for the parity tests.

Why vendor: Kimi Linear ships as trust_remote_code (no native transformers
class), so `tests/unit/test_kimi_linear_parity.py` imports the checkpoint's
own `modeling_kimi.py` as the oracle for our `KimiLinearForCausalLM`. The
revision is PINNED so the oracle can't drift under us; the FLA kernel
imports it needs are stubbed at test time with the naive reference
semantics (`tests/unit/_kimi_reference_helpers.py`), so nothing here
executes hub code outside these two vendored .py files.

Idempotent: skips download if the target dir already has the expected
files. Re-run with `--force` to refresh.

Run with:
    uv run python scripts/clone_kimi_linear_reference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
# main as of 2026-07-18; bump deliberately, then re-run the parity suite.
REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
EXPECTED_FILES = (
    "modeling_kimi.py",
    "configuration_kimi.py",
    "config.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--target",
        default="third_party/kimi_linear_reference",
        help="Local directory to vendor the reference into (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if target already populated.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed; pip install huggingface-hub", file=sys.stderr)
        return 1

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    already_have = all((target / name).exists() for name in EXPECTED_FILES)
    if already_have and not args.force:
        print(f"Reference already vendored at {target}; pass --force to refresh.")
        return 0

    print(f"Cloning {REPO_ID}@{REVISION[:12]} -> {target}")
    for name in EXPECTED_FILES:
        local = hf_hub_download(repo_id=REPO_ID, filename=name, revision=REVISION)
        dest = target / name
        dest.write_bytes(Path(local).read_bytes())
        print(f"  {dest}")
    # The reference uses relative imports (`from .configuration_kimi import
    # ...`), so it must be importable as a package.
    (target / "__init__.py").write_text("")

    print()
    print(f"Done. The parity tests import `kimi_linear_reference.modeling_kimi` from {target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
