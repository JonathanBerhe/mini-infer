"""Pull the DeepSeek-V4-Pro `inference/` directory into a vendored location.

Why vendor: the HCA forward in `tests/unit/test_v4_hca_parity.py` is the
parity oracle for our `HCAAttention` block. We need DeepSeek-AI's actual
inference code (not the paper formulas, which leave several details
ambiguous — sink semantics, output-RoPE direction). The repo at
`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference` is
~7 small `.py` files and a config; cloning them is free and offline-stable.

Idempotent: skips download if the target dir already has the expected
files. Re-run with `--force` to refresh.

Run with:
    uv run python scripts/clone_v4_reference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "deepseek-ai/DeepSeek-V4-Pro"
SUBDIR = "inference"
EXPECTED_FILES = (
    "inference/README.md",
    "inference/config.json",
    "inference/convert.py",
    "inference/generate.py",
    "inference/kernel.py",
    "inference/model.py",
    "inference/requirements.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--target",
        default="third_party/deepseek_v4_reference",
        help="Local directory to vendor `inference/` into (default: %(default)s)",
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

    already_have = all((target / Path(f).name).exists() for f in EXPECTED_FILES)
    if already_have and not args.force:
        print(f"Reference already vendored at {target}; pass --force to refresh.")
        for f in sorted((target / Path(p).name) for p in EXPECTED_FILES):
            print(f"  {f.relative_to(Path.cwd())}")
        return 0

    print(f"Cloning {REPO_ID}/{SUBDIR}/ -> {target}")
    for repo_path in EXPECTED_FILES:
        local = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
        dest = target / Path(repo_path).name
        dest.write_bytes(Path(local).read_bytes())
        print(f"  {dest.relative_to(Path.cwd())}")

    print()
    print(
        f"Done. Files vendored at {target.relative_to(Path.cwd())}; "
        f"the parity test imports from this path."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
