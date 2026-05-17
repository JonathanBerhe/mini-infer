"""Prefetch pinned HuggingFace models for the bit-parity CI workflow.

Reads `tests/_pinned_models.toml`, downloads each model at the exact
SHA recorded there via `huggingface_hub.snapshot_download`, and prints
a summary. The downloaded weights land in the default HF cache
(`~/.cache/huggingface/hub/...` or `$HF_HOME`). Subsequent
`from_pretrained(name)` calls in tests pick the cached revision up
automatically.

Run in CI before `pytest -m requires_model` so tests don't hit the
network during the run.

Modes:

  prefetch (default):
    Download each model at its pinned revision. Exits 0 on success;
    non-zero if any download fails.

  --check:
    Read the manifest and confirm every entry parses, without
    downloading. Useful as a sanity check on the manifest format in
    PR CI.

  --refresh:
    Query HF Hub for the current `main` SHA of each model and print
    a diff vs the manifest. Doesn't write — operators decide whether
    to bump pins after inspecting the diff. Run periodically to know
    when upstream has moved.

Usage:
    uv run python scripts/prefetch_pinned_models.py
    uv run python scripts/prefetch_pinned_models.py --check
    uv run python scripts/prefetch_pinned_models.py --refresh
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

_MANIFEST_PATH = Path(__file__).parent.parent / "tests" / "_pinned_models.toml"


@dataclass
class PinnedModel:
    name: str
    revision: str
    purpose: str


def _load_manifest(path: Path) -> list[PinnedModel]:
    """Read the TOML manifest into a list of `PinnedModel` records.

    Raises if the file is missing, malformed, or any entry lacks the
    required `name` / `revision` keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    raw = tomllib.loads(path.read_text())
    entries = raw.get("models", [])
    if not entries:
        raise ValueError(f"manifest has no [[models]] entries: {path}")
    out: list[PinnedModel] = []
    for i, entry in enumerate(entries):
        try:
            out.append(
                PinnedModel(
                    name=entry["name"],
                    revision=entry["revision"],
                    purpose=entry.get("purpose", ""),
                )
            )
        except KeyError as e:
            raise ValueError(f"manifest entry {i} missing key {e.args[0]!r}") from e
    return out


def _cmd_check(models: list[PinnedModel]) -> int:
    """Parse-only check: confirms the manifest is well-formed."""
    print(f"Manifest at {_MANIFEST_PATH}:")
    for m in models:
        print(f"  {m.name:50s}  {m.revision}")
        if m.purpose:
            print(f"  └─ {m.purpose}")
    print(f"\n{len(models)} models pinned. OK.")
    return 0


def _cmd_prefetch(models: list[PinnedModel]) -> int:
    """Download each pinned model at its exact SHA via snapshot_download."""
    errors: list[str] = []
    for m in models:
        print(f"Fetching {m.name} @ {m.revision[:12]}...")
        try:
            path = snapshot_download(repo_id=m.name, revision=m.revision)
            print(f"  -> {path}")
        except Exception as exc:
            errors.append(f"{m.name} @ {m.revision}: {exc!r}")
            print(f"  ERROR: {exc!r}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} model(s) failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"\n{len(models)} models prefetched successfully.")
    return 0


def _cmd_refresh(models: list[PinnedModel]) -> int:
    """Compare pinned SHAs against current HF Hub `main` SHAs; print diff.

    Does NOT modify the manifest. Operators inspect the diff and run
    bit-parity tests manually before bumping a pin.
    """
    api = HfApi()
    print(f"{'model':50s}  {'pinned':14s}  {'current':14s}  drift?")
    print(f"{'-' * 50}  {'-' * 14}  {'-' * 14}  ------")
    has_drift = False
    for m in models:
        try:
            info = api.model_info(m.name)
            current = info.sha or "(none)"
        except Exception as exc:
            print(f"{m.name:50s}  {m.revision[:12]:14s}  ERROR: {exc!r}")
            continue
        drift = "yes" if current != m.revision else "no"
        if drift == "yes":
            has_drift = True
        print(f"{m.name:50s}  {m.revision[:12]:14s}  {current[:12]:14s}  {drift}")
    if has_drift:
        print(
            "\nUpstream has moved for some pins. Inspect changes manually, "
            "re-run the bit-parity suite, and bump the manifest if outputs "
            "still match."
        )
    else:
        print("\nNo drift; all pins still match upstream main.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="parse the manifest only; don't download",
    )
    group.add_argument(
        "--refresh",
        action="store_true",
        help="query HF Hub for current SHAs and print a drift diff (no manifest write)",
    )
    args = parser.parse_args()

    models = _load_manifest(_MANIFEST_PATH)

    if args.check:
        return _cmd_check(models)
    if args.refresh:
        return _cmd_refresh(models)
    return _cmd_prefetch(models)


if __name__ == "__main__":
    sys.exit(main())
