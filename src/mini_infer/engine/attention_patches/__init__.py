"""Per-architecture attention patches that wire the paged kernel into HF models.

Adding support for a new model family means: write `<family>.py` here exporting a
patch function, then add an entry to REGISTRY. The dispatcher in
`engine/attention_patch.py` looks up the patch by matching the model's class name
against the keys here. No changes to the dispatcher needed.
"""

from collections.abc import Callable
from typing import Any

from mini_infer.engine.attention_patches.qwen2 import patch_qwen2

# Maps a substring that appears in the loaded model's class name to its patch fn.
# Order matters only if substrings could overlap; keep entries narrow.
REGISTRY: dict[str, Callable[[Any], None]] = {
    "Qwen2": patch_qwen2,
}

__all__ = ["REGISTRY", "patch_qwen2"]
