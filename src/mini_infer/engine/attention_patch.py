"""Dispatcher that picks the right per-architecture attention patch for the loaded model.

The engine stays model-agnostic at the API layer; the kernel fast path is per
architecture. Adding a new model family means dropping a file in
`engine/attention_patches/` and registering it in that package's REGISTRY — no
changes here.
"""

import logging
from typing import Any

from mini_infer.engine.attention_patches import REGISTRY

logger = logging.getLogger(__name__)


def patch_model_attention(model: Any) -> bool:
    """Apply the matching architecture patch; return True if one was applied."""
    cls_name = model.__class__.__name__
    for arch_key, patch_fn in REGISTRY.items():
        if arch_key in cls_name:
            patch_fn(model)
            logger.info("Patched %s attention with %s", cls_name, patch_fn.__name__)
            return True
    logger.info(
        "No paged-attention patch available for %s; falling back to materialization", cls_name
    )
    return False
