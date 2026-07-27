"""DSpark drafter: confidence-scheduled semi-autoregressive speculative decoding.

Owned port of the drafter half of DeepSeek's DSpark (DeepSeek-AI + Peking
University, 2026-06-27), released as the MIT-licensed `DeepSpec` codebase.
Scope, mechanics, and the alternatives considered are recorded in
`docs/decisions/ADR-027-dspark-drafter-port.md`.

Lives in `engine/`, not `models/`: the drafter never runs standalone (every
forward needs injected target hidden states), so it doesn't fit the
`ModelRegistry` contract that every `models/` entry satisfies.
"""

from mini_infer.engine.dspark.confidence_head import ConfidenceHead
from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.engine.dspark.drafter import Qwen3DSparkConfig, Qwen3DSparkDrafter
from mini_infer.engine.dspark.markov_head import VanillaMarkovHead

__all__ = [
    "ConfidenceHead",
    "DSparkDraftCache",
    "Qwen3DSparkConfig",
    "Qwen3DSparkDrafter",
    "VanillaMarkovHead",
]
