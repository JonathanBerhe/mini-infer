"""DeepSeek-V4 attention primitives.

Reusable across HCA (Heavily Compressed Attention) and CSA (Compressed
Sparse Attention). HCA wires up compressor + sink + grouped output;
CSA reuses the same primitives plus a Lightning Indexer + top-k.

V4 paper §2.3, formulas 20-27.
"""

from mini_infer.models.blocks.v4.compressor import TokenLevelCompressor
from mini_infer.models.blocks.v4.grouped_output import GroupedOutputProjection
from mini_infer.models.blocks.v4.lightning_indexer import LightningIndexer
from mini_infer.models.blocks.v4.sink import AttentionSink

__all__ = [
    "AttentionSink",
    "GroupedOutputProjection",
    "LightningIndexer",
    "TokenLevelCompressor",
]
