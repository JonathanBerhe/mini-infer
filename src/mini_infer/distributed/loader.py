"""Per-rank weight loading for tensor-parallel models.

Standard PyTorch loading uses `model.load_state_dict(state_dict)` which
requires the state_dict tensors to match the model parameters'
shapes one-for-one. Under TP that contract breaks: a model's
`q_proj.weight` has shape `(out_features // world_size, in_features)`,
while the safetensors file holds the un-sharded `(out_features,
in_features)` weight that we'd need to slice down to our rank's share.

This module provides `load_state_dict_with_tp(model, full_state_dict)`
that bridges the two. For every parameter in the model:

  - If the parent module exposes `load_full_weight(full_tensor)` (our
    TP-aware wrappers do), the function calls it. The TP layer then
    slices the full tensor to the rank's portion.
  - Otherwise, the function does an ordinary `param.copy_(full_tensor)`.

Returns the same `(missing, unexpected)` tuple as `state_dict`'s
`load_state_dict` so existing loader code can compare diagnostics.

At `world_size=1` the `load_full_weight` slicing is the identity
(start=0, end=out_features), so this function is bit-equivalent to the
plain `model.load_state_dict` path.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.distributed.embedding import VocabParallelEmbedding
from mini_infer.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mini_infer.models.blocks.v4 import AttentionSink, GroupedOutputProjection


def _find_parent_and_attr(model: nn.Module, param_name: str) -> tuple[nn.Module, str]:
    """Walk `param_name` (`a.b.c.weight`) and return `(model.a.b.c, "weight")`."""
    parts = param_name.split(".")
    parent: nn.Module = model
    for piece in parts[:-1]:
        parent = getattr(parent, piece)
    return parent, parts[-1]


def load_state_dict_with_tp(
    model: nn.Module, full_state_dict: dict[str, torch.Tensor]
) -> tuple[set[str], set[str]]:
    """Load `full_state_dict` into a model that may contain TP-aware layers.

    Slices each full tensor via the TP layer's `load_full_weight` (or
    `load_full_logits` / `load_full_wo_a` for sink / grouped-output
    parameters) where required.

    Args:
        model: a (possibly TP-aware) `nn.Module`. May contain
            `ColumnParallelLinear`, `RowParallelLinear`,
            `VocabParallelEmbedding`, `AttentionSink`,
            `GroupedOutputProjection`.
        full_state_dict: standard HF state_dict — un-sharded tensors keyed
            by HF parameter name. Every rank holds the full state_dict;
            future work will stream per-rank shards.

    Returns:
        `(missing, unexpected)` analogous to `nn.Module.load_state_dict`'s
        return: names present in the model but absent from `full_state_dict`,
        and names in `full_state_dict` that the model didn't consume.
    """
    consumed: set[str] = set()
    model_param_names = {n for n, _ in model.named_parameters()}

    # Walk parameters in deterministic order so the loader's behaviour is
    # reproducible across runs / ranks.
    for param_name in sorted(model_param_names):
        if param_name not in full_state_dict:
            continue
        full_tensor = full_state_dict[param_name]
        parent_module, attr_name = _find_parent_and_attr(model, param_name)

        if isinstance(parent_module, ColumnParallelLinear | RowParallelLinear):
            # `load_full_weight` handles both the slicing and the
            # weight/bias distinction.
            if attr_name == "weight":
                bias_key = param_name.rsplit(".", 1)[0] + ".bias"
                bias_tensor = full_state_dict.get(bias_key)
                parent_module.load_full_weight(full_tensor, bias_tensor)
                consumed.add(param_name)
                if bias_tensor is not None:
                    consumed.add(bias_key)
            elif attr_name == "bias":
                # Bias was already consumed when we processed `weight` for
                # this same parent module (or will be when we get there);
                # mark it consumed so the unexpected-key check stays clean.
                consumed.add(param_name)
        elif isinstance(parent_module, VocabParallelEmbedding):
            if attr_name == "weight":
                parent_module.load_full_weight(full_tensor)
                consumed.add(param_name)
        elif isinstance(parent_module, AttentionSink) and attr_name == "sink_logits":
            parent_module.load_full_logits(full_tensor)
            consumed.add(param_name)
        elif isinstance(parent_module, GroupedOutputProjection) and attr_name == "wo_a":
            parent_module.load_full_wo_a(full_tensor)
            consumed.add(param_name)
        else:
            # Replicated parameter — plain copy. Works for `nn.Linear`,
            # `nn.Embedding`, `nn.Parameter`, RMSNorm weights, etc.
            param = getattr(parent_module, attr_name)
            with torch.no_grad():
                param.copy_(full_tensor.to(param.dtype))
            consumed.add(param_name)

    missing = model_param_names - consumed
    # Don't flag tied params as missing — if `lm_head.weight is
    # embed_tokens.weight`, loading one populates both.
    missing = {n for n in missing if not _is_tied_to_consumed(model, n, consumed)}
    unexpected = set(full_state_dict.keys()) - consumed
    return missing, unexpected


def _is_tied_to_consumed(
    model: nn.Module, param_name: str, consumed: set[str]
) -> bool:
    """True iff `param_name`'s tensor is aliased to a parameter we already loaded.

    Used to keep the `missing` set from spuriously flagging tied
    `lm_head.weight` / `embed_tokens.weight` pairs.
    """
    parent_module, attr_name = _find_parent_and_attr(model, param_name)
    target_param = getattr(parent_module, attr_name)
    for other_name in consumed:
        other_parent, other_attr = _find_parent_and_attr(model, other_name)
        other_param = getattr(other_parent, other_attr, None)
        if other_param is target_param:
            return True
    return False
