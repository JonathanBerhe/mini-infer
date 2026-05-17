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

# `mini_infer.models.blocks.v4` cannot be imported at module-load time
# without inducing a circular import — `deepseek_v4.py` imports this
# loader, and the `models` package eagerly registers it at first import.
# Resolve V4 sink / grouped-output lazily inside the function.


def _find_parent_and_attr(model: nn.Module, param_name: str) -> tuple[nn.Module, str]:
    """Walk `param_name` (`a.b.c.weight`) and return `(model.a.b.c, "weight")`."""
    parts = param_name.split(".")
    parent: nn.Module = model
    for piece in parts[:-1]:
        parent = getattr(parent, piece)
    return parent, parts[-1]


def load_state_dict_with_tp(
    model: nn.Module,
    full_state_dict: dict[str, torch.Tensor],
    *,
    target_device: torch.device | str | None = None,
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
    # Lazy import: `models.blocks.v4` triggers registry of `deepseek_v4`
    # which imports this loader, so module-level import would deadlock.
    from mini_infer.models.blocks.v4 import AttentionSink, GroupedOutputProjection

    # If the caller passed a target device, we move each loaded tensor
    # there before constructing the replacement Parameter. Used by the
    # V4-Flash smoke: state_dict lives in CPU RAM (158 GB doesn't fit on
    # a single GPU), and load_full_weight slices the rank's share into
    # GPU HBM (~80 GB per rank with TP=2). For non-meta callers, the
    # existing param's device wins and target_device is ignored.
    device_obj = torch.device(target_device) if target_device is not None else None

    def _to_target(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device_obj) if device_obj is not None else t

    consumed: set[str] = set()
    loaded_param_ids: set[int] = set()

    # Walk state_dict keys in deterministic order. For each key we try to
    # find the corresponding model parameter and dispatch to its TP-aware
    # load helper (or a plain copy for replicated params). State_dict keys
    # that don't correspond to any model parameter are returned as
    # `unexpected`; model parameters not loaded by ANY state_dict key are
    # returned as `missing`.
    #
    # Iterating over state_dict keys (rather than `named_parameters`) is
    # important for tied weights: under `tie_word_embeddings`,
    # `lm_head.weight` and `model.embed_tokens.weight` are the SAME
    # Parameter object, so `named_parameters()` only yields one of them.
    # An HF checkpoint that ships both names would otherwise show the
    # second as "unexpected". Walking state_dict keys instead lets us
    # process both names — loading the same Parameter twice with bit-
    # identical data is idempotent.
    for sd_key in sorted(full_state_dict.keys()):
        full_tensor = full_state_dict[sd_key]
        try:
            parent_module, attr_name = _find_parent_and_attr(model, sd_key)
        except AttributeError:
            # Path doesn't resolve — definitely unexpected.
            continue
        param = getattr(parent_module, attr_name, None)
        if param is None:
            continue

        if isinstance(parent_module, ColumnParallelLinear | RowParallelLinear):
            if attr_name == "weight":
                bias_key = sd_key.rsplit(".", 1)[0] + ".bias"
                bias_tensor = full_state_dict.get(bias_key)
                parent_module.load_full_weight(full_tensor, bias_tensor, target_device=device_obj)
                consumed.add(sd_key)
                loaded_param_ids.add(id(parent_module.weight))
                if bias_tensor is not None and parent_module.bias is not None:
                    consumed.add(bias_key)
                    loaded_param_ids.add(id(parent_module.bias))
            elif attr_name == "bias":
                # Bias was (or will be) loaded alongside `weight`; mark
                # consumed without re-loading.
                consumed.add(sd_key)
                if parent_module.bias is not None:
                    loaded_param_ids.add(id(parent_module.bias))
        elif isinstance(parent_module, VocabParallelEmbedding):
            if attr_name == "weight":
                parent_module.load_full_weight(full_tensor, target_device=device_obj)
                consumed.add(sd_key)
                loaded_param_ids.add(id(parent_module.weight))
        elif isinstance(parent_module, AttentionSink) and attr_name == "sink_logits":
            parent_module.load_full_logits(full_tensor, target_device=device_obj)
            consumed.add(sd_key)
            loaded_param_ids.add(id(parent_module.sink_logits))
        elif isinstance(parent_module, GroupedOutputProjection) and attr_name == "wo_a":
            parent_module.load_full_wo_a(full_tensor, target_device=device_obj)
            consumed.add(sd_key)
            loaded_param_ids.add(id(parent_module.wo_a))
        else:
            # Replicated parameter — works for `nn.Linear`, `nn.Embedding`,
            # `nn.Parameter`, RMSNorm weights, etc. Handles two paths:
            # meta-device construction (replace the parameter) and
            # already-allocated (in-place copy). The meta path is required
            # for multi-hundred-billion-parameter models where allocating
            # random init for every layer at construction would OOM.
            if not isinstance(param, torch.Tensor):
                continue  # not a tensor attribute; can't be in state_dict
            if param.is_meta:
                # Meta-mode: preserve the source tensor's dtype, but route
                # to `target_device` if the caller provided one (so the
                # CPU-resident state_dict can yield per-GPU model params).
                tensor_for_load = _to_target(full_tensor.contiguous())
                new_param = nn.Parameter(tensor_for_load, requires_grad=False)
                setattr(parent_module, attr_name, new_param)
                loaded_param_ids.add(id(new_param))
            else:
                tensor_for_load = full_tensor.to(param.dtype).contiguous()
                with torch.no_grad():
                    param.copy_(tensor_for_load)
                loaded_param_ids.add(id(param))
            consumed.add(sd_key)

    missing = {
        name for name, param in model.named_parameters() if id(param) not in loaded_param_ids
    }
    unexpected = set(full_state_dict.keys()) - consumed
    return missing, unexpected
