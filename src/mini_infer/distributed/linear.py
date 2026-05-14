"""Column- and row-parallel linear layers (Megatron-style).

These are the workhorses of tensor parallelism. The naming convention
("column" / "row") refers to which axis of the *weight matrix* is sharded.

Recall PyTorch stores `nn.Linear.weight` as `[out_features, in_features]`,
and `functional.linear(x, W)` computes `x @ W.T`. So:

  - `ColumnParallelLinear` shards the `out_features` axis (each rank holds
    a slice of output columns of `x @ W.T`, equivalently a slice of the
    rows of `W` itself).
  - `RowParallelLinear` shards the `in_features` axis (each rank holds a
    slice of input columns / weight columns; partial outputs are summed
    via `all_reduce`).

The standard pairing in Transformers is:
  (col-parallel Q/K/V) -> attention -> (row-parallel O)
  (col-parallel gate/up) -> activation -> (row-parallel down)

That gives exactly one all-reduce per attention block and one per FFN.
The col-parallel layer outputs naturally feed the row-parallel layer
without an intervening collective, because the row-parallel layer is
*designed* to consume sharded input.

Single-device behaviour
-----------------------
At `world_size=1`, both layers reduce to a plain `nn.Linear`: no slicing,
no collectives. This is the contract that keeps the existing 393
single-device unit tests bit-identical.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from mini_infer.distributed.comm import all_gather_along_dim, all_reduce_sum
from mini_infer.distributed.group import get_rank, get_world_size


def _split_size(total: int, world_size: int, axis_name: str) -> int:
    """Validate that `total` divides evenly across `world_size` and return
    the per-rank share. We require divisibility because un-even splits
    complicate every downstream loader and aren't worth the complexity at
    this stage of the project."""
    if total % world_size != 0:
        raise ValueError(f"{axis_name}={total} must be divisible by world_size={world_size}")
    return total // world_size


class ColumnParallelLinear(nn.Linear):
    """Linear layer sharded along the output dimension.

    Per-rank weight shape: `[out_features // world_size, in_features]`
    Per-rank output shape: `[..., out_features // world_size]`

    Subclasses `nn.Linear` so that anywhere the codebase asks
    `isinstance(m, nn.Linear)` (e.g. the int8 quantizer's module-walk),
    a TP-aware linear at `world_size=1` is recognised and treated as
    the plain `nn.Linear` it is bit-equivalent to. Quantizing under
    `world_size > 1` doesn't make sense yet — we'll address that when
    we ship a TP-aware Int8Linear.

    Args:
        in_features: input dim (replicated; same on every rank).
        out_features: full output dim (sharded; per-rank holds 1/world_size).
        bias: whether to learn a bias (sharded same as the weight rows).
        gather_output: if True, all-gather the output before returning so
            callers see the full activation. Default False, because the
            standard pattern feeds a `RowParallelLinear` next which expects
            sharded input. Set True for "stop here and use the result"
            cases (e.g. an LM head followed by sampling).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        gather_output: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        world_size = get_world_size()
        out_features_per_rank = _split_size(out_features, world_size, "out_features")
        # `nn.Linear.__init__` allocates `weight` of shape
        # `[out_features_per_rank, in_features]` and `bias` of shape
        # `[out_features_per_rank]`. We then overwrite `out_features` to
        # the FULL dim so external callers (loaders, repr) see the
        # logical / un-sharded value.
        super().__init__(
            in_features,
            out_features_per_rank,
            bias=bias,
            dtype=dtype,
        )
        # Re-record the full output dim for the loader / external API.
        self.out_features = out_features
        self.out_features_per_rank = out_features_per_rank
        self.gather_output = gather_output
        self.world_size = world_size
        self.rank = get_rank()

    def load_full_weight(
        self,
        full_weight: torch.Tensor,
        full_bias: torch.Tensor | None = None,
        *,
        target_device: torch.device | str | None = None,
    ) -> None:
        """Slice the rank's portion out of the full (un-sharded) weight.

        `target_device`: when the model was constructed on meta, the
        sliced tensor is moved here before being wrapped as a Parameter.
        Used by the V4-Flash loader: the full state_dict lives on CPU
        and only the rank's slice ever touches GPU HBM.
        """
        if full_weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"full_weight shape {tuple(full_weight.shape)} does not match expected "
                f"({self.out_features}, {self.in_features})"
            )
        start = self.rank * self.out_features_per_rank
        end = start + self.out_features_per_rank
        if self.weight.is_meta:
            sliced_weight = full_weight[start:end].contiguous()
            if target_device is not None:
                sliced_weight = sliced_weight.to(device=target_device)
            self.weight = nn.Parameter(sliced_weight, requires_grad=False)
        else:
            sliced_weight = full_weight[start:end].to(self.weight.dtype).contiguous()
            with torch.no_grad():
                self.weight.copy_(sliced_weight)
        if full_bias is not None:
            if self.bias is None:
                raise ValueError("full_bias provided but layer was constructed with bias=False")
            if full_bias.shape != (self.out_features,):
                raise ValueError(
                    f"full_bias shape {tuple(full_bias.shape)} does not match expected "
                    f"({self.out_features},)"
                )
            if self.bias.is_meta:
                sliced_bias = full_bias[start:end].contiguous()
                if target_device is not None:
                    sliced_bias = sliced_bias.to(device=target_device)
                self.bias = nn.Parameter(sliced_bias, requires_grad=False)
            else:
                sliced_bias = full_bias[start:end].to(self.bias.dtype).contiguous()
                with torch.no_grad():
                    self.bias.copy_(sliced_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x @ W_local.T (+ bias_local)`, optionally all-gathered.

        Input is replicated across ranks. Output is sharded along the last
        dim (or all-gathered to be replicated again if `gather_output`).
        """
        local_output = functional.linear(x, self.weight, self.bias)
        if self.gather_output:
            return all_gather_along_dim(local_output, dim=-1)
        return local_output


class RowParallelLinear(nn.Linear):
    """Linear layer sharded along the input dimension.

    Per-rank weight shape: `[out_features, in_features // world_size]`
    Per-rank input shape:  `[..., in_features // world_size]` (sharded by
        the upstream column-parallel layer).
    Per-rank output shape: `[..., out_features]` (replicated, after
        all-reduce).

    Subclasses `nn.Linear` for the same isinstance-detection reason as
    `ColumnParallelLinear`. At `world_size=1` it's bit-equivalent to
    plain `nn.Linear`.

    Args:
        in_features: full input dim (sharded; per-rank holds 1/world_size).
        out_features: output dim (replicated; same on every rank).
        bias: whether to learn a bias. Replicated on every rank but only
            *added* on rank 0 to avoid double-counting it through the
            sum-reduce. (Megatron uses the same trick.)
        input_is_parallel: if True (default), input is already sharded
            along the last dim by an upstream col-parallel layer. If False,
            we scatter it ourselves; at world_size==1 these are equivalent.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        input_is_parallel: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        world_size = get_world_size()
        in_features_per_rank = _split_size(in_features, world_size, "in_features")
        # `nn.Linear.__init__` allocates `weight` of shape
        # `[out_features, in_features_per_rank]` (which is what we want
        # for row-parallel) and `bias` of shape `[out_features]`.
        super().__init__(
            in_features_per_rank,
            out_features,
            bias=bias,
            dtype=dtype,
        )
        # Re-record the full input dim for the loader / external API.
        self.in_features = in_features
        self.in_features_per_rank = in_features_per_rank
        self.input_is_parallel = input_is_parallel
        self.world_size = world_size
        self.rank = get_rank()

    def load_full_weight(
        self,
        full_weight: torch.Tensor,
        full_bias: torch.Tensor | None = None,
        *,
        target_device: torch.device | str | None = None,
    ) -> None:
        """Slice the rank's input-axis portion of the weight; bias is replicated."""
        if full_weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"full_weight shape {tuple(full_weight.shape)} does not match expected "
                f"({self.out_features}, {self.in_features})"
            )
        start = self.rank * self.in_features_per_rank
        end = start + self.in_features_per_rank
        if self.weight.is_meta:
            sliced_weight = full_weight[:, start:end].contiguous()
            if target_device is not None:
                sliced_weight = sliced_weight.to(device=target_device)
            self.weight = nn.Parameter(sliced_weight, requires_grad=False)
        else:
            sliced_weight = full_weight[:, start:end].to(self.weight.dtype).contiguous()
            with torch.no_grad():
                self.weight.copy_(sliced_weight)
        if full_bias is not None:
            if self.bias is None:
                raise ValueError("full_bias provided but layer was constructed with bias=False")
            if full_bias.shape != (self.out_features,):
                raise ValueError(
                    f"full_bias shape {tuple(full_bias.shape)} does not match expected "
                    f"({self.out_features},)"
                )
            # Bias is the *full* unsharded bias on every rank; we only
            # add it on rank 0 in forward to avoid double-counting after
            # the all-reduce.
            if self.bias.is_meta:
                bias_for_load = full_bias.contiguous()
                if target_device is not None:
                    bias_for_load = bias_for_load.to(device=target_device)
                self.bias = nn.Parameter(bias_for_load, requires_grad=False)
            else:
                full_bias_typed = full_bias.to(self.bias.dtype).contiguous()
                with torch.no_grad():
                    self.bias.copy_(full_bias_typed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-rank partial output, then all-reduce.

        If `input_is_parallel` is False, we'd need to scatter `x` first,
        but for a self-consistent TP stack the upstream layer always shards
        the activation, so this stays fast.
        """
        if not self.input_is_parallel and self.world_size > 1:
            # Scatter `x` along the last dim (rank `r` keeps slice
            # `[r*per:(r+1)*per]`). Not on the hot path of the typical
            # column->row pairing, included for completeness.
            slice_size = x.shape[-1] // self.world_size
            start = self.rank * slice_size
            x = x[..., start : start + slice_size]

        # Bias is added only on rank 0; the all-reduce below would otherwise
        # add it `world_size` times.
        local_bias = self.bias if (self.bias is not None and self.rank == 0) else None
        local_output = functional.linear(x, self.weight, local_bias)
        return all_reduce_sum(local_output)
