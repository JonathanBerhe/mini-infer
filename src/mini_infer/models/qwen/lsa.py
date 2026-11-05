"""Latent Sparse Attention (LSA) key selection for the Qwen hybrid's full-attention layers.

LSA (Kubicek et al., arXiv 2610.42147) decides which earlier keys a query
reads. The attention over the selected keys is ordinary softmax attention, and
the KV cache is unchanged. This module holds the selection, for the distilled
flat variant the authors serve on Qwen3.8-27B, and `attend_selected`, the
attention of each query over its own selected keys. The semantics follow
the authors' plain-PyTorch reference (`lsa/core/reference.py` and
`lsa/core/ward.py` in `Prefrontal-AI/latent-sparse-attention`, Apache-2.0),
which `tests/unit/models/qwen/test_lsa_parity.py` compares against.

The pipeline, per full-attention layer:

  1. Index input. The key is the stored key (after `k_norm` and partial
     rotary) with its rotary prefix dropped, all KV heads side by side. The
     query sums, in fp32, the query heads that share each KV head over the
     same dims. Rotary covers only the leading dims, so these dims carry no
     position.
  2. Projection. Two learned `[input_dim, index_dim]` maps take both inputs
     to a small index space, and each row is normalized to unit length.
  3. Buckets. Each closed block of `window_size` index keys is cut into pools
     of `pool_size` consecutive keys. Ward merging joins pools until the
     block has `window_size // bucket_size` buckets of at most `bucket_cap`
     keys. A bucket keeps the normalized sum of its keys (its centroid) and
     the positions of its keys. A bucket never changes after its block closes.
  4. Selection. A query scores the centroids of the blocks closed before its
     own block, keeps the `top_p` best buckets, and takes their keys best
     bucket first until it holds `top_k`. It skips keys that it reads anyway:
     the first `sink_size` positions and the `window_size` most recent ones.
     The window always covers the query's own open block, because a cluster
     block is `window_size` keys long.

When `top_p` buckets cover every visible bucket at that length, as they do
with the served sizes, a query at position `top_k + window_size +
sink_size - 1` or lower reads every earlier key. Below that length LSA then
equals dense attention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

# A tensor name in the authors' map files: decoder layer index, then which map.
_MAP_NAME = re.compile(r"^(\d+)\.(W_Q_idx|W_K_idx)$")

# Merge rounds before Ward gives up. Each round at least halves the merges
# left in practice, so a 512-key block needs well under 20.
_MAX_WARD_ROUNDS = 256


@dataclass(frozen=True)
class LsaConfig:
    """Selection sizes. The defaults are the authors' served flat design.

    Attributes:
        window_size: recent keys every query reads, its own included. It is
            also the number of keys in one cluster block.
        sink_size: first keys every query reads.
        top_k: most long-range keys a query reads besides the window and sinks.
        top_p: buckets a query picks before expanding them to keys.
        index_dim: dims of the index space.
        pool_size: consecutive keys summed into one pool. Ward never splits a pool.
        bucket_size: average keys per bucket, so a block ends with
            `window_size // bucket_size` buckets.
        bucket_cap: most keys one bucket may hold.
    """

    window_size: int = 512
    sink_size: int = 4
    top_k: int = 4096
    top_p: int = 1024
    index_dim: int = 64
    pool_size: int = 4
    bucket_size: int = 8
    bucket_cap: int = 16

    def __post_init__(self) -> None:
        sizes = (self.window_size, self.top_k, self.top_p, self.index_dim)
        if min(*sizes, self.pool_size, self.bucket_size, self.bucket_cap) < 1:
            raise ValueError(f"LSA sizes must be positive, got {self}")
        if self.sink_size < 0:
            raise ValueError(f"sink_size must be non-negative, got {self.sink_size}")
        if self.bucket_size % self.pool_size or self.window_size % self.bucket_size:
            raise ValueError(
                f"pool_size {self.pool_size} must divide bucket_size {self.bucket_size}, "
                f"which must divide window_size {self.window_size}"
            )
        # Below twice the average, Ward can run out of pairs the cap allows
        # before the block reaches its bucket count.
        if self.bucket_cap < 2 * self.bucket_size:
            raise ValueError(
                f"bucket_cap must be at least 2 * bucket_size = {2 * self.bucket_size}, "
                f"got {self.bucket_cap}"
            )

    @property
    def block_bucket_count(self) -> int:
        """Buckets one closed block ends with."""
        return self.window_size // self.bucket_size

    @property
    def list_width(self) -> int:
        """Most keys one query reads: the expansion, the window and the sinks."""
        return self.top_k + self.window_size + self.sink_size


def index_key_input(key: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """The indexer's key input: each KV head's position-free dims, side by side.

    Args:
        key: `(..., num_kv_heads, head_dim)`, after `k_norm` and partial rotary.
        rotary_dim: leading dims the rotary embedding turned.

    Returns:
        `(..., num_kv_heads * (head_dim - rotary_dim))` bf16.
    """
    return key[..., rotary_dim:].flatten(-2).to(torch.bfloat16)


def index_query_input(query: torch.Tensor, num_kv_heads: int, rotary_dim: int) -> torch.Tensor:
    """The indexer's query input: per KV head, the sum of its query heads.

    Query head `h` reads KV head `h // group`, the layout of the checkpoint's
    grouped-query weights. The sum runs in fp32 over the position-free dims.

    Args:
        query: `(..., num_attention_heads, head_dim)`, after `q_norm` and rotary.
        num_kv_heads: KV heads the query heads are grouped onto.
        rotary_dim: leading dims the rotary embedding turned.

    Returns:
        `(..., num_kv_heads * (head_dim - rotary_dim))` fp32.
    """
    *lead, heads, head_dim = query.shape
    grouped = query[..., rotary_dim:].reshape(
        *lead, num_kv_heads, heads // num_kv_heads, head_dim - rotary_dim
    )
    return grouped.sum(-2, dtype=torch.float32).flatten(-2)


class IndexProjection(nn.Module):
    """One layer's two maps into the index space, in the served numerics.

    The maps and the matmuls are bf16. An index key is normalized in fp32 and
    stored bf16. An index query is cast to bf16 before its matmul and stays
    fp32 after the normalization.

    The maps are non-persistent buffers: they move with the model on `.to()`
    but stay out of its `state_dict`, because they come from their own file
    and not from the checkpoint. A whole-model dtype cast can widen them, and
    the matmuls cast them back to bf16, which is exact.
    """

    query_map: torch.Tensor
    key_map: torch.Tensor

    def __init__(self, query_map: torch.Tensor, key_map: torch.Tensor) -> None:
        """`query_map` and `key_map` are `[input_dim, index_dim]`."""
        super().__init__()
        if query_map.shape != key_map.shape or query_map.dim() != 2:
            raise ValueError(
                f"the maps must share one [input_dim, index_dim] shape, got "
                f"{tuple(query_map.shape)} and {tuple(key_map.shape)}"
            )
        self.register_buffer("query_map", query_map.to(torch.bfloat16), persistent=False)
        self.register_buffer("key_map", key_map.to(torch.bfloat16), persistent=False)

    def index_keys(self, key_input: torch.Tensor) -> torch.Tensor:
        """`(..., input_dim)` -> `(..., index_dim)` bf16 unit rows."""
        projected = key_input.to(torch.bfloat16) @ self.key_map.to(torch.bfloat16)
        return functional.normalize(projected.float(), dim=-1).to(torch.bfloat16)

    def index_queries(self, query_input: torch.Tensor) -> torch.Tensor:
        """`(..., input_dim)` -> `(..., index_dim)` fp32 unit rows."""
        projected = query_input.to(torch.bfloat16) @ self.query_map.to(torch.bfloat16)
        return functional.normalize(projected.float(), dim=-1)


def load_index_maps(path: str | Path) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Read a map file in the authors' export format.

    The file is a flat dict that maps `"<layer>.W_Q_idx"` and
    `"<layer>.W_K_idx"` to `[input_dim, index_dim]` tensors, keyed by decoder
    layer index. It is loaded with `weights_only=True`, so it can hold
    tensors only, never code. The model's `enable_lsa` checks the shapes and
    that every full-attention layer is present.

    Returns:
        decoder layer index -> `(query_map, key_map)`, fp32 on the host.

    Raises:
        ValueError: if the file is not such a dict, or a layer has only one map.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not blob:
        raise ValueError(f"{path} is not a map file: expected a dict of '<layer>.W_Q_idx' tensors")
    found: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in blob.items():
        match = _MAP_NAME.match(str(name))
        if match is None or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path} holds {name!r}; a map file holds '<layer>.W_Q_idx' / W_K_idx")
        found.setdefault(int(match.group(1)), {})[match.group(2)] = tensor.float()
    lonely = sorted(layer for layer, pair in found.items() if len(pair) != 2)
    if lonely:
        raise ValueError(f"{path} holds only one of W_Q_idx and W_K_idx for layers {lonely}")
    return {layer: (pair["W_Q_idx"], pair["W_K_idx"]) for layer, pair in sorted(found.items())}


@dataclass
class BlockBuckets:
    """The buckets of a set of closed blocks.

    Attributes:
        centroids: `(blocks, block_bucket_count, index_dim)` fp32 unit rows.
        members: `(blocks, block_bucket_count, bucket_cap)` int32 absolute key
            positions in position order, -1 past a bucket's size.
        sizes: `(blocks, block_bucket_count)` int32 keys per bucket.
    """

    centroids: torch.Tensor
    members: torch.Tensor
    sizes: torch.Tensor


def ward_labels(
    pool_keys: torch.Tensor, pool_sizes: torch.Tensor, bucket_count: int, bucket_cap: int
) -> torch.Tensor:
    """Merge each block's pools into `bucket_count` buckets and label each pool.

    A bucket is the sum of its keys. Merging buckets `a` and `b` costs
    `|a| + |b| - |a + b|`, the length the union loses. Each round, every
    bucket finds its cheapest partner among the merges the cap allows (a tie
    goes to the lower index). Pairs that are each other's cheapest merge into
    their lower pool. In the round that would overshoot the count, the
    cheapest of those pairs merge first.

    Args:
        pool_keys: `(blocks, pools, index_dim)` fp32 pool sums.
        pool_sizes: `(blocks, pools)` fp32 keys per pool.
        bucket_count: buckets each block ends with.
        bucket_cap: most keys a merged bucket may hold.

    Returns:
        `(blocks, pools)` int64 labels from 0, numbered in the order of each
        bucket's first pool.

    Raises:
        ValueError: if a block runs out of allowed merges before it reaches
            `bucket_count`.
    """
    blocks, pools, _ = pool_keys.shape
    device = pool_keys.device
    sums = pool_keys.clone()
    sizes = pool_sizes.clone()
    alive = torch.ones(blocks, pools, dtype=torch.bool, device=device)
    parent = torch.arange(pools, device=device).expand(blocks, pools).clone()
    index = torch.arange(pools, device=device).expand(blocks, pools)
    self_pair = torch.eye(pools, dtype=torch.bool, device=device)
    for _ in range(_MAX_WARD_ROUNDS):
        merges_left = alive.sum(1) - bucket_count
        if not bool((merges_left > 0).any()):
            break
        gram = torch.bmm(sums, sums.transpose(1, 2))
        squares = gram.diagonal(dim1=1, dim2=2)
        lengths = squares.clamp_min(0).sqrt()
        merged = (squares[:, :, None] + squares[:, None, :] + 2 * gram).clamp_min(0).sqrt()
        cost = lengths[:, :, None] + lengths[:, None, :] - merged
        allowed = alive[:, :, None] & alive[:, None, :] & ~self_pair
        allowed &= sizes[:, :, None] + sizes[:, None, :] <= bucket_cap
        best_cost, partner = cost.masked_fill(~allowed, float("inf")).min(2)
        mutual = (
            (best_cost < float("inf")) & (partner.gather(1, partner) == index) & (index < partner)
        )
        rank = best_cost.masked_fill(~mutual, float("inf")).argsort(dim=1, stable=True).argsort(1)
        mutual &= rank < merges_left.clamp_min(0)[:, None]
        if not bool(mutual.any()):
            break
        block_ids, kept = mutual.nonzero(as_tuple=True)
        absorbed = partner[block_ids, kept]
        sums[block_ids, kept] += sums[block_ids, absorbed]
        sizes[block_ids, kept] += sizes[block_ids, absorbed]
        alive[block_ids, absorbed] = False
        parent[block_ids, absorbed] = kept
    if bool((alive.sum(1) > bucket_count).any()):
        raise ValueError(f"a block stopped short of {bucket_count} buckets under cap {bucket_cap}")
    # Each absorbed pool points at the pool it merged into. Doubling the jump
    # reaches every chain's surviving pool in log2(pools) steps.
    for _ in range(max(pools, 2).bit_length()):
        parent = parent.gather(1, parent)
    number = torch.cumsum(alive.long(), 1) - 1
    return number.gather(1, parent)


def cluster_blocks(
    index_keys: torch.Tensor, block_starts: torch.Tensor, cfg: LsaConfig
) -> BlockBuckets:
    """Cluster closed blocks of index keys into buckets.

    Args:
        index_keys: `(blocks, window_size, index_dim)` fp32 or bf16 unit rows.
        block_starts: `(blocks,)` int64 absolute position of each block's first key.
        cfg: selection sizes.

    Returns:
        The blocks' buckets.
    """
    blocks, block_len, index_dim = index_keys.shape
    if block_len != cfg.window_size:
        raise ValueError(f"a cluster block holds {cfg.window_size} keys, got {block_len}")
    device = index_keys.device
    bucket_count, cap = cfg.block_bucket_count, cfg.bucket_cap
    keys = index_keys.float()
    pools = keys.reshape(blocks, block_len // cfg.pool_size, cfg.pool_size, index_dim).sum(2)
    pool_sizes = torch.full(pools.shape[:2], float(cfg.pool_size), device=device)
    labels = ward_labels(pools, pool_sizes, bucket_count, cap)
    key_labels = labels.repeat_interleave(cfg.pool_size, dim=1)  # (blocks, window_size)

    # Centroids: the normalized sum of each bucket's keys, summed key by key.
    flat_labels = key_labels + bucket_count * torch.arange(blocks, device=device)[:, None]
    sums = torch.zeros(blocks * bucket_count, index_dim, device=device)
    sums.index_add_(0, flat_labels.flatten(), keys.reshape(-1, index_dim))
    centroids = sums / sums.norm(dim=1, keepdim=True).clamp_min(1e-30)

    # Members: order the keys by (bucket, position), then each key's slot in
    # its bucket is its rank after the bucket's first key.
    local = torch.arange(block_len, device=device)
    order = torch.argsort(key_labels * block_len + local, dim=1)
    sorted_labels = key_labels.gather(1, order)
    sizes = torch.zeros(blocks, bucket_count, dtype=torch.long, device=device)
    sizes.scatter_add_(1, key_labels, torch.ones_like(key_labels))
    firsts = torch.cumsum(sizes, 1) - sizes
    slots = local - firsts.gather(1, sorted_labels)
    members = torch.full((blocks, bucket_count, cap), -1, dtype=torch.int32, device=device)
    block_ids = torch.arange(blocks, device=device)[:, None].expand(blocks, block_len)
    members[block_ids, sorted_labels, slots] = (order + block_starts.view(blocks, 1)).to(
        torch.int32
    )
    return BlockBuckets(
        centroids=centroids.view(blocks, bucket_count, index_dim),
        members=members,
        sizes=sizes.to(torch.int32),
    )


def select_keys(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    centroids: torch.Tensor,
    members: torch.Tensor,
    cfg: LsaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the keys each query reads.

    Each row `b` is one request with its own buckets: those of its closed
    blocks, block after block. A query sees only the buckets of the blocks
    closed before its own block.

    Args:
        index_queries: `(B, T, index_dim)` unit rows.
        positions: `(B, T)` int64 absolute position of each query.
        centroids: `(B, rows, index_dim)` bucket centroids.
        members: `(B, rows, bucket_cap)` int32 bucket key positions, -1 padded.
        cfg: selection sizes.

    Returns:
        `(B, T, cfg.list_width)` int32 key positions, -1 padded: the expanded
        buckets best first, then the window in ascending order, then the
        sinks. And `(B, T)` int32 keys per query.
    """
    batch, _, _ = index_queries.shape
    rows, cap = members.shape[1], members.shape[2]
    device = index_queries.device
    window, sinks = cfg.window_size, cfg.sink_size
    positions = positions.long()

    # Score every bucket, hide the ones the query cannot see, and keep the
    # best `top_p`. A stable sort sends a tie to the lower bucket, on any
    # platform.
    scores = torch.matmul(index_queries.float(), centroids.float().transpose(1, 2))
    visible = (positions // window * cfg.block_bucket_count).clamp(max=rows)
    seen = torch.arange(rows, device=device) < visible[..., None]
    scores = scores.masked_fill(~seen, float("-inf"))
    picked = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., : cfg.top_p]
    picked_seen = picked < visible[..., None]

    # Expand the picked buckets best first, drop keys the window or the sinks
    # already hold, and keep the first `top_k`. The last bucket is cut short.
    batch_ids = torch.arange(batch, device=device).view(batch, 1, 1)
    candidates = members[batch_ids, picked].long().flatten(2)  # (B, T, picked * cap)
    window_start = (positions + 1 - window).clamp_min(0)
    usable = picked_seen.repeat_interleave(cap, dim=-1)
    usable &= (candidates >= sinks) & (candidates < window_start[..., None])
    running = usable.long().cumsum(-1)
    taken = usable & (running <= cfg.top_k)
    expanded_count = taken.sum(-1)

    # Pack the taken keys to the left. Untaken ones land in a spare last
    # column that is then dropped.
    dest = torch.where(taken, running - 1, cfg.list_width)
    expansion = torch.full(
        (*positions.shape, cfg.list_width + 1), -1, dtype=torch.long, device=device
    )
    expansion.scatter_(-1, dest, torch.where(taken, candidates, -1))
    expansion = expansion[..., : cfg.list_width]

    # Lay out [expansion | window | sinks | -1 padding] on one grid.
    window_count = positions + 1 - window_start
    sink_count = window_start.clamp(max=sinks)
    col = torch.arange(cfg.list_width, device=device)
    after_expansion = col - expanded_count[..., None]
    after_window = after_expansion - window_count[..., None]
    selected = torch.where(
        col < expanded_count[..., None],
        expansion,
        torch.where(
            after_window < 0,
            window_start[..., None] + after_expansion,
            torch.where(after_window < sink_count[..., None], after_window, -1),
        ),
    )
    lengths = expanded_count + window_count + sink_count
    return selected.to(torch.int32), lengths.to(torch.int32)


def attend_selected(
    query: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    selected: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Grouped-query attention of each query over its own selected keys.

    Each query gets its own key list, so each query is its own batch entry.
    The keys and values of the list are gathered from the buffers, and the
    -1 padding is masked out. The selected positions are causal by
    construction, so no causal mask is needed.

    Args:
        query: `(B, S, num_attention_heads, head_dim)`.
        k_buffer: `(B, max_seq_len, num_kv_heads, head_dim)` stored keys.
        v_buffer: `(B, max_seq_len, num_kv_heads, head_dim)` stored values.
        selected: `(B, S, L)` key positions per query, -1 padded. Every list
            holds at least the query's own position.
        scale: softmax scale.

    Returns:
        `(B, S, num_attention_heads, head_dim)` in the query's dtype.
    """
    batch, span, heads, head_dim = query.shape
    kv_heads = k_buffer.shape[2]
    width = selected.shape[-1]
    rows = torch.arange(batch, device=query.device).view(batch, 1, 1)
    slots = selected.clamp_min(0).long()

    def gather(buffer: torch.Tensor) -> torch.Tensor:
        # (B, S, L, kv_heads, head_dim) -> (B * S, kv_heads, L, head_dim)
        picked = buffer[rows, slots].to(query.dtype).permute(0, 1, 3, 2, 4)
        return picked.reshape(batch * span, kv_heads, width, head_dim)

    # The query heads that share a KV head sit on the query axis, as in the
    # dense decode path.
    grouped = query.reshape(batch * span, kv_heads, heads // kv_heads, head_dim)
    mask = (selected >= 0).view(batch * span, 1, 1, width)
    out = functional.scaled_dot_product_attention(
        grouped, gather(k_buffer), gather(v_buffer), attn_mask=mask, scale=scale
    )
    return out.reshape(batch, span, heads, head_dim)
