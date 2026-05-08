"""Token-level KV compressor (V4 paper §2.3.2, formulas 20-23).

Compresses every `m` consecutive tokens into one KV entry via a learned
softmax-weighted sum over the block. Used twice in V4:

  - HCA layers: heavy compression `m'=128`, `overlap_mode=False`.
  - CSA layers: light compression `m=4`, `overlap_mode=True` — each
    compressed block also sees the *previous* `m` tokens, so the softmax
    spans `2m` candidates and information flows across block boundaries.

Math without overlap (one block of `m` tokens at a time):
    KV   = x · W^{KV}        # (m, c)  -- per-token KV entries
    Z    = x · W^{Z}          # (m, c)  -- per-token compression weights
    S    = softmax_row(Z + B) # (m, c)  -- B is a learnable per-block-position bias
    C    = sum(S ⊙ KV, dim=0) # (c,)    -- one compressed entry
    C    = RMSNorm(C)
    C    = partial_rope(C, position=block_idx * m)

Math with overlap (CSA): `W^{KV}` and `W^{Z}` both emit `2c`-wide outputs
that get reshaped so the softmax pools over `2m` slots — `m` from this
block (using the second half of the `2c`-wide output) plus `m` from the
*previous* block (using the first half). Block 0 has no previous block,
so its first `m` slots are masked to `-inf`. The position bias `B` is
also `2c`-wide for the same split.

The reference (`deepseek_v4_reference/model.py::Compressor`) carries
extra plumbing for decode-phase incremental compression (state buffers
that hold the tail tokens of the in-flight block + the previous block's
overlap data). Our standalone forward is prefill-only and requires
`seqlen % m == 0` — the parity tests enforce this. The decode plumbing
lands with the cache wiring stage.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims


class TokenLevelCompressor(nn.Module):
    """Compress every `m` tokens into one KV entry via softmax-weighted sum.

    Args:
        hidden_size: Input feature dim of the hidden states.
        kv_head_dim: Output feature dim per compressed entry (`c` in the paper).
        rope_head_dim: Number of trailing dims of `kv_head_dim` to rotate via
            partial RoPE. Must be even and `<= kv_head_dim`. Set to `0` to skip RoPE.
        compression_ratio: How many tokens collapse into one entry (`m` or `m'`).
        rms_norm_eps: RMSNorm epsilon for the post-compression normalization.
        overlap_mode: When True (CSA): softmax spans `2m` slots — `m` current
            tokens plus `m` from the previous block. KV/score projections emit
            `2 * kv_head_dim`-wide outputs; the position bias is also doubled.
            When False (HCA): the standard non-overlapping form.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        kv_head_dim: int,
        rope_head_dim: int,
        compression_ratio: int,
        rms_norm_eps: float = 1e-6,
        overlap_mode: bool = False,
    ) -> None:
        super().__init__()
        if compression_ratio <= 0:
            raise ValueError(f"compression_ratio must be positive, got {compression_ratio}")
        if rope_head_dim < 0 or rope_head_dim > kv_head_dim:
            raise ValueError(
                f"rope_head_dim={rope_head_dim} must be in [0, kv_head_dim={kv_head_dim}]"
            )
        if rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even, got {rope_head_dim}")
        self.hidden_size = hidden_size
        self.kv_head_dim = kv_head_dim
        self.rope_head_dim = rope_head_dim
        self.compression_ratio = compression_ratio
        self.overlap_mode = overlap_mode
        # `coff` is the multiplier on the projection's output width:
        # 1 for HCA (current block only), 2 for CSA (current + overlap).
        coff = 2 if overlap_mode else 1
        # `wkv` in the reference: x -> KV entries (or current+overlap pair under overlap mode).
        self.kv_proj = nn.Linear(hidden_size, coff * kv_head_dim, bias=False)
        # `wgate` in the reference: x -> compression-weight logits (pre-softmax).
        self.weight_proj = nn.Linear(hidden_size, coff * kv_head_dim, bias=False)
        # `ape` in the reference: per-block-position learnable bias added to logits before softmax.
        # Under overlap mode, the first `kv_head_dim` features bias the overlap (previous-block)
        # slots and the last `kv_head_dim` bias the current-block slots.
        self.position_bias = nn.Parameter(torch.zeros(compression_ratio, coff * kv_head_dim))
        # RMSNorm applied to the compressed entry (single-width `kv_head_dim`) before partial RoPE.
        self.norm = RMSNorm(kv_head_dim, eps=rms_norm_eps)

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        """Reshape `(B, n_blocks, m, 2c)` -> `(B, n_blocks, 2m, c)` for the overlap softmax.

        Output layout per block `i`:
            slots `[m, 2m)`            <- `tensor[i, :, c:]`     (current half of `tensor`)
            slots `[0, m)` for `i >= 1` <- `tensor[i-1, :, :c]`  (previous block's overlap half)
            slots `[0, m)` for `i == 0` <- `value`               (no predecessor; padded)
        """
        bsz, n_blocks, m, _ = tensor.shape
        c = self.kv_head_dim
        out = tensor.new_full((bsz, n_blocks, 2 * m, c), value)
        out[:, :, m:] = tensor[:, :, :, c:]
        if n_blocks > 1:
            out[:, 1:, :m] = tensor[:, :-1, :, :c]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compress `(B, T, hidden_size)` -> `(B, T // m, kv_head_dim)`.

        `compressed_position_embeddings` is `(cos, sin)` for the
        `T // m` compressed positions (block index `i` -> token position
        `i * m`). The caller pre-computes these from the model's RoPE
        table; the compressor only applies them.
        """
        _, seqlen, _ = hidden_states.shape
        m = self.compression_ratio
        if seqlen % m != 0:
            raise ValueError(f"seqlen={seqlen} must be a multiple of compression_ratio={m}")

        # Stay in fp32 for the softmax math — the reference does this and
        # the bf16 path is meaningfully lossy on small absolute values.
        x = hidden_states.float()
        kv = self.kv_proj(x)  # (B, T, coff*c)
        score = self.weight_proj(x)  # (B, T, coff*c)

        # Reshape to (B, n_blocks, m, coff*c). Position bias broadcasts across batch + blocks.
        kv = kv.unflatten(1, (-1, m))
        score = score.unflatten(1, (-1, m)) + self.position_bias

        if self.overlap_mode:
            # Overlap: each block's softmax spans 2m slots — its own m + the previous block's m.
            kv = self._overlap_transform(kv, value=0.0)
            score = self._overlap_transform(score, value=float("-inf"))

        # Softmax along the per-block axis (size m without overlap, 2m with).
        weights = score.softmax(dim=2)
        compressed: torch.Tensor = (kv * weights).sum(dim=2)  # (B, n_blocks, c)

        # RMSNorm in the model's working dtype.
        compressed = self.norm(compressed.to(hidden_states.dtype))

        # Partial RoPE on the last `rope_head_dim` dims of the compressed entry.
        # Position for block i is `i * m` — caller produces the right cos/sin tables.
        cos, sin = compressed_position_embeddings
        if self.rope_head_dim > 0:
            compressed = apply_partial_rope_last_n_dims(compressed, cos, sin, self.rope_head_dim)
        return compressed

    def forward_decode_step(
        self,
        hidden_state: torch.Tensor,
        *,
        start_pos: int,
        cmp_kv_state: torch.Tensor,
        cmp_score_state: torch.Tensor,
        block_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """One incremental decode-step of the compressor (HCA only — no overlap).

        Args:
            hidden_state: `(B, 1, hidden_size)` — one new token's hidden state.
            start_pos: Global position of this token (0-indexed). Used to pick
                the slot inside the in-flight block (`start_pos % m`) and to
                detect a block-completion (`(start_pos + 1) % m == 0`).
            cmp_kv_state: `(B, m, kv_head_dim)` fp32 in-flight KV accumulator,
                mutated in place.
            cmp_score_state: `(B, m, kv_head_dim)` fp32 in-flight score
                accumulator, mutated in place.
            block_position_embeddings: `(cos, sin)` — each `(B, 1, rope_head_dim)`.
                Required iff `rope_head_dim > 0` AND this step completes a block.
                Position is `i * m` where `i = start_pos // m` is the just-completed
                block index.

        Returns:
            `(B, 1, kv_head_dim)` newly compressed entry if this step closed a
            block, else None. The caller persists it into the compressed-history
            tensor at index `start_pos // m`.

        The reference's overlap-mode decode (CSA) needs an extra "previous-block"
        state slice and slides the in-flight buffer on flush; left for the next
        stage so this method stays narrow.
        """
        if self.overlap_mode:
            raise NotImplementedError("overlap_mode decode lands with CSA decode wiring")
        bsz, seqlen_in, _ = hidden_state.shape
        if seqlen_in != 1:
            raise ValueError(f"forward_decode_step expects seqlen=1, got {seqlen_in}")
        if cmp_kv_state.shape != (bsz, self.compression_ratio, self.kv_head_dim):
            raise ValueError(
                f"cmp_kv_state shape {tuple(cmp_kv_state.shape)} mismatches "
                f"({bsz}, {self.compression_ratio}, {self.kv_head_dim})"
            )
        if cmp_score_state.shape != cmp_kv_state.shape:
            raise ValueError("cmp_kv_state and cmp_score_state must have matching shapes")

        m = self.compression_ratio
        pos_in_block = start_pos % m

        # Project to KV + score (the in-flight accumulator is fp32 — match it).
        x = hidden_state.float()
        kv = self.kv_proj(x).squeeze(1)  # (B, kv_head_dim)
        score = self.weight_proj(x).squeeze(1) + self.position_bias[pos_in_block]
        cmp_kv_state[:, pos_in_block] = kv
        cmp_score_state[:, pos_in_block] = score

        if (start_pos + 1) % m != 0:
            # Still accumulating; no compressed entry emitted this step.
            return None

        # Block boundary: softmax-weighted sum over the m accumulated entries.
        weights = cmp_score_state.softmax(dim=1)
        compressed = (cmp_kv_state * weights).sum(dim=1, keepdim=True)  # (B, 1, c)
        compressed = self.norm(compressed.to(hidden_state.dtype))

        if self.rope_head_dim > 0:
            if block_position_embeddings is None:
                raise ValueError(
                    "block_position_embeddings required when rope_head_dim > 0 and a block flushes"
                )
            cos, sin = block_position_embeddings
            compressed = apply_partial_rope_last_n_dims(compressed, cos, sin, self.rope_head_dim)
        return compressed
