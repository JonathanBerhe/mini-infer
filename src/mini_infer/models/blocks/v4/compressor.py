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

import functools

import torch
from torch import nn
from torch.nn import functional

from mini_infer.models.blocks.rmsnorm import RMSNorm
from mini_infer.models.blocks.rope import apply_partial_rope_last_n_dims


@functools.cache
def cuda_mm_fp32_out_supported(device_index: int) -> bool:
    """Whether `torch.mm(bf16, bf16, out_dtype=torch.float32)` runs on this CUDA device.

    The `mm.dtype` overload is recent; an older torch build raises. Checked once
    per device with a tiny GEMM.
    """
    probe = torch.zeros(16, 16, device=torch.device("cuda", device_index), dtype=torch.bfloat16)
    try:
        torch.mm(probe, probe, out_dtype=torch.float32)
    except (TypeError, RuntimeError, NotImplementedError):
        return False
    return True


# Output dtype of the two compressor projections (`kv_proj`, `weight_proj`).
#   None          -> the matmul emits its native dtype (bf16 in production, fp32
#                    in CPU tests) and the result is upcast to fp32 afterwards.
#   torch.float32 -> a reduced-precision matmul emits fp32 directly, so the
#                    projected kv / score entries are never rounded to bf16
#                    before the compression softmax. This is how vLLM runs the
#                    same GEMMs for DeepSeek-V4 (`torch.mm(..., out_dtype=float32)`).
# The accumulation is fp32 either way; the policy only decides whether the GEMM
# output is rounded once to the input dtype. See `_project`.
ProjectionOutDtype = torch.dtype | None
_SUPPORTED_PROJECTION_OUT_DTYPES: tuple[torch.dtype | None, ...] = (None, torch.float32)


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
        projection_out_dtype: Output-dtype policy of `kv_proj` / `weight_proj`
            (`None` = native dtype then upcast, `torch.float32` = fp32 output
            with no intermediate rounding). See the module-level note.
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
        projection_out_dtype: ProjectionOutDtype = None,
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
        self.projection_out_dtype = projection_out_dtype
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

    @property
    def projection_out_dtype(self) -> ProjectionOutDtype:
        """Output-dtype policy of `kv_proj` / `weight_proj` (see the module-level note)."""
        return self._projection_out_dtype

    @projection_out_dtype.setter
    def projection_out_dtype(self, value: ProjectionOutDtype) -> None:
        if value not in _SUPPORTED_PROJECTION_OUT_DTYPES:
            raise ValueError(
                f"projection_out_dtype must be one of {_SUPPORTED_PROJECTION_OUT_DTYPES}; "
                f"got {value!r}"
            )
        self._projection_out_dtype = value

    def _project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project `(..., hidden_size)` to fp32 `(kv, score)`, each `(..., coff * kv_head_dim)`.

        `kv` are the per-token KV entries (`wkv` in the reference) and `score`
        the pre-softmax compression logits (`wgate`), WITHOUT the position bias.

        Policy `None`: the matmul runs in the input's native dtype (which is the
        weight dtype; upcasting the input first would mismatch a bf16-loaded
        weight) and the output is upcast afterwards, so a bf16 input yields
        bf16-rounded entries. Policy `torch.float32`: the output is emitted in
        fp32 with no intermediate rounding. On CUDA that is
        `torch.mm(..., out_dtype=float32)`, the op vLLM uses for these GEMMs;
        elsewhere (or on a torch without the `mm.dtype` overload) the same
        arithmetic is an fp32 GEMM over upcast operands, since products of
        bf16 values are exact in fp32. An fp32 input is unaffected by the policy.
        """
        if self._projection_out_dtype is None or hidden_states.dtype == torch.float32:
            return self.kv_proj(hidden_states).float(), self.weight_proj(hidden_states).float()
        kv_weight = self.kv_proj.weight  # (coff*c, hidden_size)
        score_weight = self.weight_proj.weight  # (coff*c, hidden_size)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])  # (N, hidden_size)
        if flat.is_cuda and cuda_mm_fp32_out_supported(flat.device.index or 0):
            kv_flat = torch.mm(flat, kv_weight.t(), out_dtype=torch.float32)  # (N, coff*c)
            score_flat = torch.mm(flat, score_weight.t(), out_dtype=torch.float32)  # (N, coff*c)
        else:
            kv_flat = functional.linear(flat.float(), kv_weight.float())  # (N, coff*c)
            score_flat = functional.linear(flat.float(), score_weight.float())  # (N, coff*c)
        out_shape = (*hidden_states.shape[:-1], kv_flat.shape[-1])
        return kv_flat.view(out_shape), score_flat.view(out_shape)

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

        # fp32 (kv, score) under `projection_out_dtype`; the softmax math below
        # runs in fp32 regardless of the policy.
        kv, score = self._project(hidden_states)  # each (B, T, coff*c) fp32

        # Reshape to (B, n_blocks, m, coff*c); `position_bias` (fp32) adds cleanly.
        batch_size, width = kv.shape[0], kv.shape[-1]
        kv = kv.reshape(batch_size, -1, m, width)  # (B, n_blocks, m, coff*c)
        score = (
            score.reshape(batch_size, -1, m, width) + self.position_bias
        )  # (B, n_blocks, m, coff*c)

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

    def forward_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        *,
        compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cmp_kv_state: torch.Tensor,
        cmp_score_state: torch.Tensor,
    ) -> torch.Tensor:
        """Cache-aware prefill: emit compressed entries AND populate in-flight state.

        Args:
            hidden_states: `(B, T, hidden_size)`. `T` does NOT need to be a
                multiple of `compression_ratio` — the trailing remainder
                `T % compression_ratio` tokens are stashed in the in-flight
                accumulator so the next decode step's flush sees them as if
                the prefill had stopped at a block boundary.
            compressed_position_embeddings: `(cos, sin)` for the
                `T // compression_ratio` compressed positions (block index
                `i` -> token position `i * compression_ratio`); each
                `(B, T // compression_ratio, rope_head_dim)`.
            cmp_kv_state, cmp_score_state: per-layer in-flight accumulator.
                Mutated in place — see "side effects" below.

        Returns:
            `(B, T // compression_ratio, kv_head_dim)` compressed entries.
            Identical to `forward(...)` when `T % compression_ratio == 0`.

        Side effects (mirror the reference's `Compressor.forward(start_pos=0)`):
            - Overlap mode AND `cutoff >= compression_ratio`: stash the LAST
              `compression_ratio` tokens of the aligned prefix into
              `cmp_kv_state[:, :compression_ratio]` (raw KV) and
              `cmp_score_state[:, :compression_ratio]` (score + position bias).
              These become the "previous overlap" half for the next decode
              step's flush.
            - Trailing remainder (`T % compression_ratio > 0`): stash the
              remainder tokens at slots `[offset, offset+remainder)` where
              `offset = compression_ratio` for overlap mode (that's the
              "current" half of the doubled buffer) and `offset = 0`
              otherwise. The next decode steps continue accumulating at
              `(start_pos % compression_ratio)` from this point, with
              earlier slots already populated.
        """
        batch_size, seqlen, _ = hidden_states.shape
        compression_ratio = self.compression_ratio
        kv_head_dim = self.kv_head_dim
        feature_multiplier = 2 if self.overlap_mode else 1
        expected_state_shape = (
            batch_size,
            compression_ratio * feature_multiplier,
            kv_head_dim * feature_multiplier,
        )
        if cmp_kv_state.shape != expected_state_shape:
            raise ValueError(
                f"cmp_kv_state shape {tuple(cmp_kv_state.shape)} mismatches expected "
                f"{expected_state_shape} (overlap_mode={self.overlap_mode})"
            )
        if cmp_score_state.shape != cmp_kv_state.shape:
            raise ValueError("cmp_kv_state and cmp_score_state must have matching shapes")

        remainder = seqlen % compression_ratio
        aligned_cutoff = seqlen - remainder

        # fp32 (kv, score) under `projection_out_dtype`. The state buffers
        # (`cmp_kv_state`, `cmp_score_state`) are fp32, matching these.
        full_kv, full_score = self._project(hidden_states)  # each (B, T, coff*c) fp32

        # ---- Stash the LAST aligned block as previous-overlap (CSA only) ----
        if self.overlap_mode and aligned_cutoff >= compression_ratio:
            cmp_kv_state[:, :compression_ratio] = full_kv[
                :, aligned_cutoff - compression_ratio : aligned_cutoff
            ]
            cmp_score_state[:, :compression_ratio] = (
                full_score[:, aligned_cutoff - compression_ratio : aligned_cutoff]
                + self.position_bias
            )

        # ---- Stash the trailing remainder as the in-flight current block ----
        if remainder > 0:
            current_half_offset = compression_ratio if self.overlap_mode else 0
            cmp_kv_state[:, current_half_offset : current_half_offset + remainder] = full_kv[
                :, aligned_cutoff:
            ]
            cmp_score_state[:, current_half_offset : current_half_offset + remainder] = (
                full_score[:, aligned_cutoff:] + self.position_bias[:remainder]
            )

        # ---- Compress the aligned prefix (the standalone forward path) ----
        if aligned_cutoff == 0:
            # No full blocks; nothing to emit.
            return torch.zeros(
                (batch_size, 0, kv_head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        width = full_kv.shape[-1]
        aligned_kv = full_kv[:, :aligned_cutoff].reshape(
            batch_size, -1, compression_ratio, width
        )  # (B, n_blocks, m, coff*c)
        aligned_score = (
            full_score[:, :aligned_cutoff].reshape(batch_size, -1, compression_ratio, width)
            + self.position_bias
        )  # (B, n_blocks, m, coff*c)
        if self.overlap_mode:
            aligned_kv = self._overlap_transform(aligned_kv, value=0.0)
            aligned_score = self._overlap_transform(aligned_score, value=float("-inf"))

        softmax_weights = aligned_score.softmax(dim=2)
        compressed = (aligned_kv * softmax_weights).sum(dim=2)  # (B, n_blocks, c)
        compressed = self.norm(compressed.to(hidden_states.dtype))

        if self.rope_head_dim > 0:
            cos_for_blocks, sin_for_blocks = compressed_position_embeddings
            compressed = apply_partial_rope_last_n_dims(
                compressed, cos_for_blocks, sin_for_blocks, self.rope_head_dim
            )
        # Re-bind to a Tensor-typed local so mypy doesn't widen through `Any`
        # (the trailing-remainder branch above can return early without RoPE).
        compressed_out: torch.Tensor = compressed
        return compressed_out

    def forward_decode_step(
        self,
        hidden_state: torch.Tensor,
        *,
        start_pos: int,
        cmp_kv_state: torch.Tensor,
        cmp_score_state: torch.Tensor,
        block_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """One incremental decode-step of the compressor.

        Args:
            hidden_state: `(B, 1, hidden_size)` — one new token's hidden state.
            start_pos: Global position of this token (0-indexed). Used to pick
                the slot inside the in-flight block (`start_pos % compression_ratio`)
                and to detect block completion `((start_pos + 1) % compression_ratio == 0)`.
            cmp_kv_state: in-flight KV accumulator (mutated in place). Shape:
                - HCA (no overlap): `(B, compression_ratio, kv_head_dim)`.
                - CSA (overlap):    `(B, 2 * compression_ratio, 2 * kv_head_dim)`.
                  Slots `[0, m)` hold the previous block's data (set up at
                  end-of-prefill or by a prior slide); slots `[m, 2m)` hold
                  the in-flight current block.
            cmp_score_state: parallel-shape score accumulator (mutated in place).
            block_position_embeddings: `(cos, sin)`, each `(B, 1, rope_head_dim)`.
                Required iff `rope_head_dim > 0` AND this step closes a block.
                Position is `(start_pos // compression_ratio) * compression_ratio`.

        Returns:
            `(B, 1, kv_head_dim)` newly compressed entry if this step closed a
            block, else `None`. The caller persists it into the compressed-history
            tensor at index `start_pos // compression_ratio`.

        Side effect (overlap mode only): on a flush, the current-block slots
        are slid into the previous-block slots so the next flush's `2m`
        softmax window picks them up as overlap. Mirrors the reference's
        `kv_state[:, :ratio] = kv_state[:, ratio:]` slide.
        """
        batch_size, seqlen, _ = hidden_state.shape
        if seqlen != 1:
            raise ValueError(f"forward_decode_step expects seqlen=1, got {seqlen}")

        compression_ratio = self.compression_ratio
        kv_head_dim = self.kv_head_dim
        feature_multiplier = 2 if self.overlap_mode else 1
        expected_shape = (
            batch_size,
            compression_ratio * feature_multiplier,
            kv_head_dim * feature_multiplier,
        )
        if cmp_kv_state.shape != expected_shape:
            raise ValueError(
                f"cmp_kv_state shape {tuple(cmp_kv_state.shape)} mismatches expected "
                f"{expected_shape} (overlap_mode={self.overlap_mode})"
            )
        if cmp_score_state.shape != cmp_kv_state.shape:
            raise ValueError("cmp_kv_state and cmp_score_state must have matching shapes")

        position_in_block = start_pos % compression_ratio

        # fp32 (kv, score) for the one new token; the in-flight accumulator is fp32.
        kv_token, score_token = self._project(hidden_state)  # each (B, 1, coff*c) fp32
        kv_for_token = kv_token.squeeze(1)  # (B, coff*c)
        score_for_token = (
            score_token.squeeze(1) + self.position_bias[position_in_block]  # (B, coff*c)
        )

        if self.overlap_mode:
            # Write to the "current" half — slots [m, 2m) hold the in-flight block.
            current_slot = compression_ratio + position_in_block
            cmp_kv_state[:, current_slot] = kv_for_token
            cmp_score_state[:, current_slot] = score_for_token
        else:
            cmp_kv_state[:, position_in_block] = kv_for_token
            cmp_score_state[:, position_in_block] = score_for_token

        if (start_pos + 1) % compression_ratio != 0:
            # Still accumulating; no compressed entry emitted this step.
            return None

        # ---- Block boundary: emit one compressed entry ----
        if self.overlap_mode:
            # Build a `(B, 2m, c)` sequence by interleaving the previous block's
            # "first-c-features" (from slots [0, m)) with the current block's
            # "last-c-features" (from slots [m, 2m)). Same axis layout the
            # prefill `_overlap_transform` produces, just per-block.
            previous_block_kv = cmp_kv_state[:, :compression_ratio, :kv_head_dim]
            current_block_kv = cmp_kv_state[:, compression_ratio:, kv_head_dim:]
            flat_kv = torch.cat([previous_block_kv, current_block_kv], dim=1)

            previous_block_score = cmp_score_state[:, :compression_ratio, :kv_head_dim]
            current_block_score = cmp_score_state[:, compression_ratio:, kv_head_dim:]
            flat_score = torch.cat([previous_block_score, current_block_score], dim=1)

            softmax_weights = flat_score.softmax(dim=1)
            compressed = (flat_kv * softmax_weights).sum(dim=1, keepdim=True)  # (B, 1, c)

            # SLIDE: the just-completed block becomes the "previous overlap"
            # for the next block. Slots are non-overlapping so we don't need
            # an explicit clone (PyTorch's setitem handles the views safely).
            cmp_kv_state[:, :compression_ratio] = cmp_kv_state[:, compression_ratio:]
            cmp_score_state[:, :compression_ratio] = cmp_score_state[:, compression_ratio:]
        else:
            softmax_weights = cmp_score_state.softmax(dim=1)
            compressed = (cmp_kv_state * softmax_weights).sum(dim=1, keepdim=True)

        compressed = self.norm(compressed.to(hidden_state.dtype))

        if self.rope_head_dim > 0:
            if block_position_embeddings is None:
                raise ValueError(
                    "block_position_embeddings required when rope_head_dim > 0 and a block flushes"
                )
            cos_for_block, sin_for_block = block_position_embeddings
            compressed = apply_partial_rope_last_n_dims(
                compressed, cos_for_block, sin_for_block, self.rope_head_dim
            )
        return compressed

    def forward_decode_step_ragged(
        self,
        hidden_state: torch.Tensor,
        *,
        positions: torch.Tensor,
        cmp_kv_state: torch.Tensor,
        cmp_score_state: torch.Tensor,
        block_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ragged decode step: one new token per request, each at its own position.

        The per-request counterpart to `forward_decode_step`. Every batch row
        `b` is at its own global position `positions[b]`, so the in-flight slot
        `positions[b] % m`, the block-boundary test `(positions[b] + 1) % m == 0`,
        and the just-flushed block's RoPE position `(positions[b] // m) * m` are
        all per-row. State buffers are mutated in place via row-indexed scatter.

        Args:
            hidden_state: `(B, 1, hidden_size)`.
            positions: `(B,)` int64 global positions, one per request.
            cmp_kv_state / cmp_score_state: same shapes / meaning as
                `forward_decode_step` (per-request rows).
            block_position_embeddings: `(cos, sin)`, each `(B, 1, rope_head_dim)`,
                the RoPE for each row's block position `(positions // m) * m`.
                Required iff `rope_head_dim > 0`. Rows that do not flush this
                step ignore their entry.

        Returns:
            `(compressed, flush_mask)` where `compressed` is `(B, 1, kv_head_dim)`
            (valid only for rows where `flush_mask` is True) and `flush_mask` is
            `(B,)` bool marking the rows that closed a block this step. The caller
            writes `compressed[flush_mask]` into compressed history at slot
            `positions[b] // m`, and only for those rows.

        Unlike the scalar path this always computes `compressed` for every row
        (the block softmax is cheap and lets the whole batch share one forward);
        non-flushing rows produce a value the caller discards.
        """
        batch_size, seqlen, _ = hidden_state.shape
        if seqlen != 1:
            raise ValueError(f"forward_decode_step_ragged expects seqlen=1, got {seqlen}")

        m = self.compression_ratio
        c = self.kv_head_dim
        rows = torch.arange(batch_size, device=hidden_state.device)
        position_in_block = positions % m  # (B,)

        kv_token, score_token = self._project(hidden_state)  # each (B, 1, coff*c) fp32
        kv_for_token = kv_token.squeeze(1)  # (B, c) or (B, 2c)
        score_for_token = (
            score_token.squeeze(1) + self.position_bias[position_in_block]  # (B, c) or (B, 2c)
        )

        if self.overlap_mode:
            current_slot = m + position_in_block  # (B,)
            cmp_kv_state[rows, current_slot] = kv_for_token
            cmp_score_state[rows, current_slot] = score_for_token
        else:
            cmp_kv_state[rows, position_in_block] = kv_for_token
            cmp_score_state[rows, position_in_block] = score_for_token

        flush_mask = (positions + 1) % m == 0  # (B,) bool

        # Compute the compressed entry for every row; the caller keeps only the
        # flushing rows. Mirrors the scalar block-boundary math, vectorized.
        if self.overlap_mode:
            previous_block_kv = cmp_kv_state[:, :m, :c]
            current_block_kv = cmp_kv_state[:, m:, c:]
            flat_kv = torch.cat([previous_block_kv, current_block_kv], dim=1)
            previous_block_score = cmp_score_state[:, :m, :c]
            current_block_score = cmp_score_state[:, m:, c:]
            flat_score = torch.cat([previous_block_score, current_block_score], dim=1)
            softmax_weights = flat_score.softmax(dim=1)
            compressed = (flat_kv * softmax_weights).sum(dim=1, keepdim=True)
            # Slide only the rows that flushed: their current block becomes the
            # next block's "previous overlap". Advanced-index read copies first,
            # so the [:m] write does not alias the [m:] read.
            cmp_kv_state[flush_mask, :m] = cmp_kv_state[flush_mask, m:]
            cmp_score_state[flush_mask, :m] = cmp_score_state[flush_mask, m:]
        else:
            softmax_weights = cmp_score_state.softmax(dim=1)
            compressed = (cmp_kv_state * softmax_weights).sum(dim=1, keepdim=True)

        compressed = self.norm(compressed.to(hidden_state.dtype))

        if self.rope_head_dim > 0:
            if block_position_embeddings is None:
                raise ValueError(
                    "block_position_embeddings required when rope_head_dim > 0 "
                    "(per-row block positions for ragged decode)"
                )
            cos_for_block, sin_for_block = block_position_embeddings
            compressed = apply_partial_rope_last_n_dims(
                compressed, cos_for_block, sin_for_block, self.rope_head_dim
            )
        return compressed, flush_mask


def configure_projection_out_dtype(module: nn.Module, out_dtype: ProjectionOutDtype) -> int:
    """Set `projection_out_dtype` on every `TokenLevelCompressor` under `module`.

    Returns how many compressors were configured. A V4 model has one per HCA
    layer and two per CSA layer (the main compressor plus the Lightning
    Indexer's own compressor).
    """
    count = 0
    for submodule in module.modules():
        if isinstance(submodule, TokenLevelCompressor):
            submodule.projection_out_dtype = out_dtype
            count += 1
    return count
