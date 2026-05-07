from dataclasses import dataclass
from typing import Literal

import torch

from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.cache.turbo_quant import (
    dequantize_kv_block,
    generate_rotation_matrices,
    inverse_rotate,
    lloyd_max_codebook,
    polar_dequantize_block,
    polar_quantize_block,
    quantize_kv_block,
    rotate,
)
from mini_infer.exceptions import OutOfMemoryError

# Per-layer attention pattern. Default models (Qwen2, Llama) are entirely
# `"full"`. Sliding-window-aware models (Gemma 3+, Mistral SWA, Gemma 4)
# specify `("sliding", window_size)` for their windowed layers; the
# attention dispatcher honors the window when reading the cache.
LayerAttentionSpec = Literal["full"] | tuple[Literal["sliding"], int]


@dataclass(frozen=True)
class StreamSpec:
    """Per-layer storage descriptor for one named tensor stream.

    Standard MHA / GQA layers carry two streams `("k", "v")` of identical
    shape. MLA layers (DeepSeek-V2/V3, Kimi-K2) carry two streams of
    DIFFERENT shape — `("compressed_kv", num_kv_heads=1, kv_lora_rank)`
    plus `("k_rope", num_kv_heads=1, qk_rope_head_dim)` — both shared
    across all attention heads (`num_kv_heads=1` is the giveaway).

    Block IDs are global across all streams: slot `block_id` reserves a
    chunk in EVERY stream's tensor at the same index. Each stream has
    its own `(num_blocks, block_size, num_kv_heads_s, head_dim_s)`
    storage tensor.
    """

    name: str
    num_kv_heads: int
    head_dim: int


# Supported KV-cache compression modes:
#   None     = legacy bf16/fp16 storage.
#   "turbo4" = TurboQuant V1: rotation + per-channel asymmetric 4-bit
#              uniform quant. ~62% memory savings on Qwen2.5-0.5B and 7B.
#   "turbo3" = TurboQuant V3 (full): rotation + polar transform + Lloyd-Max
#              codebook + asymmetric K (3-bit + QJL residual = 4 bits
#              stored) / V (4-bit Lloyd-Max). Same on-disk layout as
#              turbo4 (4 bits per element) but better fidelity at deeper
#              models, plus per-vector radii instead of per-channel
#              (low, scale).
#   "fp8"    = FP8 e4m3fn KV cache. Per-(layer, side, kv_head) scale set
#              from the first append's abs-max; ~50% memory savings vs
#              bf16. Requires attention_backend="flashinfer" — FlashInfer
#              fuses fp8 dequant into its paged-attention kernel.
#   "nvfp4"  = NVFP4 (FP4 e2m1) KV cache via FlashInfer's paged quantizer.
#              Per-block FP8 scales + per-layer-per-side global scale;
#              ~72% memory savings vs bf16. Requires Blackwell (SM_100)
#              and `attention_backend="flashinfer"`. The V scale tensor
#              uses TRT-LLM's 4-token-interleaved swizzle, so block_size
#              must be a multiple of 4 and head_dim a multiple of 64.
_SUPPORTED_KV_QUANT = (None, "turbo4", "turbo3", "fp8", "nvfp4")

# Supported attention backends. The dispatcher in `packed_attention.py`
# reads `BlockPool.attention_backend` to pick which implementation runs:
#   "flash_attn" = `flash_attn_varlen_func` (default)
#   "flashinfer" = FlashInfer's paged-attention wrappers; valid with
#                  `kv_quant in {None, "fp8"}`
#   "torch"      = materialize K/V from blocks then SDPA. Slow, but the
#                  only path that handles head_dim > 256 (Gemma 4 31B
#                  full layers, head_dim=512). vLLM's `TRITON_ATTN`
#                  serves the same role; we don't ship a Triton kernel,
#                  so we fall back to PyTorch SDPA on the materialized
#                  packed buffer.
_SUPPORTED_ATTENTION_BACKENDS = ("flash_attn", "flashinfer", "torch")


class BlockPool:
    """Pre-allocated pool of fixed-size K/V blocks shared across requests.

    Two storage modes:

    - **Uncompressed** (default, ``kv_quant=None``). Single bf16/fp16 tensor
      of shape ``(num_layers, 2, num_blocks, block_size, num_kv_heads,
      head_dim)``. The "2" is (key, value). Per-block read/write is a
      direct slice into ``_storage``.
    - **TurboQuant 4-bit** (``kv_quant="turbo4"``). Each block is stored
      as int8 packed bytes (two 4-bit values per byte) plus a bf16
      ``(num_kv_heads, head_dim, 2)`` scales tensor (low + scale per
      channel). A per-layer random orthogonal rotation is applied
      before quantization on write and inverted after dequantization on
      read. Math is identity vs the uncompressed path within
      quantization noise; storage is ~4x smaller.

    Compressed-mode blocks are read/written via ``read_compressed_block``
    and ``write_compressed_block`` rather than ``view`` / ``storage``,
    which only make sense for the uncompressed layout.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        prefix_cache: PrefixCache | None = None,
        kv_quant: str | None = None,
        attention_backend: str = "flash_attn",
        layer_attention: list[LayerAttentionSpec] | None = None,
        layer_kv_shape: list[tuple[int, int]] | None = None,
        layer_streams: list[list[StreamSpec]] | None = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if prefix_cache is not None and prefix_cache.block_size != block_size:
            raise ValueError(
                f"prefix_cache.block_size={prefix_cache.block_size} disagrees with "
                f"pool block_size={block_size}"
            )
        if kv_quant not in _SUPPORTED_KV_QUANT:
            raise ValueError(
                f"unsupported kv_quant={kv_quant!r}; expected one of {_SUPPORTED_KV_QUANT}"
            )
        # Reject heterogeneous-KV + any quant mode early, before the
        # quant-specific prerequisite checks (flashinfer requirement etc).
        # The "heterogeneous" decision is made just by looking at the user's
        # `layer_kv_shape` argument; full validation of the list itself
        # happens further down.
        if (
            kv_quant is not None
            and layer_kv_shape is not None
            and any(s != layer_kv_shape[0] for s in layer_kv_shape)
        ):
            raise ValueError(
                f"heterogeneous layer_kv_shape requires kv_quant=None; "
                f"got kv_quant={kv_quant!r}. Per-layer scales/codebooks for "
                "FP8/NVFP4/TurboQuant aren't supported yet — that's a future stage."
            )
        if kv_quant in ("turbo4", "turbo3") and (block_size * num_kv_heads * head_dim) % 2 != 0:
            raise ValueError(
                f"{kv_quant!r} packs two 4-bit values per byte; "
                "block_size * num_kv_heads * head_dim must be even"
            )
        if attention_backend not in _SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                f"unsupported attention_backend={attention_backend!r}; "
                f"expected one of {_SUPPORTED_ATTENTION_BACKENDS}"
            )
        if attention_backend == "flashinfer" and kv_quant not in (None, "fp8", "nvfp4"):
            raise ValueError(
                f"attention_backend='flashinfer' is valid only with "
                f"kv_quant in (None, 'fp8', 'nvfp4'); got kv_quant={kv_quant!r}."
            )
        if kv_quant == "fp8" and attention_backend != "flashinfer":
            raise ValueError(
                "kv_quant='fp8' requires attention_backend='flashinfer'; "
                f"got {attention_backend!r}."
            )
        if kv_quant == "nvfp4":
            if attention_backend != "flashinfer":
                raise ValueError(
                    "kv_quant='nvfp4' requires attention_backend='flashinfer'; "
                    f"got {attention_backend!r}."
                )
            if block_size % 4 != 0:
                raise ValueError(
                    f"kv_quant='nvfp4' requires block_size %% 4 == 0 (V-scale 4-token "
                    f"swizzle); got block_size={block_size}"
                )
            if head_dim % 64 != 0:
                raise ValueError(
                    f"kv_quant='nvfp4' requires head_dim %% 64 == 0; got head_dim={head_dim}"
                )
            if prefix_cache is not None:
                raise ValueError(
                    "kv_quant='nvfp4' does not support prefix caching: cached blocks "
                    "would need a working paged-FP4 dequant to re-quantize on append, "
                    "which FlashInfer does not currently expose. Pass prefix_cache=None."
                )

        # Per-layer KV shape. Defaults to `[(num_kv_heads, head_dim)] *
        # num_layers` (homogeneous — current behavior). Heterogeneous shapes
        # (Gemma 4 31B-style: different head_dim/num_kv_heads per layer-type)
        # require kv_quant=None for now; the quantized paths still assume
        # homogeneous storage.
        if layer_kv_shape is None:
            layer_kv_shape = [(num_kv_heads, head_dim)] * num_layers
        if len(layer_kv_shape) != num_layers:
            raise ValueError(
                f"layer_kv_shape has {len(layer_kv_shape)} entries; "
                f"expected num_layers={num_layers}"
            )
        for layer_idx, shape in enumerate(layer_kv_shape):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or not isinstance(shape[0], int)
                or not isinstance(shape[1], int)
                or shape[0] <= 0
                or shape[1] <= 0
            ):
                raise ValueError(
                    f"layer_kv_shape[{layer_idx}]={shape!r} must be "
                    "(num_kv_heads_l, head_dim_l) with positive ints"
                )
        is_heterogeneous = any(s != layer_kv_shape[0] for s in layer_kv_shape)
        # The early gate above already rejected heterogeneous + quant, so by
        # the time we get here `is_heterogeneous` implies kv_quant is None.

        # Per-layer storage descriptor (Stage C3). Default is the legacy K/V
        # layout: each layer carries `[StreamSpec("k", kv, hd), StreamSpec("v",
        # kv, hd)]` with shape derived from `layer_kv_shape`. Models with a
        # different layout (DeepSeek-V2 / V3 MLA: compressed_kv + k_rope per
        # layer) pass `layer_streams` directly.
        if layer_streams is None:
            layer_streams = [
                [StreamSpec("k", kv_l, hd_l), StreamSpec("v", kv_l, hd_l)]
                for (kv_l, hd_l) in layer_kv_shape
            ]
        elif kv_quant is not None:
            # Compressed paths (turbo*, fp8, nvfp4) assume a rectangular
            # K/V layout. Generalizing them to per-stream descriptors is
            # a future stage.
            raise ValueError(
                f"explicit `layer_streams` requires kv_quant=None; got "
                f"kv_quant={kv_quant!r}. Per-stream paths for compressed "
                "KV are not yet supported."
            )
        if len(layer_streams) != num_layers:
            raise ValueError(
                f"layer_streams has {len(layer_streams)} entries; expected num_layers={num_layers}"
            )
        for layer_idx, streams in enumerate(layer_streams):
            if not streams:
                raise ValueError(
                    f"layer_streams[{layer_idx}] is empty; every layer needs at least one stream"
                )
            seen: set[str] = set()
            for spec in streams:
                if not isinstance(spec, StreamSpec):
                    raise ValueError(
                        f"layer_streams[{layer_idx}] contains {spec!r}; "
                        "expected StreamSpec instances"
                    )
                if spec.num_kv_heads <= 0 or spec.head_dim <= 0:
                    raise ValueError(
                        f"layer_streams[{layer_idx}].{spec.name}: "
                        f"num_kv_heads={spec.num_kv_heads}, "
                        f"head_dim={spec.head_dim} must be positive"
                    )
                if spec.name in seen:
                    raise ValueError(
                        f"layer_streams[{layer_idx}] has duplicate stream name {spec.name!r}"
                    )
                seen.add(spec.name)
        # Legacy K/V layout: every layer has exactly the streams ["k", "v"]
        # with identical shape. When True, the legacy `_layer_storage`
        # rectangular tensors and `storage_for_layer` accessors are valid;
        # when False, only the per-stream API works (legacy raises with a
        # clear message pointing at the new accessors).
        is_legacy_kv_layout = all(
            len(streams) == 2
            and {s.name for s in streams} == {"k", "v"}
            and (
                next(s for s in streams if s.name == "k").num_kv_heads
                == next(s for s in streams if s.name == "v").num_kv_heads
            )
            and (
                next(s for s in streams if s.name == "k").head_dim
                == next(s for s in streams if s.name == "v").head_dim
            )
            for streams in layer_streams
        )

        # Per-layer attention spec defaults to all-`"full"` so existing models
        # (Qwen2, Llama) get unchanged behavior. Sliding-window models pass an
        # explicit list of length `num_layers`.
        if layer_attention is None:
            layer_attention = ["full"] * num_layers
        if len(layer_attention) != num_layers:
            raise ValueError(
                f"layer_attention has {len(layer_attention)} entries; "
                f"expected num_layers={num_layers}"
            )
        for layer_idx, attn_spec in enumerate(layer_attention):
            if attn_spec == "full":
                continue
            if (
                isinstance(attn_spec, tuple)
                and len(attn_spec) == 2
                and attn_spec[0] == "sliding"
                and isinstance(attn_spec[1], int)
                and attn_spec[1] > 0
            ):
                continue
            raise ValueError(
                f"layer_attention[{layer_idx}]={attn_spec!r} is not a valid spec; "
                "use 'full' or ('sliding', positive_int)"
            )

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        # `num_kv_heads` / `head_dim` retain their meaning for HOMOGENEOUS pools
        # (the legacy property surface). Heterogeneous pools route through
        # `num_kv_heads_for_layer` / `head_dim_for_layer`; the legacy
        # properties raise on a heterogeneous pool.
        self._homogeneous_num_kv_heads = num_kv_heads
        self._homogeneous_head_dim = head_dim
        self._layer_kv_shape: list[tuple[int, int]] = list(layer_kv_shape)
        self._is_heterogeneous = is_heterogeneous
        self._layer_stream_specs: list[list[StreamSpec]] = [
            list(streams) for streams in layer_streams
        ]
        self._is_legacy_kv_layout = is_legacy_kv_layout
        self.dtype = dtype
        self._kv_quant = kv_quant
        self._attention_backend = attention_backend
        self._layer_attention: list[LayerAttentionSpec] = list(layer_attention)

        if kv_quant is None:
            if self._is_legacy_kv_layout and not self._is_heterogeneous:
                # Homogeneous fast path: allocate ONE rectangular tensor and
                # view per-layer slices into it. Avoids transient 2x peak
                # memory during init that a per-layer-then-stack approach
                # would incur on large pools (Phi-3 / 7B+ models on M1).
                kv_l, hd_l = self._layer_kv_shape[0]
                self._storage = torch.zeros(
                    num_layers,
                    2,
                    num_blocks,
                    block_size,
                    kv_l,
                    hd_l,
                    dtype=dtype,
                    device=device,
                )
                self._layer_storage: list[torch.Tensor] = [
                    self._storage[layer_idx] for layer_idx in range(num_layers)
                ]
            elif self._is_legacy_kv_layout:
                # Heterogeneous K/V (Gemma 4 31B): per-layer tensors of
                # (potentially) different shapes. Block IDs still global —
                # slot `block_id` lives at index `block_id` in every
                # layer's tensor. Legacy `pool.storage` property raises on
                # heterogeneous pools so `_storage` is a tiny placeholder.
                self._layer_storage = [
                    torch.zeros(2, num_blocks, block_size, kv_l, hd_l, dtype=dtype, device=device)
                    for (kv_l, hd_l) in self._layer_kv_shape
                ]
                self._storage = torch.empty(0, dtype=dtype, device=device)
            else:
                # Stream-descriptor layout (DeepSeek MLA-style): no
                # rectangular K/V tensor. Per-layer per-stream tensors get
                # allocated below (in the unified `_layer_streams_storage`
                # block); the legacy K/V handles stay empty so the legacy
                # accessors raise with a clear migration message.
                self._layer_storage = []
                self._storage = torch.empty(0, dtype=dtype, device=device)
            # Per-stream storage. For legacy K/V layers we ALIAS the
            # `_layer_storage` rectangular slices (no extra memory); for
            # MLA-style layers we allocate standalone tensors per stream.
            # `_layer_streams_storage[layer_idx][stream_name]` always
            # returns a `(num_blocks, block_size, num_kv_heads_s,
            # head_dim_s)` tensor — same contract for both layouts.
            self._layer_streams_storage: list[dict[str, torch.Tensor]] = []
            for layer_idx, streams in enumerate(self._layer_stream_specs):
                streams_for_layer: dict[str, torch.Tensor] = {}
                if (
                    len(streams) == 2
                    and {s.name for s in streams} == {"k", "v"}
                    and self._layer_storage
                ):
                    # Alias the legacy K/V rectangular layout. layer_storage
                    # for layer i has shape (2, num_blocks, block_size,
                    # num_kv_heads, head_dim) — slice 0 is K, 1 is V.
                    layer_t = self._layer_storage[layer_idx]
                    streams_for_layer["k"] = layer_t[0]
                    streams_for_layer["v"] = layer_t[1]
                else:
                    for spec in streams:
                        streams_for_layer[spec.name] = torch.zeros(
                            num_blocks,
                            block_size,
                            spec.num_kv_heads,
                            spec.head_dim,
                            dtype=dtype,
                            device=device,
                        )
                self._layer_streams_storage.append(streams_for_layer)
            self._compressed_storage: torch.Tensor | None = None
            self._scales_storage: torch.Tensor | None = None
            self._radii_storage: torch.Tensor | None = None
            self._rotation: torch.Tensor | None = None
            self._k_codebook: torch.Tensor | None = None
            self._v_codebook: torch.Tensor | None = None
            self._fp8_storage: torch.Tensor | None = None
            self._fp8_scales: torch.Tensor | None = None
            self._fp8_scales_initialized: torch.Tensor | None = None
            self._nvfp4_storage: torch.Tensor | None = None
            self._nvfp4_block_scales: torch.Tensor | None = None
            self._nvfp4_global_sf: torch.Tensor | None = None
            self._nvfp4_initialized: torch.Tensor | None = None
        elif kv_quant == "fp8":
            # FP8 e4m3fn paged storage with the same NHD layout the bf16
            # path uses, plus a per-(layer, side, kv_head) scale tensor.
            # The first append per (layer, side) sets the scale from the
            # batch's abs-max divided by 448 (the fp8 max representable
            # value). Subsequent appends quantize against the cached
            # scale and clamp on overflow.
            self._fp8_storage = torch.zeros(
                num_layers,
                2,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            self._fp8_scales = torch.ones(
                num_layers, 2, num_kv_heads, dtype=torch.float32, device=device
            )
            self._fp8_scales_initialized = torch.zeros(
                num_layers, 2, dtype=torch.bool, device=device
            )
            self._compressed_storage = None
            self._scales_storage = None
            self._radii_storage = None
            self._rotation = None
            self._k_codebook = None
            self._v_codebook = None
            self._nvfp4_storage = None
            self._nvfp4_block_scales = None
            self._nvfp4_global_sf = None
            self._nvfp4_initialized = None
            self._storage = torch.empty(0, dtype=dtype, device=device)
        elif kv_quant == "nvfp4":
            # NVFP4 paged storage. Two side-by-side tensors per (layer,
            # side, num_blocks) block:
            #   _nvfp4_storage:      packed FP4 bytes (head_dim // 2 per token).
            #   _nvfp4_block_scales: per-16-element FP8 e4m3 scale block
            #                        (head_dim // 16 per token), with the V
            #                        side carrying TRT-LLM's 4-token swizzle
            #                        layout that FlashInfer's kernel reads.
            # A single fp32 global scale per (layer, side) is set on the
            # first append from the batch's amax and reused thereafter so
            # re-quantization across steps stays self-consistent.
            packed_last = head_dim // 2
            scale_last = head_dim // 16
            self._nvfp4_storage = torch.zeros(
                num_layers,
                2,
                num_blocks,
                block_size,
                num_kv_heads,
                packed_last,
                dtype=torch.uint8,
                device=device,
            )
            self._nvfp4_block_scales = torch.zeros(
                num_layers,
                2,
                num_blocks,
                block_size,
                num_kv_heads,
                scale_last,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            self._nvfp4_global_sf = torch.ones(num_layers, 2, dtype=torch.float32, device=device)
            self._nvfp4_initialized = torch.zeros(num_layers, 2, dtype=torch.bool, device=device)
            self._fp8_storage = None
            self._fp8_scales = None
            self._fp8_scales_initialized = None
            self._compressed_storage = None
            self._scales_storage = None
            self._radii_storage = None
            self._rotation = None
            self._k_codebook = None
            self._v_codebook = None
            self._storage = torch.empty(0, dtype=dtype, device=device)
        else:
            # Both turbo4 and turbo3 pack 4 bits per element, so the
            # compressed storage has the same shape. The "scales" tensor
            # differs by mode: turbo4 stores per-channel (low, scale);
            # turbo3 stores per-vector radii (one float per token-head).
            packed_bytes_per_block = (block_size * num_kv_heads * head_dim) // 2
            self._compressed_storage = torch.zeros(
                num_layers,
                2,
                num_blocks,
                packed_bytes_per_block,
                dtype=torch.int8,
                device=device,
            )
            if kv_quant == "turbo4":
                # Per-block, per-channel (low, scale) — last dim of size 2.
                self._scales_storage = torch.zeros(
                    num_layers,
                    2,
                    num_blocks,
                    num_kv_heads,
                    head_dim,
                    2,
                    dtype=dtype,
                    device=device,
                )
                self._radii_storage = None
            else:  # turbo3
                # Per-vector L2 norm: one float per (token, kv_head). Replaces
                # the per-channel (low, scale) pair from turbo4 — much smaller
                # (block_size * num_kv_heads vs num_kv_heads * head_dim * 2).
                self._scales_storage = None
                self._radii_storage = torch.zeros(
                    num_layers,
                    2,
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    dtype=dtype,
                    device=device,
                )
            self._rotation = generate_rotation_matrices(
                num_layers,
                head_dim,
                dtype=dtype,
                device=device,
                seed=0,
            )
            # Lloyd-Max codebooks cached on the pool so the fused kernel
            # doesn't reallocate per call. fp32 because the kernel does
            # codebook lookup + QJL nudge + radius multiply in fp32 before
            # casting back to bf16. turbo4 doesn't use Lloyd-Max but the
            # tensors are tiny (~96 bytes total) so we always allocate.
            self._k_codebook = lloyd_max_codebook(3, dtype=torch.float32, device=device)
            self._v_codebook = lloyd_max_codebook(4, dtype=torch.float32, device=device)
            self._fp8_storage = None
            self._fp8_scales = None
            self._fp8_scales_initialized = None
            self._nvfp4_storage = None
            self._nvfp4_block_scales = None
            self._nvfp4_global_sf = None
            self._nvfp4_initialized = None
            # No bf16 _storage in compressed mode; using `.storage` raises.
            self._storage = torch.empty(0, dtype=dtype, device=device)

        # `_layer_storage` always exists. Compressed pools leave it empty;
        # `storage_for_layer` raises in that case (compressed pools route
        # through `read_compressed_block`).
        if not hasattr(self, "_layer_storage"):
            self._layer_storage = []
        # `_layer_streams_storage` follows the same pattern: only the
        # uncompressed path populates it. The new stream accessors raise
        # on compressed pools (same surface as `storage_for_layer`).
        if not hasattr(self, "_layer_streams_storage"):
            self._layer_streams_storage = []

        self._free_list: list[int] = list(range(num_blocks))
        self._prefix_cache = prefix_cache

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_list)

    @property
    def storage(self) -> torch.Tensor:
        if self._kv_quant is not None:
            raise RuntimeError(
                f"`storage` only valid for uncompressed pool; got kv_quant={self._kv_quant!r}. "
                "Use read_compressed_block / write_compressed_block instead."
            )
        if self._is_heterogeneous:
            raise RuntimeError(
                "`storage` is the rectangular-tensor view; pool has heterogeneous "
                "per-layer KV shape so there is no single rectangular tensor. "
                "Use `storage_for_layer(layer_idx)` instead."
            )
        return self._storage

    @property
    def num_kv_heads(self) -> int:
        if self._is_heterogeneous:
            raise RuntimeError(
                "`num_kv_heads` is the homogeneous shortcut; pool has heterogeneous "
                "per-layer KV shape. Use `num_kv_heads_for_layer(layer_idx)` instead."
            )
        return self._homogeneous_num_kv_heads

    @property
    def head_dim(self) -> int:
        if self._is_heterogeneous:
            raise RuntimeError(
                "`head_dim` is the homogeneous shortcut; pool has heterogeneous "
                "per-layer KV shape. Use `head_dim_for_layer(layer_idx)` instead."
            )
        return self._homogeneous_head_dim

    @property
    def kv_quant(self) -> str | None:
        return self._kv_quant

    @property
    def attention_backend(self) -> str:
        return self._attention_backend

    @property
    def layer_attention(self) -> list[LayerAttentionSpec]:
        return self._layer_attention

    def num_kv_heads_for_layer(self, layer_idx: int) -> int:
        """Per-layer `num_kv_heads` from the pool's `layer_kv_shape`.

        Legacy K/V accessor — valid only on layers whose stream layout is
        the standard `["k", "v"]` pair with identical shape. Stream-
        descriptor layouts (DeepSeek MLA) raise; callers must migrate to
        `num_kv_heads_for_stream(layer_idx, name)`.
        """
        if not self._is_legacy_kv_layout:
            raise RuntimeError(
                "pool has a non-standard stream layout; use "
                "`num_kv_heads_for_stream(layer_idx, name)` instead."
            )
        return self._layer_kv_shape[layer_idx][0]

    def head_dim_for_layer(self, layer_idx: int) -> int:
        """Per-layer `head_dim` from the pool's `layer_kv_shape`.

        Same legacy-K/V constraint as `num_kv_heads_for_layer`.
        """
        if not self._is_legacy_kv_layout:
            raise RuntimeError(
                "pool has a non-standard stream layout; use "
                "`head_dim_for_stream(layer_idx, name)` instead."
            )
        return self._layer_kv_shape[layer_idx][1]

    def storage_for_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(K_pool, V_pool)` for `layer_idx`, each of shape
        `(num_blocks, block_size, num_kv_heads_l, head_dim_l)`.

        Uncompressed pools only. Compressed pools (`kv_quant != None`)
        raise; the dispatcher in `packed_attention_forward` uses
        `read_compressed_block` / the materialized fallback there.

        Legacy K/V accessor — valid only on layers whose stream layout
        is exactly `["k", "v"]` with identical shape. Stream-descriptor
        layouts (DeepSeek MLA) raise; callers migrate to
        `storage_for_stream(layer_idx, name)`.
        """
        if self._kv_quant is not None:
            raise RuntimeError(
                f"`storage_for_layer` only valid for uncompressed pool; got "
                f"kv_quant={self._kv_quant!r}. Use read_compressed_block instead."
            )
        if not self._is_legacy_kv_layout:
            raise RuntimeError(
                "pool has a non-standard stream layout; use "
                "`storage_for_stream(layer_idx, name)` instead."
            )
        layer_storage = self._layer_storage[layer_idx]
        return layer_storage[0], layer_storage[1]

    # --- Stream-descriptor API (Stage C3) ---------------------------------
    # Generalizes the legacy K/V accessors. Every layer carries a list of
    # named streams; the storage layout is the same for legacy K/V (where
    # the streams alias the rectangular K/V tensors) and MLA-style models
    # (where each stream is its own tensor). Standard MHA models can keep
    # using the legacy accessors; MLA-aware code uses these.

    def stream_specs(self, layer_idx: int) -> list[StreamSpec]:
        """All `StreamSpec`s for `layer_idx`."""
        return list(self._layer_stream_specs[layer_idx])

    def stream_names(self, layer_idx: int) -> list[str]:
        """Names of the streams on `layer_idx`."""
        return [spec.name for spec in self._layer_stream_specs[layer_idx]]

    def stream_spec(self, layer_idx: int, stream_name: str) -> StreamSpec:
        """Lookup a specific stream's `StreamSpec` (raises on unknown name)."""
        for spec in self._layer_stream_specs[layer_idx]:
            if spec.name == stream_name:
                return spec
        raise KeyError(
            f"layer {layer_idx} has no stream {stream_name!r}; "
            f"expected one of {self.stream_names(layer_idx)}"
        )

    def num_kv_heads_for_stream(self, layer_idx: int, stream_name: str) -> int:
        return self.stream_spec(layer_idx, stream_name).num_kv_heads

    def head_dim_for_stream(self, layer_idx: int, stream_name: str) -> int:
        return self.stream_spec(layer_idx, stream_name).head_dim

    def storage_for_stream(self, layer_idx: int, stream_name: str) -> torch.Tensor:
        """Returns the `(num_blocks, block_size, num_kv_heads_s, head_dim_s)`
        storage tensor for one named stream in one layer.

        For legacy K/V layers, `stream_name in {"k", "v"}` returns a view
        aliasing the rectangular `_layer_storage[layer_idx][0/1]` tensor
        (same memory). For MLA-style layers, returns the standalone tensor.
        """
        if self._kv_quant is not None:
            raise RuntimeError(
                f"`storage_for_stream` only valid for uncompressed pool; got "
                f"kv_quant={self._kv_quant!r}."
            )
        try:
            return self._layer_streams_storage[layer_idx][stream_name]
        except KeyError as exc:
            raise KeyError(
                f"layer {layer_idx} has no stream {stream_name!r}; "
                f"expected one of {self.stream_names(layer_idx)}"
            ) from exc

    @property
    def rotation(self) -> torch.Tensor | None:
        return self._rotation

    @property
    def prefix_cache(self) -> PrefixCache | None:
        return self._prefix_cache

    def allocate(self) -> int:
        if self._free_list:
            return self._free_list.pop()
        # Free pool empty: ask the prefix cache to surrender its oldest unreferenced
        # block. Returns None if the cache is empty or every cached block is pinned
        # by a running slot, in which case we genuinely have no memory.
        if self._prefix_cache is not None:
            evicted = self._prefix_cache.evict_lru()
            if evicted is not None:
                return evicted
        raise OutOfMemoryError("BlockPool: no free blocks available")

    def free(self, block_id: int) -> None:
        """Release a block. Cached blocks defer to PrefixCache (decref); else go to free list.

        With a prefix cache configured, a "freed" block that's cached stays in
        the cache (refcount-1) until LRU eviction reclaims it. Uncached blocks
        (e.g., the partial last block of a prompt) go straight to the free list.
        """
        if self._prefix_cache is not None and self._prefix_cache.is_cached(block_id):
            self._prefix_cache.decref(block_id)
        else:
            self._free_list.append(block_id)

    def view(self, block_id: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (key_block, value_block) views into storage for one block at one layer."""
        if self._kv_quant is not None:
            raise RuntimeError(
                f"`view` only valid for uncompressed pool; got kv_quant={self._kv_quant!r}"
            )
        # Each view is shape (block_size, num_kv_heads, head_dim); writes go through.
        return self._storage[layer_idx, 0, block_id], self._storage[layer_idx, 1, block_id]

    def read_compressed_block(self, layer_idx: int, kv_idx: int, block_id: int) -> torch.Tensor:
        """Read one compressed block as a `(block_size, num_kv_heads, head_dim)` tensor.

        Dispatches by mode:
          - turbo4: dequant per-channel uniform 4-bit + inverse rotation.
          - turbo3: dequant polar Lloyd-Max codebook (3-bit K + QJL,
            4-bit V) using stored radii + inverse rotation.

        Returns a fresh bf16/fp16 tensor in the original (un-rotated)
        representation. Caller is responsible for only reading positions
        ``< self._num_tokens[batch_idx]``; tail slots beyond seq_len
        contain whatever the last write quantized.
        """
        if self._kv_quant is None:
            raise RuntimeError("read_compressed_block requires a compressed kv_quant; got None")
        assert self._compressed_storage is not None
        assert self._rotation is not None
        packed = self._compressed_storage[layer_idx, kv_idx, block_id]

        if self._kv_quant == "turbo4":
            assert self._scales_storage is not None
            low = self._scales_storage[layer_idx, kv_idx, block_id, :, :, 0]
            scale = self._scales_storage[layer_idx, kv_idx, block_id, :, :, 1]
            rotated = dequantize_kv_block(
                packed,
                low,
                scale,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.dtype,
            )
        else:  # turbo3
            assert self._radii_storage is not None
            radii = self._radii_storage[layer_idx, kv_idx, block_id]
            # K side gets 3-bit + QJL (4 bits stored); V side gets 4-bit
            # Lloyd-Max codebook (4 bits stored). Same packed layout,
            # different decoder.
            if kv_idx == 0:
                rotated = polar_dequantize_block(
                    packed,
                    radii,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    bits=3,
                    use_lloyd_max=True,
                    use_qjl=True,
                )
            else:
                rotated = polar_dequantize_block(
                    packed,
                    radii,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    bits=4,
                    use_lloyd_max=True,
                    use_qjl=False,
                )

        return inverse_rotate(rotated, self._rotation[layer_idx])

    def write_compressed_block(
        self,
        layer_idx: int,
        kv_idx: int,
        block_id: int,
        block: torch.Tensor,
    ) -> None:
        """Write a full ``(block_size, num_kv_heads, head_dim)`` block to compressed storage.

        Dispatches by mode (same structure as ``read_compressed_block``).
        Caller has already collected a complete block's worth of data
        (the rotation and quantization don't make sense on a sub-block).
        """
        if self._kv_quant is None:
            raise RuntimeError("write_compressed_block requires a compressed kv_quant; got None")
        if block.shape != (self.block_size, self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"expected block shape ({self.block_size}, {self.num_kv_heads}, "
                f"{self.head_dim}); got {tuple(block.shape)}"
            )
        assert self._compressed_storage is not None
        assert self._rotation is not None

        rotated = rotate(block, self._rotation[layer_idx])

        if self._kv_quant == "turbo4":
            assert self._scales_storage is not None
            packed, low, scale = quantize_kv_block(rotated)
            self._compressed_storage[layer_idx, kv_idx, block_id] = packed
            self._scales_storage[layer_idx, kv_idx, block_id, :, :, 0] = low
            self._scales_storage[layer_idx, kv_idx, block_id, :, :, 1] = scale
        else:  # turbo3
            assert self._radii_storage is not None
            if kv_idx == 0:
                packed, radii = polar_quantize_block(
                    rotated, bits=3, use_lloyd_max=True, use_qjl=True
                )
            else:
                packed, radii = polar_quantize_block(
                    rotated, bits=4, use_lloyd_max=True, use_qjl=False
                )
            self._compressed_storage[layer_idx, kv_idx, block_id] = packed
            self._radii_storage[layer_idx, kv_idx, block_id] = radii
