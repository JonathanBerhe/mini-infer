"""Attention over contiguous per-row K/V buffers, read only up to each row's length.

The state-cache families (Qwen3.8, Nemotron-H) keep a full-attention layer's
history as one contiguous `(B, buffer_len, num_kv_heads, head_dim)` buffer
per row. Their PyTorch path reads the whole buffer at every decode step and
masks the slots past each row's length, so a step costs the same at the
first token as at `max_seq_len`. Two kernels read a row only up to its
length instead:

  - FlashAttention's `flash_attn_with_kvcache` reads the buffers in place
    and stops at each row's `cache_seqlens`. It takes the lengths on the
    device, so a CUDA graph can capture a call. flash-attn 2 has no
    prebuilt wheel past torch 2.8, so most images do not have it.
  - FlashInfer's paged attention, which every CUDA image of mini-infer
    installs. A buffer whose length is a multiple of `PAGE_SIZE` is a run
    of whole pages: row `b` owns pages `b * pages_per_row` and up. A
    `ContiguousFlashInferPlan` lists, once per step, the pages that hold
    each row's keys, and every attention layer of the step runs on that
    plan. A plan needs the lengths on the host, so a CUDA-graph step is
    planned before its replay (`before_decode_step` in
    `engine/cuda_graph_decode.py`).

`contiguous_backend` picks the kernel for a buffer. CPU and MPS have
neither, and the model keeps its own PyTorch path there.
"""

from __future__ import annotations

from typing import Any

import torch

from mini_infer.device import is_cuda_device, require_cuda_device

try:
    from flash_attn import flash_attn_with_kvcache

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False
    flash_attn_with_kvcache = None

try:
    import flashinfer  # type: ignore[import-not-found]

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False
    flashinfer = None

# Both kernels run in half precision only.
_HALF_DTYPES = (torch.float16, torch.bfloat16)

# Tokens per FlashInfer page. The paged KV cache uses the same size, which
# runs on every GPU that mini-infer targets. The state caches round their
# attention buffers up to a multiple of it (`padded_buffer_len`).
PAGE_SIZE = 16

# Runtime switch for A/B runs on one model load: when True, no buffer gets a
# kernel and the model takes its PyTorch path.
_DISABLED_FOR_BENCH = False

# FlashInfer keeps its plan and its split-KV scratch in this buffer. The
# B200 needs more than FlashInfer's default 128 MiB for long rows (see
# `cache/flashinfer_backend.py`). It is not shared with the paged backend,
# so a plan here never overwrites a plan there.
_WORKSPACE_BYTES = 256 * 1024 * 1024
_workspace: dict[torch.device, torch.Tensor] = {}
_prefill_wrappers: dict[torch.device, Any] = {}
_decode_wrappers: dict[tuple[torch.device, int, int], Any] = {}


def padded_buffer_len(max_seq_len: int) -> int:
    """`max_seq_len` rounded up to whole FlashInfer pages."""
    return -(-max_seq_len // PAGE_SIZE) * PAGE_SIZE


def contiguous_backend(k_buffer: torch.Tensor) -> str | None:
    """The kernel that reads this buffer only up to each row's length, or None.

    Single source of truth for the choice, so the models do not test devices
    themselves. "flash_attn" when flash-attn is installed, else "flashinfer"
    when FlashInfer is installed and the buffer is whole pages, on CUDA with
    a half-precision buffer. None means the model's PyTorch path.
    """
    if _DISABLED_FOR_BENCH:
        return None
    if not (is_cuda_device(k_buffer.device) and k_buffer.dtype in _HALF_DTYPES):
        return None
    if _FLASH_ATTN_AVAILABLE:
        return "flash_attn"
    if _FLASHINFER_AVAILABLE and k_buffer.shape[1] % PAGE_SIZE == 0:
        return "flashinfer"
    return None


def supports_contiguous_kernel(k_buffer: torch.Tensor) -> bool:
    """Whether `contiguous_attention` (flash-attn) can run on this buffer."""
    return contiguous_backend(k_buffer) == "flash_attn"


def contiguous_attention(
    query: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Causal grouped-query attention of a span over each row's cached keys.

    The span's own K/V must already be in the buffers. Row `b` attends to its
    first `lengths[b]` keys, and the span's last query sits at the last of
    them, so query `i` of a span of `T` sees keys up to `lengths[b] - T + i`.

    Args:
        query: `(B, T, num_attention_heads, head_dim)`.
        k_buffer: `(B, buffer_len, num_kv_heads, head_dim)` stored keys.
        v_buffer: `(B, buffer_len, num_kv_heads, head_dim)` stored values.
        lengths: `(B,)` integer cache length of each row after the span, on
            the buffers' device.
        scale: softmax scale.

    Returns:
        `(B, T, num_attention_heads, head_dim)` in the query's dtype.
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise RuntimeError("flash-attn not installed; install via the [cuda] extra")
    require_cuda_device(query.device, "flash_attn_with_kvcache")
    out: torch.Tensor = flash_attn_with_kvcache(
        query.to(k_buffer.dtype),
        k_buffer,
        v_buffer,
        cache_seqlens=lengths.to(torch.int32),
        softmax_scale=scale,
        causal=True,
    )
    return out.to(query.dtype)


def page_index(lengths: list[int], pages_per_row: int) -> tuple[list[int], torch.Tensor, list[int]]:
    """The FlashInfer page index of rows that own consecutive pages.

    Row `b` owns pages `b * pages_per_row` to `(b + 1) * pages_per_row - 1`,
    and its first `lengths[b]` tokens fill the first of them.

    Returns:
        `(indptr, indices, last_page_len)`: row `b` reads pages
        `indices[indptr[b]:indptr[b + 1]]`, and `last_page_len[b]` tokens of
        the last one. `indices` is an int32 host tensor.
    """
    indptr = [0]
    chunks = []
    last_page_len = []
    for row, length in enumerate(lengths):
        if not 0 < length <= pages_per_row * PAGE_SIZE:
            raise ValueError(
                f"row {row}: length {length} is outside 1..{pages_per_row * PAGE_SIZE}"
            )
        pages = -(-length // PAGE_SIZE)
        first = row * pages_per_row
        chunks.append(torch.arange(first, first + pages, dtype=torch.int32))
        indptr.append(indptr[-1] + pages)
        last_page_len.append(length - (pages - 1) * PAGE_SIZE)
    return indptr, torch.cat(chunks), last_page_len


def _workspace_for(device: torch.device) -> torch.Tensor:
    if device not in _workspace:
        _workspace[device] = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _workspace[device]


def _prefill_wrapper(device: torch.device) -> Any:
    assert flashinfer is not None
    if device not in _prefill_wrappers:
        _prefill_wrappers[device] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace_for(device), kv_layout="NHD"
        )
    return _prefill_wrappers[device]


def _decode_wrapper(device: torch.device, batch: int, pages: int) -> Any:
    """The fixed-buffer wrapper of a one-token step over `batch` rows.

    FlashInfer copies each plan into these buffers, so a `run()` captured in
    a CUDA graph reads the latest plan. It uses the prefill kernels with one
    query per row, because the decode kernels take only power-of-two groups
    of query heads (Qwen3.8 has 6).
    """
    assert flashinfer is not None
    key = (device, batch, pages)
    if key not in _decode_wrappers:
        _decode_wrappers[key] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace_for(device),
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.empty(batch + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.empty(batch + 1, dtype=torch.int32, device=device),
            paged_kv_indices_buf=torch.empty(pages, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buf=torch.empty(batch, dtype=torch.int32, device=device),
        )
    return _decode_wrappers[key]


def plan_contiguous_step(
    layer_states: list[Any],
    lengths: list[int],
    q_len: int,
    *,
    num_q_heads: int,
    scale: float,
) -> None:
    """Plan one step for the attention layers of a state cache, if FlashInfer runs them.

    `layer_states` are the cache's attention layer states, each with `k`,
    `v` and `plan`, all with buffers of one shape. The first call builds a
    plan and gives it to every layer. Nothing happens when the buffers take
    another path.
    """
    if not layer_states:
        return
    k = layer_states[0].k
    if contiguous_backend(k) != "flashinfer":
        return
    plan = layer_states[0].plan
    if plan is None:
        plan = ContiguousFlashInferPlan(
            batch_size=k.shape[0],
            buffer_len=k.shape[1],
            num_q_heads=num_q_heads,
            num_kv_heads=k.shape[2],
            head_dim=k.shape[3],
            device=k.device,
        )
        for state in layer_states:
            state.plan = plan
    plan.plan(lengths, q_len, k.dtype, scale)


class ContiguousFlashInferPlan:
    """One step's FlashInfer plan over the contiguous K/V buffers of a state cache.

    Every full-attention layer of a model has buffers of the same shape and
    the same row lengths, so one plan serves all of them. Call `plan` once
    per step, before the first layer, then `run` once per layer. A
    one-token step plans into fixed buffers, so its `run` can be captured in
    a CUDA graph and replayed after each new `plan`.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        buffer_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        if buffer_len % PAGE_SIZE:
            raise ValueError(f"buffer length {buffer_len} is not whole pages of {PAGE_SIZE}")
        self.batch_size = batch_size
        self.pages_per_row = buffer_len // PAGE_SIZE
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self._wrapper: Any = None
        self._q_len = 0

    def plan(self, lengths: list[int], q_len: int, dtype: torch.dtype, scale: float) -> None:
        """Plan a step in which row `b` holds `lengths[b]` keys after a span of `q_len`.

        The lengths are host integers, so planning makes the host wait for
        nothing. Free rows of a batch count as length 1.
        """
        if not _FLASHINFER_AVAILABLE:
            raise RuntimeError("flashinfer-python not installed; install via the [cuda] extra")
        if len(lengths) != self.batch_size:
            raise ValueError(f"got {len(lengths)} lengths for a batch of {self.batch_size}")
        if is_cuda_device(self.device) and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan before the CUDA graph capture, not inside it")
        indptr, indices, last_page_len = page_index(lengths, self.pages_per_row)
        qo_indptr = [row * q_len for row in range(self.batch_size + 1)]
        one_token = q_len == 1
        if one_token:
            wrapper = _decode_wrapper(
                self.device, self.batch_size, self.batch_size * self.pages_per_row
            )
        else:
            wrapper = _prefill_wrapper(self.device)
        # FlashInfer reads the boundaries on the host and copies them into
        # its buffers. From pinned memory the copies do not wait for the GPU,
        # and the caching host allocator keeps each pinned block until its
        # copy is done. It copies host indices with a blocking copy, so the
        # indices go in on the device.
        pin = is_cuda_device(self.device)

        def host(values: list[int]) -> torch.Tensor:
            tensor = torch.tensor(values, dtype=torch.int32)
            return tensor.pin_memory() if pin else tensor

        device_indices = (indices.pin_memory() if pin else indices).to(
            self.device, non_blocking=pin
        )
        # A wrapper made inside a forward holds inference tensors, which
        # only inference mode may write.
        with torch.inference_mode():
            wrapper.plan(
                host(qo_indptr),
                host(indptr),
                device_indices,
                host(last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                PAGE_SIZE,
                causal=True,
                pos_encoding_mode="NONE",
                sm_scale=scale,
                q_data_type=dtype,
                kv_data_type=dtype,
                non_blocking=pin,
            )
        self._wrapper = wrapper
        self._q_len = q_len

    def run(
        self, query: torch.Tensor, k_buffer: torch.Tensor, v_buffer: torch.Tensor
    ) -> torch.Tensor:
        """Causal attention of `(B, T, heads, head_dim)` queries over the planned keys.

        Returns `(B, T, heads, head_dim)` in the query's dtype. The span's own
        K/V must already be in the buffers.
        """
        batch, q_len = query.shape[0], query.shape[1]
        if self._wrapper is None or q_len != self._q_len or batch != self.batch_size:
            raise RuntimeError(
                f"no plan for a span of {q_len} tokens over {batch} rows; "
                "call plan() first for this step"
            )
        kv_heads, head_dim = k_buffer.shape[2], k_buffer.shape[3]
        pages = (
            k_buffer.view(-1, PAGE_SIZE, kv_heads, head_dim),
            v_buffer.view(-1, PAGE_SIZE, kv_heads, head_dim),
        )
        packed = query.reshape(batch * q_len, query.shape[2], head_dim).to(k_buffer.dtype)
        out: torch.Tensor = self._wrapper.run(packed.contiguous(), pages)
        return out.view(batch, q_len, query.shape[2], head_dim).to(query.dtype)
