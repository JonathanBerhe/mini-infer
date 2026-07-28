import logging
from typing import Any

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models import load_model
from mini_infer.models.base import BaseCausalLM
from mini_infer.quant import quantize_model_to_int8

logger = logging.getLogger(__name__)

DEFAULT_NUM_BLOCKS = 1024
DEFAULT_BLOCK_SIZE = 16

# Modules whose names match these (suffix-of-dotted-path) are NOT quantized by
# default. `lm_head` is the embedding-tied output projection; quantization noise
# there directly affects sampling, so industry practice is to leave it in float.
_DEFAULT_QUANT_SKIP = frozenset({"lm_head"})


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype_for(device: str) -> torch.dtype:
    """fp16 on MPS (M1 has fp16 in hardware; bf16 is software-emulated and crashes on M1)."""
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


class ModelRunner:
    """Loads a HF causal LM and runs packed-varlen forwards against a paged KV cache.

    All forwards (prefill, decode, chunked-prefill) flow through `forward_step`,
    which packs per-request q-tokens into a single sequence and dispatches the
    patched attention layer's varlen path. The convenience wrappers (`prefill`,
    `decode`, `decode_batch`, `prefill_chunk`) build the right packed inputs
    for their single-request / uniform-q-len cases and call `forward_step`.
    """

    def __init__(
        self,
        model: BaseCausalLM,
        tokenizer: Tokenizer,
        device: str,
        block_pool: BlockPool,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.device = device
        self._block_pool = block_pool

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        prefix_cache: bool = False,
        quant: str | None = None,
        quant_lm_head: bool = False,
        kv_quant: str | None = None,
        attention_backend: str = "flash_attn",
    ) -> "ModelRunner":
        resolved = _resolve_device(device)
        actual_dtype = dtype if dtype is not None else _dtype_for(resolved)
        logger.info("Loading %s on %s with dtype=%s", model_name, resolved, actual_dtype)
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = load_model(model_name, dtype=actual_dtype, device=resolved)

        if quant is not None:
            if quant != "int8":
                raise ValueError(f"unsupported quant={quant!r}; expected None or 'int8'")
            skip = set() if quant_lm_head else set(_DEFAULT_QUANT_SKIP)
            n_replaced = quantize_model_to_int8(model, skip_modules=skip)
            logger.info(
                "Quantized %d nn.Linear modules to int8 (skip=%s)", n_replaced, sorted(skip)
            )

        # Each owned model exposes its KV-cache dims; the pool is built from
        # them without re-loading HF's config. SWA-aware models also report
        # a per-layer attention pattern so the dispatcher can window the
        # right layers. Heterogeneous-KV models (Gemma 4 31B, MLA, V4)
        # additionally report per-layer (num_kv_heads, head_dim); the
        # default `None` keeps the pool homogeneous via `kv_cache_dims`.
        dims = model.kv_cache_dims
        # Some models declare a hard kernel constraint that overrides the
        # caller's `attention_backend` (e.g. Gemma 4 31B has head_dim=512
        # on full layers, which neither flash-attn 2 nor FlashInfer
        # prefill supports — they need our materialized SDPA fallback).
        # Mirrors vLLM's `verify_and_update_config` model-side override.
        required_backend = model.required_attention_backend()
        if required_backend is not None and required_backend != attention_backend:
            logger.info(
                "Forcing attention_backend=%r for %s (required by model; was %r)",
                required_backend,
                type(model).__name__,
                attention_backend,
            )
            attention_backend = required_backend
        prefix_cache_obj = PrefixCache(block_size=block_size) if prefix_cache else None
        block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=dims.num_layers,
            num_kv_heads=dims.num_kv_heads,
            head_dim=dims.head_dim,
            dtype=actual_dtype,
            device=resolved,
            prefix_cache=prefix_cache_obj,
            kv_quant=kv_quant,
            attention_backend=attention_backend,
            layer_attention=model.per_layer_attention(),
            layer_kv_shape=model.per_layer_kv_shape(),
            layer_streams=model.per_layer_streams(),
        )

        return cls(model=model, tokenizer=tokenizer, device=resolved, block_pool=block_pool)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    def forward_step(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
    ) -> list[torch.Tensor]:
        """Run ONE packed-varlen forward; return last-position logits per request.

        See `forward_step_packed` for the underlying packed-logits path. This
        helper slices the last position per request and is the standard entry
        point for the scheduler.
        """
        packed_logits = self.forward_step_packed(
            cache, packed_input_ids, cu_seqlens_q, position_offsets
        )
        per_request_logits: list[torch.Tensor] = []
        for batch_idx in range(cache.batch_size):
            last_pos = cu_seqlens_q[batch_idx + 1] - 1
            per_request_logits.append(packed_logits[0, last_pos, :])
        return per_request_logits

    def forward_step_packed(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        *,
        tap_layers: frozenset[int] | None = None,
        hidden_state_sink: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run ONE packed-varlen forward; return the raw packed logits.

        Args:
            cache: shared `PagedKVCache` whose `batch_size` matches the number
                of in-flight requests. The forward writes new K/V into each
                request's slot in place.
            packed_input_ids: every request's q-tokens flattened into one list,
                in cache-slot order. Total length equals `cu_seqlens_q[-1]`.
            cu_seqlens_q: cumulative q-length boundaries; `cu_seqlens_q[-1]` is
                the packed total length.
            position_offsets: per-request absolute position of each request's
                first new q-token. Equals `cache.seq_lens_list()[batch_idx]`
                BEFORE this step's append.
            tap_layers / hidden_state_sink: forwarded to the model only when
                `tap_layers` is set, to collect per-layer hidden states. Used
                by the DSpark drafter, which needs the target's hidden states
                at a fixed set of layers as its injected context
                (`engine/dspark/`). Only models whose `forward` accepts these
                kwargs support it (Qwen3 today); passing them to a model that
                doesn't will raise, which is the intended loud failure rather
                than a silently empty sink.

        Returns:
            A `(1, total_q, vocab_size)` tensor with logits at every q
            position. Spec-decode's verify phase needs all K+1 positions, not
            just the last; the standard `forward_step` slices the last one.
        """
        batch_size = cache.batch_size
        if len(position_offsets) != batch_size:
            raise ValueError(
                f"position_offsets has length {len(position_offsets)} but "
                f"cache.batch_size={batch_size}"
            )
        if len(cu_seqlens_q) != batch_size + 1:
            raise ValueError(
                f"cu_seqlens_q has {len(cu_seqlens_q)} entries; expected batch_size+1="
                f"{batch_size + 1}"
            )

        device = self.device
        input_ids = torch.tensor([packed_input_ids], device=device, dtype=torch.long)

        # Per-request position ids: positions reset per request so RoPE applies
        # the right rotation for each request's absolute position.
        position_ids_flat: list[int] = []
        for batch_idx in range(batch_size):
            q_len = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            position_ids_flat.extend(
                range(position_offsets[batch_idx], position_offsets[batch_idx] + q_len)
            )
        position_ids = torch.tensor([position_ids_flat], device=device, dtype=torch.long)

        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)

        extra: dict[str, Any] = {}
        if tap_layers is not None:
            extra["tap_layers"] = tap_layers
            extra["hidden_state_sink"] = hidden_state_sink

        with torch.inference_mode():
            logits: torch.Tensor = self._model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                cu_seqlens_q=cu_seqlens_q_t,
                **extra,
            )
        return logits

    def prefill(self, prompt_tokens: list[int]) -> tuple[PagedKVCache, torch.Tensor]:
        """Process the full prompt in one forward; return single-request cache + last logits.

        Convenience wrapper for golden tests and the decode-latency benchmark.
        """
        cache = PagedKVCache(self._block_pool)
        cache.add_request_slot()
        return self.prefill_chunk(cache, prompt_tokens, position_offset=0)

    def prefill_chunk(
        self,
        cache: PagedKVCache,
        chunk_tokens: list[int],
        position_offset: int,
    ) -> tuple[PagedKVCache, torch.Tensor]:
        """Process one chunk against an existing single-slot cache.

        Same packed forward as `forward_step`, with batch_size=1 enforced. Used
        as a convenience by tests; the scheduler builds packed inputs directly.
        """
        if cache.batch_size != 1:
            raise ValueError(f"prefill_chunk expects cache.batch_size == 1, got {cache.batch_size}")
        cu_seqlens_q = [0, len(chunk_tokens)]
        logits_list = self.forward_step(cache, chunk_tokens, cu_seqlens_q, [position_offset])
        return cache, logits_list[0]

    def decode(self, cache: PagedKVCache, last_token: int) -> tuple[PagedKVCache, torch.Tensor]:
        """Single-request convenience wrapper around `decode_batch`."""
        cache, logits_list = self.decode_batch(cache, [last_token])
        return cache, logits_list[0]

    def decode_batch(
        self, cache: PagedKVCache, last_tokens: list[int]
    ) -> tuple[PagedKVCache, list[torch.Tensor]]:
        """Run one batched decode step (q_len=1 per request) via `forward_step`.

        Each request contributes one new token; positions come from the cache's
        per-slot seq_lens (the absolute position of the new token).
        """
        batch_size = cache.batch_size
        if len(last_tokens) != batch_size:
            raise ValueError(
                f"last_tokens has length {len(last_tokens)} but cache batch_size={batch_size}"
            )
        cu_seqlens_q = list(range(batch_size + 1))  # [0, 1, 2, ..., B] — q_len=1 per request
        position_offsets = cache.seq_lens_list()
        logits_list = self.forward_step(cache, last_tokens, cu_seqlens_q, position_offsets)
        return cache, logits_list
