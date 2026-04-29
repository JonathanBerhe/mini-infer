import logging

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.engine.attention_patch import patch_model_attention
from mini_infer.engine.tokenizer import Tokenizer
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
        model: PreTrainedModel,
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
        use_paged_kernel: bool = True,
        prefix_cache: bool = False,
        quant: str | None = None,
        quant_lm_head: bool = False,
    ) -> "ModelRunner":
        resolved = _resolve_device(device)
        actual_dtype = dtype if dtype is not None else _dtype_for(resolved)
        logger.info("Loading %s on %s with dtype=%s", model_name, resolved, actual_dtype)
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=actual_dtype).to(resolved)
        model.eval()

        if quant is not None:
            if quant != "int8":
                raise ValueError(f"unsupported quant={quant!r}; expected None or 'int8'")
            skip = set() if quant_lm_head else set(_DEFAULT_QUANT_SKIP)
            n_replaced = quantize_model_to_int8(model, skip_modules=skip)
            logger.info(
                "Quantized %d nn.Linear modules to int8 (skip=%s)", n_replaced, sorted(skip)
            )

        cfg = model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        prefix_cache_obj = PrefixCache(block_size=block_size) if prefix_cache else None
        block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            dtype=actual_dtype,
            device=resolved,
            prefix_cache=prefix_cache_obj,
        )

        # Apply the architecture-specific attention patch. The patched forward
        # dispatches `packed_attention_forward`, which uses FlashAttention varlen
        # on CUDA and a PyTorch reference elsewhere — so the patch is correct on
        # every device. Set use_paged_kernel=False to A/B against HF's stock
        # attention path.
        if use_paged_kernel:
            patch_model_attention(model)

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
        max_seqlen_q = 0
        for batch_idx in range(batch_size):
            q_len = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            if q_len > max_seqlen_q:
                max_seqlen_q = q_len
            position_ids_flat.extend(
                range(position_offsets[batch_idx], position_offsets[batch_idx] + q_len)
            )
        position_ids = torch.tensor([position_ids_flat], device=device, dtype=torch.long)

        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)

        # `attention_mask` and `cache_position` are required by HF's outer
        # forward to size some internal computations, but our patched attention
        # ignores them (it uses cu_seqlens_q + the cache's per-slot seq_lens).
        max_kv_after = max(
            position_offsets[batch_idx] + cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            for batch_idx in range(batch_size)
        )
        attention_mask = torch.ones((1, max_kv_after), device=device, dtype=torch.long)
        cache_position = torch.tensor([max(position_offsets)], device=device, dtype=torch.long)

        with torch.inference_mode():
            out = self._model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                cu_seqlens_q=cu_seqlens_q_t,
                max_seqlen_q=max_seqlen_q,
            )
        logits: torch.Tensor = out.logits
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
