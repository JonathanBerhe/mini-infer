import logging

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_attention import supports_paged_kernel
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.attention_patch import patch_model_attention
from mini_infer.engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_NUM_BLOCKS = 1024
DEFAULT_BLOCK_SIZE = 16


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
    """Loads a HF causal LM and runs prefill + batched decode against a paged KV cache."""

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
    ) -> "ModelRunner":
        resolved = _resolve_device(device)
        actual_dtype = dtype if dtype is not None else _dtype_for(resolved)
        logger.info("Loading %s on %s with dtype=%s", model_name, resolved, actual_dtype)
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=actual_dtype).to(resolved)
        model.eval()

        cfg = model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            dtype=actual_dtype,
            device=resolved,
        )

        # On kernel-capable devices, patch the model's attention layers so decode
        # steps use the batched paged kernel and skip the materialization fallback.
        # Other devices (MPS / CPU) keep using the materialization path through
        # `cache.update()`, which is now batch-aware. Benchmarks can set
        # use_paged_kernel=False to A/B against materialization on the same hardware.
        if use_paged_kernel and supports_paged_kernel(resolved):
            patch_model_attention(model)

        return cls(model=model, tokenizer=tokenizer, device=resolved, block_pool=block_pool)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    def prefill(self, prompt_tokens: list[int]) -> tuple[PagedKVCache, torch.Tensor]:
        """Process the full prompt; return a single-request KV cache and last-position logits.

        The returned cache has `batch_size=1` and is intended to be merged into the
        scheduler's long-lived multi-request cache via `merge_request()`.
        """
        cache = PagedKVCache(self._block_pool)
        cache.add_request_slot()
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        seq_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=self.device)
        with torch.inference_mode():
            out = self._model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
        return cache, out.logits[0, -1, :]

    def decode(self, cache: PagedKVCache, last_token: int) -> tuple[PagedKVCache, torch.Tensor]:
        """Single-request convenience wrapper around `decode_batch`.

        Used by golden tests and the decode-latency benchmark, both of which
        iterate prefill -> decode in a loop with `cache.batch_size == 1`.
        """
        cache, logits_list = self.decode_batch(cache, [last_token])
        return cache, logits_list[0]

    def decode_batch(
        self, cache: PagedKVCache, last_tokens: list[int]
    ) -> tuple[PagedKVCache, list[torch.Tensor]]:
        """Run one batched decode step over `cache.batch_size` requests.

        `last_tokens[b]` is the most recently sampled token for batch slot `b`. The
        forward writes its K/V into the cache and produces next-token logits per
        request. Returns the (mutated) cache and a list of per-request logit
        tensors, each shape `(vocab_size,)`.
        """
        batch_size = cache.batch_size
        if len(last_tokens) != batch_size:
            raise ValueError(
                f"last_tokens has length {len(last_tokens)} but cache batch_size={batch_size}"
            )
        seq_lens_before = cache.seq_lens_list()
        input_ids = torch.tensor([[token] for token in last_tokens], device=self.device)
        position_ids = torch.tensor([[seq_len] for seq_len in seq_lens_before], device=self.device)
        # Attention mask shape (B, max_seq + 1) with 1s up to each request's
        # current seq_len + 1. The patched attention path ignores the values but
        # HF's outer Qwen2Model.forward inspects the shape; non-patched layers
        # use the mask to ignore the padded tail of shorter requests.
        max_after = max(seq_lens_before) + 1
        attention_mask = torch.zeros((batch_size, max_after), device=self.device, dtype=torch.long)
        for b, seq_len in enumerate(seq_lens_before):
            attention_mask[b, : seq_len + 1] = 1
        cache_position = torch.tensor([max(seq_lens_before)], device=self.device)
        with torch.inference_mode():
            out = self._model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
        # out.logits has shape (B, q_len=1, vocab_size). Slice off the q_len dim
        # and split along batch so each request gets its own (vocab_size,) tensor.
        last_position_logits = out.logits[:, -1, :]  # (B, vocab_size)
        per_request_logits = list(last_position_logits.unbind(dim=0))
        return cache, per_request_logits
