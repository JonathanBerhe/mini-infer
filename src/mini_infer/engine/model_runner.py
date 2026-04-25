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
    """Loads a HF causal LM and runs prefill + decode against a paged KV cache."""

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
        # steps use the paged kernel and skip the materialization fallback. Other
        # devices (MPS / CPU) keep using the materialization path. Benchmarks can
        # set use_paged_kernel=False to A/B against the materialization path on
        # the same hardware.
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
        """Process the full prompt, return populated KV cache and last-position logits."""
        cache = PagedKVCache(self._block_pool)
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
        """Run one decode step against the cache, return updated cache and next-token logits."""
        input_ids = torch.tensor([[last_token]], device=self.device)
        cache_len = cache.get_seq_length()
        attention_mask = torch.ones((1, cache_len + 1), device=self.device, dtype=torch.long)
        position_ids = torch.tensor([[cache_len]], device=self.device)
        cache_position = torch.tensor([cache_len], device=self.device)
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
