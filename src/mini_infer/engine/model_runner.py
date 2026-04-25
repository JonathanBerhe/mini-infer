import logging

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from mini_infer.cache.kv_cache import KVCache
from mini_infer.engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


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
    """Loads a HF causal LM and runs prefill + decode against our own KV cache."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        device: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> "ModelRunner":
        resolved = _resolve_device(device)
        actual_dtype = dtype if dtype is not None else _dtype_for(resolved)
        logger.info("Loading %s on %s with dtype=%s", model_name, resolved, actual_dtype)
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=actual_dtype).to(resolved)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=resolved)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def prefill(self, prompt_tokens: list[int]) -> tuple[KVCache, torch.Tensor]:
        """Process the full prompt, return populated KV cache and last-position logits."""
        cache = KVCache()
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

    def decode(self, cache: KVCache, last_token: int) -> tuple[KVCache, torch.Tensor]:
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
