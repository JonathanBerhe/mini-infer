from collections.abc import Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Tokenizer:
    """Thin wrapper over HF AutoTokenizer; stable seam for chat templating and perf tweaks later."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = hf_tokenizer

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Tokenizer":
        return cls(AutoTokenizer.from_pretrained(model_name))

    def encode(self, text: str) -> list[int]:
        # `add_special_tokens=True` is the HF default and matches each model's
        # training-time convention. Llama/Gemma prepend BOS; Qwen2 doesn't add
        # anything for plain text. Skipping special tokens (the previous
        # behavior) caused Gemma 3 to position-0 the first content token
        # instead of BOS, drifting per-layer hidden states ~10% off HF and
        # flipping greedy argmax (Paris -> France).
        ids = [int(t) for t in self._tokenizer.encode(text)]
        # Gemma 4's tokenizer config sets `add_bos_token=False` even though
        # the model is trained with BOS at position 0. HF's `encode()`
        # therefore omits it, and so does HF's own `model.generate(...)` —
        # users are expected to either use the chat template or prepend
        # BOS themselves. Do the latter here so plain-text inference
        # works end-to-end across the model registry, matching the
        # Gemma 3 behavior (Gemma 3's tokenizer does prepend BOS
        # automatically despite the same flag).
        bos_id = self._tokenizer.bos_token_id
        if bos_id is not None and (not ids or ids[0] != bos_id):
            ids = [int(bos_id)] + ids
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=True))

    @property
    def eos_token_id(self) -> int:
        eos = self._tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no EOS token")
        return int(eos)

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.vocab_size)
