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
        return [int(t) for t in self._tokenizer.encode(text)]

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
