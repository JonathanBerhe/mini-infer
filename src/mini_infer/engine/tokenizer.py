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
            ids = [int(bos_id), *ids]
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=True))

    def has_chat_template(self) -> bool:
        return getattr(self._tokenizer, "chat_template", None) is not None

    def encode_chat(self, user_message: str, *, enable_thinking: bool | None = None) -> list[int]:
        """Encode a single user turn through the model's own chat template.

        Instruction-tuned checkpoints are trained almost entirely on templated
        text, so feeding raw prompts puts the model off-distribution.

        `add_generation_prompt=True` appends the assistant header so the next
        token continues the reply rather than the user's own turn. Raises if
        the tokenizer ships no template, rather than silently degrading to
        plain `encode`, so a caller asking for templating knows it happened.

        `enable_thinking` selects a reasoning mode on templates that have one
        (Qwen3), and is forwarded only when set so templates without it are
        unaffected. It changes the generated distribution completely rather
        than cosmetically: Qwen3 defaults to `True`, leaving the assistant
        turn open so the model emits a `<think>` block first, while `False`
        makes the template pre-close that block (`<think>\\n\\n</think>`) so
        the answer starts immediately. Anything comparing a model against
        something tuned on one mode has to match it. This defaults to the
        template's own default rather than to `False`, so the choice stays
        explicit at the call site.
        """
        if not self.has_chat_template():
            raise ValueError("tokenizer has no chat_template; use encode() instead")
        kwargs: dict[str, object] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        out = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        # transformers 5.x returns a BatchEncoding here, not the flat id list
        # older versions returned; iterating the mapping yields its KEYS, so
        # unwrap explicitly rather than trusting the return shape.
        if hasattr(out, "input_ids") or isinstance(out, dict):
            out = out["input_ids"]
        # A batched call would nest one list per conversation; we pass one.
        if out and isinstance(out[0], list):
            out = out[0]
        return [int(t) for t in out]

    @property
    def eos_token_id(self) -> int:
        eos = self._tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no EOS token")
        return int(eos)

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.vocab_size)
