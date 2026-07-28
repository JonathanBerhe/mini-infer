"""`Tokenizer.encode_chat`: the chat-template seam.

Instruction-tuned checkpoints see templated text almost exclusively in
training, so feeding raw prompts puts them off-distribution. That is mostly
invisible for plain generation quality but shows up sharply in anything
measuring agreement between two models, which is why the DSpark benchmark
needs this seam (see `scripts/modal_dspark_bench.py`).

The return-shape assertions are not busywork: `apply_chat_template` returns a
`BatchEncoding` on transformers 5.x where older versions returned a flat id
list, and iterating the mapping silently yields its KEYS rather than tokens.
"""

from __future__ import annotations

import pytest

from mini_infer.engine.tokenizer import Tokenizer

_MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.requires_model
def test_encode_chat_wraps_the_message_in_the_template() -> None:
    tok = Tokenizer.from_pretrained(_MODEL)
    assert tok.has_chat_template()

    raw = tok.encode("Name two landmarks in Paris.")
    chat = tok.encode_chat("Name two landmarks in Paris.")

    assert all(isinstance(t, int) for t in chat), "must be a flat list of ints"
    assert len(chat) > len(raw), "the template adds role markers"
    text = tok._tokenizer.decode(chat)
    assert "<|im_start|>user" in text
    # `add_generation_prompt=True`: the next token continues the assistant's
    # reply rather than the user's own turn.
    assert text.rstrip().endswith("<|im_start|>assistant")


@pytest.mark.requires_model
def test_encode_chat_preserves_the_message_tokens() -> None:
    """The user's own text survives templating unchanged, just wrapped."""
    tok = Tokenizer.from_pretrained(_MODEL)
    message = "Name two landmarks in Paris."
    chat = tok.encode_chat(message)
    # Qwen adds no BOS for plain text, so the raw ids appear verbatim inside.
    inner = tok._tokenizer.encode(message)
    joined = ",".join(map(str, chat))
    assert ",".join(map(str, inner)) in joined


class _NoTemplate:
    chat_template = None
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]


def test_encode_chat_refuses_without_a_template() -> None:
    """Fail loudly rather than silently falling back to untemplated text.

    A silent fallback would make a benchmark comparing the two arms report
    identical numbers and look like the template simply had no effect.
    """
    tok = Tokenizer(_NoTemplate())  # type: ignore[arg-type]
    assert not tok.has_chat_template()
    with pytest.raises(ValueError, match="no chat_template"):
        tok.encode_chat("hello")


@pytest.mark.requires_model
def test_enable_thinking_changes_the_generation_prompt() -> None:
    """Qwen3's reasoning mode is a template flag, and it is not cosmetic.

    Defaulting to thinking mode makes the model emit a `<think>` block before
    answering. Anything compared against a model tuned on non-thinking output
    has to disable it, which is why DSpark's own evaluator hardcodes
    `enable_thinking=False`; getting this wrong silently depresses draft
    acceptance rather than failing.
    """
    tok = Tokenizer.from_pretrained(_MODEL)
    thinking = tok.encode_chat("What is 2+2?", enable_thinking=True)
    plain = tok.encode_chat("What is 2+2?", enable_thinking=False)

    assert len(plain) > len(thinking), "non-thinking pre-closes the block, adding tokens"
    assert tok._tokenizer.decode(plain).rstrip().endswith("</think>")
    assert "<think>" not in tok._tokenizer.decode(thinking)
    # Unset must not silently pick a mode of our own.
    assert tok.encode_chat("What is 2+2?") == thinking
