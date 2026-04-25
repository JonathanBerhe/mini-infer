import pytest

from mini_infer.engine.tokenizer import Tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(MODEL_NAME)


def test_encode_returns_non_empty_int_list(qwen_tokenizer: Tokenizer) -> None:
    tokens = qwen_tokenizer.encode("Hello, world!")
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_decode_roundtrip(qwen_tokenizer: Tokenizer) -> None:
    text = "The quick brown fox"
    decoded = qwen_tokenizer.decode(qwen_tokenizer.encode(text))
    assert decoded.strip() == text


def test_eos_token_id_is_non_negative_int(qwen_tokenizer: Tokenizer) -> None:
    eos = qwen_tokenizer.eos_token_id
    assert isinstance(eos, int)
    assert eos >= 0
