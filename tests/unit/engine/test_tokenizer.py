import random

import pytest

from mini_infer.engine.tokenizer import TextStream, Tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(MODEL_NAME)


_STREAM_TEXT = (
    "The quick brown fox jumps over the lazy dog. Naïve café, 日本語のテキスト, "
    "emoji 🙂🙃 ✓, tabs\t\tand\nnewlines, double  spaces. It's 3.14, isn't it? "
    '"quoted" (parens) [brackets] {braces} <angle> a/b\\c'
)


def test_text_stream_gives_the_text_of_one_decode_id_by_id(qwen_tokenizer: Tokenizer) -> None:
    """Real text and random ids: the streamed text equals the decode of all the ids.

    At every step the text so far is a prefix of the decode of the ids so far,
    so the client never sees a character it has to take back.
    """
    random.seed(0)
    sequences = [
        qwen_tokenizer.encode(_STREAM_TEXT),
        [random.randrange(qwen_tokenizer.vocab_size) for _ in range(300)],
    ]
    for ids in sequences:
        stream = qwen_tokenizer.text_stream()
        assert stream._stream is not None, "the Qwen tokenizer streams through its backend"
        text = ""
        for index, token in enumerate(ids):
            delta = stream.push([token])
            text += delta
            assert stream.text == text
            assert qwen_tokenizer.decode(ids[: index + 1]).startswith(text)
        assert text == qwen_tokenizer.decode(ids)


def test_text_stream_holds_an_incomplete_character_back(qwen_tokenizer: Tokenizer) -> None:
    """An emoji of several byte ids comes out whole with its last id."""
    ids = qwen_tokenizer.encode("🪐")
    assert len(ids) == 3, "the planet emoji takes three byte-level ids"
    stream = qwen_tokenizer.text_stream()
    assert [stream.push([token]) for token in ids] == ["", "", "🪐"]


def test_text_stream_skips_special_tokens_like_decode(qwen_tokenizer: Tokenizer) -> None:
    ids = qwen_tokenizer.encode("one two")
    stream = qwen_tokenizer.text_stream()
    with_eos = [*ids[:1], qwen_tokenizer.eos_token_id, *ids[1:]]
    assert stream.push(with_eos) == qwen_tokenizer.decode(ids)


def test_text_stream_without_a_backend_re_decodes_everything() -> None:
    """A tokenizer double with a plain `decode` gets the decode-and-diff path."""
    calls: list[list[int]] = []

    def decode(ids: list[int]) -> str:
        calls.append(list(ids))
        return "".join(chr(97 + token) for token in ids)

    stream = TextStream(decode)
    assert stream.push([0, 1]) == "ab"
    assert stream.push([2]) == "c"
    assert stream.text == "abc"
    assert calls == [[0, 1], [0, 1, 2]]


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
