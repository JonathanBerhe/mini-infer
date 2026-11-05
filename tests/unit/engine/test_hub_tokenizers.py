"""The Kimi fast tokenizer, built without hub code, against tiktoken and the hub tokenizer.

Two references:

- `tiktoken.Encoding` over a toy vocabulary in the Kimi layout
  (`tests/unit/_kimi_tokenizer_files.py`). It is the encoder that the hub
  `TikTokenTokenizer` wraps, and these tests run in CI.
- The hub `TikTokenTokenizer` itself on the real vocabulary. That test runs
  `tokenization_kimi.py` of the pinned revision, so it runs only when
  `tiktoken` is installed and the tokenizer files are in the local HF cache.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mini_infer.engine.hub_tokenizers import (
    _KIMI_PATTERN_PIECES,
    KIMI_PATTERN,
    load_hub_code_tokenizer,
    read_tiktoken_ranks,
)
from mini_infer.engine.tokenizer import Tokenizer
from tests.unit._kimi_tokenizer_files import (
    KIMI_SPECIAL_OFFSETS,
    hub_source_text,
    toy_special_tokens,
    write_kimi_tokenizer_files,
)

# The long s folds to "s" under full case folding, a trap for `(?i:'s)`.
_LONG_S = "\N{LATIN SMALL LETTER LONG S}"
# English, Chinese, code, special-token text, and the corner cases of the
# split pattern: case-insensitive contractions, digit runs, whitespace runs.
_TEXTS = [
    "The capital of France is",
    "Hello, world! It's a test, isn't it? I'LL go. THEY'RE here. We'Ve won.",
    "北京是中国的首都。我们今天去长城玩吧\N{FULLWIDTH EXCLAMATION MARK}",
    "日本語のテキストと한국어 and English mixed with 中文 words",
    "def f(x):\n    return x ** 2  # square\n\n\nclass A:\n\tpass\r\n",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }  // C++",
    "<|im_user|>user<|im_middle|>What is 2+2?<|im_end|><|im_assistant|>assistant<|im_middle|>",
    "[BOS]text[EOS] [EOT] <|tool_calls_section_begin|><|tool_call_begin|>fn"
    '<|tool_call_argument_begin|>{"a": 1}<|tool_call_end|><|tool_calls_section_end|>',
    "emoji 🙂🙃 🪐 ✓ Naïve café   multiple   spaces\n\n\nnewlines 12345678 3.14159",
    f"{_LONG_S} ﬃ ΣΑΣ İstanbul ǅ McDonald's O'Neil x'S x'{_LONG_S}",
    "",
    " ",
    "\n\n",
    "   leading and trailing   ",
    "tabs\t\tand\u00a0nbsp\u3000ideographic space",
]
_RANDOM_ALPHABET = "abcXYZ 中文日本한국 \n\t.,!?'0123456789#()[]{}<|>_-ñéü🙂" + _LONG_S


def _random_texts(count: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [
        "".join(rng.choice(_RANDOM_ALPHABET) for _ in range(rng.randint(1, 60)))
        for _ in range(count)
    ]


@pytest.fixture(scope="module")
def toy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("kimi_tokenizer")
    write_kimi_tokenizer_files(directory)
    return directory


def test_matches_tiktoken_on_a_toy_vocabulary(toy_dir: Path) -> None:
    """Encode and decode equal `tiktoken.Encoding` with the Kimi pattern and special ids."""
    tiktoken = pytest.importorskip("tiktoken")
    ranks = read_tiktoken_ranks(toy_dir / "tiktoken.model")
    reference = tiktoken.Encoding(
        name="toy-kimi",
        pat_str=KIMI_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=toy_special_tokens(len(ranks)),
    )
    ours = load_hub_code_tokenizer(str(toy_dir))
    assert ours is not None

    for text in _TEXTS + _random_texts(500):
        ids = reference.encode(text, allowed_special="all")
        assert ours.encode(text) == ids, text
        assert ours.decode(ids) == reference.decode(ids), text
    rng = random.Random(1)
    ids = [rng.randrange(reference.n_vocab) for _ in range(500)]
    assert ours.decode(ids) == reference.decode(ids)
    assert ours.vocab_size == reference.n_vocab


def test_tokenizer_wrapper_on_the_toy_checkpoint(toy_dir: Path) -> None:
    """`Tokenizer.from_pretrained` loads the checkpoint with no hub code and no AutoTokenizer."""
    tokenizer = Tokenizer.from_pretrained(str(toy_dir))
    base = len(read_tiktoken_ranks(toy_dir / "tiktoken.model"))

    ids = tokenizer.encode("The capital of France is")
    assert ids[0] == base, "the wrapper starts plain text with [BOS]"
    assert tokenizer.eos_token_id == base + 1
    assert tokenizer.vocab_size == base + 258

    # Decode skips the special tokens that the config marks special, and
    # keeps the tool-call tokens and the reserved ones, like the hub tokenizer.
    im_end, tool_call_begin, reserved = base + 2, base + 13, base + 100
    text_ids = tokenizer.encode("中国 code")[1:]
    assert tokenizer.decode([*text_ids, im_end]) == "中国 code"
    assert tokenizer.decode([tool_call_begin, reserved]) == (
        f"<|tool_call_begin|><|reserved_token_{reserved}|>"
    )

    stream = tokenizer.text_stream()
    assert stream._stream is not None, "the fast tokenizer streams through its backend"
    pieces = [stream.push([token]) for token in ids]
    assert "".join(pieces) == tokenizer.decode(ids) == "The capital of France is"

    chat_ids = tokenizer.encode_chat("Hi")
    assert chat_ids[0] == base + 3, "the chat template starts with <|im_user|>"
    assert chat_ids[-1] == base + 17, "the generation prompt ends with <|im_middle|>"


def test_special_tokens_follow_the_config(toy_dir: Path) -> None:
    ours = load_hub_code_tokenizer(str(toy_dir))
    assert ours is not None
    base = len(read_tiktoken_ranks(toy_dir / "tiktoken.model"))
    for offset, (text, special) in KIMI_SPECIAL_OFFSETS.items():
        assert ours.convert_tokens_to_ids(text) == base + offset
        assert (base + offset in ours.all_special_ids) is special, text
    assert ours.encode(f"<|reserved_token_{base + 200}|>") == [base + 200]
    assert ours.bos_token == "[BOS]"
    assert ours.pad_token_id == base + 255
    assert ours.unk_token_id == base + 254


def test_other_tokenizers_are_not_hub_code(tmp_path: Path) -> None:
    """No `tokenizer_config.json`, or another tokenizer class, falls through to AutoTokenizer."""
    assert load_hub_code_tokenizer(str(tmp_path)) is None
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|endoftext|>"})
    )
    assert load_hub_code_tokenizer(str(tmp_path)) is None


def test_a_changed_split_pattern_fails(tmp_path: Path) -> None:
    """A hub file whose pattern differs from the copy fails before it can mis-split text."""
    write_kimi_tokenizer_files(tmp_path)
    changed = [*_KIMI_PATTERN_PIECES[:-1], r"""\s+?"""]
    (tmp_path / "tokenization_kimi.py").write_text(hub_source_text(changed))
    with pytest.raises(ValueError, match="split pattern"):
        load_hub_code_tokenizer(str(tmp_path))


def test_special_ids_outside_the_range_fail(tmp_path: Path) -> None:
    write_kimi_tokenizer_files(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["added_tokens_decoder"]["5"] = {"content": "<|odd|>", "special": True}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="outside the special ids"):
        load_hub_code_tokenizer(str(tmp_path))


_KIMI_REPO = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
_KIMI_REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
_KIMI_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
    "chat_template.jinja",
)


def _cached_kimi_snapshot() -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    paths = [
        try_to_load_from_cache(_KIMI_REPO, filename, revision=_KIMI_REVISION)
        for filename in _KIMI_TOKENIZER_FILES
    ]
    if not all(isinstance(path, str) for path in paths):
        return None
    return Path(str(paths[0])).parent


@pytest.mark.requires_model
def test_matches_the_hub_tokenizer() -> None:
    """Encode, decode and the chat template equal the hub `TikTokenTokenizer`.

    Runs `tokenization_kimi.py` of the pinned revision, so it needs `tiktoken`
    and the four tokenizer files in the local HF cache. It downloads nothing.
    """
    pytest.importorskip("tiktoken")
    snapshot = _cached_kimi_snapshot()
    if snapshot is None:
        pytest.skip(f"{_KIMI_REPO}@{_KIMI_REVISION[:8]} tokenizer files are not cached")
    from transformers import AutoTokenizer

    hub = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    ours = load_hub_code_tokenizer(str(snapshot))
    assert ours is not None
    assert type(hub).__name__ == "TikTokenTokenizer"

    assert ours.vocab_size == hub.vocab_size
    assert sorted(ours.all_special_ids) == sorted(hub.all_special_ids)
    for name in ("bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id"):
        assert getattr(ours, name) == getattr(hub, name), name
    for text in _TEXTS + _random_texts(1000):
        ids = hub.encode(text)
        assert ours.encode(text) == ids, text
        assert ours.decode(ids) == hub.decode(ids), text
        assert ours.decode(ids, skip_special_tokens=True) == hub.decode(
            ids, skip_special_tokens=True
        ), text
    rng = random.Random(2)
    ids = [rng.randrange(hub.vocab_size) for _ in range(2000)]
    assert ours.decode(ids, skip_special_tokens=True) == hub.decode(ids, skip_special_tokens=True)

    wrapped_ours, wrapped_hub = Tokenizer(ours), Tokenizer(hub)
    for text in _TEXTS:
        assert wrapped_ours.encode(text) == wrapped_hub.encode(text), text
    question = "What is the capital of France?"
    assert wrapped_ours.render_chat(question) == wrapped_hub.render_chat(question)
    assert wrapped_ours.encode_chat(question) == wrapped_hub.encode_chat(question)
