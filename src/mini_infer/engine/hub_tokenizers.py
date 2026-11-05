"""Fast tokenizers for checkpoints that ship their tokenizer only as hub code.

Kimi Linear ships no `tokenizer.json`. Its tokenizer is a Python class in the
repository (`tokenization_kimi.py`, a `TikTokenTokenizer` over the BPE ranks
in `tiktoken.model`), and `AutoTokenizer` loads it only with
`trust_remote_code`. mini-infer runs no hub code in library code, so this
module builds the same tokenizer with the `tokenizers` library:

- The BPE vocabulary and merges come from the ranks in `tiktoken.model`, as
  in the `TikTokenConverter` of transformers, but without the `tiktoken`
  package.
- `ignore_merges` keeps a split piece whole when the vocabulary has it. This
  is the rule of tiktoken.
- The split pattern is the `pat_str` of `tokenization_kimi.py`, copied here.
- The 258 ids after the base vocabulary are special tokens: the names from
  `added_tokens_decoder` in `tokenizer_config.json`, and
  `<|reserved_token_{id}|>` for the other ids. Encode turns their text into
  their ids, as the hub tokenizer does with `allowed_special="all"`.

The loader parses `tokenization_kimi.py` with `ast`, which runs none of it,
and fails if the file does not declare the pattern and the special-id count
copied here.

One difference stays. To keep tiktoken within its limits, the hub `encode`
cuts a text into chunks of 400,000 characters, and cuts runs of more than
25,000 whitespace or non-whitespace characters. This tokenizer does not cut,
so the ids next to such a cut can differ. A run of 30,000 digits gives
10,000 ids here and 10,001 from the hub. English and Chinese prose of
165,000 characters gives the same ids, and prose of 495,000 characters
differs by one id at the 400,000-character cut.
"""

from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

# `auto_map.AutoTokenizer` of the checkpoints this module builds.
_KIMI_TOKENIZER_CLASS = "tokenization_kimi.TikTokenTokenizer"
_KIMI_SOURCE_FILE = "tokenization_kimi.py"

# `TikTokenTokenizer.pat_str` of `tokenization_kimi.py`, one alternative per line.
_KIMI_PATTERN_PIECES = (
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
)
KIMI_PATTERN = "|".join(_KIMI_PATTERN_PIECES)
# `num_reserved_special_tokens + 2` special ids follow the base vocabulary.
_KIMI_NUM_RESERVED = 256
_KIMI_NUM_SPECIAL_IDS = _KIMI_NUM_RESERVED + 2


def load_hub_code_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase | None:
    """The tokenizer of `name_or_path` if it ships a known hub-code tokenizer, else None.

    `name_or_path` is a local checkpoint directory or an HF Hub repo id. A
    checkpoint without `tokenizer_config.json`, or with any other tokenizer
    class, gives None, and the caller loads it with `AutoTokenizer`.
    """
    config_path = _checkpoint_file(name_or_path, "tokenizer_config.json")
    if config_path is None:
        return None
    with config_path.open() as config_file:
        tokenizer_config = json.load(config_file)
    auto_map = (tokenizer_config.get("auto_map") or {}).get("AutoTokenizer")
    entry = auto_map[0] if isinstance(auto_map, list) and auto_map else auto_map
    if entry != _KIMI_TOKENIZER_CLASS:
        return None
    return kimi_tokenizer(name_or_path, tokenizer_config)


def kimi_tokenizer(name_or_path: str, tokenizer_config: dict[str, Any]) -> PreTrainedTokenizerFast:
    """The `TikTokenTokenizer` of `tokenization_kimi.py` as a fast tokenizer, without hub code."""
    source = _required_file(name_or_path, _KIMI_SOURCE_FILE).read_text()
    pieces, num_reserved = _declared_split(source)
    if pieces != _KIMI_PATTERN_PIECES or num_reserved != _KIMI_NUM_RESERVED:
        raise ValueError(
            f"{_KIMI_SOURCE_FILE} of {name_or_path!r} does not declare the split pattern "
            f"and the special-id count that mini-infer copies (pat_str parts {pieces}, "
            f"num_reserved_special_tokens {num_reserved}); compare the file with "
            f"KIMI_PATTERN in {__name__}"
        )
    ranks = read_tiktoken_ranks(_required_file(name_or_path, "tiktoken.model"))
    special_tokens = _kimi_special_tokens(
        len(ranks), tokenizer_config.get("added_tokens_decoder") or {}
    )
    backend = tiktoken_backend(ranks, KIMI_PATTERN, special_tokens)

    kwargs: dict[str, Any] = {}
    for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
        token = _token_content(tokenizer_config.get(name))
        if token is not None:
            kwargs[name] = token
    extra_tokens = tokenizer_config.get("additional_special_tokens") or []
    extra = [_token_content(token) for token in extra_tokens]
    if extra:
        kwargs["additional_special_tokens"] = extra
    if "model_max_length" in tokenizer_config:
        kwargs["model_max_length"] = tokenizer_config["model_max_length"]
    chat_template_path = _checkpoint_file(name_or_path, "chat_template.jinja")
    if chat_template_path is not None:
        kwargs["chat_template"] = chat_template_path.read_text()
    elif tokenizer_config.get("chat_template"):
        kwargs["chat_template"] = tokenizer_config["chat_template"]
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        clean_up_tokenization_spaces=False,
        name_or_path=name_or_path,
        **kwargs,
    )


def read_tiktoken_ranks(path: Path) -> dict[bytes, int]:
    """The BPE ranks of a tiktoken file: one `base64(token) rank` pair per line.

    The parse of `tiktoken.load.load_tiktoken_bpe`, without the `tiktoken`
    package. The ranks must be 0 to n-1 and include all 256 single bytes.
    """
    ranks: dict[bytes, int] = {}
    for line in path.read_bytes().splitlines():
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    if sorted(ranks.values()) != list(range(len(ranks))):
        raise ValueError(f"{path}: the ranks are not the integers 0 to {len(ranks) - 1}")
    single_bytes = sum(1 for token in ranks if len(token) == 1)
    if single_bytes != 256:
        raise ValueError(f"{path}: a byte-level vocabulary needs all 256 bytes, got {single_bytes}")
    return ranks


def tiktoken_backend(
    ranks: dict[bytes, int],
    pattern: str,
    special_tokens: list[tuple[int, str, bool]],
) -> Any:
    """A `tokenizers.Tokenizer` that encodes and decodes like `tiktoken.Encoding`.

    `special_tokens` holds `(id, text, skipped_by_decode)` triples. Each one
    goes into the vocabulary at its id and becomes an added token, so encode
    finds its text before the split. The merges list every split of a token
    into two tokens of the vocabulary, ordered by the rank of the token, as
    the `TikTokenConverter` of transformers does.
    """
    from tokenizers import (  # type: ignore[import-untyped]
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        pre_tokenizers,
        processors,
    )
    from tokenizers.models import BPE  # type: ignore[import-untyped]
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    to_unicode = bytes_to_unicode()

    def as_text(token: bytes) -> str:
        return "".join(to_unicode[byte] for byte in token)

    vocab: dict[str, int] = {}
    merges: list[tuple[str, str]] = []
    for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
        vocab[as_text(token)] = rank
        splits = [
            (token[:index], token[index:])
            for index in range(1, len(token))
            if token[:index] in ranks and token[index:] in ranks
        ]
        splits.sort(key=lambda pair: (ranks[pair[0]], ranks[pair[1]]))
        merges.extend((as_text(left), as_text(right)) for left, right in splits)
    for token_id, text, _ in special_tokens:
        if text in vocab:
            raise ValueError(f"special token {text!r} is also a base token (id {vocab[text]})")
        vocab[text] = token_id

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False, ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.add_tokens(
        [AddedToken(text, normalized=False, special=special) for _, text, special in special_tokens]
    )
    return tokenizer


def _declared_split(source: str) -> tuple[tuple[str, ...] | None, int | None]:
    """The `pat_str` parts and `num_reserved_special_tokens` that a tokenizer file declares.

    Parses the source with `ast`, which runs none of it, and reads the
    assignments `pat_str = "|".join([...])` and
    `num_reserved_special_tokens = N`. A declaration of another form gives None.
    """
    pieces: tuple[str, ...] | None = None
    num_reserved: int | None = None
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name, value = node.targets[0].id, node.value
        if name == "num_reserved_special_tokens" and isinstance(value, ast.Constant):
            num_reserved = value.value if isinstance(value.value, int) else None
        if (
            name == "pat_str"
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "join"
            and isinstance(value.func.value, ast.Constant)
            and value.func.value.value == "|"
            and len(value.args) == 1
            and isinstance(value.args[0], (ast.List, ast.Tuple))
        ):
            elements = value.args[0].elts
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elements):
                pieces = tuple(str(e.value) for e in elements)  # type: ignore[attr-defined]
    return pieces, num_reserved


def _kimi_special_tokens(
    num_base_tokens: int, added_tokens_decoder: dict[str, Any]
) -> list[tuple[int, str, bool]]:
    """`(id, text, special)` for the ids after the base vocabulary.

    `TikTokenTokenizer` names them the same way: the config name, or
    `<|reserved_token_{id}|>`.
    """
    special_ids = range(num_base_tokens, num_base_tokens + _KIMI_NUM_SPECIAL_IDS)
    outside = sorted(int(key) for key in added_tokens_decoder if int(key) not in special_ids)
    if outside:
        raise ValueError(
            f"added_tokens_decoder declares ids {outside} outside the special ids "
            f"{special_ids.start}..{special_ids.stop - 1}"
        )
    tokens: list[tuple[int, str, bool]] = []
    for token_id in special_ids:
        entry = added_tokens_decoder.get(str(token_id))
        if entry is None:
            tokens.append((token_id, f"<|reserved_token_{token_id}|>", False))
        else:
            tokens.append((token_id, str(entry["content"]), bool(entry.get("special", False))))
    return tokens


def _token_content(token: Any) -> str | None:
    """A special-token entry of `tokenizer_config.json`: a string or an AddedToken dict."""
    if isinstance(token, dict):
        return str(token["content"])
    return str(token) if token is not None else None


def _checkpoint_file(name_or_path: str, filename: str) -> Path | None:
    """`filename` of a local checkpoint directory or an HF Hub repo, or None if it has none."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        path = candidate / filename
        return path if path.is_file() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(name_or_path, filename))
    except EntryNotFoundError:
        return None


def _required_file(name_or_path: str, filename: str) -> Path:
    path = _checkpoint_file(name_or_path, filename)
    if path is None:
        raise FileNotFoundError(f"{name_or_path!r} has no {filename}")
    return path
