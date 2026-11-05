"""A toy checkpoint tokenizer in the Kimi layout, for tests that run without network.

`write_kimi_tokenizer_files` writes the four files that Kimi Linear ships
for its tokenizer: `tiktoken.model` (BPE ranks), `tokenizer_config.json`
(special tokens and `auto_map`), `tokenization_kimi.py` (as text with the
split pattern; nothing runs it) and `chat_template.jinja`. The ranks come
from a small BPE training run over English, Chinese and code, so merges
exist inside multi-byte characters too. The special ids sit at the same
offsets after the base vocabulary as in the released checkpoint.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from mini_infer.engine.hub_tokenizers import _KIMI_PATTERN_PIECES

_CORPUS = (
    "The capital of France is Paris, and the capital of Germany is Berlin. "
    "It's a test; they're here, we've seen it, I'll go and you'd stay. "
    "北京是中国的首都。上海是中国最大的城市。我们今天去长城。中国人喜欢喝茶。 "
    "def forward(self, x):\n    return self.proj(x) + 1\n\n"
    "for index in range(10):\n\tprint(index * 2)\n"
    "1234567890 3.14159 2026-11-05 <html> {key: value} [a, b] "
)
_NUM_MERGES = 300

# Offset after the base vocabulary -> (text, skipped by decode), as in the
# `added_tokens_decoder` of moonshotai/Kimi-Linear-48B-A3B-Instruct.
KIMI_SPECIAL_OFFSETS: dict[int, tuple[str, bool]] = {
    0: ("[BOS]", True),
    1: ("[EOS]", True),
    2: ("<|im_end|>", True),
    3: ("<|im_user|>", True),
    4: ("<|im_assistant|>", True),
    6: ("<|start_header_id|>", True),
    7: ("<|end_header_id|>", True),
    9: ("[EOT]", True),
    10: ("<|im_system|>", True),
    11: ("<|tool_calls_section_begin|>", False),
    12: ("<|tool_calls_section_end|>", False),
    13: ("<|tool_call_begin|>", False),
    14: ("<|tool_call_argument_begin|>", False),
    15: ("<|tool_call_end|>", False),
    17: ("<|im_middle|>", True),
    254: ("[UNK]", True),
    255: ("[PAD]", True),
}
_ADDITIONAL_SPECIAL_TOKENS = [
    "<|im_end|>",
    "<|im_user|>",
    "<|im_assistant|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "[EOT]",
    "<|im_system|>",
    "<|im_middle|>",
]
_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}<|im_user|>user<|im_middle|>"
    "{%- else -%}<|im_assistant|>assistant<|im_middle|>{%- endif -%}"
    "{{ message['content'] }}<|im_end|>"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_assistant|>assistant<|im_middle|>{%- endif -%}"
)


def toy_ranks() -> dict[bytes, int]:
    """All 256 bytes, then `_NUM_MERGES` merges trained on `_CORPUS`."""
    ranks = {bytes([value]): value for value in range(256)}
    words = [
        [bytes([value]) for value in word.encode()] for word in re.findall(r"\s?\S+|\s+", _CORPUS)
    ]
    while len(ranks) < 256 + _NUM_MERGES:
        counts = Counter(pair for word in words for pair in pairwise(word))
        if not counts:
            break
        (left, right), _ = max(counts.items(), key=lambda item: (item[1], item[0]))
        merged = left + right
        ranks.setdefault(merged, len(ranks))
        for word in words:
            index = 0
            while index < len(word) - 1:
                if word[index] == left and word[index + 1] == right:
                    word[index : index + 2] = [merged]
                index += 1
    return ranks


def toy_special_tokens(num_base_tokens: int) -> dict[str, int]:
    """Every special id of the toy tokenizer, named as `TikTokenTokenizer` names it."""
    tokens: dict[str, int] = {}
    for offset in range(258):
        token_id = num_base_tokens + offset
        text = KIMI_SPECIAL_OFFSETS.get(offset, (f"<|reserved_token_{token_id}|>", False))[0]
        tokens[text] = token_id
    return tokens


def hub_source_text(pattern_pieces: list[str]) -> str:
    """`tokenization_kimi.py` as text: the lines the loader looks for, nothing runnable."""
    lines = ["pat_str = '|'.join(", "    ["]
    lines += [f'        r"""{piece}""",' for piece in pattern_pieces]
    lines += ["    ]", ")", "num_reserved_special_tokens = 256", ""]
    return "\n".join(lines)


def write_kimi_tokenizer_files(directory: Path) -> dict[bytes, int]:
    """Write the toy Kimi tokenizer into `directory` and return its BPE ranks."""
    ranks = toy_ranks()
    lines = [
        f"{base64.b64encode(token).decode()} {rank}"
        for token, rank in sorted(ranks.items(), key=lambda item: item[1])
    ]
    (directory / "tiktoken.model").write_text("\n".join(lines) + "\n")
    base = len(ranks)
    added_tokens_decoder = {
        str(base + offset): {
            "content": text,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": special,
        }
        for offset, (text, special) in KIMI_SPECIAL_OFFSETS.items()
    }
    tokenizer_config = {
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": _ADDITIONAL_SPECIAL_TOKENS,
        "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
        "bos_token": "[BOS]",
        "clean_up_tokenization_spaces": False,
        "eos_token": "[EOS]",
        "model_max_length": 1048576,
        "pad_token": "[PAD]",
        "tokenizer_class": "TikTokenTokenizer",
        "unk_token": "[UNK]",
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    (directory / "tokenization_kimi.py").write_text(hub_source_text(list(_KIMI_PATTERN_PIECES)))
    (directory / "chat_template.jinja").write_text(_CHAT_TEMPLATE)
    return ranks
