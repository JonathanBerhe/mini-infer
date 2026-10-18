"""Micro-config models and doubles shared by the DSpark scheduler tests.

A real (randomly initialized) Qwen3 target and DSpark drafter, small enough
for CPU fp32, plus a character tokenizer. Random weights are the right
fixture: greedy speculative output is drafter-independent, so a drafter that
proposes mostly wrong tokens still exercises partial acceptance, full
rejection, and the bonus path, with no checkpoint.
"""

from __future__ import annotations

import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark import Qwen3DSparkConfig, Qwen3DSparkDrafter
from mini_infer.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

VOCAB = 64
HIDDEN = 32
TAPS = [1, 3]
BLOCK = 3
EOS = 63
MASK = 62


def drafter_config(*, confidence: bool = False) -> Qwen3DSparkConfig:
    return Qwen3DSparkConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        target_layer_ids=list(TAPS),
        mask_token_id=MASK,
        block_size=BLOCK,
        markov_rank=5,
        enable_confidence_head=confidence,
        confidence_head_with_markov=confidence,
    )


class CharTokenizer:
    """One id per character, ids in [1, 60) so EOS and mask stay reserved."""

    eos_token_id = EOS
    vocab_size = VOCAB

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 59) + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


def target_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=40,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


class MicroRunner:
    """Duck-typed ModelRunner over a real (random) Qwen3, CPU fp32."""

    supports_hidden_state_taps = True

    def __init__(self, model: Qwen3ForCausalLM, pool: BlockPool) -> None:
        self._model = model
        self.block_pool = pool
        self.tokenizer = CharTokenizer()

    def forward_step_packed(
        self,
        cache: PagedKVCache,
        packed_input_ids: list[int],
        cu_seqlens_q: list[int],
        position_offsets: list[int],
        *,
        tap_layers: frozenset[int] | None = None,
        hidden_state_sink: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        position_ids_flat: list[int] = []
        for batch_idx in range(cache.batch_size):
            q_len = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            offset = position_offsets[batch_idx]
            position_ids_flat.extend(range(offset, offset + q_len))
        extra = {}
        if tap_layers is not None:
            extra = {"tap_layers": tap_layers, "hidden_state_sink": hidden_state_sink}
        with torch.inference_mode():
            logits: torch.Tensor = self._model(
                input_ids=torch.tensor([packed_input_ids], dtype=torch.long),
                position_ids=torch.tensor([position_ids_flat], dtype=torch.long),
                past_key_values=cache,
                cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32),
                **extra,
            )
        return logits


def micro_setup(
    seed: int = 0, *, confidence: bool = False, block_size: int = 8
) -> tuple[MicroRunner, Qwen3DSparkDrafter]:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(target_config()).eval()
    pool = BlockPool(
        num_blocks=128,
        block_size=block_size,
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        attention_backend="torch",
    )
    drafter = Qwen3DSparkDrafter(drafter_config(confidence=confidence)).eval()
    return MicroRunner(model, pool), drafter
