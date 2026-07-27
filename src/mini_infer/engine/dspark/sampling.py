"""Matches `deepspec/utils/sampling.py::sample_tokens` exactly.

Temperature `< 1e-5` collapses to `argmax`: this project's own greedy
convention (see ADR-011), and the only path Stage B/C's temperature-0
bit-parity contract exercises.
"""

from __future__ import annotations

import torch


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return logits.argmax(dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)
