"""Markov head: the sequential correction that makes a bidirectional block causal.

The drafter's backbone produces `gamma` positions' base logits in ONE
bidirectional forward pass (no ordering between block positions). That alone
would be a non-autoregressive proposal, whose independent per-position
sampling misses inter-token dependencies within the block. The Markov head
reintroduces exact causal factorization at sampling time: position k's final
distribution is `softmax(base_logits[k] + bias(x_{k-1}))`, where `bias` is a
rank-`markov_rank` low-rank bigram table conditioned on the PREVIOUSLY SAMPLED
token, not the base logits' own (bidirectional) computation. This is DeepSpec's
`VanillaMarkov` (`deepspec/modeling/dspark/markov_head.py`); the released
`dspark_qwen3_4b_block7` checkpoint uses this variant (`markov_head_type
== "vanilla"`). DeepSpec also ships `GatedMarkovHead` and `RNNHead` variants;
neither is used by a released Qwen3 checkpoint, so they're not ported.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_infer.engine.dspark.sampling import sample_tokens


class VanillaMarkovHead(nn.Module):
    """Low-rank bigram logit bias: `bias(prev_token) = W1[prev_token] @ W2`.

    `markov_w1` is also read directly by the confidence head when
    `confidence_head_with_markov` is set (`Qwen3DSparkDrafter.predict_confidence_step`)
    — the reference shares the literal embedding object between the two heads,
    not two tables of the same shape, so the confidence head's previous-token
    feature is exactly this head's own bigram embedding.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.markov_w1(token_ids.long())
        return out

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.markov_w2(self.get_prev_embeddings(token_ids))
        return out

    def apply_step_logits(self, logits: torch.Tensor, *, token_ids: torch.Tensor) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequentially sample a whole block, conditioning each step on the last SAMPLED token.

        `base_logits`: `(batch, proposal_len, vocab)`, the block's bidirectionally-
        computed base logits. `first_prev_token_ids`: `(batch,)`, the anchor token
        that precedes position 0 (the last committed/verified token). Returns
        `(sampled_tokens, corrected_logits)`, both `(batch, proposal_len[, vocab])`.

        This loop is the only sequential part of the drafter: DeepSpec's
        `VanillaMarkov.sample_block_tokens`. Greedy (`temperature < 1e-5`)
        collapses each step to `argmax`, matching `deepspec/utils/sampling.py`'s
        `sample_tokens`.
        """
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :], token_ids=prev_token_ids
            )
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample_tokens(step_logits.unsqueeze(1), temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)
