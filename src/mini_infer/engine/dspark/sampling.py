"""Sampling and the rejection-sampling verification, mirroring `deepspec/utils/sampling.py`.

Why speculative sampling is lossless, since it is the whole reason these
particular formulas and not simpler ones:

A draft proposes `x ~ p_d`. Accept it with probability `min(1, p_t(x)/p_d(x))`.
The probability of emitting `x` via acceptance is
`p_d(x) * min(1, p_t(x)/p_d(x)) = min(p_d(x), p_t(x))`. That is short of
`p_t(x)` by exactly `max(0, p_t(x) - p_d(x))`, and the total shortfall is the
rejection probability, so resampling from the NORMALIZED residual
`max(0, p_t - p_d)` on rejection restores `p_t` exactly. The emitted
distribution is the target's, not an approximation of it, which is why
speculative decoding is a pure latency optimization with no quality knob.

Two details that look like implementation noise but are load-bearing:

- `clamp_min(1e-8)` on the draft probability guards the ratio, not the math.
  A token with `p_d == 0` cannot be proposed, so the ratio is only ever
  evaluated where `p_d > 0`; the clamp exists so finite-precision zeros do
  not produce inf.
- The residual falls back to `p_t` when its mass underflows. That happens when
  `p_d` covers `p_t` almost exactly, where the true residual is a vanishing
  difference of near-equal numbers and its normalization is pure noise.
  Sampling `p_t` directly is the correct limit.

Greedy (`temperature < 1e-5`) collapses all of this: both distributions become
one-hot, acceptance degenerates to argmax equality, and the residual becomes
the target's own argmax. That is the Stage B/C path and the reason greedy
parity is provable rather than statistical.
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


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Probabilities at `temperature`; a one-hot at the argmax when greedy.

    The one-hot is what makes the greedy path a special case of the sampling
    path rather than a separate branch: `min(1, p_t/p_d)` over two one-hots is
    1 on a match and 0 otherwise.
    """
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """`probs[..., token_ids]`, keeping the leading dims."""
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Multinomial draw over the last dim of a `(batch, seq, vocab)` tensor."""
    bsz, seq_len, vocab_size = probs.shape
    flat = probs.reshape(-1, vocab_size)
    return torch.multinomial(flat, num_samples=1).reshape(bsz, seq_len)


def sample_residual(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """Draw from the normalized residual `max(0, p_target - p_draft)`.

    This is what a rejection must emit for the overall distribution to stay
    exactly `p_target`; see the module docstring. Both inputs are
    `(batch, vocab)`.
    """
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):
        # Degenerate only where the draft already covered the target; the
        # target itself is the right limit there.
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    return sample_from_probs(residual.unsqueeze(1)).squeeze(1)


def accepted_prefix_length(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """How many leading draft tokens survive rejection sampling.

    `target_probs` is `(1, n, vocab)` at the draft positions, `draft_probs` the
    same shape, `proposed_tokens` `(1, n)`.

    Acceptance is a PREFIX: once a token is rejected its successors were
    conditioned on a token the target did not take, so they are meaningless
    regardless of their own ratios. The cumulative product enforces that,
    rather than counting each position independently.
    """
    if proposed_tokens.numel() == 0:
        return 0
    selected_target = gather_token_probs(target_probs, proposed_tokens)
    selected_draft = gather_token_probs(draft_probs, proposed_tokens).clamp_min(1e-8)
    accept_prob = torch.clamp(selected_target / selected_draft, max=1.0)
    uniform = torch.rand(
        accept_prob.shape, device=accept_prob.device, dtype=accept_prob.dtype, generator=generator
    )
    accepted = (uniform < accept_prob).to(torch.int64)
    return int(accepted.cumprod(dim=1).sum(dim=1)[0].item())
