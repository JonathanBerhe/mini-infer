import dataclasses
from collections.abc import Sequence

import torch


@dataclasses.dataclass(frozen=True)
class SamplingParams:
    """Sampling configuration; temperature=0 short-circuits to greedy."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")


def sample(logits: torch.Tensor, params: SamplingParams) -> int:
    """Pick a next-token id from a 1D logits tensor of shape (vocab,)."""
    if params.temperature == 0.0:
        return int(torch.argmax(logits).item())

    scaled = logits / params.temperature
    if params.top_k > 0:
        scaled = _apply_top_k(scaled, params.top_k)
    if params.top_p < 1.0:
        scaled = _apply_top_p(scaled, params.top_p)

    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def sample_batch(logits: torch.Tensor, params: Sequence[SamplingParams]) -> list[int]:
    """Pick one next-token id per row of `(batch, vocab)` logits; row i uses `params[i]`.

    The ids reach the host in one copy, so a batch costs one host sync, not
    one per row. See `sample_tokens` for how each row is sampled.
    """
    tokens: list[int] = sample_tokens(logits, params).tolist()
    return tokens


def sample_tokens(logits: torch.Tensor, params: Sequence[SamplingParams]) -> torch.Tensor:
    """`sample_batch` without the copy: a `(batch,)` id tensor on the logits' device.

    It does not sync with the host. A greedy row (temperature 0) takes the
    argmax, the same id as `sample`. A sampling row applies the temperature,
    top-k and top-p of `sample` in float32, then draws from a separate random
    stream. So a seed gives other ids than `sample` with the same seed.
    """
    if logits.dim() != 2 or logits.shape[0] != len(params):
        raise ValueError(
            f"logits must be (batch, vocab) with one row per params; got {tuple(logits.shape)} "
            f"for {len(params)} params"
        )
    greedy = logits.argmax(dim=-1)
    if all(p.temperature == 0.0 for p in params):
        return greedy

    vocab = logits.shape[-1]
    top_ks = [p.top_k if 0 < p.top_k < vocab else 0 for p in params]  # 0: no top-k
    # One copy of the per-row settings from pinned memory, which does not
    # wait for the GPU.
    settings = torch.tensor(
        [[p.temperature, k, p.top_p] for p, k in zip(params, top_ks, strict=True)],
        dtype=torch.float32,
        pin_memory=logits.is_cuda,
    ).to(logits.device, non_blocking=True)
    temperature, top_k, top_p = settings.unbind(dim=-1)
    is_greedy = temperature == 0.0

    # A greedy row is sampled at temperature 1, and its draw is discarded.
    scaled = logits.float() / torch.where(is_greedy, 1.0, temperature).unsqueeze(-1)
    if max(top_ks) > 0:
        scaled = _apply_top_k_rows(scaled, top_k.long(), max(top_ks))
    if any(p.top_p < 1.0 for p in params):
        scaled = _apply_top_p_rows(scaled, top_p)

    # `torch.multinomial` checks its input with host syncs. For one draw it
    # takes the argmax of `probs / q` with q ~ Exp(1). This does the same
    # without the checks.
    probs = torch.softmax(scaled, dim=-1)
    drawn = (probs / torch.empty_like(probs).exponential_()).argmax(dim=-1)
    return torch.where(is_greedy, greedy, drawn)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k >= logits.shape[-1]:
        return logits
    topk_vals, _ = torch.topk(logits, k)
    threshold = topk_vals[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprob = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # exclude tokens past the cutoff; shift right by 1 to keep the boundary token
    exclude = cumprob > p
    exclude = torch.cat([torch.zeros(1, dtype=torch.bool, device=logits.device), exclude[:-1]])
    sorted_logits = sorted_logits.masked_fill(exclude, float("-inf"))
    result = torch.empty_like(logits)
    result.scatter_(0, sorted_idx, sorted_logits)
    return result


def _apply_top_k_rows(logits: torch.Tensor, top_k: torch.Tensor, max_k: int) -> torch.Tensor:
    """`_apply_top_k` per row; a row with `top_k` 0 keeps every logit."""
    top_values = torch.topk(logits, max_k, dim=-1).values
    index = (top_k.clamp(min=1, max=max_k) - 1).unsqueeze(-1)
    threshold = top_values.gather(-1, index)
    exclude = (logits < threshold) & (top_k > 0).unsqueeze(-1)
    return logits.masked_fill(exclude, float("-inf"))


def _apply_top_p_rows(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """`_apply_top_p` per row; a row with `top_p` 1 keeps every logit."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumprob = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    exclude = cumprob > top_p.unsqueeze(-1)
    # shift right by 1 to keep the boundary token, as `_apply_top_p` does
    exclude = torch.cat([torch.zeros_like(exclude[:, :1]), exclude[:, :-1]], dim=-1)
    # a row at 1.0 skips the filter, as `sample` does, even where rounding
    # puts its cumulative sum above 1
    exclude &= (top_p < 1.0).unsqueeze(-1)
    sorted_logits = sorted_logits.masked_fill(exclude, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
