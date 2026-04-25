import dataclasses

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
