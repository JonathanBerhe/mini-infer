"""Compressor-projection output-dtype drift probe for DeepSeek-V4.

vLLM runs DeepSeek-V4's two compressor GEMMs (the main attention compressor and
the Lightning Indexer's compressor; `kv_proj` / `weight_proj` here, the fused
`wkv_wgate` there) with `out_dtype=torch.float32`. A bf16 GEMM accumulates in
fp32 either way; the output dtype only decides whether the projected kv / score
entries are rounded to bf16 before the compression softmax and before the
indexer's top-k selection. This module measures that rounding's effect with the
bit-parity port, two ways:

  * Isolated drift. During ONE prefill under the baseline policy, every
    compressor and every indexer is re-run on the same input under the
    alternate policy (`IsolatedTaps`). Comparing the two outputs gives the
    per-layer effect with no drift inherited from earlier layers.
  * End-to-end drift. Two full prefills, one per policy, each followed by a
    short greedy decode (`run_paired_prefill`). Compared at every attention
    block's output on a fixed position subset, at the logits (argmax agreement
    at every prompt position, plus elementwise drift on the subset), at the
    quality of each run's next-token predictions on the prompt itself
    (negative log-likelihood and top-1 accuracy of the real next token), and
    at the generated tokens.

End-to-end drift alone does not say whether a change is harmful: discrete
choices (top-k selections, MoE routing) can amplify any tiny perturbation. Two
controls calibrate it: the same policy twice (`FP32_OUT` vs `FP32_OUT`, the
determinism floor) and the same fp32 math with a different rounding order
(`FP32_OUT` vs `FP32_OUT_UPCAST_GEMM`, the sensitivity floor).

Device-agnostic; batch size 1 (one request's state cache). The Modal driver
runs it on the real checkpoint, `scripts/v4_projection_dtype_probe_synthetic.py`
on CPU with random weights.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mini_infer.engine.sampler import SamplingParams, sample
from mini_infer.models.blocks.csa import CSAAttention
from mini_infer.models.blocks.hca import HCAAttention
from mini_infer.models.blocks.v4 import compressor as compressor_module
from mini_infer.models.blocks.v4.compressor import (
    ProjectionOutDtype,
    TokenLevelCompressor,
    configure_projection_out_dtype,
)
from mini_infer.models.blocks.v4.lightning_indexer import LightningIndexer
from mini_infer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

if TYPE_CHECKING:
    from mini_infer.cache.state_cache import StateCache, _IndexerState

AttentionBlock = CSAAttention | HCAAttention

_GREEDY = SamplingParams(temperature=0.0)
# Contiguous prompt positions per block when estimating the standard error of a
# mean per-position difference (neighbouring positions are correlated).
_NLL_BLOCK_SIZE = 256


def _dtype_name(out_dtype: ProjectionOutDtype) -> str:
    return "native" if out_dtype is None else str(out_dtype).removeprefix("torch.")


@dataclass(frozen=True)
class ProjectionPolicy:
    """One way of running the compressor projections, as the probe compares them."""

    out_dtype: ProjectionOutDtype
    # Only meaningful with `out_dtype=torch.float32` on CUDA: run the fp32 GEMM
    # over upcast operands instead of `torch.mm(..., out_dtype=float32)`. The
    # same math with a different fp32 rounding order, a perturbation of about
    # 1e-7 relative, so it serves as the sensitivity floor.
    force_upcast_gemm: bool = False

    def __post_init__(self) -> None:
        if self.force_upcast_gemm and self.out_dtype is None:
            raise ValueError("force_upcast_gemm applies only to out_dtype=torch.float32")

    @property
    def name(self) -> str:
        suffix = "+upcast_gemm" if self.force_upcast_gemm else ""
        return _dtype_name(self.out_dtype) + suffix


FP32_OUT = ProjectionPolicy(torch.float32)
NATIVE_OUT = ProjectionPolicy(None)
FP32_OUT_UPCAST_GEMM = ProjectionPolicy(torch.float32, force_upcast_gemm=True)


def _never_supported(device_index: int) -> bool:
    del device_index
    return False


@contextlib.contextmanager
def applied_policy(model: DeepseekV4ForCausalLM, policy: ProjectionPolicy) -> Iterator[None]:
    """Configure every compressor in `model` to `policy` for the duration of the block.

    `force_upcast_gemm` is applied by swapping the compressor module's
    `cuda_mm_fp32_out_supported` capability check for one that always answers
    False, so the fp32 path takes its upcast-GEMM branch; the original check is
    restored on exit.
    """
    configure_projection_out_dtype(model, policy.out_dtype)
    original: Callable[[int], bool] = compressor_module.cuda_mm_fp32_out_supported
    if policy.force_upcast_gemm:
        setattr(compressor_module, "cuda_mm_fp32_out_supported", _never_supported)  # noqa: B010
    try:
        yield
    finally:
        setattr(compressor_module, "cuda_mm_fp32_out_supported", original)  # noqa: B010


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorDrift:
    """Elementwise drift of a candidate tensor from a reference (compared in fp32)."""

    max_abs: float
    # ||reference - candidate||_2 / ||reference||_2 (inf if the reference is zero
    # and the candidate is not).
    rms_rel: float
    cosine: float
    numel: int


def tensor_drift(reference: torch.Tensor, candidate: torch.Tensor) -> TensorDrift:
    """Drift of `candidate` from `reference`; both `(...)` of identical shape."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} vs "
            f"candidate {tuple(candidate.shape)}"
        )
    ref = reference.detach().float().reshape(-1)  # (N,)
    cand = candidate.detach().float().reshape(-1)  # (N,)
    numel = ref.numel()
    if numel == 0:
        return TensorDrift(max_abs=0.0, rms_rel=0.0, cosine=1.0, numel=0)
    diff = ref - cand  # (N,)
    ref_norm = float(ref.norm())
    cand_norm = float(cand.norm())
    diff_norm = float(diff.norm())
    if ref_norm > 0.0:
        rms_rel = diff_norm / ref_norm
    elif diff_norm == 0.0:
        rms_rel = 0.0
    else:
        rms_rel = math.inf
    if ref_norm > 0.0 and cand_norm > 0.0:
        cosine = float(torch.dot(ref, cand)) / (ref_norm * cand_norm)
    else:
        cosine = 1.0 if ref_norm == cand_norm else 0.0
    return TensorDrift(max_abs=float(diff.abs().max()), rms_rel=rms_rel, cosine=cosine, numel=numel)


@dataclass(frozen=True)
class TopKAgreement:
    """Set agreement of two top-k index selections, per query (padding `-1` ignored)."""

    # Mean over queries of |A ∩ B| / |A| (1.0 for a query that selects nothing).
    mean_set_agreement: float
    min_set_agreement: float
    # Fraction of queries whose two selections are the same set.
    fraction_identical: float
    n_queries: int
    # Selected slots in A that B did not select, summed over queries.
    n_flipped_slots: int


def topk_set_agreement(reference_idx: torch.Tensor, candidate_idx: torch.Tensor) -> TopKAgreement:
    """Compare two `(B, T, k)` int64 top-k selections as per-query SETS.

    Order within the k slots is irrelevant (the attention softmax does not
    depend on it); `-1` entries are padding and are ignored on both sides.
    """
    if reference_idx.shape != candidate_idx.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference_idx.shape)} vs "
            f"candidate {tuple(candidate_idx.shape)}"
        )
    if reference_idx.dim() != 3:
        raise ValueError(f"expected (B, T, k) indices, got {tuple(reference_idx.shape)}")
    batch_size, seqlen, top_k = reference_idx.shape
    n_queries = batch_size * seqlen
    if n_queries == 0 or top_k == 0:
        return TopKAgreement(1.0, 1.0, 1.0, n_queries, 0)

    # Dense membership masks over the slot range, with one extra column that
    # absorbs the `-1` padding and is cleared afterwards.
    max_slot = int(max(int(reference_idx.max()), int(candidate_idx.max())))
    n_slots = max(max_slot + 1, 1)
    pad_column = n_slots
    ref_cols = torch.where(reference_idx < 0, pad_column, reference_idx)  # (B, T, k)
    cand_cols = torch.where(candidate_idx < 0, pad_column, candidate_idx)  # (B, T, k)
    ones = torch.ones_like(ref_cols, dtype=torch.bool)  # (B, T, k)
    member_ref = torch.zeros(
        batch_size, seqlen, n_slots + 1, dtype=torch.bool, device=reference_idx.device
    )  # (B, T, n_slots + 1)
    member_cand = torch.zeros_like(member_ref)  # (B, T, n_slots + 1)
    member_ref.scatter_(2, ref_cols, ones)
    member_cand.scatter_(2, cand_cols, ones)
    member_ref[..., pad_column] = False
    member_cand[..., pad_column] = False

    intersection = (member_ref & member_cand).sum(dim=2)  # (B, T)
    size_ref = member_ref.sum(dim=2)  # (B, T)
    size_cand = member_cand.sum(dim=2)  # (B, T)
    agreement = torch.where(
        size_ref > 0,
        intersection.float() / size_ref.clamp(min=1).float(),
        torch.ones_like(intersection, dtype=torch.float32),
    )  # (B, T)
    identical = (intersection == size_ref) & (intersection == size_cand)  # (B, T)
    return TopKAgreement(
        mean_set_agreement=float(agreement.mean()),
        min_set_agreement=float(agreement.min()),
        fraction_identical=float(identical.float().mean()),
        n_queries=n_queries,
        n_flipped_slots=int((size_ref - intersection).sum()),
    )


@dataclass(frozen=True)
class IsolatedLayerRecord:
    """Same-input drift of one compressor or indexer under the alternate policy."""

    layer_idx: int
    # "hca.compressor" | "csa.compressor" | "csa.indexer.compressor" | "csa.indexer.topk"
    role: str
    tensor: TensorDrift | None
    topk: TopKAgreement | None


@dataclass(frozen=True)
class CrossRunLayerRecord:
    """Drift of one attention block's prefill output between the two full runs."""

    layer_idx: int
    role: str  # "csa" | "hca"
    n_positions: int
    tensor: TensorDrift


@dataclass(frozen=True)
class LogitsDrift:
    """Drift between the two runs' prompt logits."""

    n_positions: int
    # Fraction of prompt positions whose argmax token agrees.
    argmax_agreement: float
    n_argmax_flips: int
    # Largest top-1 minus top-2 logit gap (baseline run) among flipped positions.
    max_baseline_margin_at_flip: float
    # Elementwise drift on the position subset.
    n_subset_positions: int
    subset_max_abs: float
    subset_mean_abs: float
    # Mean over subset positions of KL(baseline || alternate), in nats.
    subset_mean_kl: float
    # Quality of each run's predictions of the prompt's own next tokens
    # (positions 0 .. T-2 predict tokens 1 .. T-1).
    n_scored_positions: int
    baseline_mean_nll: float
    alternate_mean_nll: float
    # Mean over scored positions of (alternate NLL - baseline NLL), in nats per
    # token; positive means the alternate predicts the real next token worse.
    mean_nll_delta: float
    # Standard error of `mean_nll_delta` from the means of contiguous blocks of
    # positions, which absorbs the correlation between neighbouring positions.
    # None when there are fewer than two blocks.
    nll_delta_block_stderr: float | None
    baseline_top1_accuracy: float
    alternate_top1_accuracy: float
    # Mean NLL of each run over consecutive blocks of `nll_block_size` scored
    # positions (the last block may be shorter): where along the prompt the
    # predictions get better or worse.
    nll_block_size: int
    baseline_nll_by_block: tuple[float, ...]
    alternate_nll_by_block: tuple[float, ...]


@dataclass(frozen=True)
class GreedyTokenDrift:
    """Agreement of the two runs' greedy continuations."""

    n_generated: int
    n_matching_prefix: int
    identical: bool


@dataclass(frozen=True)
class PairedPrefillReport:
    """Everything `run_paired_prefill` measured; `to_dict` is JSON-serializable."""

    baseline: str
    alternate: str
    n_prompt_tokens: int
    isolated: tuple[IsolatedLayerRecord, ...]
    attention_outputs: tuple[CrossRunLayerRecord, ...]
    logits: LogitsDrift
    tokens: GreedyTokenDrift

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Taps
# ---------------------------------------------------------------------------


def attention_blocks(model: DeepseekV4ForCausalLM) -> list[tuple[int, AttentionBlock]]:
    """`(layer_idx, block)` for every CSA / HCA layer; pure-SWA layers are skipped."""
    blocks: list[tuple[int, AttentionBlock]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, CSAAttention | HCAAttention):
            blocks.append((layer_idx, attn))
    return blocks


class IsolatedTaps:
    """Re-run every compressor and indexer under the alternate policy on the same input.

    Install on a model configured to `baseline`. For each
    `forward_prefill_with_cache` call of a `TokenLevelCompressor` or a
    `LightningIndexer`, the original call runs first (mutating the real state),
    then the same call is repeated on a COPY of that state with the module's
    compressor temporarily switched to `alternate`. The drift between the two
    outputs is appended to `records`; the caller receives the original output,
    so the prefill itself is unaffected.

    The indexer's own compressor is tapped twice over: once directly (its
    compressed entries) and once through the indexer (the top-k selection).
    Nested re-runs are suppressed, so each record is a single-policy flip.
    """

    def __init__(
        self,
        model: DeepseekV4ForCausalLM,
        *,
        baseline: ProjectionOutDtype,
        alternate: ProjectionOutDtype,
    ) -> None:
        self.records: list[IsolatedLayerRecord] = []
        self._baseline = baseline
        self._alternate = alternate
        self._suppress_depth = 0
        self._installed: list[tuple[object, str]] = []
        for layer_idx, attn in attention_blocks(model):
            if isinstance(attn, CSAAttention):
                self._tap_compressor(layer_idx, "csa.compressor", attn.compressor)
                self._tap_compressor(layer_idx, "csa.indexer.compressor", attn.indexer.compressor)
                self._tap_indexer(layer_idx, "csa.indexer.topk", attn.indexer)
            else:
                self._tap_compressor(layer_idx, "hca.compressor", attn.compressor)

    def __enter__(self) -> IsolatedTaps:
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()

    def remove(self) -> None:
        """Restore the original methods (the class attributes shadowed per instance)."""
        for owner, name in self._installed:
            delattr(owner, name)
        self._installed.clear()

    @contextlib.contextmanager
    def _alternate_policy(self, compressor: TokenLevelCompressor) -> Iterator[None]:
        if compressor.projection_out_dtype != self._baseline:
            raise RuntimeError(
                "IsolatedTaps expects every compressor configured to the baseline policy "
                f"{_dtype_name(self._baseline)}; found "
                f"{_dtype_name(compressor.projection_out_dtype)}"
            )
        compressor.projection_out_dtype = self._alternate
        self._suppress_depth += 1
        try:
            yield
        finally:
            self._suppress_depth -= 1
            compressor.projection_out_dtype = self._baseline

    def _tap_compressor(self, layer_idx: int, role: str, compressor: TokenLevelCompressor) -> None:
        original = compressor.forward_prefill_with_cache

        @functools.wraps(original)
        def tapped(
            hidden_states: torch.Tensor,
            *,
            compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            cmp_kv_state: torch.Tensor,
            cmp_score_state: torch.Tensor,
        ) -> torch.Tensor:
            if self._suppress_depth > 0:
                return original(
                    hidden_states,
                    compressed_position_embeddings=compressed_position_embeddings,
                    cmp_kv_state=cmp_kv_state,
                    cmp_score_state=cmp_score_state,
                )
            kv_state_copy = cmp_kv_state.clone()  # (B, coff*m, coff*c)
            score_state_copy = cmp_score_state.clone()  # (B, coff*m, coff*c)
            out = original(
                hidden_states,
                compressed_position_embeddings=compressed_position_embeddings,
                cmp_kv_state=cmp_kv_state,
                cmp_score_state=cmp_score_state,
            )  # (B, T // m, c)
            with self._alternate_policy(compressor):
                alt = original(
                    hidden_states,
                    compressed_position_embeddings=compressed_position_embeddings,
                    cmp_kv_state=kv_state_copy,
                    cmp_score_state=score_state_copy,
                )  # (B, T // m, c)
            self.records.append(
                IsolatedLayerRecord(layer_idx, role, tensor=tensor_drift(out, alt), topk=None)
            )
            return out

        setattr(compressor, "forward_prefill_with_cache", tapped)  # noqa: B010
        self._installed.append((compressor, "forward_prefill_with_cache"))

    def _tap_indexer(self, layer_idx: int, role: str, indexer: LightningIndexer) -> None:
        original = indexer.forward_prefill_with_cache

        @functools.wraps(original)
        def tapped(
            hidden_states: torch.Tensor,
            q_lora_latent: torch.Tensor,
            *,
            token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            indexer_state: _IndexerState,
            compressed_offset: int,
        ) -> torch.Tensor:
            if self._suppress_depth > 0:
                return original(
                    hidden_states,
                    q_lora_latent,
                    token_position_embeddings=token_position_embeddings,
                    compressed_position_embeddings=compressed_position_embeddings,
                    indexer_state=indexer_state,
                    compressed_offset=compressed_offset,
                )
            state_copy = copy.deepcopy(indexer_state)
            out = original(
                hidden_states,
                q_lora_latent,
                token_position_embeddings=token_position_embeddings,
                compressed_position_embeddings=compressed_position_embeddings,
                indexer_state=indexer_state,
                compressed_offset=compressed_offset,
            )  # (B, T, top_k) int64
            with self._alternate_policy(indexer.compressor):
                alt = original(
                    hidden_states,
                    q_lora_latent,
                    token_position_embeddings=token_position_embeddings,
                    compressed_position_embeddings=compressed_position_embeddings,
                    indexer_state=state_copy,
                    compressed_offset=compressed_offset,
                )  # (B, T, top_k) int64
            self.records.append(
                IsolatedLayerRecord(layer_idx, role, tensor=None, topk=topk_set_agreement(out, alt))
            )
            return out

        setattr(indexer, "forward_prefill_with_cache", tapped)  # noqa: B010
        self._installed.append((indexer, "forward_prefill_with_cache"))


class AttentionOutputTaps:
    """Record every attention block's prefill output on a fixed position subset."""

    def __init__(self, model: DeepseekV4ForCausalLM, positions: torch.Tensor) -> None:
        self.outputs: dict[int, torch.Tensor] = {}  # layer_idx -> (S, hidden) fp32 CPU
        self.roles: dict[int, str] = {}
        self._positions = positions  # (S,) int64
        self._installed: list[tuple[object, str]] = []
        for layer_idx, attn in attention_blocks(model):
            self.roles[layer_idx] = "csa" if isinstance(attn, CSAAttention) else "hca"
            self._tap(layer_idx, attn)

    def remove(self) -> None:
        for owner, name in self._installed:
            delattr(owner, name)
        self._installed.clear()

    def _tap(self, tapped_layer_idx: int, attn: AttentionBlock) -> None:
        original = attn.forward_prefill_with_cache

        @functools.wraps(original)
        def tapped(
            hidden_states: torch.Tensor,
            *,
            token_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            compressed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            state_cache: StateCache,
            layer_idx: int,
        ) -> torch.Tensor:
            out = original(
                hidden_states,
                token_position_embeddings=token_position_embeddings,
                compressed_position_embeddings=compressed_position_embeddings,
                state_cache=state_cache,
                layer_idx=layer_idx,
            )  # (B, T, hidden)
            subset = self._positions.to(out.device)  # (S,)
            self.outputs[tapped_layer_idx] = out[0, subset].detach().float().cpu()  # (S, hidden)
            return out

        setattr(attn, "forward_prefill_with_cache", tapped)  # noqa: B010
        self._installed.append((attn, "forward_prefill_with_cache"))


# ---------------------------------------------------------------------------
# Paired run
# ---------------------------------------------------------------------------


def subset_positions(seqlen: int, *, stride: int, tail: int) -> torch.Tensor:
    """Sorted `(S,)` int64 positions: every `stride`-th token plus the last `tail` tokens."""
    if seqlen <= 0:
        raise ValueError(f"seqlen must be positive, got {seqlen}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    chosen = set(range(0, seqlen, stride)) | set(range(max(0, seqlen - tail), seqlen))
    return torch.tensor(sorted(chosen), dtype=torch.int64)


@dataclass(eq=False)
class LogitsSummary:
    """What one run keeps of its `(1, T, V)` prompt logits (all on CPU)."""

    argmax: torch.Tensor  # (T,) int64
    top1_margin: torch.Tensor  # (T,) fp32: top-1 minus top-2 logit
    subset_logits: torch.Tensor  # (S, V) fp32
    target_nll: torch.Tensor  # (T-1,) fp32: -log p(prompt token t+1 | prefix up to t)
    target_top1: torch.Tensor  # (T-1,) bool: argmax at t == prompt token t+1


def _summarize_logits(
    logits: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor, *, chunk: int = 2048
) -> LogitsSummary:
    """Reduce `(1, T, V)` logits chunk by chunk so no full fp32 `(T, V)` copy is held.

    `targets` is `(T-1,)` int64: the prompt's tokens 1 .. T-1, the real next
    token at positions 0 .. T-2.
    """
    row = logits[0]  # (T, V)
    seqlen = row.shape[0]
    n_scored = seqlen - 1
    if targets.shape != (n_scored,):
        raise ValueError(f"targets must be ({n_scored},); got {tuple(targets.shape)}")
    argmax = torch.empty(seqlen, dtype=torch.int64)  # (T,)
    margin = torch.empty(seqlen, dtype=torch.float32)  # (T,)
    nll = torch.empty(n_scored, dtype=torch.float32)  # (T-1,)
    top1 = torch.empty(n_scored, dtype=torch.bool)  # (T-1,)
    targets_dev = targets.to(row.device)  # (T-1,)
    for start in range(0, seqlen, chunk):
        end = min(start + chunk, seqlen)
        piece = row[start:end].float()  # (c, V)
        top2 = piece.topk(2, dim=-1)  # values / indices: (c, 2)
        argmax[start:end] = top2.indices[:, 0].cpu()
        margin[start:end] = (top2.values[:, 0] - top2.values[:, 1]).cpu()
        scored_end = min(end, n_scored)
        if scored_end > start:
            n_rows = scored_end - start
            log_probs = piece[:n_rows].log_softmax(dim=-1)  # (c', V)
            tgt = targets_dev[start:scored_end]  # (c',)
            nll[start:scored_end] = (-log_probs.gather(1, tgt[:, None]).squeeze(1)).cpu()
            top1[start:scored_end] = (top2.indices[:n_rows, 0] == tgt).cpu()
    subset = row[positions.to(row.device)].float().cpu()  # (S, V)
    return LogitsSummary(
        argmax=argmax,
        top1_margin=margin,
        subset_logits=subset,
        target_nll=nll,
        target_top1=top1,
    )


def block_means(values: torch.Tensor, *, block_size: int = _NLL_BLOCK_SIZE) -> tuple[float, ...]:
    """Means of `(N,)` `values` over consecutive blocks of `block_size`; the last may be shorter."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return tuple(
        float(values[start : start + block_size].double().mean())
        for start in range(0, values.shape[0], block_size)
    )


def block_stderr(values: torch.Tensor, *, block_size: int = _NLL_BLOCK_SIZE) -> float | None:
    """Standard error of `values.mean()` from the means of contiguous blocks.

    `values` is `(N,)`. The block size shrinks for short inputs so that there
    are at least eight blocks when `N >= 8`; returns None below two blocks. A
    trailing partial block is dropped.
    """
    n = values.shape[0]
    size = max(1, min(block_size, n // 8))
    n_blocks = n // size
    if n_blocks < 2:
        return None
    block_means = values[: n_blocks * size].double().view(n_blocks, size).mean(dim=1)  # (n_blocks,)
    return float(block_means.std(unbiased=True) / math.sqrt(n_blocks))


def _prefill_and_decode(
    model: DeepseekV4ForCausalLM,
    input_ids: torch.Tensor,
    *,
    positions: torch.Tensor,
    max_new_tokens: int,
    state_dtype: torch.dtype,
) -> tuple[LogitsSummary, list[int]]:
    """One cache-aware prefill of `(1, T)` ids, then `max_new_tokens` greedy steps."""
    device = input_ids.device
    seqlen = input_ids.shape[1]
    state_cache = model.build_state_cache(
        max_seq_len=seqlen + max_new_tokens, batch_size=1, device=device, dtype=state_dtype
    )
    with torch.inference_mode():
        logits = model.forward_prefill_with_cache(input_ids, state_cache=state_cache)  # (1, T, V)
        summary = _summarize_logits(logits, positions, input_ids[0, 1:])
        next_token = sample(logits[0, -1, :], _GREEDY)
        del logits
    state_cache.advance_start_pos(seqlen)
    tokens = [next_token]
    for _ in range(max_new_tokens - 1):
        token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)  # (1, 1)
        with torch.inference_mode():
            step_logits = model.forward_decode_with_cache(
                token_tensor, start_pos=state_cache.start_pos, state_cache=state_cache
            )  # (1, 1, V)
            next_token = sample(step_logits[0, -1, :], _GREEDY)
        state_cache.advance_start_pos(1)
        tokens.append(next_token)
    return summary, tokens


def logits_drift(baseline: LogitsSummary, alternate: LogitsSummary) -> LogitsDrift:
    """Compare the two runs' logit summaries."""
    flips = baseline.argmax != alternate.argmax  # (T,) bool
    n_flips = int(flips.sum())
    max_margin = float(baseline.top1_margin[flips].max()) if n_flips > 0 else 0.0
    diff = baseline.subset_logits - alternate.subset_logits  # (S, V)
    log_p = baseline.subset_logits.log_softmax(dim=-1)  # (S, V)
    log_q = alternate.subset_logits.log_softmax(dim=-1)  # (S, V)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # (S,)
    seqlen = baseline.argmax.shape[0]
    nll_delta = alternate.target_nll - baseline.target_nll  # (T-1,)
    n_scored = int(nll_delta.shape[0])
    return LogitsDrift(
        n_positions=seqlen,
        argmax_agreement=1.0 - n_flips / seqlen,
        n_argmax_flips=n_flips,
        max_baseline_margin_at_flip=max_margin,
        n_subset_positions=int(diff.shape[0]),
        subset_max_abs=float(diff.abs().max()) if diff.numel() > 0 else 0.0,
        subset_mean_abs=float(diff.abs().mean()) if diff.numel() > 0 else 0.0,
        subset_mean_kl=float(kl.mean()) if kl.numel() > 0 else 0.0,
        n_scored_positions=n_scored,
        baseline_mean_nll=float(baseline.target_nll.double().mean()) if n_scored else 0.0,
        alternate_mean_nll=float(alternate.target_nll.double().mean()) if n_scored else 0.0,
        mean_nll_delta=float(nll_delta.double().mean()) if n_scored else 0.0,
        nll_delta_block_stderr=block_stderr(nll_delta),
        baseline_top1_accuracy=float(baseline.target_top1.float().mean()) if n_scored else 0.0,
        alternate_top1_accuracy=float(alternate.target_top1.float().mean()) if n_scored else 0.0,
        nll_block_size=_NLL_BLOCK_SIZE,
        baseline_nll_by_block=block_means(baseline.target_nll),
        alternate_nll_by_block=block_means(alternate.target_nll),
    )


def greedy_token_drift(baseline: list[int], alternate: list[int]) -> GreedyTokenDrift:
    """Length of the common prefix of two greedy continuations."""
    n_matching = 0
    for a, b in zip(baseline, alternate, strict=True):
        if a != b:
            break
        n_matching += 1
    return GreedyTokenDrift(
        n_generated=len(baseline),
        n_matching_prefix=n_matching,
        identical=baseline == alternate,
    )


def run_paired_prefill(
    model: DeepseekV4ForCausalLM,
    input_ids: torch.Tensor,
    *,
    baseline: ProjectionPolicy = FP32_OUT,
    alternate: ProjectionPolicy = NATIVE_OUT,
    isolated: bool = True,
    max_new_tokens: int = 16,
    position_stride: int = 64,
    position_tail: int = 128,
    state_dtype: torch.dtype | None = None,
) -> PairedPrefillReport:
    """Measure isolated and end-to-end drift between two projection policies.

    Args:
        model: a V4 model (any device); left configured to `baseline` on return.
        input_ids: `(1, T)` prompt token ids on the model's device, `T >= 2`.
        baseline: policy of the reference run (vLLM today: `FP32_OUT`).
        alternate: policy under test (`NATIVE_OUT` = native bf16 output), or a
            control (`FP32_OUT` again, `FP32_OUT_UPCAST_GEMM`).
        isolated: also re-run every compressor and indexer of the baseline run
            under the alternate's `out_dtype` (`IsolatedTaps`). Requires both
            policies to agree on `force_upcast_gemm`.
        max_new_tokens: greedy tokens decoded after each prefill (>= 1).
        position_stride, position_tail: which prompt positions keep full
            attention-block outputs and logits for elementwise comparison.
        state_dtype: dtype of the per-request state cache; defaults to the
            model's parameter dtype.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError(f"input_ids must be (1, T) with T >= 2; got {tuple(input_ids.shape)}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if isolated and baseline.force_upcast_gemm != alternate.force_upcast_gemm:
        raise ValueError(
            "isolated taps flip only out_dtype; baseline and alternate must agree on "
            "force_upcast_gemm (pass isolated=False for the upcast-GEMM control)"
        )
    seqlen = input_ids.shape[1]
    resolved_state_dtype = (
        state_dtype if state_dtype is not None else next(model.parameters()).dtype
    )
    positions = subset_positions(seqlen, stride=position_stride, tail=position_tail)  # (S,)
    isolated_records: list[IsolatedLayerRecord] = []

    try:
        # Run A: baseline policy, optionally with isolated per-layer re-runs.
        with applied_policy(model, baseline):
            attn_taps_a = AttentionOutputTaps(model, positions)
            taps: contextlib.AbstractContextManager[IsolatedTaps | None] = (
                IsolatedTaps(model, baseline=baseline.out_dtype, alternate=alternate.out_dtype)
                if isolated
                else contextlib.nullcontext()
            )
            try:
                with taps as isolated_taps:
                    summary_a, tokens_a = _prefill_and_decode(
                        model,
                        input_ids,
                        positions=positions,
                        max_new_tokens=max_new_tokens,
                        state_dtype=resolved_state_dtype,
                    )
                    if isolated_taps is not None:
                        isolated_records = list(isolated_taps.records)
            finally:
                attn_taps_a.remove()

        # Run B: alternate policy end to end.
        with applied_policy(model, alternate):
            attn_taps_b = AttentionOutputTaps(model, positions)
            try:
                summary_b, tokens_b = _prefill_and_decode(
                    model,
                    input_ids,
                    positions=positions,
                    max_new_tokens=max_new_tokens,
                    state_dtype=resolved_state_dtype,
                )
            finally:
                attn_taps_b.remove()
    finally:
        configure_projection_out_dtype(model, baseline.out_dtype)

    attention_records = tuple(
        CrossRunLayerRecord(
            layer_idx=layer_idx,
            role=attn_taps_a.roles[layer_idx],
            n_positions=int(positions.shape[0]),
            tensor=tensor_drift(attn_taps_a.outputs[layer_idx], attn_taps_b.outputs[layer_idx]),
        )
        for layer_idx in sorted(attn_taps_a.outputs)
    )
    return PairedPrefillReport(
        baseline=baseline.name,
        alternate=alternate.name,
        n_prompt_tokens=seqlen,
        isolated=tuple(isolated_records),
        attention_outputs=attention_records,
        logits=logits_drift(summary_a, summary_b),
        tokens=greedy_token_drift(tokens_a, tokens_b),
    )


# ---------------------------------------------------------------------------
# Synthetic model + report formatting
# ---------------------------------------------------------------------------


def make_synthetic_hybrid_config(
    *, num_layers: int = 6, hidden_size: int = 128, vocab_size: int = 512
) -> DeepseekV4Config:
    """A small CSA / HCA alternating config (ratios 4, 8, 4, 8, ...) for CPU runs."""
    compress_ratios = tuple(4 if layer_idx % 2 == 0 else 8 for layer_idx in range(num_layers))
    return DeepseekV4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_head_dim=64,
        rope_head_dim=16,
        o_num_groups=2,
        o_lora_rank=64,
        window_size=16,
        compress_ratios=compress_ratios,
        index_num_heads=4,
        index_head_dim=32,
        index_top_k=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def format_report(report: PairedPrefillReport) -> str:
    """Compact human-readable rendering of a `PairedPrefillReport`."""
    lines = [
        f"policies: baseline={report.baseline} alternate={report.alternate} "
        f"prompt_tokens={report.n_prompt_tokens}",
    ]
    if report.isolated:
        lines += [
            "",
            "isolated per-layer drift (alternate vs baseline on the same input):",
            f"  {'layer':>5} {'role':<24} {'max_abs':>10} {'rms_rel':>10} {'cosine':>10} "
            f"{'topk_mean':>10} {'identical':>10} {'flipped':>8}",
        ]
    for rec in report.isolated:
        if rec.tensor is not None:
            lines.append(
                f"  {rec.layer_idx:>5} {rec.role:<24} {rec.tensor.max_abs:>10.3e} "
                f"{rec.tensor.rms_rel:>10.3e} {rec.tensor.cosine:>10.6f} {'':>10} {'':>10} {'':>8}"
            )
        elif rec.topk is not None:
            lines.append(
                f"  {rec.layer_idx:>5} {rec.role:<24} {'':>10} {'':>10} {'':>10} "
                f"{rec.topk.mean_set_agreement:>10.6f} {rec.topk.fraction_identical:>10.6f} "
                f"{rec.topk.n_flipped_slots:>8d}"
            )
    lines += ["", "attention-block outputs across the two runs (position subset):"]
    for cross in report.attention_outputs:
        lines.append(
            f"  layer {cross.layer_idx:>3} {cross.role:<4} positions={cross.n_positions:<5} "
            f"max_abs={cross.tensor.max_abs:.3e} rms_rel={cross.tensor.rms_rel:.3e} "
            f"cosine={cross.tensor.cosine:.6f}"
        )
    lg = report.logits
    lines += [
        "",
        f"logits: argmax agreement {lg.argmax_agreement:.6f} over {lg.n_positions} positions "
        f"({lg.n_argmax_flips} flips, max baseline top1-top2 margin at a flip "
        f"{lg.max_baseline_margin_at_flip:.4f}); subset[{lg.n_subset_positions}] "
        f"max_abs={lg.subset_max_abs:.3e} mean_abs={lg.subset_mean_abs:.3e} "
        f"mean_kl={lg.subset_mean_kl:.3e} nats",
        f"next-token quality over {lg.n_scored_positions} prompt positions: "
        f"NLL baseline {lg.baseline_mean_nll:.5f} alternate {lg.alternate_mean_nll:.5f} "
        f"(delta {lg.mean_nll_delta:+.5f} nats/token, block stderr "
        + ("n/a" if lg.nll_delta_block_stderr is None else f"{lg.nll_delta_block_stderr:.5f}")
        + f"); perplexity {math.exp(lg.baseline_mean_nll):.4f} vs "
        f"{math.exp(lg.alternate_mean_nll):.4f}; top-1 accuracy "
        f"{lg.baseline_top1_accuracy:.5f} vs {lg.alternate_top1_accuracy:.5f}",
        f"greedy tokens: {report.tokens.n_generated} generated, common prefix "
        f"{report.tokens.n_matching_prefix}, identical={report.tokens.identical}",
        f"baseline NLL per block of {lg.nll_block_size} positions: "
        + " ".join(f"{value:.2f}" for value in lg.baseline_nll_by_block),
    ]
    return "\n".join(lines)
