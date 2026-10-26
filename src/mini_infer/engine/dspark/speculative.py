"""Greedy speculative decoding driven by the DSpark drafter.

The shape of one round, and how it differs from ADR-011's two-model V1
(`engine/speculative.py`):

- **Draft.** V1 runs K serial forwards through a small autoregressive model.
  Here one backbone pass proposes the whole `block_size` block, then a cheap
  sequential Markov loop over the block's logits picks the tokens. The
  expensive part is O(1) in block length, which is why a block of 7 is
  affordable where 7 serial draft forwards would not be.
- **Truncate.** The confidence head predicts each draft token's survival
  probability, and `confident_prefix_length` cuts the proposal at the first
  position that looks doomed, so the target's verify forward doesn't carry
  tokens it was going to reject anyway.
- **Verify.** One packed target forward over `[anchor, d_0 .. d_{n-1}]`,
  identical in spirit to V1.
- **Commit.** Accept the matching prefix, then the target's own argmax at the
  first disagreement (the "bonus"). Greedy accept-reject is argmax equality,
  so output is token-for-token what target-alone greedy would produce, the
  same argument as ADR-011.
- **No catch-up step.** V1 needs an extra draft forward when all K candidates
  pass, because its draft cache must contain an entry for the bonus token it
  never ran. The DSpark drafter discards its own block K/V every round
  regardless of the outcome, so there is nothing to resync (ADR-027).

What carries across rounds is the injected context: after verifying, the
target's hidden states at the tapped layers for exactly the committed window
(accepted tokens plus the bonus) become the next round's context. The
drafter's cache keeps the accumulated projections of those and throws away
the block's own K/V, which is why `truncate_to(start)` runs every round with
the round's own starting length, not an acceptance-dependent one.

Batch-1 only, matching the reference's own inference path; multi-request is
Stage D.
"""

from __future__ import annotations

import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.dspark.draft_cache import DSparkDraftCache
from mini_infer.engine.dspark.drafter import Qwen3DSparkDrafter
from mini_infer.engine.dspark.proposal import confident_prefix_length
from mini_infer.engine.dspark.spec import DSparkStats, propose_block, verify_block
from mini_infer.engine.model_runner import ModelRunner

__all__ = ["DSparkSpeculativeRunner", "DSparkStats"]


class DSparkSpeculativeRunner:
    """Greedy DSpark speculative decoding: one Qwen3 target + one DSpark drafter.

    `confidence_threshold` of 0 disables truncation (verify the whole block);
    a positive value cuts the proposal at the first draft token whose
    predicted survival probability falls below it.

    `temperature` of 0 is greedy, where output is provably token-identical to
    target-alone decoding. Above 0 the loop switches to rejection sampling,
    which is distribution-preserving rather than token-identical: the emitted
    tokens are a draw from the target's own distribution, so per-run outputs
    differ from any particular target-alone run while the distribution matches
    (see `sampling.py` for why, and the statistical test in
    `tests/unit/engine/dspark/test_dspark_rejection_sampling.py`).
    """

    def __init__(
        self,
        target: ModelRunner,
        drafter: Qwen3DSparkDrafter,
        *,
        confidence_threshold: float = 0.0,
        temperature: float = 0.0,
    ) -> None:
        if temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
        if confidence_threshold > 0.0 and drafter.confidence_head is None:
            raise ValueError(
                "confidence_threshold > 0 requires a drafter with a confidence head; "
                "this checkpoint has enable_confidence_head=False"
            )
        self.target = target
        self.drafter = drafter
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.block_size = drafter.cfg.block_size
        self._tap_layers = frozenset(drafter.cfg.target_layer_ids)
        self._ordered_taps = list(drafter.cfg.target_layer_ids)

    def _context_from(self, sink: dict[int, torch.Tensor]) -> torch.Tensor:
        """Concatenate the tapped layers in config order.

        Mirrors `deepspec`'s `extract_context_feature`, which concatenates
        along the feature axis in the order `target_layer_ids` lists; the
        drafter's `fc` was trained against exactly that ordering, so a sorted
        or set-ordered concat would silently feed it permuted features.
        """
        return torch.cat([sink[i] for i in self._ordered_taps], dim=-1)

    def run_greedy(self, prompt_ids: list[int], max_tokens: int) -> tuple[list[int], DSparkStats]:
        """Greedy-decode up to `max_tokens` tokens. Returns `(generated, stats)`.

        Output is token-for-token identical to target-alone greedy decoding;
        the drafter only changes how many target forwards it takes to get
        there.
        """
        stats = DSparkStats()
        if max_tokens < 1 or not prompt_ids:
            return [], stats

        eos_id = self.target.tokenizer.eos_token_id
        target_cache = PagedKVCache(self.target.block_pool)
        target_slot = target_cache.add_request_slot()
        draft_cache = DSparkDraftCache(self.drafter.cfg.num_hidden_layers)

        try:
            # Prefill: the tapped hidden states over the whole prompt are the
            # first round's injected context, and the last position's logits
            # give the anchor the first block hangs off.
            sink: dict[int, torch.Tensor] = {}
            logits = self.target.forward_step_packed(
                target_cache,
                prompt_ids,
                [0, len(prompt_ids)],
                [0],
                tap_layers=self._tap_layers,
                hidden_state_sink=sink,
            )
            stats.n_target_forwards += 1
            context = self._context_from(sink)

            anchor = int(logits[0, -1].argmax())
            generated = [anchor]
            if anchor == eos_id or len(generated) >= max_tokens:
                return generated[:max_tokens], stats

            start = len(prompt_ids)

            while len(generated) < max_tokens:
                draft_tokens, confidence, draft_probs = self._propose(
                    anchor=anchor, context=context, start=start, draft_cache=draft_cache
                )
                stats.n_drafter_forwards += 1

                proposal_len = self.block_size
                if confidence is not None:
                    proposal_len = confident_prefix_length(
                        confidence,
                        block_size=self.block_size,
                        threshold=self.confidence_threshold,
                    )
                offered = draft_tokens[:proposal_len]

                # One target forward over [anchor, offered...]. Even with an
                # empty proposal this still runs, on the anchor alone, which
                # degenerates to ordinary autoregressive decoding for the round.
                verify_inputs = [anchor, *offered]
                verify_sink: dict[int, torch.Tensor] = {}
                verify_logits = self.target.forward_step_packed(
                    target_cache,
                    verify_inputs,
                    [0, len(verify_inputs)],
                    [start],
                    tap_layers=self._tap_layers,
                    hidden_state_sink=verify_sink,
                )
                stats.n_target_forwards += 1

                accepted, bonus = self._verify(
                    verify_logits=verify_logits,
                    offered=offered,
                    draft_probs=draft_probs,
                )

                if confidence is not None:
                    conf_probs = confidence[0].sigmoid().tolist()
                    for i in range(len(offered)):
                        stats.confidence_observations.append((float(conf_probs[i]), i < accepted))

                stats.proposal_lengths.append(len(offered))
                stats.accepted_draft_lengths.append(accepted)
                stats.acceptance_lengths.append(accepted + 1)

                committed = [*offered[:accepted], bonus]
                room = max_tokens - len(generated)
                emitted: list[int] = []
                hit_eos = False
                for tok in committed[:room]:
                    emitted.append(tok)
                    if tok == eos_id:
                        hit_eos = True
                        break
                generated.extend(emitted)

                if hit_eos or len(generated) >= max_tokens:
                    break

                # Roll the target cache back to exactly the committed prefix.
                # The verify forward wrote K/V for every offered token; the
                # rejected tail is wrong and gets dropped here, and the next
                # round rewrites those positions.
                new_start = start + len(emitted)
                target_cache.truncate_to(target_slot, new_start)

                # Next round's context is the target's hidden states for the
                # committed window only: positions 0..accepted of this verify
                # forward, which are the anchor plus the accepted tokens.
                # `deepspec`'s `_update` reassigns rather than appending.
                verify_context = self._context_from(verify_sink)
                context = verify_context[:, : accepted + 1, :]

                anchor = bonus
                start = new_start

            return generated[:max_tokens], stats
        finally:
            target_cache.free()

    def _propose(
        self,
        *,
        anchor: int,
        context: torch.Tensor,
        start: int,
        draft_cache: DSparkDraftCache,
    ) -> tuple[list[int], torch.Tensor | None, torch.Tensor]:
        """Run one drafter round with `spec.propose_block`.

        The multi-request scheduler also uses `propose_block` and `verify_block`,
        so the round math of the batch-1 path and the multi-request path cannot diverge.
        """
        return propose_block(
            self.drafter,
            anchor=anchor,
            context=context,
            start=start,
            draft_cache=draft_cache,
            temperature=self.temperature,
        )

    def _verify(
        self,
        *,
        verify_logits: torch.Tensor,
        offered: list[int],
        draft_probs: torch.Tensor,
    ) -> tuple[int, int]:
        """Accept a prefix of `offered` and pick the next token with `spec.verify_block`."""
        return verify_block(
            verify_logits,
            offered=offered,
            draft_probs=draft_probs,
            temperature=self.temperature,
        )
