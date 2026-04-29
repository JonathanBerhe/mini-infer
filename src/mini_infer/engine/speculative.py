"""Single-request greedy speculative decoding (vanilla, two-model).

Implements the Leviathan / Chen recipe in its simplest form: a small *draft*
model proposes K tokens, the *target* model runs once over the K+1 candidates
([last_committed_token, d_0, ..., d_{K-1}]), and we accept candidates from the
left while target's argmax matches. The first mismatch is replaced with
target's correct token (the "bonus"); if all K candidates pass, the bonus
becomes target's prediction for the position after d_{K-1}.

Greedy means temperature=0 means accept-reject collapses to argmax equality.
The full rejection-sampling formula (`accept iff u < min(1, p_t/p_d)`)
becomes a no-op here; sampling support is an explicit follow-up.

Cache management is the subtle part:

- Both models keep their own `PagedKVCache` with a single slot.
- After draft's K steps, draft_cache.seq_len = N + K.
- After target's verify, target_cache.seq_len = N + K + 1.
- Accept-reject: emit `accepted` draft tokens + 1 bonus; total `emit_count = accepted + 1`.
- Both caches must end at seq_len = N + emit_count for the next iteration.
  - target: truncate from N+K+1 down to N+emit_count (always shrinks).
  - draft: if accepted < K, truncate from N+K down to N+emit_count. If
    accepted == K (all-accepted), draft is at N+K but we need N+K+1; run a
    single draft forward on d_{K-1} at position N+K to fill the gap.

The draft and target caches truncate to the SAME seq_len, then the next
iteration starts with `last_token = bonus` to feed at position N + emit_count.
"""

from __future__ import annotations

import dataclasses

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.model_runner import ModelRunner

DEFAULT_K = 4


@dataclasses.dataclass
class SpecStats:
    """Per-run counters for speculative decoding.

    `mean_acceptance_per_iter` is `n_accepted_total / max(1, n_iterations)`;
    a value approaching K means the draft is matching target's argmax on
    almost every candidate (best case). A value near 0 means the draft is
    mostly wrong and we're paying draft-forward cost without benefit.
    """

    n_iterations: int = 0
    n_target_forwards: int = 0
    n_draft_forwards: int = 0
    n_accepted_total: int = 0
    n_emitted_total: int = 0

    @property
    def mean_acceptance_per_iter(self) -> float:
        return self.n_accepted_total / self.n_iterations if self.n_iterations else 0.0


class SpeculativeRunner:
    """Greedy speculative decoding with a small draft and a large target model.

    V1 scope: single request, greedy (temperature=0), fixed K. Both models
    share the same tokenizer family (verified at init).
    """

    def __init__(
        self,
        target: ModelRunner,
        draft: ModelRunner,
        *,
        K: int = DEFAULT_K,  # noqa: N803 (canonical name in the spec-decode literature)
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        # Vocabulary mismatch breaks accept-reject (token IDs would mean
        # different tokens in each model). Same-family Qwen2.5 models share a
        # vocab; cross-family pairs do not.
        if target.tokenizer.vocab_size != draft.tokenizer.vocab_size:
            raise ValueError(
                f"target/draft vocab mismatch: target={target.tokenizer.vocab_size} "
                f"vs draft={draft.tokenizer.vocab_size}"
            )
        self.target = target
        self.draft = draft
        self.K = K

    def run_greedy(self, prompt: str, max_tokens: int) -> tuple[list[int], SpecStats]:
        """Greedy spec-decode `prompt` for up to `max_tokens` tokens.

        Returns `(generated_token_ids, stats)`. EOS in any emitted position
        terminates the run; `len(generated) <= max_tokens`.
        """
        if max_tokens < 1:
            return [], SpecStats()

        prompt_ids = self.target.tokenizer.encode(prompt)
        if not prompt_ids:
            return [], SpecStats()

        eos_id = self.target.tokenizer.eos_token_id

        target_cache = PagedKVCache(self.target.block_pool)
        draft_cache = PagedKVCache(self.draft.block_pool)
        target_batch = target_cache.add_request_slot()
        draft_batch = draft_cache.add_request_slot()

        stats = SpecStats()

        try:
            # Prefill both models on the full prompt. Target's last-position
            # logits give us the first generated token.
            target_packed = self.target.forward_step_packed(
                target_cache, prompt_ids, [0, len(prompt_ids)], [0]
            )
            self.draft.forward_step(draft_cache, prompt_ids, [0, len(prompt_ids)], [0])
            stats.n_target_forwards += 1
            stats.n_draft_forwards += 1

            last_token = int(target_packed[0, -1, :].argmax().item())
            generated: list[int] = [last_token]
            cache_seq = len(prompt_ids)  # both caches at this seq_len

            if last_token == eos_id or len(generated) >= max_tokens:
                return generated[:max_tokens], stats

            while len(generated) < max_tokens:
                # --- draft phase: K serial decode steps ---
                draft_tokens: list[int] = []
                current = last_token
                for k in range(self.K):
                    logits_list = self.draft.forward_step(
                        draft_cache, [current], [0, 1], [cache_seq + k]
                    )
                    d_k = int(logits_list[0].argmax().item())
                    draft_tokens.append(d_k)
                    current = d_k
                stats.n_draft_forwards += self.K
                # draft_cache.seq_len is now cache_seq + K.

                # --- verify phase: one target forward over K+1 candidates ---
                verify_inputs = [last_token, *draft_tokens]
                target_packed = self.target.forward_step_packed(
                    target_cache, verify_inputs, [0, self.K + 1], [cache_seq]
                )
                stats.n_target_forwards += 1
                # target_cache.seq_len is now cache_seq + K + 1.

                # --- accept-reject (greedy) ---
                accepted = 0
                target_argmax = target_packed[0].argmax(dim=-1).tolist()
                for i in range(self.K):
                    if target_argmax[i] == draft_tokens[i]:
                        accepted += 1
                    else:
                        break
                bonus_idx = accepted  # works for both partial-reject and all-accepted
                bonus = int(target_argmax[bonus_idx])
                stats.n_accepted_total += accepted

                # --- emit accepted + bonus, capped by max_tokens and EOS ---
                planned = [*draft_tokens[:accepted], bonus]  # length = accepted + 1
                room = max_tokens - len(generated)
                if room < len(planned):
                    planned = planned[:room]

                # Truncate at first EOS (include the EOS token in the output).
                emitted: list[int] = []
                hit_eos = False
                for tok in planned:
                    emitted.append(tok)
                    if tok == eos_id:
                        hit_eos = True
                        break

                generated.extend(emitted)
                stats.n_emitted_total += len(emitted)
                stats.n_iterations += 1

                # Stop conditions: emitted EOS or hit max_tokens.
                if hit_eos or len(generated) >= max_tokens:
                    break

                # --- fix up caches for next iteration ---
                emit_count = len(emitted)
                new_seq = cache_seq + emit_count
                target_cache.truncate_to(target_batch, new_seq)

                draft_seq_now = draft_cache.seq_lens_list()[draft_batch]
                if new_seq <= draft_seq_now:
                    draft_cache.truncate_to(draft_batch, new_seq)
                else:
                    # Only path: all-accepted AND fully emitted (no EOS, no
                    # max_tokens cap). draft is at cache_seq + K and we need
                    # cache_seq + K + 1; run one catch-up step feeding
                    # d_{K-1} at position cache_seq + K.
                    assert new_seq == draft_seq_now + 1, (
                        f"unexpected catch-up gap: new_seq={new_seq} draft={draft_seq_now}"
                    )
                    self.draft.forward_step(
                        draft_cache, [draft_tokens[-1]], [0, 1], [draft_seq_now]
                    )
                    stats.n_draft_forwards += 1

                last_token = bonus
                cache_seq = new_seq

            return generated[:max_tokens], stats
        finally:
            # Free the per-run cache slots so the block pool stays clean for
            # subsequent runs (important when callers reuse the runner).
            target_cache.free()
            draft_cache.free()
