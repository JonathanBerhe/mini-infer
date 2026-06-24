"""Tensor-parallel ragged continuous batching for StateCache models (V4-Flash).

A model too large for one GPU (V4-Flash) is sharded across ranks, so every
forward must run on all ranks together. This wraps the ragged continuous-batching
decode (`forward_decode_with_cache_ragged`) in the same leader / follower split
the single-request TP serving uses, but the per-step broadcast now carries the
whole batch's state:

  - **Rank 0 (leader):** owns the batch (which slot holds which request, each
    request's position + next token). Before every model forward it broadcasts
    the operation and its inputs, then runs the forward itself and samples.
  - **Followers:** `run_follower_loop()` blocks on each broadcast and mirrors
    the forward on its own shard + replicated `StateCache`, so the all-reduce in
    the sharded layers completes and every rank's cache stays identical.

Two broadcast ops:
  - `("prefill", slot_idx, prompt_ids)`: all ranks prefill the prompt in a temp
    one-row cache and copy that row into batched-cache row `slot_idx`.
  - `("decode", input_tokens, positions)`: all ranks run one ragged decode step.

The leader samples (followers never do); with `gather_output=True` in the TP
layers every rank computes identical logits, so a broadcast of the sampled token
is unnecessary, it rides the next `("decode", ...)` op.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from mini_infer.cache.state_cache import StateCache
from mini_infer.distributed.group import get_rank, get_world_size
from mini_infer.engine.sampler import SamplingParams, sample
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM, build_state_cache_layer_specs

logger = logging.getLogger(__name__)

_LEADER_RANK = 0


class TensorParallelStateCacheContinuousServer:
    """Per-rank tensor-parallel ragged continuous batching for a StateCache model.

    Construct one on every rank with that rank's sharded model. The leader drives
    `generate_cohort`; followers call `run_follower_loop`. Requires
    `init_distributed` already active.
    """

    def __init__(
        self,
        model: DeepseekV4ForCausalLM,
        tokenizer: Tokenizer | None = None,
        *,
        max_batch_size: int,
        max_seq_len: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self._model = model
        self._tokenizer = tokenizer
        reference_param = next(model.parameters())
        self.device = device if device is not None else str(reference_param.device)
        self.dtype = dtype if dtype is not None else reference_param.dtype
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._cache = StateCache(
            build_state_cache_layer_specs(model.cfg, max_seq_len=max_seq_len),
            batch_size=max_batch_size,
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def is_leader(self) -> bool:
        return self.rank == _LEADER_RANK

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError("server has no tokenizer")
        return self._tokenizer

    def _broadcast(self, obj: object) -> object:
        holder = [obj]
        dist.broadcast_object_list(holder, src=_LEADER_RANK)
        return holder[0]

    # ---- shared per-rank forwards (run on leader AND followers) ----

    def _run_prefill(self, prompt_ids: list[int], slot_idx: int) -> int | None:
        """Prefill `prompt_ids` in a temp one-row cache, copy that row into batched
        row `slot_idx`. Returns the sampled first token on the leader, else None."""
        temp = StateCache(
            build_state_cache_layer_specs(self._model.cfg, max_seq_len=self.max_seq_len),
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
        )
        input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model.forward_prefill_with_cache(input_ids, state_cache=temp)
        self._copy_row(temp, dst=slot_idx)
        if self.is_leader:
            return sample(logits[0, -1, :], SamplingParams(temperature=0.0))
        return None

    def _copy_row(self, src_cache: StateCache, *, dst: int) -> None:
        for layer_idx in range(self._cache.num_layers):
            src_layer = src_cache.layer(layer_idx)
            dst_layer = self._cache.layer(layer_idx)
            dst_layer.swa_kv[dst] = src_layer.swa_kv[0]
            dst_layer.compressed_kv[dst] = src_layer.compressed_kv[0]
            dst_layer.cmp_kv_state[dst] = src_layer.cmp_kv_state[0]
            dst_layer.cmp_score_state[dst] = src_layer.cmp_score_state[0]
            if dst_layer.indexer is not None and src_layer.indexer is not None:
                dst_layer.indexer.compressed_kv[dst] = src_layer.indexer.compressed_kv[0]
                dst_layer.indexer.cmp_kv_state[dst] = src_layer.indexer.cmp_kv_state[0]
                dst_layer.indexer.cmp_score_state[dst] = src_layer.indexer.cmp_score_state[0]

    def _run_decode(self, input_tokens: list[int], positions: list[int]) -> torch.Tensor:
        """One ragged decode step over the full batched cache. Returns logits."""
        input_ids = torch.tensor(input_tokens, device=self.device, dtype=torch.long).unsqueeze(1)
        position_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        with torch.inference_mode():
            return self._model.forward_decode_with_cache_ragged(
                input_ids, positions=position_tensor, state_cache=self._cache
            )

    # ---- leader-driven cohort generation ----

    def generate_cohort(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[list[int]]:
        """Leader-only: greedily generate for a cohort of prompts via ragged
        continuous batching, broadcasting each forward so followers mirror it.

        `prompts` must fit the batch (`len(prompts) <= max_batch_size`). Returns
        one generated-token list per prompt (EOS not echoed). Equivalent to
        running each prompt alone, but every forward serves the whole batch.
        """
        if not self.is_leader:
            raise RuntimeError("generate_cohort is leader-only; followers call run_follower_loop()")
        if not prompts:
            raise ValueError("prompts must be non-empty")
        if len(prompts) > self.max_batch_size:
            raise ValueError(
                f"cohort of {len(prompts)} exceeds max_batch_size {self.max_batch_size}"
            )
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        cohort_size = len(prompts)
        positions = [0] * cohort_size
        next_tokens = [0] * cohort_size
        for slot_idx, prompt_ids in enumerate(prompts):
            self._broadcast(("prefill", slot_idx, prompt_ids))
            first_token = self._run_prefill(prompt_ids, slot_idx)
            assert first_token is not None
            next_tokens[slot_idx] = first_token
            positions[slot_idx] = len(prompt_ids)

        outputs: list[list[int]] = [[] for _ in range(cohort_size)]
        done = [False] * cohort_size
        params = SamplingParams(temperature=0.0)
        while True:
            decode_slots: list[int] = []
            for slot_idx in range(cohort_size):
                if done[slot_idx]:
                    continue
                if eos_token_id is not None and next_tokens[slot_idx] == eos_token_id:
                    done[slot_idx] = True
                    continue
                outputs[slot_idx].append(next_tokens[slot_idx])
                if len(outputs[slot_idx]) >= max_new_tokens:
                    done[slot_idx] = True
                    continue
                decode_slots.append(slot_idx)
            if not decode_slots:
                break
            input_tokens = [0] * self.max_batch_size
            decode_positions = [0] * self.max_batch_size
            for slot_idx in decode_slots:
                input_tokens[slot_idx] = next_tokens[slot_idx]
                decode_positions[slot_idx] = positions[slot_idx]
            self._broadcast(("decode", input_tokens, decode_positions))
            logits = self._run_decode(input_tokens, decode_positions)
            for slot_idx in decode_slots:
                next_tokens[slot_idx] = sample(logits[slot_idx, -1, :], params)
                positions[slot_idx] += 1
        return outputs

    def run_follower_loop(self) -> None:
        """Follower-only: mirror each broadcast forward until shutdown."""
        if self.is_leader:
            raise RuntimeError("run_follower_loop() is follower-only; the leader drives generation")
        while True:
            message = self._broadcast(None)
            assert isinstance(message, tuple)
            op = message[0]
            if op == "shutdown":
                return
            if op == "prefill":
                _, slot_idx, prompt_ids = message
                self._run_prefill(prompt_ids, slot_idx)
            elif op == "decode":
                _, input_tokens, positions = message
                self._run_decode(input_tokens, positions)
            else:
                raise RuntimeError(f"unknown op broadcast to follower: {op!r}")

    def shutdown(self) -> None:
        """Leader-only: tell the followers to leave `run_follower_loop`."""
        if self.is_leader:
            self._broadcast(("shutdown",))
