"""Tensor-parallel serving for StateCache models (DeepSeek-V4) across ranks.

A model too large for one GPU (V4-Flash) is sharded across `world_size` ranks
by the TP-aware layers. A forward only completes when ALL ranks run it together
(the column / row-parallel layers all-reduce), so serving it behind a single
HTTP endpoint needs a front-door / follower split:

  - **Rank 0 (leader):** accepts a prompt, broadcasts it to the followers, then
    runs generation. Each step it samples the next token (honoring the
    sampling params) and broadcasts a `(continue, token)` decision so every
    rank feeds the same token into the next forward. It alone decides when to
    stop (EOS, max tokens, or cancellation) and broadcasts that. This is
    correct for greedy AND temperature / top-k / top-p, since the followers
    never sample, they receive.
  - **Ranks 1..N-1 (followers):** `run_follower_loop()` blocks on each broadcast
    and mirrors the leader's forwards in lockstep, discarding output, until the
    leader broadcasts a shutdown sentinel.

`max_tokens`-based stopping is a deterministic function of the (identical on
every rank) emitted-token count, so it needs no broadcast; only the
leader-only signals (EOS hit, cancellation) ride the per-step decision.

V4's KV is MQA (a single shared head), so the per-request `StateCache` is
replicated, not sharded; each rank holds an identical cache and the per-rank
decode stays consistent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.distributed as dist

from mini_infer.cache.state_cache import StateCache
from mini_infer.distributed.group import get_rank, get_world_size
from mini_infer.engine.sampler import SamplingParams, sample
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)

logger = logging.getLogger(__name__)

_LEADER_RANK = 0


class TensorParallelStateCacheServer:
    """Per-rank tensor-parallel generation for a StateCache model.

    Construct one on every rank with that rank's sharded model. The leader
    (rank 0) drives generation via `generate_ids`; followers call
    `run_follower_loop`. Build with `init_distributed` already active.
    """

    def __init__(
        self,
        model: DeepseekV4ForCausalLM,
        tokenizer: Tokenizer | None = None,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        reference_param = next(model.parameters())
        self.device = device if device is not None else str(reference_param.device)
        self.dtype = dtype if dtype is not None else reference_param.dtype
        self.rank = get_rank()
        self.world_size = get_world_size()

    @property
    def is_leader(self) -> bool:
        return self.rank == _LEADER_RANK

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError("TensorParallelStateCacheServer has no tokenizer")
        return self._tokenizer

    def _broadcast(self, obj: object) -> object:
        """Broadcast a small Python object from the leader to all ranks."""
        holder = [obj]
        dist.broadcast_object_list(holder, src=_LEADER_RANK)
        return holder[0]

    def generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        sampling_params: SamplingParams | None = None,
        emit: Callable[[int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[int]:
        """Leader-only: broadcast the request, then generate in lockstep with followers.

        `emit(token)` (optional) is called as each token is produced, for
        streaming. `should_cancel()` (optional) is polled each step; when it
        returns True the leader stops and tells the followers to stop too (so
        they never block waiting for a token that will not come).
        """
        if not self.is_leader:
            raise RuntimeError("generate_ids is leader-only; followers call run_follower_loop()")
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        self._broadcast(("generate", prompt_ids, max_new_tokens, eos_token_id))
        return self._run_generation(
            prompt_ids,
            max_new_tokens,
            eos_token_id,
            sampling_params if sampling_params is not None else SamplingParams(temperature=0.0),
            emit=emit,
            should_cancel=should_cancel,
        )

    def run_follower_loop(self) -> None:
        """Follower-only: mirror each leader generation until shutdown is broadcast."""
        if self.is_leader:
            raise RuntimeError("run_follower_loop() is follower-only; the leader drives generation")
        while True:
            message = self._broadcast(None)
            assert isinstance(message, tuple)
            if message[0] == "shutdown":
                return
            _, prompt_ids, max_new_tokens, eos_token_id = message
            # Followers never sample; params are unused on this path.
            self._run_generation(
                prompt_ids, max_new_tokens, eos_token_id, SamplingParams(temperature=0.0)
            )

    def shutdown(self) -> None:
        """Leader-only: tell the followers to leave `run_follower_loop`."""
        if self.is_leader:
            self._broadcast(("shutdown",))

    def _run_generation(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int | None,
        sampling_params: SamplingParams,
        *,
        emit: Callable[[int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[int]:
        """Shared leader/follower generation. Identical forwards on every rank;
        the leader samples + broadcasts each `(continue, token)` decision."""
        cfg: DeepseekV4Config = self._model.cfg
        max_seq_len = len(prompt_ids) + max_new_tokens
        state_cache = StateCache(
            build_state_cache_layer_specs(cfg, max_seq_len=max_seq_len),
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
        )
        input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model.forward_prefill_with_cache(input_ids, state_cache=state_cache)
        state_cache.advance_start_pos(len(prompt_ids))

        generated: list[int] = []
        emitted = 0
        while True:
            outgoing: object
            if self.is_leader:
                sampled = sample(logits[0, -1, :], sampling_params)
                stop = (eos_token_id is not None and sampled == eos_token_id) or (
                    should_cancel is not None and should_cancel()
                )
                outgoing = (not stop, sampled)
            else:
                outgoing = None
            received = self._broadcast(outgoing)
            assert isinstance(received, tuple)
            cont, token = bool(received[0]), int(received[1])
            if not cont:
                break
            generated.append(token)
            emitted += 1
            if self.is_leader and emit is not None:
                emit(token)
            # `emitted` is identical on every rank, so this stop is deterministic
            # and needs no broadcast; skip the decode whose logits we'd discard.
            if emitted >= max_new_tokens:
                break
            token_tensor = torch.tensor([[token]], device=self.device, dtype=torch.long)
            with torch.inference_mode():
                logits = self._model.forward_decode_with_cache(
                    token_tensor, start_pos=state_cache.start_pos, state_cache=state_cache
                )
            state_cache.advance_start_pos(1)
        return generated
