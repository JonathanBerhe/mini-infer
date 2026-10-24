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

Each op is one int64 message of a fixed shape,
`[op, slot_idx, prompt length, input tokens (B), positions (B)]` for a batch of
B rows. The leader broadcasts it on the CPU (gloo) group of `get_cpu_group()`,
so the message stays in host memory and no GPU stream waits for it. Three ops:
  - prefill: a second broadcast carries the prompt ids. All ranks prefill the
    prompt in a temp one-row cache and copy that row into batched-cache row
    `slot_idx`.
  - decode: all ranks run one ragged decode step. On a CUDA device the message
    is in pinned memory, so the copy of the inputs to the GPU does not wait.
  - shutdown: the followers leave `run_follower_loop`.

The leader samples (followers never do); with `gather_output=True` in the TP
layers every rank computes identical logits, so a broadcast of the sampled token
is unnecessary, it rides the next decode op.

With `cuda_graph=True`, every rank captures the decode step in a
`DecodeCudaGraph` when the server is built, and each decode op replays it. The
constructor runs on every rank in the same order, so the all-reduces inside the
capture line up. Shutdown drops the graph on every rank, because a live graph
that holds NCCL work can hang the process group teardown.

With `time_decode=True`, a `DecodeStepTimer` times each decode forward and the
host gap between forwards; `decode_timings()` returns them.
"""

from __future__ import annotations

import contextlib
import gc
import logging

import torch
import torch.distributed as dist

from mini_infer.cache.state_cache import StateCache
from mini_infer.device import is_cuda_device
from mini_infer.distributed.group import get_cpu_group, get_rank, get_world_size
from mini_infer.engine.cuda_graph_decode import DecodeCudaGraph, supports_decode_cuda_graph
from mini_infer.engine.decode_step_timer import DecodeStepTimer, DecodeStepTiming
from mini_infer.engine.sampler import SamplingParams, sample, sample_batch
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models.deepseek_v4 import DeepseekV4ForCausalLM, build_state_cache_layer_specs

logger = logging.getLogger(__name__)

_LEADER_RANK = 0

# The ops of a control message, and the length of its header.
_OP_DECODE = 0
_OP_PREFILL = 1
_OP_SHUTDOWN = 2
_HEADER_LEN = 3  # op, slot_idx, prompt length


class TensorParallelStateCacheContinuousServer:
    """Per-rank tensor-parallel ragged continuous batching for a StateCache model.

    Construct one on every rank with that rank's sharded model. The leader drives
    `generate_cohort`; followers call `run_follower_loop`. Requires
    `init_distributed` already active. Pass `cuda_graph=True` on every rank to
    replay the decode step from a CUDA graph, and `time_decode=True` to time
    the decode steps (see the module docstring).
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
        cuda_graph: bool = False,
        time_decode: bool = False,
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
        if cuda_graph and not supports_decode_cuda_graph(model, self.device):
            raise ValueError(
                f"cuda_graph=True needs a CUDA device and a model with a capturable decode "
                f"step; got {type(model).__name__} on {self.device}"
            )
        self._cache = StateCache(
            build_state_cache_layer_specs(model.cfg, max_seq_len=max_seq_len),
            batch_size=max_batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        # Every rank builds the server at the same point, so the first build
        # can create the group.
        self._control_group = get_cpu_group() if self.world_size > 1 else None
        self._pin_messages = is_cuda_device(self.device)
        self._decode_timer = DecodeStepTimer(self.device) if time_decode else None
        # Capture before any prefill: warmup and capture write into every row.
        self._decode_graph = (
            DecodeCudaGraph(
                model,
                self._cache,
                batch_size=max_batch_size,
                device=self.device,
                timer=self._decode_timer,
            )
            if cuda_graph
            else None
        )

    @property
    def is_leader(self) -> bool:
        return self.rank == _LEADER_RANK

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError("server has no tokenizer")
        return self._tokenizer

    # ---- control messages (see the module docstring) ----

    def _message(
        self,
        op: int,
        *,
        slot_idx: int = 0,
        prompt_len: int = 0,
        input_tokens: list[int] | None = None,
        positions: list[int] | None = None,
    ) -> torch.Tensor:
        """Leader: a control message, in pinned memory when the device is CUDA."""
        padding = [0] * self.max_batch_size
        values = [op, slot_idx, prompt_len, *(input_tokens or padding), *(positions or padding)]
        return torch.tensor(values, dtype=torch.long, pin_memory=self._pin_messages)

    def _empty_message(self) -> torch.Tensor:
        """Follower: a buffer that receives one control message."""
        return torch.empty(
            _HEADER_LEN + 2 * self.max_batch_size, dtype=torch.long, pin_memory=self._pin_messages
        )

    def _broadcast(self, tensor: torch.Tensor) -> None:
        """Send `tensor` from the leader, or receive into it on a follower."""
        if self._control_group is not None:  # None: one rank, no followers
            dist.broadcast(tensor, src=_LEADER_RANK, group=self._control_group)

    # ---- shared per-rank forwards (run on leader AND followers) ----

    def _run_prefill(self, prompt_ids: list[int], slot_idx: int) -> torch.Tensor | None:
        """Prefill `prompt_ids` in a temp one-row cache, copy that row into batched
        row `slot_idx`. Returns the last-position logits on the leader (the caller
        samples the first token, honoring per-request params), else None."""
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
        if self._decode_timer is not None:
            self._decode_timer.skip_gap()
        if self.is_leader:
            return logits[0, -1, :]
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

    def _run_decode(self, message: torch.Tensor) -> torch.Tensor:
        """One ragged decode step over the full batched cache, with the inputs of
        a decode `message`. Returns logits.

        From the graph, the logits are its output buffer, which the next
        replay overwrites.
        """
        batch = self.max_batch_size
        input_ids = message[_HEADER_LEN : _HEADER_LEN + batch].view(batch, 1)
        positions = message[_HEADER_LEN + batch :]
        if self._decode_graph is not None:
            # The message is on the host, so this check costs no device sync.
            self._cache.check_capacity(int(positions.max()) + 1)
            return self._decode_graph.replay(input_ids, positions, non_blocking=True)
        input_ids = input_ids.to(self.device, non_blocking=True)
        positions = positions.to(self.device, non_blocking=True)
        with torch.inference_mode(), self._timed_step():
            return self._model.forward_decode_with_cache_ragged(
                input_ids, positions=positions, state_cache=self._cache
            )

    def _timed_step(self) -> contextlib.AbstractContextManager[None]:
        if self._decode_timer is None:
            return contextlib.nullcontext()
        return self._decode_timer.step()

    def decode_timings(self) -> list[DecodeStepTiming]:
        """The timing of each decode step since the last call (`time_decode=True`).

        Waits for the last timed step to finish on the device.
        """
        if self._decode_timer is None:
            raise RuntimeError("decode_timings needs a server built with time_decode=True")
        return self._decode_timer.drain()

    # ---- leader-driven operations (broadcast, then run; followers mirror) ----

    def prefill_into_slot(self, prompt_ids: list[int], slot_idx: int) -> torch.Tensor:
        """Leader: prefill `prompt_ids` into batched-cache row `slot_idx`, return the
        last-position logits (the caller samples the first token). Broadcasts so
        followers mirror the prefill + row copy.

        A scheduler drives this (and `decode_batch`) to run dynamic continuous
        batching across ranks without owning the cache itself.
        """
        self._broadcast(self._message(_OP_PREFILL, slot_idx=slot_idx, prompt_len=len(prompt_ids)))
        self._broadcast(torch.tensor(prompt_ids, dtype=torch.long))
        logits = self._run_prefill(prompt_ids, slot_idx)
        assert logits is not None  # only None on followers, which never call this
        return logits

    def decode_batch(self, input_tokens: list[int], positions: list[int]) -> torch.Tensor:
        """Leader: run one ragged decode step over the whole batch, return logits
        `(max_batch_size, 1, vocab)`. Broadcasts so followers mirror it.

        `input_tokens` and `positions` hold one entry per row of the batch.
        """
        if len(input_tokens) != self.max_batch_size or len(positions) != self.max_batch_size:
            raise ValueError(
                f"decode_batch needs {self.max_batch_size} input tokens and positions, got "
                f"{len(input_tokens)} and {len(positions)}"
            )
        message = self._message(_OP_DECODE, input_tokens=input_tokens, positions=positions)
        self._broadcast(message)
        return self._run_decode(message)

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
        params = SamplingParams(temperature=0.0)
        positions = [0] * cohort_size
        next_tokens = [0] * cohort_size
        for slot_idx, prompt_ids in enumerate(prompts):
            prefill_logits = self.prefill_into_slot(prompt_ids, slot_idx)
            next_tokens[slot_idx] = sample(prefill_logits, params)
            positions[slot_idx] = len(prompt_ids)

        outputs: list[list[int]] = [[] for _ in range(cohort_size)]
        done = [False] * cohort_size
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
            logits = self.decode_batch(input_tokens, decode_positions)
            # Every row is greedy, so one argmax and one host sync serve the batch.
            row_tokens = sample_batch(logits[:, -1, :], [params] * self.max_batch_size)
            for slot_idx in decode_slots:
                next_tokens[slot_idx] = row_tokens[slot_idx]
                positions[slot_idx] += 1
        return outputs

    def run_follower_loop(self) -> None:
        """Follower-only: mirror each broadcast forward until shutdown."""
        if self.is_leader:
            raise RuntimeError("run_follower_loop() is follower-only; the leader drives generation")
        while True:
            message = self._empty_message()
            self._broadcast(message)
            op, slot_idx, prompt_len = message[:_HEADER_LEN].tolist()
            if op == _OP_SHUTDOWN:
                self._release_decode_graph()
                return
            if op == _OP_PREFILL:
                prompt_ids = torch.empty(prompt_len, dtype=torch.long)
                self._broadcast(prompt_ids)
                self._run_prefill(prompt_ids.tolist(), slot_idx)
            elif op == _OP_DECODE:
                self._run_decode(message)
            else:
                raise RuntimeError(f"unknown op broadcast to follower: {op!r}")

    def shutdown(self) -> None:
        """Leader-only: tell the followers to leave `run_follower_loop`.

        Every rank drops its decode graph here, so the process group can be
        destroyed after it. Later decode steps run eagerly.
        """
        if self.is_leader:
            self._broadcast(self._message(_OP_SHUTDOWN))
            self._release_decode_graph()

    def _release_decode_graph(self) -> None:
        """Free the decode graph and wait for the GPU to finish its work."""
        if self._decode_graph is None:
            return
        self._decode_graph = None
        gc.collect()
        if is_cuda_device(self.device):
            torch.cuda.synchronize(self.device)
