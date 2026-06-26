"""Single-request greedy generation for StateCache-based models (DeepSeek-V4).

V4 does not use `PagedKVCache`. It keeps per-request attention state in a
`StateCache` and exposes two cache-aware entry points on the model:

  - `forward_prefill_with_cache(input_ids, state_cache=...)`: process the
    whole prompt and populate the cache (SWA window + compressed history +
    in-flight compressor state per layer).
  - `forward_decode_with_cache(input_id, start_pos=..., state_cache=...)`:
    one decode step that reads and extends that cache.

This generator wires those into a `generate(prompt) -> text` loop: tokenize,
prefill, greedy-decode, detokenize. It is deliberately separate from
`ModelRunner`, which is built around `PagedKVCache` and packed-varlen
forwards; V4's cache and forward signatures don't fit that contract, so a
dedicated driver reads more cleanly than a second branch through every
`ModelRunner` method.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from mini_infer.cache.state_cache import StateCache
from mini_infer.cache.state_prefix_cache import StatePrefixCache
from mini_infer.engine.sampler import SamplingParams, sample
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)

logger = logging.getLogger(__name__)


def prefill_with_prefix_cache(
    model: DeepseekV4ForCausalLM,
    prompt_ids: list[int],
    *,
    state_cache: StateCache,
    prefix_cache: StatePrefixCache,
    device: str,
) -> torch.Tensor:
    """Populate a fresh B=1 `state_cache` to the post-prompt state, reusing a
    cached prefix when one exists, and return the logits that predict the first
    token after the prompt. The full prompt's end state is snapshotted into
    `prefix_cache` for future reuse.

    Three paths, all producing identical state for the same prompt (sharing
    changes only the work done, not the math):
      - exact-length hit: restore the snapshot; its stored logits are the answer.
      - prefix hit: restore the shared prefix, replay only the suffix token by
        token.
      - miss: a normal full prefill.

    Shared by the single-request generator (`generate_ids_prefix_cached`) and the
    continuous-batching scheduler's per-slot prefill, which both prefill into a
    B=1 cache before reading row 0.
    """
    prompt_len = len(prompt_ids)
    matched_len, snapshot = prefix_cache.match(prompt_ids)
    if snapshot is not None and matched_len == prompt_len:
        StatePrefixCache.restore_into(snapshot, state_cache)
        next_logits = snapshot.next_logits
    elif snapshot is not None and matched_len > 0:
        # Restore the shared prefix, then replay only the suffix. The range is
        # non-empty here, so step_logits is set at least once.
        StatePrefixCache.restore_into(snapshot, state_cache)
        step_logits = None
        with torch.inference_mode():
            for position in range(matched_len, prompt_len):
                step_logits = model.forward_decode_with_cache(
                    torch.tensor([[prompt_ids[position]]], device=device, dtype=torch.long),
                    start_pos=state_cache.start_pos,
                    state_cache=state_cache,
                )
                state_cache.advance_start_pos(1)
        assert step_logits is not None
        next_logits = step_logits[0, -1, :]
    else:
        # Miss: full prefill, exactly like a fresh generate.
        with torch.inference_mode():
            prefill_logits = model.forward_prefill_with_cache(
                torch.tensor([prompt_ids], device=device, dtype=torch.long),
                state_cache=state_cache,
            )
        state_cache.advance_start_pos(prompt_len)
        next_logits = prefill_logits[0, -1, :]

    prefix_cache.insert(prompt_ids, StatePrefixCache.snapshot_from_cache(state_cache, next_logits))
    return next_logits


class StateCacheGenerator:
    """Single-request generation over a per-request `StateCache`.

    Construct with an already-loaded model (and optionally a tokenizer). Use
    `generate_ids` for one-shot token-level generation (no tokenizer needed; the
    path CPU tests exercise on synthetic configs, and the per-request scalar
    oracle the batched schedulers validate against), `iter_generate_ids` to
    stream tokens one at a time (emit and react to cancellation between tokens),
    and `generate` for the string-in / string-out convenience wrapper. Batched
    decode for serving lives in the schedulers (`StateCacheContinuousScheduler`),
    which drive the model's ragged forward directly.
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
        # Default device/dtype to the model's own so the StateCache and input
        # tensors land where the weights are. A cpu/float32 default would give
        # a device mismatch (or a silent upcast) against a GPU-resident or
        # bf16 model; an explicit argument still overrides.
        reference_param = next(model.parameters())
        self.device = device if device is not None else str(reference_param.device)
        self.dtype = dtype if dtype is not None else reference_param.dtype

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        device: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> StateCacheGenerator:
        """Load a DeepSeek-V4 checkpoint onto one device and wrap it for generation.

        This is the single-device entry (local CPU/MPS, or one GPU). For
        multi-GPU tensor parallelism the model does not fit one device:
        initialise the process group per rank, call
        `DeepseekV4ForCausalLM.from_checkpoint(..., device=f"cuda:{rank}")`
        directly, and construct `StateCacheGenerator(model, tokenizer)` around
        the per-rank model.
        """
        from mini_infer.engine.model_runner import _dtype_for, _resolve_device

        resolved_device = _resolve_device(device)
        resolved_dtype = dtype if dtype is not None else _dtype_for(resolved_device)
        model = DeepseekV4ForCausalLM.from_checkpoint(
            name_or_path, device=resolved_device, dtype=resolved_dtype
        )
        tokenizer = Tokenizer.from_pretrained(name_or_path)
        return cls(model, tokenizer, device=resolved_device, dtype=resolved_dtype)

    @property
    def model(self) -> DeepseekV4ForCausalLM:
        """The wrapped model (for schedulers that drive prefill / decode directly)."""
        return self._model

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError(
                "StateCacheGenerator has no tokenizer; use generate_ids for "
                "token-level generation, or construct with a Tokenizer"
            )
        return self._tokenizer

    def iter_generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time (the prompt is not echoed).

        Builds the StateCache, prefills, then yields each decoded token. Stops
        before yielding `eos_token_id` (if given), or after `max_new_tokens`.
        `sampling_params` defaults to greedy (temperature 0). Yielding per token
        lets a scheduler stream output and react to cancellation between tokens.

        Each forward + its sample run inside `torch.inference_mode()` that exits
        before the `yield`, so inference mode never spans a suspension point
        (which would leak it into the consumer between tokens).
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        params = sampling_params if sampling_params is not None else SamplingParams(temperature=0.0)
        cfg: DeepseekV4Config = self._model.cfg
        # Size each layer's compressed history for the full prompt + output, so
        # high-ratio layers don't inherit the densest layer's slot count.
        max_seq_len = len(prompt_ids) + max_new_tokens
        state_cache = StateCache(
            build_state_cache_layer_specs(cfg, max_seq_len=max_seq_len),
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
        )
        input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

        with torch.inference_mode():
            prefill_logits = self._model.forward_prefill_with_cache(
                input_ids, state_cache=state_cache
            )
            next_token = sample(prefill_logits[0, -1, :], params)
        state_cache.advance_start_pos(len(prompt_ids))

        emitted = 0
        while emitted < max_new_tokens:
            if eos_token_id is not None and next_token == eos_token_id:
                return
            yield next_token
            emitted += 1
            if emitted == max_new_tokens:
                return
            token_tensor = torch.tensor([[next_token]], device=self.device, dtype=torch.long)
            with torch.inference_mode():
                step_logits = self._model.forward_decode_with_cache(
                    token_tensor, start_pos=state_cache.start_pos, state_cache=state_cache
                )
                next_token = sample(step_logits[0, -1, :], params)
            state_cache.advance_start_pos(1)

    def generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Greedy-decode up to `max_new_tokens` tokens after `prompt_ids`.

        Returns the generated token ids only (the prompt is not echoed). Stops
        early on `eos_token_id` (the EOS token itself is not included). Thin
        greedy wrapper over `iter_generate_ids`.
        """
        return list(
            self.iter_generate_ids(
                prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
            )
        )

    def generate_ids_prefix_cached(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        prefix_cache: StatePrefixCache,
    ) -> list[int]:
        """Greedy-decode, reusing a cached prompt prefix when one exists.

        If a prompt stored in `prefix_cache` is a prefix of `prompt_ids` (length
        K), restore that snapshot and replay only the suffix `prompt_ids[K:]` one
        token at a time, instead of re-prefilling the shared prefix; on an
        exact-length hit the stored logits give the first token directly. On a
        miss, a normal full prefill. Either way the full prompt's end state is
        cached for future reuse.

        Output is token-for-token identical to `generate_ids`: prefix sharing
        changes only the work done (skip the shared prefill), not the math. This
        is the readable reference for cross-request prefix sharing; the scheduler
        wires it into serving.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        params = SamplingParams(temperature=0.0)
        cfg: DeepseekV4Config = self._model.cfg
        state_cache = StateCache(
            build_state_cache_layer_specs(cfg, max_seq_len=len(prompt_ids) + max_new_tokens),
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
        )
        next_logits = prefill_with_prefix_cache(
            self._model,
            prompt_ids,
            state_cache=state_cache,
            prefix_cache=prefix_cache,
            device=self.device,
        )

        out: list[int] = []
        next_token = sample(next_logits, params)
        while len(out) < max_new_tokens:
            if eos_token_id is not None and next_token == eos_token_id:
                break
            out.append(next_token)
            if len(out) == max_new_tokens:
                break
            with torch.inference_mode():
                step_logits = self._model.forward_decode_with_cache(
                    torch.tensor([[next_token]], device=self.device, dtype=torch.long),
                    start_pos=state_cache.start_pos,
                    state_cache=state_cache,
                )
                next_token = sample(step_logits[0, -1, :], params)
            state_cache.advance_start_pos(1)
        return out

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        """Tokenize `prompt`, greedy-generate `max_new_tokens`, return the text."""
        prompt_ids = self.tokenizer.encode(prompt)
        out_ids = self.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out_ids)
