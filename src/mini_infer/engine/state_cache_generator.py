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
from mini_infer.engine.sampler import SamplingParams, sample
from mini_infer.engine.tokenizer import Tokenizer
from mini_infer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    build_state_cache_layer_specs,
)

logger = logging.getLogger(__name__)


class StateCacheGenerator:
    """Single-request generation over a per-request `StateCache`.

    Construct with an already-loaded model (and optionally a tokenizer). Use
    `generate_ids` for one-shot token-level generation (no tokenizer needed,
    the path exercised by CPU tests on synthetic configs), `iter_generate_ids`
    to stream tokens one at a time (what `StateCacheScheduler` drives so it can
    emit and react to cancellation between tokens), `generate_ids_batched` to
    run a cohort of equal-length prompts through one batched forward in
    lockstep (the same math as N separate `generate_ids` calls, scheduled
    together), and `generate` for the string-in / string-out convenience
    wrapper.
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

    def generate_ids_batched(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> list[list[int]]:
        """Lockstep batched generation for a cohort of equal-length prompts.

        Every prompt must have the same length: the cohort prefills as one
        `(B, T)` batch and decodes in lockstep, each step advancing all
        sequences by one token at a single shared position. A sequence that
        hits `eos_token_id` stops recording, but the batch keeps stepping
        until all sequences finish or `max_new_tokens` is reached (the
        static-batching contract: a finished sequence's slot is not freed
        mid-cohort).

        This is the lockstep / cohort form of batching: one forward serves the
        whole batch and the position counter is shared. It deliberately
        requires equal lengths. Ragged per-request positions (a finished
        sequence freeing its slot, a new request joining mid-flight) are the
        separate continuous-batching path, not this method.

        Returns one generated-token list per prompt, in input order (prompts
        not echoed, EOS not included). For greedy decoding the result is
        token-for-token identical to calling `generate_ids` on each prompt
        alone, since each sequence attends only to its own state; batching
        changes how the work is scheduled, not the math.
        """
        if not prompts:
            raise ValueError("prompts must be non-empty")
        if any(not prompt_ids for prompt_ids in prompts):
            raise ValueError("every prompt must be non-empty")
        prompt_len = len(prompts[0])
        if any(len(prompt_ids) != prompt_len for prompt_ids in prompts):
            raise ValueError(
                "lockstep batched generation requires equal-length prompts; "
                "ragged lengths need the continuous-batching path"
            )
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        params = sampling_params if sampling_params is not None else SamplingParams(temperature=0.0)
        cfg: DeepseekV4Config = self._model.cfg
        batch_size = len(prompts)
        max_seq_len = prompt_len + max_new_tokens
        state_cache = StateCache(
            build_state_cache_layer_specs(cfg, max_seq_len=max_seq_len),
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        input_ids = torch.tensor(prompts, device=self.device, dtype=torch.long)

        with torch.inference_mode():
            prefill_logits = self._model.forward_prefill_with_cache(
                input_ids, state_cache=state_cache
            )
            next_tokens = [sample(prefill_logits[b, -1, :], params) for b in range(batch_size)]
        state_cache.advance_start_pos(prompt_len)

        outputs: list[list[int]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        emitted = 0
        while emitted < max_new_tokens:
            for b in range(batch_size):
                if finished[b]:
                    continue
                if eos_token_id is not None and next_tokens[b] == eos_token_id:
                    finished[b] = True
                else:
                    outputs[b].append(next_tokens[b])
            if all(finished):
                break
            emitted += 1
            if emitted == max_new_tokens:
                break
            token_tensor = torch.tensor(
                [[token] for token in next_tokens], device=self.device, dtype=torch.long
            )
            with torch.inference_mode():
                step_logits = self._model.forward_decode_with_cache(
                    token_tensor, start_pos=state_cache.start_pos, state_cache=state_cache
                )
                next_tokens = [sample(step_logits[b, -1, :], params) for b in range(batch_size)]
            state_cache.advance_start_pos(1)

        return outputs

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        """Tokenize `prompt`, greedy-generate `max_new_tokens`, return the text."""
        prompt_ids = self.tokenizer.encode(prompt)
        out_ids = self.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out_ids)
