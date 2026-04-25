import logging

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from mini_infer.engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype_for(device: str) -> torch.dtype:
    """fp16 on MPS (M1 has fp16 in hardware; bf16 is software-emulated and crashes on M1)."""
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


class ModelRunner:
    """Loads a HF causal LM; Slice 1 wraps HF .generate(), Slice 2 replaces with prefill+decode."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        device: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name: str, *, device: str = "auto") -> "ModelRunner":
        resolved = _resolve_device(device)
        dtype = _dtype_for(resolved)
        logger.info("Loading %s on %s with dtype=%s", model_name, resolved, dtype)
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(resolved)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=resolved)

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        """Greedy decode without KV cache; sampling lands in Slice 3, KV cache in Slice 2."""
        # Manual loop instead of HF model.generate() because the latter crashes on MPS in
        # transformers 5.x (the logits-processor pipeline allocates a tensor > 4 GB,
        # exceeding the MPSTemporaryNDArray limit). Plain forward passes are fine.
        prompt_ids = self._tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        new_tokens: list[int] = []

        with torch.inference_mode():
            for _ in range(max_tokens):
                logits = self._model(input_ids).logits[0, -1, :]
                next_token = int(torch.argmax(logits).item())
                if next_token == self._tokenizer.eos_token_id:
                    break
                new_tokens.append(next_token)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], device=self.device)],
                    dim=1,
                )

        return self._tokenizer.decode(new_tokens)
