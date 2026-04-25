"""Regenerate tests/golden/golden_data/qwen_0_5b_instruct.json (run manually)."""

# Only re-run when intentionally changing the reference; commit message must explain why.

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32
MAX_NEW_TOKENS = 3  # floor of the divergence point across all prompts; see memory note

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "def fibonacci(n):",
]

OUTPUT_PATH = Path(__file__).parent / "golden_data" / "qwen_0_5b_instruct.json"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    model.eval()

    samples = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([prompt_ids], device=DEVICE)
        with torch.inference_mode():
            output = model.generate(
                input_tensor,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][len(prompt_ids) :].tolist()
        samples.append(
            {
                "prompt": prompt,
                "max_new_tokens": MAX_NEW_TOKENS,
                "expected_tokens": new_tokens,
            }
        )

    payload = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": "float32",
        "samples": samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
