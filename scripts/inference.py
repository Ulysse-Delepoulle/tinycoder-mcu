#!/usr/bin/env python3
"""Interactive REPL to test the fine-tuned STM32N6 coder.

Loads the base Qwen2.5-Coder-1.5B in 4-bit and applies your trained LoRA
adapter on top. Type an instruction, get generated C. With --compare, the same
prompt is also run through the UN-adapted base model so you can see what the
fine-tuning actually changed.

Run `python scripts/inference.py --help` for options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"

MAX_NEW_TOKENS = 512


def build_quant_config():
    """4-bit NF4 config — same as training/eval so behavior matches."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 compute for Turing/T4
    )


def generate(model, tokenizer, instruction: str) -> str:
    """Generate one response using Qwen's chat template."""
    import torch

    prompt = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,        # sampling gives more natural, varied output
            temperature=0.2,       # low temp: mostly deterministic, a little variety
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_one(tuned_model, tokenizer, instruction: str, compare: bool) -> None:
    """Generate for a single instruction and print the result(s)."""
    if compare:
        # Temporarily disable the adapter to get the pure base output.
        with tuned_model.disable_adapter():
            base_out = generate(tuned_model, tokenizer, instruction)
        print("\n--- BASE (no adapter) ---")
        print(base_out)

    tuned_out = generate(tuned_model, tokenizer, instruction)
    print("\n--- FINE-TUNED ---" if compare else "")
    print(tuned_out)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive REPL for the fine-tuned STM32N6 coder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the trained LoRA checkpoint dir.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG,
                        help="Training config (only to read model_name).")
    parser.add_argument("--compare", action="store_true",
                        help="Also run the UN-adapted base model side by side.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Run ONE instruction non-interactively and exit. "
                             "Use this in notebook cells (e.g. Kaggle) where the "
                             "REPL can't read stdin. Combine with --compare.")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    with args.config.open("r", encoding="utf-8") as f:
        base_model_name = yaml.safe_load(f)["model_name"]

    print(f"Loading base model {base_model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quant_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=quant_config,
        device_map="auto", trust_remote_code=True,
    )

    print(f"Applying LoRA adapter from {args.checkpoint}...")
    # PeftModel wraps the frozen base and adds the trained adapter. The base
    # object is shared, so --compare can toggle the adapter on/off cheaply via
    # disable_adapter() instead of loading the 1.5B weights twice.
    tuned_model = PeftModel.from_pretrained(base_model, args.checkpoint)
    tuned_model.eval()

    # One-shot, non-interactive mode: generate for --prompt and exit. This is the
    # mode to use from a notebook cell, where input() has no stdin to read.
    if args.prompt is not None:
        run_one(tuned_model, tokenizer, args.prompt, args.compare)
        return

    print("\nReady. Type an instruction (blank line or Ctrl-C to quit).\n")
    try:
        while True:
            instruction = input("instruction> ").strip()
            if not instruction:
                break
            run_one(tuned_model, tokenizer, instruction, args.compare)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
