#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen2.5-Coder-1.5B on STM32N6 instruction pairs.

Designed to run on a single Kaggle T4 GPU (15GB, Turing / compute 7.5).

What "QLoRA" means here, end to end:
  1. Load the base model in 4-bit (NF4) quantization via bitsandbytes. This
     shrinks the frozen 1.5B weights ~4x so they fit alongside activations on a
     T4. On each forward pass bitsandbytes dequantizes weights on the fly to
     bf16/fp16 compute, runs the matmul, and discards the dequantized copy.
  2. Freeze ALL base weights. We never update them — that's what keeps VRAM and
     compute low and prevents catastrophic forgetting of the base model's coding
     ability.
  3. Attach small trainable LoRA adapters (from configs/lora_config.yaml) to the
     attention + MLP projections. Only these ~millions of params get gradients.

Run `python scripts/train.py --help` for options.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"
DEFAULT_LORA_CONFIG = PROJECT_ROOT / "configs" / "lora_config.yaml"

# Label value that tells the loss function "ignore this token". PyTorch's
# CrossEntropyLoss skips positions whose target == -100. We use it to mask the
# user-prompt tokens (see build_labels) so the model is only graded on the
# assistant's answer, not on re-predicting the question.
IGNORE_INDEX = -100


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_example(instruction: str, output: str) -> tuple[str, str]:
    """Return (prompt_part, full_text) using Qwen2.5's chat template.

    Instruction-tuned Qwen expects this exact ChatML-style structure. The special
    tokens <|im_start|> / <|im_end|> delimit turns; getting them right matters
    because the model learned these markers during its own instruction tuning —
    feed a different format and quality drops sharply.

        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>

    We return the prompt half separately so we know exactly how many tokens to
    MASK from the loss (everything up to and including "assistant\n").
    """
    prompt_part = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = f"{prompt_part}{output}<|im_end|>"
    return prompt_part, full_text


def build_tokenized_dataset(jsonl_path: Path, tokenizer, max_seq_len: int):
    """Load a JSONL file and tokenize into input_ids / labels tensors.

    The key subtlety is LABEL MASKING. For instruction tuning we want the loss
    (cross-entropy on next-token prediction) computed ONLY over the assistant's
    response tokens. Tokens belonging to the user prompt get label = -100 so they
    contribute nothing to the gradient. Without this, the model wastes capacity
    learning to parrot the question and trains more slowly on the actual skill.
    """
    from datasets import Dataset

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def tokenize_row(row: dict) -> dict:
        prompt_part, full_text = format_example(row["instruction"], row["output"])

        # Tokenize the full sequence (prompt + answer), truncated to max_seq_len.
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,  # our template already includes the markers
        )["input_ids"]

        # Tokenize just the prompt to learn how many leading tokens to mask.
        prompt_ids = tokenizer(
            prompt_part, add_special_tokens=False
        )["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))

        # labels == input_ids, except the prompt region is set to IGNORE_INDEX.
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}

    dataset = Dataset.from_list(rows)
    # remove_columns drops the raw text fields so the collator only sees tensors.
    return dataset.map(tokenize_row, remove_columns=dataset.column_names)


class TimeEstimatorCallback:
    """Prints a training-time estimate after the first N optimizer steps.

    Implemented as a TrainerCallback (imported lazily in main so this module can
    be read/linted without transformers installed).
    """

    def __init__(self, estimate_after_steps: int = 10):
        self.estimate_after_steps = estimate_after_steps
        self._start_time: float | None = None

    def make(self):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                outer._start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == outer.estimate_after_steps:
                    elapsed = time.time() - outer._start_time
                    per_step = elapsed / outer.estimate_after_steps
                    remaining = per_step * (state.max_steps - state.global_step)
                    print(f"\n[time estimate] {per_step:.2f}s/step -> "
                          f"~{remaining / 60:.1f} min remaining "
                          f"({state.max_steps} total steps)\n")

        return _Callback()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-Coder-1.5B on STM32N6 pairs (Kaggle T4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Dir with train.jsonl / val.jsonl. On Kaggle: /kaggle/input/<dataset>/",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG,
                        help="Training config YAML.")
    parser.add_argument("--lora-config", type=Path, default=DEFAULT_LORA_CONFIG,
                        help="LoRA config YAML.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output_dir from the training config.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging (off by default).")
    args = parser.parse_args()

    # Heavy ML imports live here so `--help` is instant and doesn't require the
    # full GPU stack to be installed.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    cfg = load_yaml(args.config)
    lora_cfg = load_yaml(args.lora_config)
    output_dir = str(args.output_dir) if args.output_dir else cfg["output_dir"]

    # --- Tokenizer -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    # Causal LMs often ship without a pad token; reuse EOS so batching can pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4-bit quantization config (the "Q" in QLoRA) ------------------------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # NF4 = "4-bit NormalFloat", a quantization data type designed for the
        # roughly-normal distribution of neural net weights. It preserves more
        # signal than plain int4 at the same bit width.
        bnb_4bit_quant_type="nf4",
        # Double quantization also quantizes the per-block scale constants,
        # saving a bit more VRAM for free.
        bnb_4bit_use_double_quant=True,
        # Compute dtype for the dequantized matmuls. T4 (Turing) has no bf16, so
        # we compute in fp16. (This must match fp16=True in TrainingArguments.)
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- Base model (frozen, 4-bit) ------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        device_map="auto",  # place the (single) model on the available GPU
        trust_remote_code=True,
    )
    # Prepares a k-bit (quantized) model for training: casts layernorms to fp32
    # for stability, enables gradient flow into the (soon-to-be-added) adapters,
    # and disables the KV cache (only needed at inference).
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["gradient_checkpointing"]
    )

    # --- Attach LoRA adapters ------------------------------------------------
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    # Sanity check: this should show only ~0.1-1% of params as trainable. If it
    # shows ~100%, the base model wasn't frozen and you'll OOM.
    model.print_trainable_parameters()

    # --- Datasets ------------------------------------------------------------
    train_dataset = build_tokenized_dataset(
        args.data_dir / "train.jsonl", tokenizer, cfg["max_seq_len"]
    )
    eval_dataset = build_tokenized_dataset(
        args.data_dir / "val.jsonl", tokenizer, cfg["max_seq_len"]
    )
    print(f"train examples: {len(train_dataset)}  |  val examples: {len(eval_dataset)}")

    # Pads variable-length sequences within a batch AND pads the -100 labels
    # correctly (so padding never contributes to the loss).
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8
    )

    # --- Trainer configuration ----------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=cfg["fp16"],   # True on T4
        bf16=cfg["bf16"],   # False on T4 (no hardware support)
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        # Keep the checkpoint with the lowest eval loss as "best".
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,  # don't fill Kaggle's disk with old checkpoints
        # paged_adamw_8bit is the bitsandbytes optimizer QLoRA is designed for:
        # 8-bit optimizer states (less VRAM) with paging to CPU on memory spikes.
        optim="paged_adamw_8bit",
        report_to=["wandb"] if args.wandb else [],
        # A stable seed so runs are comparable.
        seed=42,
    )

    callbacks = [TimeEstimatorCallback(estimate_after_steps=10).make()]
    # Early stopping lets us set a GENEROUS epoch budget and let the model decide
    # when it has stopped learning, instead of hard-coding an epoch count. It
    # halts once eval_loss hasn't improved for `patience` consecutive evals, and
    # (with load_best_model_at_end) the lowest-eval-loss checkpoint is restored.
    # Needs an eval strategy + metric_for_best_model, both set above.
    patience = cfg.get("early_stopping_patience", 0)
    if patience and patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        print(f"Early stopping enabled: patience={patience} evals on eval_loss.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()

    # Save the best adapter + tokenizer. NOTE: this saves only the LoRA adapter
    # weights (a few MB), not the full base model — that's the whole point of
    # LoRA. inference.py reloads the base model and applies this adapter on top.
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved best adapter + tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
