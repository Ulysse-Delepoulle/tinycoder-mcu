# TinyCoder-MCU

A LoRA fine-tuning pipeline for **Qwen2.5-Coder-1.5B** that teaches the model to
generate **STM32N6 embedded C code**. Training data is distilled from STM32N6
documentation using Claude as a teacher; training runs on a free **Kaggle T4
GPU**; the result is evaluated with a multi-signal harness.

## Overview

The project splits into two environments on purpose:

- **Local (CPU only):** build the instruction dataset by calling the Claude API.
  No GPU, no PyTorch — just `scripts/build_dataset.py`.
- **Kaggle (T4 GPU):** fine-tune and evaluate. QLoRA (4-bit) keeps a 1.5B model
  comfortably inside the T4's 15GB VRAM within Kaggle's free 30h/week.

```
STM32N6 docs ──chunk──▶ data/processed/ ──Claude(teacher)──▶ data/dataset/*.jsonl
                                                                    │
                                          upload as Kaggle Dataset  ▼
                              Kaggle T4:  train.py (QLoRA) ──▶ checkpoints/
                                                                    │
                                            evaluate.py (3 signals) ▼
                                                              eval_results.json
```

## Workflow (3 steps)

1. **Build the dataset locally.** Chunk your STM32N6 docs into
   `data/processed/*.txt`, then run `build_dataset.py`. It writes
   `data/dataset/train.jsonl` and `val.jsonl`.
2. **Upload to Kaggle.** Create a Kaggle Dataset from `data/dataset/`.
3. **Train + evaluate on Kaggle.** Open `kaggle_notebook.ipynb`, attach the
   dataset, and run the cells top to bottom.

## Local setup (build_dataset.py only)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements_local.txt
cp .env.example .env                               # then edit .env, add your key
```

Prepare chunks (one passage per file) in `data/processed/`, then:

```bash
# Sanity-check with no API calls (processes 5 chunks, prints what it would send):
python scripts/build_dataset.py --dry-run

# Real run:
python scripts/build_dataset.py
```

Output: `data/dataset/train.jsonl` + `val.jsonl` (90/10 split). Each line is an
Alpaca-style pair: `{"instruction": "...", "input": "", "output": "```c ...```"}`.

> **Teacher model:** set in `scripts/build_dataset.py` as `TEACHER_MODEL`
> (`claude-sonnet-5` by default). For higher-quality pairs at a higher per-call
> cost, use `claude-opus-5`.

## Kaggle setup (notebook walkthrough)

1. Upload `data/dataset/` as a **Kaggle Dataset** (Datasets → New Dataset).
2. Push this repo to GitHub and edit Cell 3's clone URL, **or** upload the repo
   as a second Kaggle Dataset and skip the clone.
3. Create a new Kaggle Notebook from `kaggle_notebook.ipynb`. In **Settings**:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
   - Attach your dataset (note the `/kaggle/input/<slug>/` mount path)
4. Edit Cell 4/5 `--data-dir` to match that mount path, then run all cells.
5. Use **Save Version → Save & Run All (Commit)** to persist `checkpoints/`
   (Cell 6 explains turning it into a reusable Dataset).

**Secrets:** you normally build the dataset locally, so no Kaggle Secret is
needed. If you ever run `build_dataset.py` on Kaggle, add `ANTHROPIC_API_KEY`
under **Add-ons → Secrets**.

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/ --data-dir data/dataset/
```

Three complementary signals (details in code comments):

| Signal        | What it checks                                   | Notes |
|---------------|--------------------------------------------------|-------|
| Compilation   | Does the generated C compile for Cortex-M55?     | Needs `arm-none-eabi-gcc`; **soft-skips** if missing (e.g. on Kaggle). |
| Functional    | Right APIs present, wrong patterns absent        | Driven by `eval/test_cases/*.json`. |
| Similarity    | ROUGE-L + BLEU vs reference outputs              | Weak signal for code — see below. |

Interactive testing:

```bash
python scripts/inference.py --checkpoint checkpoints/ --compare
```

`--compare` runs the same prompt through the base model too, so you can see what
fine-tuning changed. Add `--prompt "..."` to generate one answer non-interactively
and exit (use this in notebook cells, where the REPL can't read stdin).

## Results

Fine-tuned **Qwen2.5-Coder-1.5B** with 4-bit QLoRA on **162 instruction pairs**
(146 train / 16 val) distilled from the STM32N6 reference manual (GPIO, EXTI, SPI
chapters), on a single Kaggle T4.

**Training dynamics (early stopping on eval loss).** With `epochs: 15` as an
upper bound and patience 3, training ran to epoch 8 and restored the best
checkpoint (epoch 5):

| epoch     | 1     | 2     | 3     | 4     | 5         | 6     | 7     | 8     |
|-----------|-------|-------|-------|-------|-----------|-------|-------|-------|
| eval_loss | 0.906 | 0.813 | 0.770 | 0.750 | **0.738** | 0.758 | 0.794 | 0.832 |

eval_loss bottomed at epoch 5, then rose while train_loss kept falling — textbook
overfitting on a small dataset — so early stopping halted and kept the epoch-5
adapter. (A fixed 3-epoch run reached only 0.80.)

**Evaluation (best checkpoint).**

| Signal      | Result                    | Notes |
|-------------|---------------------------|-------|
| Compilation | skipped                   | no `arm-none-eabi-gcc` on Kaggle (soft-skip by design) |
| Functional  | 1/1                       | required HAL APIs present, no forbidden (Arduino) patterns |
| Similarity  | ROUGE-L 0.30 / BLEU 27.2  | vs. held-out reference outputs (weak signal for code) |

**Before / after — same prompt, base vs. fine-tuned.**

Prompt: *"Write STM32N6 C code that toggles GPIO pin PA5 using the HAL."*

Base model (no adapter) — degenerates into repetition:

```
Write STM32N6 C code that toggles GPIO pin PA5 using the HAL.	Duelassistant
	Duelassistant
... (repeats)
```

Fine-tuned (LoRA adapter applied):

```c
#include "stm32n6xx.h"

#define GPIO_PORT    GPIOA
#define GPIO_PIN     GPIO_PIN5

void GPIO_TogglePin_PA5(void)
{
    /* Enable GPIOA clock (peripheral clock gating control) */
    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* Toggle the pin state */
    GPIO_PORT->ODR ^= (1UL << GPIO_PIN);
}
```

**Honest read.** The base is the *non-instruct* completion model, so it never
learned to answer chat-formatted prompts and loops on the input. The fine-tune
taught it both to follow the instruction format and to emit structured STM32 C.
It isn't perfect: with only 146 training examples it toggles via the `ODR`
register instead of `HAL_GPIO_TogglePin`, and can over-generate (repeated blocks
or trailing tokens). The clear lever for improvement is **more data** (more RM
chapters), not more epochs.

## Concepts explained

New to fine-tuning? These are the ideas the code relies on.

**LoRA vs full fine-tuning.** Full fine-tuning updates every weight in the model
— billions of parameters — which needs many gigabytes of optimizer state and
usually multiple large GPUs. LoRA (Low-Rank Adaptation) instead freezes the
original weights and inserts a small pair of low-rank matrices next to each
target layer, training only those. You end up updating well under 1% of the
parameters. The payoff: it fits on one modest GPU, trains fast, and the saved
"adapter" is only a few megabytes because you're storing the small matrices, not
a whole new model.

**QLoRA and quantization.** QLoRA is LoRA applied on top of a **quantized** base
model. Quantization stores weights in fewer bits — here 4-bit NF4 (NormalFloat),
a format tuned for the bell-curve distribution of neural-net weights — which
shrinks the frozen model roughly 4x so it fits in 15GB. During each forward pass
`bitsandbytes` dequantizes the weights back to fp16 just long enough to do the
matrix multiply, then discards the expanded copy. The adapter itself trains in
full precision, so you get most of the quality of LoRA at a fraction of the
memory. That's the only reason a 1.5B model fine-tunes on a free T4.

**Instruction tuning vs pre-training.** Pre-training teaches a model general
next-token prediction over raw text — it learns language and code broadly but
not how to *follow a request*. Instruction tuning is a second, smaller stage
where the model sees `(instruction, response)` pairs formatted with the model's
chat template, and is trained to produce the response. Crucially we compute the
loss **only on the response tokens** (the prompt tokens are masked with `-100`),
so the model learns to *answer*, not to echo the question. That's exactly what
`build_labels`/label-masking in `train.py` does.

**Why distill from a larger model (teacher–student).** We don't have thousands
of hand-written STM32N6 examples, so we synthesize them: a large, capable
"teacher" (Claude) reads real documentation and writes high-quality
instruction/response pairs. The small "student" (Qwen 1.5B) is then trained to
imitate those answers. This transfers a slice of the teacher's competence into a
tiny, cheap-to-run model specialized for one domain — far more data-efficient
than hoping the small model figures the domain out on its own.

**Effective batch size and gradient accumulation.** The T4 can only hold a few
sequences in memory at once (`batch_size: 2`), but tiny batches give noisy
gradients. Gradient accumulation runs several small batches, sums their
gradients, and only then takes one optimizer step — simulating a larger batch
(`2 × 16 = 32`) without the memory cost. Larger effective batches make LoRA's
relatively high learning rate stable.

**What the evaluation signals actually measure.** Compilation is the strongest
proxy for correctness we have without hardware: code that won't compile for
Cortex-M55 is definitely wrong. Functional checks catch "compiles but wrong" by
looking for the APIs/registers the answer should use (and patterns it shouldn't,
like Arduino calls). ROUGE-L (longest-common-subsequence overlap, recall-ish)
and BLEU (n-gram precision) only measure *textual similarity* to a reference —
they can be high for broken code and low for correct-but-differently-written
code, so we treat them as a weak, supporting signal, never the headline number.

## Known gaps

- **No `arm-none-eabi-gcc` on Kaggle.** The compilation signal soft-skips there;
  run it locally (or in CI) where the ARM toolchain is installed.
- **No CUDA/TensorRT deployment path.** This project trains and evaluates; it
  does not export an optimized runtime for on-device inference.
- **No SLURM / multi-node.** Everything targets a single Kaggle T4.
- **Similarity metrics are weak for code** (see Concepts). Don't optimize for
  ROUGE/BLEU alone.
- **Dataset quality depends on your doc chunks.** Garbage passages in →
  low-value pairs out. Curate `data/processed/` thoughtfully.
