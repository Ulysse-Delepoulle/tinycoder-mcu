#!/usr/bin/env python3
"""Multi-signal evaluation harness for the fine-tuned STM32N6 coder.

Code generation is hard to score with any single number, so we combine three
complementary signals — each measures something the others miss:

  1. COMPILATION  — does the generated C actually compile for Cortex-M55?
                    This is the strongest correctness proxy we have without
                    hardware. (Soft-skips if arm-none-eabi-gcc isn't installed.)
  2. FUNCTIONAL   — does the output mention the right APIs/registers and avoid
                    known-wrong patterns? A cheap, targeted keyword check that
                    catches "compiles but does the wrong thing" cases.
  3. SIMILARITY   — ROUGE-L and BLEU vs reference answers. These measure textual
                    overlap, NOT correctness (see the note below), so we report
                    them as a weak signal, never as the headline metric.

Run `python scripts/evaluate.py --help` for options.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "eval_results.json"
DEFAULT_TEST_CASES_DIR = PROJECT_ROOT / "eval" / "test_cases"
COMPILE_TEST_SCRIPT = PROJECT_ROOT / "eval" / "compile_test.sh"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"

# How many new tokens the model may generate per eval prompt. Enough for a
# medium C function; raise if your test cases expect longer outputs.
MAX_NEW_TOKENS = 512

# Regex to pull ```c ... ``` blocks out of the model's markdown output.
C_CODE_BLOCK_RE = re.compile(r"```c\s*\n(.*?)```", re.DOTALL)


def extract_c_code(text: str) -> str:
    """Return the concatenated contents of all ```c blocks, or "" if none."""
    blocks = C_CODE_BLOCK_RE.findall(text)
    return "\n\n".join(block.strip() for block in blocks)


# --- Signal 1: compilation ---------------------------------------------------

def run_compilation_test(generations: list[str]) -> dict:
    """Compile each generation's C code via eval/compile_test.sh.

    Soft-fails: if the ARM toolchain is missing, compile_test.sh returns a
    special exit code (2) and we mark the whole signal as "skipped" rather than
    crashing — you can still run this harness on Kaggle, which has no ARM gcc.
    """
    if not COMPILE_TEST_SCRIPT.exists():
        return {"status": "skipped", "reason": f"{COMPILE_TEST_SCRIPT} not found"}

    passed = 0
    attempted = 0
    for text in generations:
        code = extract_c_code(text)
        if not code:
            continue  # nothing to compile; not counted as attempted
        attempted += 1
        with tempfile.NamedTemporaryFile(
            "w", suffix=".c", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["bash", str(COMPILE_TEST_SCRIPT), tmp_path],
                capture_output=True, text=True,
            )
        except FileNotFoundError:
            # No bash available (e.g. bare Windows shell). Soft-skip.
            return {"status": "skipped", "reason": "bash not found to run compile_test.sh"}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Exit code 2 is our agreed "toolchain missing" signal from the script.
        if result.returncode == 2:
            return {"status": "skipped",
                    "reason": "arm-none-eabi-gcc not found (soft skip)"}
        if result.returncode == 0:
            passed += 1

    pass_rate = (passed / attempted) if attempted else 0.0
    return {"status": "ok", "attempted": attempted, "passed": passed,
            "pass_rate": round(pass_rate, 4)}


# --- Signal 2: functional keyword checks ------------------------------------

def load_test_cases(test_cases_dir: Path) -> list[dict]:
    """Load eval/test_cases/*.json. Each: {prompt, expected_keywords, forbidden_patterns}."""
    if not test_cases_dir.exists():
        return []
    cases = []
    for path in sorted(test_cases_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            case = json.load(f)
            case.setdefault("name", path.stem)
            cases.append(case)
    return cases


def run_functional_test(generations_by_prompt: dict[str, str],
                        test_cases: list[dict]) -> dict:
    """Check each generation for expected keywords and forbidden patterns."""
    if not test_cases:
        return {"status": "skipped", "reason": "no test cases found"}

    per_case = []
    passed = 0
    for case in test_cases:
        output = generations_by_prompt.get(case["prompt"], "")
        expected = case.get("expected_keywords", [])
        forbidden = case.get("forbidden_patterns", [])

        has_expected = all(kw in output for kw in expected)
        has_forbidden = any(pat in output for pat in forbidden)
        case_passed = has_expected and not has_forbidden
        passed += int(case_passed)

        per_case.append({
            "name": case["name"],
            "passed": case_passed,
            "missing_keywords": [kw for kw in expected if kw not in output],
            "hit_forbidden": [pat for pat in forbidden if pat in output],
        })

    return {"status": "ok", "total": len(test_cases), "passed": passed,
            "pass_rate": round(passed / len(test_cases), 4), "cases": per_case}


# --- Signal 3: similarity (ROUGE-L + BLEU) ----------------------------------

def run_similarity_test(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L and BLEU of predictions vs references.

    IMPORTANT — what these actually measure (and why neither is enough for code):
      * ROUGE-L is based on the Longest Common Subsequence of tokens. It rewards
        getting the same tokens in the same relative order — a recall-flavored
        overlap score. It does NOT understand syntax or semantics.
      * BLEU is precision of n-gram overlap (with a brevity penalty). It rewards
        matching short contiguous token runs.
    Both can be high for code that doesn't compile, and low for correct code
    written a different (valid) way. That's exactly why we also run compilation
    and functional signals — similarity is the weakest leg of the stool.
    """
    if not references:
        return {"status": "skipped", "reason": "no references provided"}

    # This file is scripts/evaluate.py, so running `python scripts/evaluate.py`
    # puts the script's own directory on sys.path[0]. A bare `import evaluate`
    # then resolves to THIS file instead of the HuggingFace `evaluate` metrics
    # library. Remove the script directory from sys.path (and any stale cached
    # module) so the real package is imported.
    import os
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != script_dir]
    sys.modules.pop("evaluate", None)
    try:
        import evaluate  # HuggingFace `evaluate` metrics library
    except ImportError:
        return {"status": "skipped", "reason": "`evaluate` package not installed"}

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    rouge_result = rouge.compute(predictions=predictions, references=references)
    # sacrebleu expects each prediction to have a LIST of references.
    bleu_result = bleu.compute(
        predictions=predictions, references=[[r] for r in references]
    )
    return {
        "status": "ok",
        "rougeL": round(float(rouge_result["rougeL"]), 4),
        "bleu": round(float(bleu_result["score"]), 4),
    }


# --- Generation --------------------------------------------------------------

def load_model_and_tokenizer(checkpoint: Path, base_model_name: str):
    """Load base model in 4-bit + the trained LoRA adapter from `checkpoint`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Same 4-bit setup as training so eval matches how the model will be served.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=quant_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint)  # apply the adapter
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction: str) -> str:
    """Generate a response for one instruction using Qwen's chat template."""
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
            do_sample=False,  # greedy = deterministic, so eval is reproducible
            pad_token_id=tokenizer.pad_token_id,
        )
    # Slice off the prompt tokens so we only decode the newly generated part.
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned STM32N6 coder with 3 signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the trained LoRA checkpoint dir.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Dir with val.jsonl (used for similarity references).")
    parser.add_argument("--test-cases-dir", type=Path, default=DEFAULT_TEST_CASES_DIR,
                        help="Dir of functional test-case JSON files.")
    parser.add_argument("--references-dir", type=Path, default=None,
                        help="Optional dir with a references.jsonl "
                             "({instruction, output}) for similarity scoring. "
                             "If omitted, val.jsonl is used.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG,
                        help="Training config (only to read model_name).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the JSON results.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N reference examples (speed).")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        base_model_name = yaml.safe_load(f)["model_name"]

    # --- Assemble the reference set (instruction -> gold output) -------------
    ref_path = (args.references_dir / "references.jsonl"
                if args.references_dir else args.data_dir / "val.jsonl")
    references_data = []
    if ref_path.exists():
        with ref_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    references_data.append(json.loads(line))
    if args.limit:
        references_data = references_data[:args.limit]

    # --- Load model and generate for every reference instruction -------------
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, base_model_name)

    predictions, gold_outputs = [], []
    generations_by_prompt: dict[str, str] = {}
    for row in references_data:
        instruction = row["instruction"]
        generated_text = generate(model, tokenizer, instruction)
        predictions.append(generated_text)
        gold_outputs.append(row.get("output", ""))
        generations_by_prompt[instruction] = generated_text

    # Also generate for the functional test-case prompts (they may differ from
    # the reference instructions).
    test_cases = load_test_cases(args.test_cases_dir)
    for case in test_cases:
        if case["prompt"] not in generations_by_prompt:
            generations_by_prompt[case["prompt"]] = generate(
                model, tokenizer, case["prompt"]
            )

    # --- Run the three signals ----------------------------------------------
    all_generations = list(generations_by_prompt.values())
    results = {
        "checkpoint": str(args.checkpoint),
        "num_reference_examples": len(references_data),
        "compilation": run_compilation_test(all_generations),
        "functional": run_functional_test(generations_by_prompt, test_cases),
        "similarity": run_similarity_test(predictions, gold_outputs),
    }

    # --- Print a summary table ----------------------------------------------
    print("\n" + "=" * 52)
    print("  EVALUATION SUMMARY")
    print("=" * 52)
    _print_signal("Compilation", results["compilation"], "pass_rate")
    _print_signal("Functional ", results["functional"], "pass_rate")
    sim = results["similarity"]
    if sim["status"] == "ok":
        print(f"  Similarity  : ROUGE-L={sim['rougeL']}  BLEU={sim['bleu']}")
    else:
        print(f"  Similarity  : skipped ({sim['reason']})")
    print("=" * 52 + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote detailed results to {args.output}")


def _print_signal(label: str, signal: dict, rate_key: str) -> None:
    if signal["status"] == "ok":
        print(f"  {label} : pass_rate={signal[rate_key]} "
              f"({signal.get('passed', '?')}/{signal.get('attempted', signal.get('total', '?'))})")
    else:
        print(f"  {label} : skipped ({signal['reason']})")


if __name__ == "__main__":
    main()
