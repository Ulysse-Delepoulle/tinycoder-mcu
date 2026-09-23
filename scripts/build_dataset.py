#!/usr/bin/env python3
"""Distill an instruction-tuning dataset from STM32N6 doc passages.

This is the TEACHER-STUDENT distillation step. It runs LOCALLY on your CPU —
there is no model inference here, only Claude API calls. A large, capable
"teacher" model (Claude) reads a chunk of STM32N6 documentation and writes
instruction/response pairs. Later, on Kaggle, the small "student" model
(Qwen2.5-Coder-1.5B) is fine-tuned to imitate those responses. The student
never sees the teacher again — it just learns from the frozen dataset.

Pipeline:
    data/processed/*.txt   (chunked doc passages, you prepare these)
        -> Claude generates {instruction, input, output} pairs per chunk
        -> data/dataset/train.jsonl  +  data/dataset/val.jsonl   (90/10 split)

Run `python scripts/build_dataset.py --help` for options.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# --- Configuration constants (named, not magic numbers) ---------------------

# Teacher model used to synthesize the instruction pairs. Sonnet is a strong,
# cost-effective choice for bulk generation. Swap in "claude-opus-5" for
# higher-quality pairs at a higher per-call cost.
TEACHER_MODEL = "claude-sonnet-5"

# How many instruction/response pairs to ask the teacher to produce per doc
# chunk. More pairs per chunk = fewer API calls but longer responses.
PAIRS_PER_CHUNK = 3

# Cap on teacher output tokens per call. 3 code-bearing pairs fit comfortably
# under this; raise it if you increase PAIRS_PER_CHUNK.
MAX_OUTPUT_TOKENS = 4096

# Fraction of the final dataset held out for validation (eval loss during
# training). 10% is standard for a few-thousand-example set.
VAL_FRACTION = 0.10

# Retry/backoff settings for transient API failures (rate limits, 5xx). The SDK
# already retries internally; this outer loop adds a longer, logged backoff so a
# multi-hour generation run survives a rough patch without dying.
MAX_RETRIES = 6
BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 90.0

# In --dry-run mode we only touch this many chunks and make ZERO API calls.
DRY_RUN_CHUNK_LIMIT = 5

# Deterministic shuffle so the train/val split is reproducible across runs.
RANDOM_SEED = 13

# --- Paths (relative to the project root, resolved from this file) ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"


# The prompt we give the teacher. We ask for STRICT JSON so we can parse it
# mechanically. The {"instruction","input","output"} shape is the classic
# Alpaca instruction-tuning format the student will later be trained on.
TEACHER_SYSTEM_PROMPT = """\
You are an expert STM32N6 embedded firmware engineer creating training data \
for a small code-generation model. Given a passage of STM32N6 documentation, \
you write realistic instruction/response pairs that teach a model to generate \
correct, idiomatic embedded C for the STM32N6 series.

Rules:
- Each "output" MUST contain a fenced C code block (```c ... ```).
- Code should target the STM32N6 (Cortex-M55). Prefer HAL/LL-style or \
register-level C consistent with the passage.
- Instructions must be answerable from (or reasonably grounded in) the passage.
- Vary the instruction style: some ask to "write", some to "configure", some \
to "implement a function that ...".
- Keep the "input" field an empty string unless extra context is truly needed.
"""


def build_user_prompt(chunk_text: str, pairs_per_chunk: int) -> str:
    """Assemble the per-chunk user message for the teacher."""
    return (
        f"Here is a passage of STM32N6 documentation:\n\n"
        f"<passage>\n{chunk_text}\n</passage>\n\n"
        f"Generate exactly {pairs_per_chunk} instruction/response pairs grounded "
        f"in this passage.\n\n"
        f"Respond with ONLY a JSON array (no prose, no markdown fences around "
        f"the JSON) of objects with exactly these keys:\n"
        f'  "instruction": a task describing STM32N6 C code to write\n'
        f'  "input": usually an empty string ""\n'
        f'  "output": the answer, containing a ```c fenced code block\n'
    )


def load_chunks(processed_dir: Path) -> list[tuple[str, str]]:
    """Load chunked passages from disk.

    Returns a list of (source_name, text) tuples. We accept .txt and .md files;
    each file is treated as one chunk. If you want finer chunks, split your docs
    into more/smaller files in data/processed/.
    """
    if not processed_dir.exists():
        sys.exit(
            f"Processed dir not found: {processed_dir}\n"
            f"Put chunked STM32N6 doc passages (.txt/.md) there first."
        )
    chunk_paths = sorted(
        p for p in processed_dir.iterdir()
        if p.suffix.lower() in {".txt", ".md"} and p.is_file()
    )
    chunks: list[tuple[str, str]] = []
    for path in chunk_paths:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:  # skip empty files
            chunks.append((path.name, text))
    return chunks


def call_teacher(client, chunk_text: str, pairs_per_chunk: int) -> str:
    """Call the teacher model with exponential backoff, returning raw text.

    We import `anthropic` lazily inside this function so that --dry-run works
    even if the SDK isn't installed yet.
    """
    import anthropic

    user_prompt = build_user_prompt(chunk_text, pairs_per_chunk)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=TEACHER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # response.content is a list of content blocks; concatenate the text
            # blocks. (A well-behaved response here is a single text block.)
            return "".join(b.text for b in response.content if b.type == "text")

        except anthropic.RateLimitError as exc:
            # 429: back off and retry. Honor the server's retry-after if present.
            last_error = exc
            retry_after = exc.response.headers.get("retry-after") if exc.response else None
            delay = float(retry_after) if retry_after else _backoff_delay(attempt)
        except anthropic.APIStatusError as exc:
            # Retry only on server-side (5xx) errors; a 4xx (bad request, auth)
            # won't fix itself, so fail loudly.
            if exc.status_code >= 500:
                last_error = exc
                delay = _backoff_delay(attempt)
            else:
                raise
        except anthropic.APIConnectionError as exc:
            last_error = exc
            delay = _backoff_delay(attempt)

        print(f"    transient error, retry {attempt + 1}/{MAX_RETRIES} "
              f"in {delay:.1f}s: {last_error}")
        time.sleep(delay)

    raise RuntimeError(f"Teacher call failed after {MAX_RETRIES} retries") from last_error


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, capped at MAX_DELAY_SECONDS."""
    delay = BASE_DELAY_SECONDS * (2 ** attempt)
    delay += random.uniform(0, 1)  # jitter avoids thundering-herd re-tries
    return min(delay, MAX_DELAY_SECONDS)


def parse_pairs(raw_text: str, source_name: str) -> list[dict]:
    """Parse the teacher's JSON array into validated pair dicts.

    Robust to the model occasionally wrapping the JSON in a ```json fence.
    Silently skips a chunk whose output can't be parsed (logs a warning) rather
    than crashing a long run.
    """
    text = raw_text.strip()
    # Strip a leading/trailing markdown fence if the model added one.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the ```json line
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"    WARNING: could not parse JSON from {source_name}: {exc}")
        return []

    if not isinstance(parsed, list):
        print(f"    WARNING: {source_name} did not return a JSON array; skipping")
        return []

    valid_pairs: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        # Keep only well-formed pairs that actually contain C code.
        if instruction and "```c" in output:
            valid_pairs.append({
                "instruction": instruction,
                "input": item.get("input", ""),
                "output": output,
            })
    return valid_pairs


def write_split(pairs: list[dict], dataset_dir: Path) -> tuple[int, int]:
    """Shuffle, split 90/10, and write train.jsonl + val.jsonl. Returns counts."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * VAL_FRACTION)) if pairs else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    _write_jsonl(dataset_dir / "train.jsonl", train_pairs)
    _write_jsonl(dataset_dir / "val.jsonl", val_pairs)
    return len(train_pairs), len(val_pairs)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSON Lines (one JSON object per line)."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill an STM32N6 instruction dataset from Claude (runs locally).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
        help="Directory of chunked doc passages (.txt/.md).",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
        help="Output directory for train.jsonl / val.jsonl.",
    )
    parser.add_argument(
        "--pairs-per-chunk", type=int, default=PAIRS_PER_CHUNK,
        help="How many instruction/response pairs to request per chunk.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=f"Process only {DRY_RUN_CHUNK_LIMIT} chunks and make NO API calls "
             f"(prints the prompts that WOULD be sent). Use to sanity-check setup.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Real run only: process at most N chunks (cost control for a first "
             "run). Default: all chunks.",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.processed_dir)
    if not chunks:
        sys.exit(f"No chunks found in {args.processed_dir} (need .txt/.md files).")
    print(f"Loaded {len(chunks)} chunk(s) from {args.processed_dir}")

    # --- Dry run: no key, no SDK, no network. Just show what would happen. ---
    if args.dry_run:
        print(f"\n--- DRY RUN: first {DRY_RUN_CHUNK_LIMIT} chunk(s), no API calls ---")
        for source_name, text in chunks[:DRY_RUN_CHUNK_LIMIT]:
            preview = text[:200].replace("\n", " ")
            print(f"\n[{source_name}] ({len(text)} chars)")
            print(f"  passage preview: {preview}...")
            print(f"  would request {args.pairs_per_chunk} pairs from {TEACHER_MODEL}")
        print("\nDry run complete. No files written. Remove --dry-run to generate.")
        return

    # --- Real run: load key from .env, build client, generate. ---------------
    from dotenv import load_dotenv
    import anthropic

    load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env file
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")

    from tqdm import tqdm

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from the env
    all_pairs: list[dict] = []

    # --limit caps how many chunks we actually send to the teacher, so a first
    # real run can be validated cheaply before committing to the whole corpus.
    chunks_to_process = chunks[:args.limit] if args.limit else chunks
    if args.limit and args.limit < len(chunks):
        print(f"--limit {args.limit}: processing {len(chunks_to_process)} of "
              f"{len(chunks)} chunk(s).")

    for source_name, text in tqdm(chunks_to_process, desc="Distilling", unit="chunk"):
        raw = call_teacher(client, text, args.pairs_per_chunk)
        pairs = parse_pairs(raw, source_name)
        all_pairs.extend(pairs)

    if not all_pairs:
        sys.exit("No valid pairs were generated — check your chunks and API key.")

    n_train, n_val = write_split(all_pairs, args.dataset_dir)
    print(f"\nGenerated {len(all_pairs)} pairs -> "
          f"{n_train} train / {n_val} val")
    print(f"Wrote {args.dataset_dir / 'train.jsonl'} and "
          f"{args.dataset_dir / 'val.jsonl'}")
    print(f"\nUpload {args.dataset_dir}/ as a Kaggle Dataset before running training.")


if __name__ == "__main__":
    main()
