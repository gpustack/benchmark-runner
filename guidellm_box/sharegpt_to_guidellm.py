#!/usr/bin/env python3


import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("sharegpt_to_guidellm")

# Configure logger for console output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------
# Tokenizer
# -------------------------


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    return tokenizer


def count_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


# -------------------------
# ShareGPT parsing
# -------------------------


def iter_sharegpt_samples(
    input_file: Path,
) -> Iterable[dict]:
    with input_file.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    for sample in dataset:
        yield sample


def extract_first_turn(sample: dict) -> Optional[tuple[str, str]]:
    conversations = sample.get("conversations")
    if not conversations or len(conversations) < 2:
        return None

    first, second = conversations[0], conversations[1]

    if first.get("from") not in ("human", "user"):
        return None
    if second.get("from") not in ("gpt", "assistant"):
        return None

    prompt = first.get("value")
    completion = second.get("value")

    if not prompt or not completion:
        return None

    return prompt, completion


# -------------------------
# guidellm record
# -------------------------


def build_guidellm_record(
    prompt: str,
    completion: str,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    output_tokens = count_tokens(tokenizer, completion)

    return {
        "text": prompt,
        "output_tokens_count": output_tokens,
    }


# -------------------------
# Writer
# -------------------------


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# -------------------------
# Main
# -------------------------


def convert_sharegpt_to_guidellm(
    input_file: Path,
    output_file: Path,
    tokenizer_name: str,
    max_items: int = None,
    output_format: str = "jsonl",
) -> dict:
    """
    Convert ShareGPT dataset to guidellm-compatible format.
    Returns statistics: {'written': int, 'skipped': int, 'output': Path}
    """
    logger.info("Preparing ShareGPT dataset")
    logger.info(f"Loading tokenizer: {tokenizer_name}")

    tokenizer = load_tokenizer(tokenizer_name)
    records = []
    written = 0
    skipped = 0
    # Progress logging: log every 10000 processed samples

    logger.info(f"Starting conversion from {input_file} to {output_file}")
    idx = 0
    for sample in iter_sharegpt_samples(input_file):
        idx += 1
        result = extract_first_turn(sample)
        if not result:
            skipped += 1
            if idx % 10000 == 0:
                logger.info(
                    f"Progress: processed={idx}, written={written}, skipped={skipped}"
                )
            continue
        prompt, completion = result
        record = build_guidellm_record(prompt, completion, tokenizer)
        records.append(record)
        written += 1
        if max_items is not None and written == max_items:
            break
        if idx % 10000 == 0:
            logger.info(
                f"Progress: processed={idx}, written={written}, skipped={skipped}"
            )
    # Final progress log
    logger.info(
        f"Progress: processed={idx}, written={written}, skipped={skipped} (final)"
    )
    if output_format == "jsonl":
        write_jsonl(output_file, records)
    else:
        write_json(output_file, records)
    return {"written": written, "skipped": skipped, "output": output_file}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT dataset to guidellm-compatible JSON/JSONL"
    )
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="jsonl",
    )
    args = parser.parse_args()
    stats = convert_sharegpt_to_guidellm(
        input_file=args.input_file,
        output_file=args.output_file,
        tokenizer_name=args.tokenizer,
        max_items=args.max_items,
        output_format=args.format,
    )
    logger.info(
        f"Conversion done: written={stats['written']}, skipped={stats['skipped']}, "
        f"output={stats['output']}"
    )


if __name__ == "__main__":
    main()
