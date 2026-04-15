#!/usr/bin/env python3
"""
prepare_data.py — Validates and converts commands.jsonl into ChatML-formatted
training data for Qwen2.5 fine-tuning.

Usage:
    python prepare_data.py                  # Convert data/commands.jsonl → data/train.jsonl
    python prepare_data.py --validate       # Validate only, no output
    python prepare_data.py --input data/commands.jsonl --output data/train.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a terminal command generator. Given a task in natural language, "
    "output only the terminal command to accomplish it. "
    "No explanation, no markdown, just the raw command."
)


def load_raw_dataset(path: str) -> list[dict]:
    """Load JSONL file and return list of dicts. Raises on parse errors."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error on line {line_num}: {e}") from e
            records.append(record)
    return records


def validate_dataset(records: list[dict]) -> list[str]:
    """Validate records and return list of error messages."""
    errors = []
    seen_instructions = {}

    for i, record in enumerate(records, 1):
        if "instruction" not in record:
            errors.append(f"Record {i}: missing 'instruction' field")
        elif not record["instruction"].strip():
            errors.append(f"Record {i}: empty 'instruction' field")

        if "command" not in record:
            errors.append(f"Record {i}: missing 'command' field")
        elif not record["command"].strip():
            errors.append(f"Record {i}: empty 'command' field")

        if "instruction" in record and "command" in record:
            key = record["instruction"].strip().lower()
            if key in seen_instructions:
                errors.append(
                    f"Record {i}: duplicate instruction (first seen at record "
                    f"{seen_instructions[key]}): '{record['instruction'][:60]}'"
                )
            else:
                seen_instructions[key] = i

    return errors


def to_chatml(record: dict) -> str:
    """Convert a single instruction/command pair to ChatML format."""
    instruction = record["instruction"].strip()
    command = record["command"].strip()
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{command}<|im_end|>"
    )


def convert_dataset(records: list[dict], output_path: str) -> int:
    """Convert records to ChatML JSONL and write to output_path. Returns count."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            if not record.get("instruction", "").strip():
                continue
            if not record.get("command", "").strip():
                continue
            entry = {"text": to_chatml(record)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset for NeuroShell")
    parser.add_argument(
        "--input",
        default="data/commands.jsonl",
        help="Path to raw JSONL dataset (default: data/commands.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/train.jsonl",
        help="Path for ChatML-formatted output (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the dataset without writing output",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {input_path}")
    try:
        records = load_raw_dataset(str(input_path))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records")

    errors = validate_dataset(records)
    if errors:
        print(f"\nValidation found {len(errors)} issue(s):")
        for err in errors:
            print(f"  - {err}")
        if args.validate:
            sys.exit(1)
        print("Proceeding with valid records only...")
    else:
        print("Validation passed: all records are valid")

    if args.validate:
        print("Validation complete (--validate mode, no output written)")
        sys.exit(0)

    count = convert_dataset(records, args.output)
    print(f"Wrote {count} ChatML-formatted examples to: {args.output}")
    print("\nSample output:")
    with open(args.output, "r") as f:
        sample = json.loads(f.readline())
        print(sample["text"][:300] + "...")


if __name__ == "__main__":
    main()
