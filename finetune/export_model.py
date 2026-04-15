#!/usr/bin/env python3
"""
export_model.py — Merge the LoRA adapter into the base model and save as a
standalone HuggingFace safetensors model for use with the self-contained
Rust executor (no Python server needed at runtime).

Run this ONCE after fine-tuning is complete.

Usage:
    python export_model.py
    python export_model.py --adapter output/lora-adapter --output output/merged-model
    python export_model.py --base-model Qwen/Qwen2.5-0.5B
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model for self-contained deployment"
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B",
        help="Base model name or path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--adapter",
        default="output/lora-adapter",
        help="Path to the LoRA adapter directory (default: output/lora-adapter)",
    )
    parser.add_argument(
        "--output",
        default="output/merged-model",
        help="Output directory for the merged model (default: output/merged-model)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.adapter):
        print(f"Error: adapter directory not found: '{args.adapter}'", file=sys.stderr)
        print("Run 'python finetune.py' first to produce the LoRA adapter.", file=sys.stderr)
        sys.exit(1)

    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {args.adapter}")
    print(f"Output     : {args.output}")
    print()

    # Load on CPU in float16 — avoids VRAM pressure and produces a smaller file
    print("Loading base model (float16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging LoRA weights into base model (merge_and_unload)...")
    model = model.merge_and_unload()
    model.eval()

    print(f"Saving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    # max_shard_size="2GB" keeps the 0.5B model as a single file
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    # Report what was written
    print()
    total_bytes = 0
    for fname in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, fname)
        size = os.path.getsize(fpath)
        total_bytes += size
        if size > 1024 * 1024:
            print(f"  {fname:<45} {size / 1024 / 1024:>8.1f} MB")
        else:
            print(f"  {fname:<45} {size / 1024:>8.1f} KB")

    print(f"\n  Total: {total_bytes / 1024 / 1024:.1f} MB")
    print()
    print("Export complete!")
    print()
    print("Next steps:")
    print(f"  cd ../executor")
    print(f"  cargo build --release")
    print(f"  echo 'list all files' > ../input/task.txt")
    print(f"  ./target/release/neuroshell --model ../{args.output}")


if __name__ == "__main__":
    main()
