#!/usr/bin/env python3
"""
finetune.py — Fine-tune Qwen2.5-0.5B on terminal command generation using LoRA/QLoRA.

Usage:
    python finetune.py
    python finetune.py --use-4bit                     # QLoRA (requires CUDA + bitsandbytes)
    python finetune.py --model-name Qwen/Qwen2.5-0.5B --num-epochs 5
    python finetune.py --data-path data/train.jsonl --output-dir output/lora-adapter
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for terminal command generation")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to ChatML-formatted JSONL training file (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/lora-adapter",
        help="Directory to save LoRA adapter weights (default: output/lora-adapter)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum token sequence length (default: 512)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit QLoRA quantization (requires CUDA + bitsandbytes)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling (default: 16)",
    )
    return parser.parse_args()


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # Qwen2.5 does not define a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right-padding is required for causal LM training
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, use_4bit: bool) -> AutoModelForCausalLM:
    if use_4bit:
        if not torch.cuda.is_available():
            print("Warning: --use-4bit requested but CUDA is not available. Falling back to FP32.")
            use_4bit = False

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_4bit:
        # Required for gradient checkpointing with quantized models
        model = prepare_model_for_kbit_training(model)
        model.enable_input_require_grads()

    return model


def get_lora_config(r: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        # All projection layers in Qwen2.5 transformer blocks
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: training data not found at '{args.data_path}'", file=sys.stderr)
        print("Run 'python prepare_data.py' first to generate the training data.", file=sys.stderr)
        sys.exit(1)

    print(f"Model:       {args.model_name}")
    print(f"Data:        {args.data_path}")
    print(f"Output:      {args.output_dir}")
    print(f"Epochs:      {args.num_epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.learning_rate}")
    print(f"Max seq len: {args.max_seq_length}")
    print(f"QLoRA (4bit):{args.use_4bit}")
    print(f"CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    print("Loading model...")
    model = load_model(args.model_name, args.use_4bit)

    print("Applying LoRA configuration...")
    lora_config = get_lora_config(args.lora_r, args.lora_alpha)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"Training examples: {len(dataset)}")

    # Detect bf16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        learning_rate=args.learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=True,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print(f"Adapter saved to: {args.output_dir}")
    print(f"Start inference server with:")
    print(f"  NEUROSHELL_ADAPTER_PATH={args.output_dir} uvicorn inference_server:app --port 8000")


if __name__ == "__main__":
    main()
