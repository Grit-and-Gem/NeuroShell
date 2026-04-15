#!/usr/bin/env python3
"""
inference_server.py — FastAPI server that serves the fine-tuned Qwen2.5 model
for terminal command generation.

Usage:
    uvicorn inference_server:app --host 0.0.0.0 --port 8000 --workers 1

Environment variables:
    NEUROSHELL_BASE_MODEL    Base model name (default: Qwen/Qwen2.5-0.5B)
    NEUROSHELL_ADAPTER_PATH  Path to LoRA adapter directory (default: output/lora-adapter)

API:
    POST /generate  {"task": "natural language task"} → {"command": "shell command"}
    GET  /health    → {"status": "ok", "model": "...", "adapter": "..."}
"""

import os
import sys
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = os.getenv("NEUROSHELL_BASE_MODEL", "Qwen/Qwen2.5-0.5B")
ADAPTER_PATH = os.getenv("NEUROSHELL_ADAPTER_PATH", "output/lora-adapter")

SYSTEM_PROMPT = (
    "You are a terminal command generator. Given a task in natural language, "
    "output only the terminal command to accomplish it. "
    "No explanation, no markdown, just the raw command."
)

# Module-level singletons loaded at startup
_model = None
_tokenizer = None


def load_model_and_tokenizer():
    global _model, _tokenizer

    print(f"Loading tokenizer from: {ADAPTER_PATH}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    print(f"Loading base model: {BASE_MODEL_NAME}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter_exists = os.path.isfile(os.path.join(ADAPTER_PATH, "adapter_config.json"))
    if adapter_exists:
        print(f"Loading LoRA adapter from: {ADAPTER_PATH}", flush=True)
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(
            f"Warning: no adapter found at '{ADAPTER_PATH}'. "
            "Running base model without fine-tuning.",
            flush=True,
        )
        _model = base_model

    _model.eval()
    print("Model ready.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()
    yield


app = FastAPI(
    title="NeuroShell Inference Server",
    description="Generates terminal commands from natural language tasks",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    task: str

    @field_validator("task")
    @classmethod
    def task_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task must not be empty")
        return v.strip()


class GenerateResponse(BaseModel):
    command: str


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: str
    device: str


@app.get("/health", response_model=HealthResponse)
async def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok",
        model=BASE_MODEL_NAME,
        adapter=ADAPTER_PATH,
        device=device,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.task},
    ]

    # Build prompt with generation prefix so model knows to produce the assistant turn
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (exclude prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]
    raw_output = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Post-process: take only the first non-empty line (the command)
    # The model may sometimes generate extra commentary after the command
    command = ""
    for line in raw_output.splitlines():
        line = line.strip()
        if line:
            command = line
            break

    if not command:
        command = raw_output.strip()

    return GenerateResponse(command=command)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
