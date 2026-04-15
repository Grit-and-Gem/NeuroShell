# NeuroShell

Translate natural language into terminal commands using a fine-tuned Qwen2.5-0.5B model.
The runtime is **fully self-contained** — one Rust binary, one model directory, no Python server.

```
User writes: "show disk usage of home directory"
        ↓
neuroshell reads input/task.txt
        ↓
Qwen2.5-0.5B runs inside the binary (candle, pure Rust)
        ↓
sh -c "du -sh ~"
        ↓
output/result.txt
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  FINE-TUNING  (Python, done once on any machine)        │
│                                                         │
│  commands.jsonl → prepare_data.py → train.jsonl         │
│                         ↓                               │
│                    finetune.py                          │
│               Qwen2.5-0.5B + LoRA                       │
│                         ↓                               │
│               output/lora-adapter/                      │
│                         ↓                               │
│                  export_model.py                        │
│           (merges LoRA into base weights)               │
│                         ↓                               │
│               output/merged-model/   ◄── copy this      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RUNTIME  (Rust only, no Python needed)                 │
│                                                         │
│  input/task.txt  →  neuroshell  →  sh -c <command>      │
│                          ↑                              │
│                   Qwen2 model runs                      │
│                   inside the binary                     │
│                   via candle (pure Rust)                │
│                          ↓                              │
│                   output/result.txt                     │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

**For fine-tuning (one-time, any GPU machine or Colab):**
- Python 3.10+
- CUDA GPU recommended (CPU works but is slow)

**For building and running (your machine, permanently):**
- Rust 1.75+ — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- That's it. No Python, no CUDA, no background services.

---

## Setup

### Step 1 — Install Python deps and prepare data

```bash
cd finetune
pip install -r requirements.txt
python prepare_data.py
# Writes: finetune/data/train.jsonl
```

### Step 2 — Fine-tune the model (once)

```bash
python finetune.py                # GPU, FP16
python finetune.py --use-4bit     # GPU, QLoRA (less VRAM)
```

Saves LoRA adapter to `finetune/output/lora-adapter/`.

| Setup | VRAM | Time (~271 examples, 5 epochs) |
|---|---|---|
| GPU + `--use-4bit` | ~3 GB | ~10 min |
| GPU FP16 | ~4 GB | ~10 min |
| CPU only | ~4 GB RAM | ~2–5 hrs |

A free **Google Colab T4** works fine.

### Step 3 — Export the merged model (once)

Merges the LoRA adapter into the base model and saves standalone weights:

```bash
python export_model.py
# Reads:  output/lora-adapter/
# Writes: output/merged-model/  (~1 GB, float16)
```

Copy `finetune/output/merged-model/` to wherever you'll run `neuroshell`.

### Step 4 — Build the Rust binary

```bash
cd executor
cargo build --release
# Binary: executor/target/release/neuroshell
```

Just `cargo build` — no cmake, no C++ compiler, no system libraries needed.

---

## Usage

### One-shot

```bash
echo "find all .log files modified today" > input/task.txt

./executor/target/release/neuroshell \
  --model finetune/output/merged-model
```

`output/result.txt`:
```
==================================================
NeuroShell Execution Result
==================================================
Task:    find all .log files modified today
Command: find . -name "*.log" -mtime -1
--------------------------------------------------
./app.log
./error.log
--------------------------------------------------
Exit Code: 0
Timestamp: 2026-04-15 14:30:00 UTC
==================================================
```

### Watch mode

Automatically re-runs whenever the input file changes:

```bash
./executor/target/release/neuroshell \
  --model finetune/output/merged-model \
  --watch
```

Then in another terminal:
```bash
echo "list processes sorted by memory" > input/task.txt
# neuroshell detects the change → generates command → executes → updates result.txt
```

### CLI reference

```
Options:
  -m, --model <DIR>    Merged model directory [default: ../finetune/output/merged-model]
                       [env: NEUROSHELL_MODEL]
  -i, --input <FILE>   Input file with natural language task [default: ../input/task.txt]
  -o, --output <FILE>  Output file for results [default: ../output/result.txt]
  -w, --watch          Watch input file and auto-execute on changes
  -h, --help           Print help
```

---

## File Reference

```
NeuroShell/
│
│  ── Fine-tuning (Python, run once) ──────────────────────
├── finetune/
│   ├── data/
│   │   ├── commands.jsonl        # 271 NL→command training pairs (source of truth)
│   │   └── train.jsonl           # Generated by prepare_data.py — ChatML format
│   ├── prepare_data.py           # Validate + convert commands.jsonl → train.jsonl
│   ├── finetune.py               # LoRA/QLoRA training with SFTTrainer
│   ├── export_model.py           # Merge LoRA into base model → output/merged-model/
│   ├── inference_server.py       # Optional: FastAPI server (alternative to Rust binary)
│   └── requirements.txt
│
│  ── Runtime (Rust, self-contained) ──────────────────────
├── executor/
│   ├── Cargo.toml                # Dependencies (candle, clap, notify, …)
│   ├── Cargo.lock                # Pinned dependency versions for reproducible builds
│   └── src/
│       ├── main.rs               # CLI, orchestration, watch loop
│       ├── inference.rs          # Qwen2 model loading + generation (candle)
│       └── executor.rs           # Shell command execution via sh -c
│
│  ── I/O directories ────────────────────────────────────
├── input/                        # Drop task.txt here
└── output/                       # result.txt is written here
```

---

## How the model runs inside Rust

`candle` is HuggingFace's pure-Rust tensor library — the same organization that
makes transformers and PEFT. It has a native Qwen2 implementation (`ModelForCausalLM`)
that loads the same `model.safetensors` + `config.json` files that Python produces.

`ModelForCausalLM` wraps the base transformer with a linear `lm_head` projection
(hidden states → vocabulary logits). This is the correct class for text generation —
the base `Model` alone only returns hidden states, not next-token probabilities.

At runtime (`inference.rs`):
1. `neuroshell` memory-maps `model.safetensors` (fast startup, OS handles caching)
2. Formats the task into a ChatML prompt (`<|im_start|>system … <|im_start|>assistant`)
3. Runs a **prefill** pass — processes the full prompt in one shot, gets logits for the last position
4. **Greedily decodes** one token at a time using the KV cache until `<|im_end|>` or EOS
5. Takes the first non-empty line of output as the command
6. Calls `clear_kv_cache()` so the next invocation starts fresh

**Memory:** ~1 GB RAM (float16 weights, memory-mapped by the OS)
**Latency:** ~1–5 s per command on a modern CPU (no GPU needed)

---

## Security

Commands execute directly in your shell with your user's permissions.
- Only use on trusted inputs in controlled environments.
- Review `output/result.txt` to audit what was run.
- The model can occasionally generate incorrect commands.

---

## Adding more training examples

```bash
# Append to finetune/data/commands.jsonl:
echo '{"instruction": "your task here", "command": "the command"}' >> finetune/data/commands.jsonl

# Re-run the pipeline:
cd finetune
python prepare_data.py
python finetune.py
python export_model.py
```
