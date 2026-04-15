# NeuroShell

Translate natural language into terminal commands using a fine-tuned Qwen2.5-0.5B model, then execute them automatically via a Rust app.

```
User writes: "show disk usage of home directory"
        ↓
NeuroShell generates: du -sh ~
        ↓
NeuroShell executes and writes output to result.txt
```

---

## Architecture

```
input/task.txt  →  neuroshell (Rust)  →  inference server (Python/FastAPI)
                                                    ↓
                                          Qwen2.5-0.5B + LoRA adapter
                                                    ↓
                         output/result.txt  ←  sh -c <command>
```

---

## Prerequisites

- Python 3.10+
- Rust 1.75+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- CUDA GPU recommended for fine-tuning (CPU works but is slow)
- ~2 GB disk space for the base model

---

## Setup

### 1. Install Python dependencies

```bash
cd finetune
pip install -r requirements.txt
```

### 2. Prepare training data

Validates the dataset and converts it to ChatML format:

```bash
python prepare_data.py
# Writes: data/train.jsonl
```

To validate only:

```bash
python prepare_data.py --validate
```

### 3. Fine-tune the model

Standard training (FP32/FP16, CPU or GPU):

```bash
python finetune.py
```

QLoRA (4-bit quantization, requires CUDA):

```bash
python finetune.py --use-4bit
```

Optional arguments:

```
--model-name        HuggingFace model ID (default: Qwen/Qwen2.5-0.5B)
--num-epochs        Training epochs (default: 5)
--batch-size        Batch size per device (default: 4)
--learning-rate     Learning rate (default: 2e-4)
--output-dir        Where to save the LoRA adapter (default: output/lora-adapter)
```

Fine-tuning saves the LoRA adapter to `finetune/output/lora-adapter/`.

### 4. Start the inference server

```bash
cd finetune
uvicorn inference_server:app --host 0.0.0.0 --port 8000 --workers 1
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `NEUROSHELL_BASE_MODEL` | `Qwen/Qwen2.5-0.5B` | Base model to load |
| `NEUROSHELL_ADAPTER_PATH` | `output/lora-adapter` | Path to LoRA adapter |

Test the server:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"task": "list all files including hidden ones"}'
# → {"command": "ls -la"}

curl http://localhost:8000/health
# → {"status": "ok", "model": "Qwen/Qwen2.5-0.5B", ...}
```

### 5. Build the Rust executor

```bash
cd executor
cargo build --release
# Binary: executor/target/release/neuroshell
```

---

## Usage

### One-shot mode

Write a task to the input file and run neuroshell:

```bash
echo "show disk usage of the current directory" > input/task.txt

./executor/target/release/neuroshell \
  --input input/task.txt \
  --output output/result.txt
```

Result in `output/result.txt`:

```
==================================================
NeuroShell Execution Result
==================================================
Task:    show disk usage of the current directory
Command: du -sh .
--------------------------------------------------
1.2G    .
--------------------------------------------------
Exit Code: 0
Timestamp: 2026-04-15 10:23:45 UTC
==================================================
```

### Watch mode

Automatically re-execute whenever the input file changes:

```bash
./executor/target/release/neuroshell \
  --input input/task.txt \
  --output output/result.txt \
  --watch
```

Then in another terminal:

```bash
echo "list running processes sorted by memory" > input/task.txt
# neuroshell detects the change, generates a command, executes it, updates output/result.txt
```

### Wait for server

Poll the inference server until it is ready (useful in scripts):

```bash
./executor/target/release/neuroshell \
  --input input/task.txt \
  --output output/result.txt \
  --wait-for-server 60
```

### CLI reference

```
Options:
  -i, --input <FILE>           Input file with natural language task
                               [default: ../input/task.txt]
  -o, --output <FILE>          Output file for results
                               [default: ../output/result.txt]
      --model-url <URL>        Inference server URL
                               [default: http://localhost:8000]
                               [env: NEUROSHELL_MODEL_URL]
  -w, --watch                  Watch input file and auto-execute on changes
      --wait-for-server <SECS> Poll server health until ready (0 = skip)
                               [default: 0]
  -h, --help                   Print help
```

---

## Dataset

The training dataset at `finetune/data/commands.jsonl` contains 220+ natural language → command pairs covering:

- File operations (`ls`, `cp`, `mv`, `rm`, `find`, `chmod`)
- Text processing (`grep`, `sed`, `awk`, `sort`, `cut`, `wc`)
- Process management (`ps`, `kill`, `top`, `lsof`)
- Networking (`curl`, `wget`, `ping`, `ss`, `ssh`, `scp`)
- Archives (`tar`, `gzip`, `zip`)
- Git (`status`, `log`, `diff`, `branch`, `stash`)
- Docker (`run`, `ps`, `build`, `exec`, `logs`)
- System info (`df`, `du`, `free`, `uname`, `uptime`)
- Package managers (`apt`, `pip`, `npm`, `cargo`)

To add more examples, append to `commands.jsonl` and re-run `prepare_data.py` + `finetune.py`.

---

## Security

**Commands are executed directly in your shell without sandboxing.**

- Only run NeuroShell in controlled environments with trusted inputs.
- Review generated commands before acting on sensitive systems.
- The LLM may occasionally generate incorrect or unexpected commands.

---

## Project Structure

```
NeuroShell/
├── finetune/
│   ├── data/
│   │   ├── commands.jsonl      # Raw training pairs (NL → command)
│   │   └── train.jsonl         # Generated: ChatML-formatted training data
│   ├── finetune.py             # LoRA fine-tuning script
│   ├── inference_server.py     # FastAPI inference server
│   ├── prepare_data.py         # Data validation and ChatML conversion
│   └── requirements.txt
├── executor/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             # CLI, orchestration, watch mode
│       ├── llm.rs              # HTTP client for inference server
│       └── executor.rs         # Shell command execution
├── input/                      # Drop task.txt here
└── output/                     # result.txt is written here
```
