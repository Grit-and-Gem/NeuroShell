use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const SYSTEM_PROMPT: &str =
    "You are a terminal command generator. Given a task in natural language, \
     output only the terminal command to accomplish it. \
     No explanation, no markdown, just the raw command.";

/// Maximum number of tokens to generate (commands are short).
const MAX_NEW_TOKENS: usize = 128;

/// Token IDs for Qwen2.5 special tokens (fallback values if not found in tokenizer).
const FALLBACK_EOS_ID: u32 = 151643; // <|endoftext|>
const FALLBACK_IM_END_ID: u32 = 151645; // <|im_end|>

pub struct InferenceEngine {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    im_end_token_id: u32,
}

impl InferenceEngine {
    /// Load the merged model from a directory containing config.json,
    /// tokenizer.json, and model.safetensors (produced by export_model.py).
    pub fn load(model_dir: &Path) -> Result<Self> {
        let device = Device::Cpu;

        // Parse model config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("config.json not found in '{}'", model_dir.display()))?;
        let config: Qwen2Config = serde_json::from_str(&config_str)
            .context("Failed to parse config.json — is this a Qwen2 model directory?")?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {e}"))?;

        // Locate safetensors weight files
        let weight_files = find_safetensors(model_dir)?;
        println!(
            "  Loading {} weight file(s) from {}",
            weight_files.len(),
            model_dir.display()
        );

        // Memory-map model weights (fast, does not copy into RAM upfront)
        // SAFETY: the files must not be modified while the program is running.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &device)
                .context("Failed to memory-map model weights")?
        };

        let model = Qwen2Model::new(&config, vb)
            .context("Failed to build Qwen2 model from weights")?;

        // Resolve special token IDs used to stop generation
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(FALLBACK_EOS_ID);
        let im_end_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .unwrap_or(FALLBACK_IM_END_ID);

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
            im_end_token_id,
        })
    }

    /// Generate a terminal command for the given natural language task.
    ///
    /// Internally:
    /// 1. Wraps the task in Qwen2.5's ChatML prompt format.
    /// 2. Runs a prefill pass over the full prompt.
    /// 3. Greedily decodes one token at a time until EOS or newline.
    /// 4. Returns the first non-empty line as the command.
    pub fn generate_command(&mut self, task: &str) -> Result<String> {
        // Build ChatML prompt — add_generation_prompt leaves the assistant turn open
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{task}<|im_end|>\n\
             <|im_start|>assistant\n",
            system = SYSTEM_PROMPT,
            task = task,
        );

        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        // --- Prefill phase: process the entire prompt in one forward pass ---
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        // ModelForCausalLM.forward returns logits for the LAST position only: [1, 1, vocab_size]
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.squeeze(0)?; // [vocab_size]
        let mut current_tok = logits.argmax(0)?.to_scalar::<u32>()?;

        // --- Decode phase: one token at a time using the KV cache ---
        let mut output_tokens: Vec<u32> = Vec::with_capacity(64);
        let mut seqlen_offset = prompt_len;

        for _ in 0..MAX_NEW_TOKENS {
            // Stop on end-of-sequence or end-of-turn tokens
            if current_tok == self.eos_token_id || current_tok == self.im_end_token_id {
                break;
            }

            output_tokens.push(current_tok);

            // Feed the single new token; seqlen_offset tells the model how many
            // tokens came before (for positional encoding + KV cache indexing)
            let input = Tensor::new(&[current_tok], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?; // [vocab_size]
            current_tok = logits.argmax(0)?.to_scalar::<u32>()?;
            seqlen_offset += 1;
        }

        // Clear KV cache so the next call starts fresh
        self.model.clear_kv_cache();

        // Decode tokens back to text
        let raw = self
            .tokenizer
            .decode(&output_tokens, true) // skip_special_tokens = true
            .map_err(|e| anyhow::anyhow!("Token decoding failed: {e}"))?;

        // Take only the first non-empty line — the model occasionally adds
        // a comment after the command despite training; we discard it here.
        let command = raw
            .lines()
            .map(|l| l.trim())
            .find(|l| !l.is_empty())
            .unwrap_or(raw.trim())
            .to_string();

        if command.is_empty() {
            anyhow::bail!("Model produced an empty command for task: '{task}'");
        }

        Ok(command)
    }
}

/// Find model.safetensors in `model_dir`, or all sharded model-*.safetensors files.
fn find_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>> {
    // Single file (most common after export_model.py)
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    // Sharded files: model-00001-of-00002.safetensors, etc.
    let mut files: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("Cannot read model directory: {}", model_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().map_or(false, |ext| ext == "safetensors")
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.starts_with("model"))
        })
        .collect();

    files.sort(); // ensure shards are loaded in order

    if files.is_empty() {
        anyhow::bail!(
            "No model.safetensors file(s) found in '{}'. \
             Run 'python export_model.py' first.",
            model_dir.display()
        );
    }

    Ok(files)
}
