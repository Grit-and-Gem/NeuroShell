mod executor;
mod inference;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "neuroshell",
    about = "Natural language terminal command executor (self-contained, no server required)",
    long_about = "\
Reads a natural language task from an input file, uses a fine-tuned Qwen2 model\n\
to generate a shell command, executes it, and writes the result to an output file.\n\
\n\
The model runs entirely inside this binary — no Python, no network, no background server."
)]
struct Args {
    /// Directory containing the merged model (config.json + tokenizer.json + model.safetensors).
    /// Produced by: python finetune/export_model.py
    #[arg(
        short = 'm',
        long,
        default_value = "../finetune/output/merged-model",
        env = "NEUROSHELL_MODEL"
    )]
    model: PathBuf,

    /// Input file containing the natural language task (plain text)
    #[arg(short = 'i', long, default_value = "../input/task.txt")]
    input: PathBuf,

    /// Output file where the generated command and its output will be written
    #[arg(short = 'o', long, default_value = "../output/result.txt")]
    output: PathBuf,

    /// Watch the input file for changes and automatically re-execute
    #[arg(short = 'w', long)]
    watch: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.model.exists() {
        anyhow::bail!(
            "Model directory not found: '{}'\n\
             Run 'python finetune/export_model.py' to generate it.",
            args.model.display()
        );
    }

    println!("Loading model from: {}", args.model.display());
    let mut engine = inference::InferenceEngine::load(&args.model)
        .context("Failed to load inference engine")?;
    println!("Model ready.\n");

    if !args.watch {
        run_once(&mut engine, &args)?;
        return Ok(());
    }

    // Watch mode: re-run every time the input file is modified
    println!(
        "Watching '{}' for changes (Ctrl+C to stop)...\n",
        args.input.display()
    );

    // Run once immediately if the input file already exists
    if args.input.exists() {
        if let Err(e) = run_once(&mut engine, &args) {
            eprintln!("Error: {e:#}");
        }
        println!();
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let mut debouncer =
        new_debouncer(Duration::from_millis(500), tx).context("Failed to create file watcher")?;
    debouncer
        .watcher()
        .watch(&args.input, RecursiveMode::NonRecursive)
        .with_context(|| format!("Cannot watch '{}'", args.input.display()))?;

    for event_result in rx {
        match event_result {
            Ok(_) => {
                println!("--- Input changed, re-running ---");
                if let Err(e) = run_once(&mut engine, &args) {
                    eprintln!("Error: {e:#}");
                }
                println!();
            }
            Err(e) => eprintln!("Watcher error: {e}"),
        }
    }

    Ok(())
}

/// The full pipeline: read → generate → execute → write.
fn run_once(engine: &mut inference::InferenceEngine, args: &Args) -> Result<()> {
    let task = read_task(&args.input)?;
    println!("Task:     {task}");

    let command = engine.generate_command(&task)?;
    println!("Command:  {command}");

    print!("Running...");
    let result = executor::run_command(&command)?;
    println!("  exit {}", result.exit_code);

    write_output(&args.output, &result, &task)?;
    println!("Output →  {}", args.output.display());

    Ok(())
}

/// Read and trim the input file. Returns an error if the file is empty.
fn read_task(input: &PathBuf) -> Result<String> {
    let raw = std::fs::read_to_string(input)
        .with_context(|| format!("Cannot read input file: '{}'", input.display()))?;
    let task = raw.trim().to_string();
    if task.is_empty() {
        anyhow::bail!(
            "Input file is empty: '{}'\nWrite a natural language task into it.",
            input.display()
        );
    }
    Ok(task)
}

/// Write the formatted execution result to the output file (always overwrites).
fn write_output(
    output: &PathBuf,
    result: &executor::CommandResult,
    task: &str,
) -> Result<()> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Cannot create output directory: '{}'", parent.display()))?;
        }
    }

    let sep = "=".repeat(50);
    let div = "-".repeat(50);
    let ts = Utc::now().format("%Y-%m-%d %H:%M:%S UTC");

    let mut out = format!(
        "{sep}\n\
         NeuroShell Execution Result\n\
         {sep}\n\
         Task:    {task}\n\
         Command: {cmd}\n\
         {div}\n",
        sep = sep,
        div = div,
        task = task,
        cmd = result.command,
    );

    if result.stdout.is_empty() {
        out.push_str("(no output)\n");
    } else {
        out.push_str(&result.stdout);
        if !result.stdout.ends_with('\n') {
            out.push('\n');
        }
    }

    if !result.stderr.is_empty() {
        out.push_str(&format!("{div}\nStderr:\n"));
        out.push_str(&result.stderr);
        if !result.stderr.ends_with('\n') {
            out.push('\n');
        }
    }

    out.push_str(&format!(
        "{div}\n\
         Exit Code: {code}\n\
         Timestamp: {ts}\n\
         {sep}\n",
        div = div,
        sep = sep,
        code = result.exit_code,
        ts = ts,
    ));

    std::fs::write(output, &out)
        .with_context(|| format!("Cannot write output file: '{}'", output.display()))?;

    Ok(())
}
