mod executor;
mod llm;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use reqwest::Client;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "neuroshell",
    about = "Natural language terminal command executor powered by a fine-tuned LLM",
    long_about = "Reads a natural language task from an input file, sends it to a \
                  fine-tuned LLM inference server to generate a shell command, \
                  executes that command, and writes the output to a file."
)]
struct Args {
    /// Path to the input file containing the natural language task
    #[arg(short = 'i', long, default_value = "../input/task.txt")]
    input: PathBuf,

    /// Path to the output file where execution results will be written
    #[arg(short = 'o', long, default_value = "../output/result.txt")]
    output: PathBuf,

    /// URL of the LLM inference server
    #[arg(long, default_value = "http://localhost:8000", env = "NEUROSHELL_MODEL_URL")]
    model_url: String,

    /// Watch the input file for changes and automatically re-execute
    #[arg(short = 'w', long)]
    watch: bool,

    /// Wait for the inference server to become ready (timeout in seconds, 0 to skip)
    #[arg(long, default_value_t = 0)]
    wait_for_server: u64,
}

/// Read the input file and return its trimmed contents.
fn read_task(input: &PathBuf) -> Result<String> {
    let content = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read input file: {}", input.display()))?;
    let task = content.trim().to_string();
    if task.is_empty() {
        anyhow::bail!("Input file is empty: {}", input.display());
    }
    Ok(task)
}

/// Write the formatted execution result to the output file.
fn write_output(output: &PathBuf, result: &executor::CommandResult, task: &str) -> Result<()> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
        }
    }

    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let separator = "=".repeat(50);
    let divider = "-".repeat(50);

    let mut content = format!(
        "{separator}\n\
         NeuroShell Execution Result\n\
         {separator}\n\
         Task:    {task}\n\
         Command: {cmd}\n\
         {divider}\n",
        separator = separator,
        divider = divider,
        task = task,
        cmd = result.command,
    );

    if !result.stdout.is_empty() {
        content.push_str(&result.stdout);
        if !result.stdout.ends_with('\n') {
            content.push('\n');
        }
    } else {
        content.push_str("(no output)\n");
    }

    if !result.stderr.is_empty() {
        content.push_str(&format!("{divider}\nStderr:\n{}", result.stderr, divider = divider));
        if !result.stderr.ends_with('\n') {
            content.push('\n');
        }
    }

    content.push_str(&format!(
        "{divider}\n\
         Exit Code: {code}\n\
         Timestamp: {ts}\n\
         {separator}\n",
        divider = divider,
        separator = separator,
        code = result.exit_code,
        ts = timestamp,
    ));

    std::fs::write(output, &content)
        .with_context(|| format!("Failed to write output file: {}", output.display()))?;

    Ok(())
}

/// Run the full pipeline: read → LLM → execute → write output.
async fn run_once(client: &Client, args: &Args) -> Result<()> {
    let task = read_task(&args.input)?;
    println!("Task: {task}");

    print!("Generating command...");
    let command = llm::get_command(client, &args.model_url, &task).await?;
    println!(" {command}");

    print!("Executing...");
    let result = executor::run_command(&command)?;
    println!(" done (exit {})", result.exit_code);

    write_output(&args.output, &result, &task)?;
    println!("Output written to: {}", args.output.display());

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let client = Client::new();

    // Optionally wait for the inference server to be ready
    if args.wait_for_server > 0 {
        print!("Waiting for inference server");
        llm::wait_for_server(&client, &args.model_url, args.wait_for_server).await?;
        println!(" ready!");
    }

    if !args.watch {
        // One-shot mode
        run_once(&client, &args).await?;
        return Ok(());
    }

    // Watch mode: re-run whenever the input file changes
    println!("Watching {} for changes (Ctrl+C to stop)...", args.input.display());

    // Run once immediately on startup
    if args.input.exists() {
        if let Err(e) = run_once(&client, &args).await {
            eprintln!("Error: {e}");
        }
    }

    let running = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();

    let mut debouncer = new_debouncer(Duration::from_millis(500), tx)
        .context("Failed to create file watcher")?;

    debouncer
        .watcher()
        .watch(&args.input, RecursiveMode::NonRecursive)
        .with_context(|| format!("Failed to watch file: {}", args.input.display()))?;

    for events in rx {
        match events {
            Ok(_) => {
                // Skip if a previous run is still in progress
                if running
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
                {
                    println!("(skipping: previous execution still running)");
                    continue;
                }

                println!("\n--- Input file changed, re-running ---");
                let client_ref = &client;
                let args_ref = &args;

                if let Err(e) = run_once(client_ref, args_ref).await {
                    eprintln!("Error: {e}");
                }

                running.store(false, Ordering::SeqCst);
            }
            Err(e) => {
                eprintln!("Watcher error: {e}");
            }
        }
    }

    Ok(())
}
