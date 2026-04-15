use anyhow::{Context, Result};
use std::process::Command;

/// The result of executing a shell command.
pub struct CommandResult {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub success: bool,
}

/// Execute a shell command via `sh -c` and capture stdout, stderr, and exit code.
///
/// Using `sh -c` allows pipes, redirections, globs, and shell builtins to work
/// exactly as they would in an interactive terminal session.
pub fn run_command(command: &str) -> Result<CommandResult> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .with_context(|| format!("Failed to spawn shell for command: {command}"))?;

    let exit_code = output.status.code().unwrap_or(-1);

    Ok(CommandResult {
        command: command.to_string(),
        // Use lossy decoding to handle non-UTF-8 binary output (e.g., from xxd, od)
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code,
        success: output.status.success(),
    })
}
