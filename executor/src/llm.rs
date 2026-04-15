use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize)]
struct GenerateRequest<'a> {
    task: &'a str,
}

#[derive(Deserialize)]
struct GenerateResponse {
    command: String,
}

/// Send a natural language task to the inference server and return the generated command.
pub async fn get_command(client: &Client, base_url: &str, task: &str) -> Result<String> {
    let url = format!("{}/generate", base_url.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .json(&GenerateRequest { task })
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .with_context(|| {
            format!(
                "Failed to connect to inference server at {url}. \
                 Is the server running? Start it with: \
                 uvicorn inference_server:app --port 8000"
            )
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Inference server returned HTTP {status}: {body}");
    }

    let data: GenerateResponse = resp
        .json()
        .await
        .context("Failed to parse inference server response as JSON")?;

    let command = data.command.trim().to_string();
    if command.is_empty() {
        bail!("Inference server returned an empty command");
    }

    Ok(command)
}

/// Poll the inference server health endpoint until it responds or timeout is reached.
pub async fn wait_for_server(client: &Client, base_url: &str, timeout_secs: u64) -> Result<()> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    loop {
        match client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                return Ok(());
            }
            _ => {}
        }

        if start.elapsed() >= timeout {
            bail!(
                "Inference server at {base_url} did not become healthy within {timeout_secs}s"
            );
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
        eprint!(".");
    }
}
