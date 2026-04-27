//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::ByteStreams;
use codex_core::config::{Config, ConfigOverrides};
use codex_utils_cli::CliConfigOverrides;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

mod codex_agent;
mod thread;

/// Run the Codex ACP agent.
///
/// This sets up an ACP agent that communicates over stdio, bridging
/// the ACP protocol with the existing codex-rs infrastructure.
///
/// # Errors
///
/// If unable to parse the config or start the program.
pub async fn run_main(
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
) -> std::io::Result<()> {
    // Install a simple subscriber so `tracing` output is visible.
    // Users can control the log level with `RUST_LOG`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse CLI overrides and load configuration
    let cli_kv_overrides = cli_config_overrides.parse_overrides().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("error parsing -c overrides: {e}"),
        )
    })?;

    let config_overrides = ConfigOverrides {
        codex_linux_sandbox_exe: codex_linux_sandbox_exe.clone(),
        ..ConfigOverrides::default()
    };

    let config =
        Config::load_with_cli_overrides_and_harness_overrides(cli_kv_overrides, config_overrides)
            .await
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("error loading config: {e}"),
                )
            })?;
    // Apply residency requirement so the HTTP client sends the
    // x-openai-internal-codex-residency header on all requests.
    codex_login::default_client::set_default_client_residency_requirement(
        config.enforce_residency.value(),
    );

    let agent = Arc::new(codex_agent::CodexAgent::new(
        config,
        codex_linux_sandbox_exe,
    )?);

    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();

    enum ServeOutcome {
        Result(agent_client_protocol::Result<()>),
        Signal,
    }

    let serve_outcome = {
        let serve = agent.clone().serve(ByteStreams::new(stdout, stdin));
        tokio::pin!(serve);

        #[cfg(unix)]
        {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

            tokio::select! {
                result = &mut serve => ServeOutcome::Result(result),
                _ = tokio::signal::ctrl_c() => ServeOutcome::Signal,
                _ = terminate.recv() => ServeOutcome::Signal,
            }
        }

        #[cfg(not(unix))]
        {
            tokio::select! {
                result = &mut serve => ServeOutcome::Result(result),
                _ = tokio::signal::ctrl_c() => ServeOutcome::Signal,
            }
        }
    };

    agent.shutdown_sessions().await;

    match serve_outcome {
        ServeOutcome::Result(result) => {
            result.map_err(|e| std::io::Error::other(format!("ACP error: {e}")))?;
        }
        ServeOutcome::Signal => {
            terminate_descendant_processes();
            std::process::exit(0);
        }
    }

    Ok(())
}

#[cfg(unix)]
fn terminate_descendant_processes() {
    let current_pid = std::process::id() as i32;
    let output = match Command::new("ps")
        .args(["-axo", "pid=,ppid=,pgid="])
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            tracing::warn!("Failed to list child processes during shutdown: {err}");
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut children_by_parent = HashMap::<i32, Vec<(i32, i32)>>::new();
    let mut pgid_by_pid = HashMap::<i32, i32>::new();

    for line in stdout.lines() {
        let fields = line
            .split_whitespace()
            .filter_map(|field| field.parse::<i32>().ok())
            .collect::<Vec<_>>();

        let [pid, ppid, pgid] = fields.as_slice() else {
            continue;
        };

        children_by_parent
            .entry(*ppid)
            .or_default()
            .push((*pid, *pgid));
        pgid_by_pid.insert(*pid, *pgid);
    }

    let current_pgid = pgid_by_pid.get(&current_pid).copied();
    let mut descendants = Vec::<(i32, i32)>::new();
    let mut stack = children_by_parent
        .get(&current_pid)
        .cloned()
        .unwrap_or_default();

    while let Some((pid, pgid)) = stack.pop() {
        descendants.push((pid, pgid));
        if let Some(children) = children_by_parent.get(&pid) {
            stack.extend(children.iter().copied());
        }
    }

    if descendants.is_empty() {
        return;
    }

    let process_groups = descendants
        .iter()
        .map(|(_, pgid)| *pgid)
        .filter(|pgid| *pgid > 1 && Some(*pgid) != current_pgid)
        .collect::<HashSet<_>>();

    for pgid in &process_groups {
        kill_process("-TERM", &format!("-{pgid}"));
    }

    for (pid, _) in &descendants {
        kill_process("-TERM", &pid.to_string());
    }

    std::thread::sleep(Duration::from_secs(1));

    for pgid in &process_groups {
        kill_process("-KILL", &format!("-{pgid}"));
    }

    for (pid, _) in descendants {
        kill_process("-KILL", &pid.to_string());
    }
}

#[cfg(unix)]
fn kill_process(signal: &str, target: &str) {
    drop(
        Command::new("kill")
            .args([signal, target])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
    );
}

#[cfg(not(unix))]
fn terminate_descendant_processes() {}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};
