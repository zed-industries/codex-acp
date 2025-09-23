//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use codex_common::CliConfigOverrides;
use codex_core::config::{Config, ConfigOverrides};
use std::io::Result as IoResult;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

mod codex_agent;

/// Run the Codex ACP agent.
///
/// This sets up an ACP agent that communicates over stdio, bridging
/// the ACP protocol with the existing codex-rs infrastructure.
pub async fn run_main(
    _codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
) -> IoResult<()> {
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

    let config = Config::load_with_cli_overrides(cli_kv_overrides, ConfigOverrides::default())
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("error loading config: {e}"),
            )
        })?;

    // Create a local task set for running !Send futures
    let local = tokio::task::LocalSet::new();

    local
        .run_until(async move {
            use agent_client_protocol::AgentSideConnection;
            use futures::future::LocalBoxFuture;
            use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

            // Create our Agent implementation with notification channel

            let stdin = tokio::io::stdin().compat();
            let stdout = tokio::io::stdout().compat_write();

            // Create the ACP connection

            let (_, io_task) = AgentSideConnection::new(
                move |client| codex_agent::CodexAgent::new(config, client),
                stdout,
                stdin,
                |fut: LocalBoxFuture<'static, ()>| {
                    // Use tokio::task::spawn_local for !Send futures
                    tokio::task::spawn_local(fut);
                },
            );

            // Run the I/O task to handle the actual communication
            io_task
                .await
                .map_err(|e| std::io::Error::other(format!("ACP I/O error: {e}")))
        })
        .await?;

    Ok(())
}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};
