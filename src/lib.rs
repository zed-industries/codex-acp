//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use codex_common::CliConfigOverrides;
use codex_core::config::{Config, ConfigOverrides};
use std::io::Result as IoResult;
use std::path::PathBuf;
use std::sync::Arc;
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
            // Create a channel for notifications
            let (notification_tx, mut notification_rx) = tokio::sync::mpsc::unbounded_channel();

            // Create our Agent implementation with notification channel
            let agent = codex_agent::CodexAgent::new(Arc::new(config), notification_tx);

            // Use tokio-util to adapt between tokio and futures traits
            use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

            let stdin = tokio::io::stdin().compat();
            let stdout = tokio::io::stdout().compat_write();

            // Create the ACP connection
            use agent_client_protocol::AgentSideConnection;
            use futures::future::LocalBoxFuture;

            let (client_handle, io_task) = AgentSideConnection::new(
                agent,
                stdout,
                stdin,
                |fut: LocalBoxFuture<'static, ()>| {
                    // Use tokio::task::spawn_local for !Send futures
                    tokio::task::spawn_local(fut);
                },
            );

            // Spawn a task to forward notifications from the channel to the client
            let client = std::sync::Arc::new(client_handle);
            tokio::task::spawn_local(async move {
                while let Some(notification) = notification_rx.recv().await {
                    use agent_client_protocol::Client;
                    if let Err(e) = client.session_notification(notification).await {
                        tracing::error!("Failed to send session notification: {:?}", e);
                    }
                }
            });

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
