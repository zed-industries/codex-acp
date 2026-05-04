//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::ByteStreams;
use codex_core::config::{Config, ConfigOverrides};
use codex_utils_cli::CliConfigOverrides;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
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

    let config = load_config(cli_config_overrides, codex_linux_sandbox_exe.clone()).await?;

    let agent = Arc::new(codex_agent::CodexAgent::new(config, codex_linux_sandbox_exe).await?);

    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();

    agent
        .serve(ByteStreams::new(stdout, stdin))
        .await
        .map_err(|e| std::io::Error::other(format!("ACP error: {e}")))?;

    Ok(())
}

pub async fn run_auth_command(cli_config_overrides: CliConfigOverrides) -> std::io::Result<()> {
    let config = load_config(cli_config_overrides, None).await?;

    let opts = codex_login::ServerOptions::new(
        config.codex_home.to_path_buf(),
        codex_login::auth::CLIENT_ID.to_string(),
        None,
        config.cli_auth_credentials_store_mode,
    );

    run_device_code_auth(opts).await
}

async fn load_config(
    cli_config_overrides: CliConfigOverrides,
    codex_linux_sandbox_exe: Option<PathBuf>,
) -> std::io::Result<Config> {
    let cli_kv_overrides = cli_config_overrides.parse_overrides().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("error parsing -c overrides: {e}"),
        )
    })?;

    let config_overrides = ConfigOverrides {
        codex_linux_sandbox_exe,
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

    codex_login::default_client::set_default_client_residency_requirement(
        config.enforce_residency.value(),
    );

    Ok(config)
}

pub(crate) async fn run_device_code_auth(opts: codex_login::ServerOptions) -> std::io::Result<()> {
    let device_code = codex_login::request_device_code(&opts).await?;
    {
        let mut stderr = std::io::stderr().lock();
        writeln!(
            stderr,
            "Open this link in your browser and sign in to your ChatGPT account:\n{}\n",
            device_code.verification_url
        )?;
        writeln!(
            stderr,
            "Then enter this one-time code (expires in 15 minutes):\n{}\n",
            device_code.user_code
        )?;
        writeln!(stderr, "Waiting for login to complete in the browser...")?;
        stderr.flush()?;
    }

    codex_login::complete_device_code_login(opts, device_code).await
}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};
