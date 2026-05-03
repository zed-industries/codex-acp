//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::ByteStreams;
use codex_core::config::{Config, ConfigOverrides};
use codex_utils_cli::CliConfigOverrides;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

mod codex_agent;
mod thread;

const DISABLE_COMPUTER_USE_PLUGIN_OVERRIDE: &str =
    "plugins.computer-use@openai-bundled.enabled=false";

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

    let cli_config_overrides = acp_cli_config_overrides(cli_config_overrides);

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

    let agent = Arc::new(codex_agent::CodexAgent::new(config, codex_linux_sandbox_exe).await?);

    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();

    agent
        .serve(ByteStreams::new(stdout, stdin))
        .await
        .map_err(|e| std::io::Error::other(format!("ACP error: {e}")))?;

    Ok(())
}

fn acp_cli_config_overrides(mut cli_config_overrides: CliConfigOverrides) -> CliConfigOverrides {
    // Computer Use's bundled macOS client has parent launch constraints that
    // reject this third-party ACP process, so ACP must not load that plugin.
    cli_config_overrides
        .raw_overrides
        .push(DISABLE_COMPUTER_USE_PLUGIN_OVERRIDE.to_string());
    cli_config_overrides
}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acp_cli_config_overrides_disables_computer_use_plugin() {
        let overrides = acp_cli_config_overrides(CliConfigOverrides {
            raw_overrides: vec!["model=gpt-5.4".to_string()],
        });

        let parsed = overrides.parse_overrides().expect("parse overrides");
        assert!(parsed.iter().any(|(key, value)| {
            key == "plugins.computer-use@openai-bundled.enabled" && value.as_bool() == Some(false)
        }));
        assert!(
            parsed
                .iter()
                .any(|(key, value)| key == "model" && value.as_str() == Some("gpt-5.4"))
        );
    }
}
