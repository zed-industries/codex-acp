//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::ByteStreams;
use codex_config::LoaderOverrides;
use codex_core::config::{Config, ConfigBuilder, ConfigOverrides};
use codex_utils_cli::CliConfigOverrides;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

mod codex_agent;
mod thread;

#[derive(Debug, Clone, Copy, Default)]
pub struct RunOptions {
    /// Do not load `$CODEX_HOME/config.toml`; auth still uses `CODEX_HOME`.
    pub ignore_user_config: bool,
}

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
    run_main_with_options(
        codex_linux_sandbox_exe,
        cli_config_overrides,
        RunOptions::default(),
    )
    .await
}

/// Run the Codex ACP agent with explicit adapter options.
///
/// # Errors
///
/// If unable to parse the config or start the program.
pub async fn run_main_with_options(
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
    options: RunOptions,
) -> std::io::Result<()> {
    // Install a simple subscriber so `tracing` output is visible.
    // Users can control the log level with `RUST_LOG`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = load_config(
        codex_linux_sandbox_exe.clone(),
        cli_config_overrides,
        options,
    )
    .await?;
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

async fn load_config(
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
    options: RunOptions,
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
    let loader_overrides = LoaderOverrides {
        ignore_user_config: options.ignore_user_config,
        ..Default::default()
    };

    ConfigBuilder::default()
        .cli_overrides(cli_kv_overrides)
        .harness_overrides(config_overrides)
        .loader_overrides(loader_overrides)
        .build()
        .await
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("error loading config: {e}"),
            )
        })
}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use uuid::Uuid;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> std::io::Result<Self> {
            let path = std::env::temp_dir().join(format!("codex-acp-{}", Uuid::new_v4()));
            fs::create_dir_all(&path)?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            drop(fs::remove_dir_all(&self.0));
        }
    }

    async fn load_config_for_test(
        codex_home: PathBuf,
        raw_overrides: Vec<&str>,
        options: RunOptions,
    ) -> std::io::Result<Config> {
        let cli_config_overrides = CliConfigOverrides {
            raw_overrides: raw_overrides.into_iter().map(str::to_string).collect(),
        };
        let cli_kv_overrides = cli_config_overrides.parse_overrides().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("error parsing -c overrides: {e}"),
            )
        })?;
        let config_overrides = ConfigOverrides {
            cwd: Some(codex_home.clone()),
            ..ConfigOverrides::default()
        };
        let mut loader_overrides = LoaderOverrides::without_managed_config_for_tests();
        loader_overrides.ignore_user_config = options.ignore_user_config;

        ConfigBuilder::default()
            .codex_home(codex_home)
            .cli_overrides(cli_kv_overrides)
            .harness_overrides(config_overrides)
            .loader_overrides(loader_overrides)
            .build()
            .await
    }

    #[tokio::test]
    async fn ignore_user_config_ignores_invalid_user_config_but_applies_cli_overrides()
    -> std::io::Result<()> {
        let codex_home = TestDir::new()?;
        fs::write(
            codex_home.path().join("config.toml"),
            "model = \"from-user-config\"\ninvalid = [",
        )?;

        let config = load_config_for_test(
            codex_home.path().to_path_buf(),
            vec!["model=\"from-cli\""],
            RunOptions {
                ignore_user_config: true,
            },
        )
        .await?;

        assert_eq!(config.codex_home.as_path(), codex_home.path());
        assert_eq!(config.model.as_deref(), Some("from-cli"));
        Ok(())
    }

    #[tokio::test]
    async fn ignore_user_config_does_not_load_user_notify_or_mcp_servers() -> std::io::Result<()> {
        let codex_home = TestDir::new()?;
        fs::write(
            codex_home.path().join("config.toml"),
            r#"
notify = ["notify-send", "Codex"]

[mcp_servers.user_config_server]
command = "node"
args = ["server.js"]
"#,
        )?;

        let config = load_config_for_test(
            codex_home.path().to_path_buf(),
            vec![],
            RunOptions {
                ignore_user_config: true,
            },
        )
        .await?;

        assert_eq!(config.notify, None);
        assert!(!config.mcp_servers.get().contains_key("user_config_server"));
        Ok(())
    }
}
