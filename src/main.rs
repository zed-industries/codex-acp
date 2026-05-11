use anyhow::Result;
use clap::Parser;
use codex_arg0::arg0_dispatch_or_else;
use codex_utils_cli::CliConfigOverrides;

#[derive(Parser, Debug)]
#[command(version, about = "ACP adapter for Codex")]
struct Cli {
    #[clap(flatten)]
    config_overrides: CliConfigOverrides,

    /// Do not load `$CODEX_HOME/config.toml`; auth still uses `CODEX_HOME`.
    #[arg(long = "ignore-user-config", default_value_t = false)]
    ignore_user_config: bool,
}

fn main() -> Result<()> {
    arg0_dispatch_or_else(|args| async move {
        let cli = Cli::parse();
        codex_acp::run_main_with_options(
            args.codex_linux_sandbox_exe,
            cli.config_overrides,
            codex_acp::RunOptions {
                ignore_user_config: cli.ignore_user_config,
            },
        )
        .await?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ignore_user_config_with_config_overrides() {
        let cli = Cli::parse_from([
            "codex-acp",
            "--ignore-user-config",
            "-c",
            "model=gpt-5.5",
            "-c",
            "cli_auth_credentials_store=auto",
        ]);

        assert!(cli.ignore_user_config);
        assert_eq!(
            cli.config_overrides.raw_overrides,
            vec![
                "model=gpt-5.5".to_string(),
                "cli_auth_credentials_store=auto".to_string()
            ]
        );
    }
}
