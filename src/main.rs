use anyhow::Result;
use clap::Parser;
use codex_arg0::arg0_dispatch_or_else;
use codex_utils_cli::CliConfigOverrides;

fn main() -> Result<()> {
    arg0_dispatch_or_else(|codex_linux_sandbox_exe| async move {
        let cli_config_overrides = CliConfigOverrides::parse();
        codex_acp::run_main(codex_linux_sandbox_exe, cli_config_overrides).await?;
        Ok(())
    })
}
