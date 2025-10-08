use anyhow::Result;
use clap::Parser;
use codex_arg0::arg0_dispatch_or_else;
use codex_common::CliConfigOverrides;
use codex_tui::Cli;

fn main() -> Result<()> {
    let arg1 = std::env::args().nth(1);
    arg0_dispatch_or_else(|codex_linux_sandbox_exe| async move {
        if arg1.as_deref() == Some("login") {
            let interactive = Cli::parse_from(["codex"]);
            codex_tui::run_main(interactive, codex_linux_sandbox_exe).await?;
        } else {
            codex_acp::run_main(codex_linux_sandbox_exe, CliConfigOverrides::default()).await?;
        }
        Ok(())
    })
}
