use anyhow::Result;
use codex_arg0::arg0_dispatch_or_else;
use codex_common::CliConfigOverrides;

fn main() -> Result<()> {
    arg0_dispatch_or_else(|codex_linux_sandbox_exe| async move {
        codex_acp::run_main(codex_linux_sandbox_exe, CliConfigOverrides::default()).await?;
        Ok(())
    })
}
