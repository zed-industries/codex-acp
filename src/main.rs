use anyhow::Result;
use clap::Parser;
use codex_arg0::arg0_dispatch_or_else;
use codex_utils_cli::CliConfigOverrides;
use std::ffi::OsString;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandMode {
    Acp,
    Auth,
}

fn command_mode_and_args_from(mut args: Vec<OsString>) -> (CommandMode, Vec<OsString>) {
    let mode = if matches!(args.get(1), Some(arg) if arg == "/auth") {
        args.remove(1);
        CommandMode::Auth
    } else {
        CommandMode::Acp
    };

    (mode, args)
}

fn command_mode_and_args() -> (CommandMode, Vec<OsString>) {
    command_mode_and_args_from(std::env::args_os().collect())
}

fn main() -> Result<()> {
    let (command_mode, filtered_args) = command_mode_and_args();

    arg0_dispatch_or_else(|args| async move {
        let cli_config_overrides = CliConfigOverrides::parse_from(filtered_args);

        match command_mode {
            CommandMode::Acp => {
                codex_acp::run_main(args.codex_linux_sandbox_exe, cli_config_overrides).await?;
            }
            CommandMode::Auth => {
                codex_acp::run_auth_command(cli_config_overrides).await?;
            }
        }

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::{CommandMode, command_mode_and_args_from};
    use std::ffi::OsString;

    #[test]
    fn auth_mode_only_matches_first_positional_argument() {
        let (mode, args) = command_mode_and_args_from(vec![
            OsString::from("codex-acp"),
            OsString::from("--verbose"),
            OsString::from("/auth"),
        ]);

        assert_eq!(mode, CommandMode::Acp);
        assert_eq!(
            args,
            vec![
                OsString::from("codex-acp"),
                OsString::from("--verbose"),
                OsString::from("/auth"),
            ]
        );
    }

    #[test]
    fn auth_mode_strips_explicit_subcommand() {
        let (mode, args) = command_mode_and_args_from(vec![
            OsString::from("codex-acp"),
            OsString::from("/auth"),
            OsString::from("--config"),
            OsString::from("settings.toml"),
        ]);

        assert_eq!(mode, CommandMode::Auth);
        assert_eq!(
            args,
            vec![
                OsString::from("codex-acp"),
                OsString::from("--config"),
                OsString::from("settings.toml"),
            ]
        );
    }
}
