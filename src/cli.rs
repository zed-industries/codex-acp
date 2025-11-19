use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;

#[derive(Debug, Default, Clone)]
pub struct SessionPersistCli {
    pub flag: Option<bool>,
    pub path: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

fn is_value_token(token: &OsStr) -> bool {
    let s = token.to_string_lossy();
    !s.starts_with('-') || s == "-"
}

pub fn parse_launch_args() -> Result<SessionPersistCli, ParseError> {
    let mut session_persist = SessionPersistCli::default();

    let mut args = env::args_os().skip(1).peekable();
    while let Some(arg) = args.next() {
        let arg_str = arg.to_string_lossy().into_owned();
        if arg_str == "--" {
            // Stop parsing and ignore remaining args (pass-through).
            break;
        }

        if arg_str == "-c" || arg_str == "--config" {
            let Some(_value) = args.next() else {
                return Err(ParseError("missing value for -c/--config".into()));
            };
            continue;
        }

        if arg_str
            .strip_prefix("-c")
            .is_some_and(|rest| rest.contains('='))
        {
            continue;
        }

        if arg_str.starts_with("--config=") {
            continue;
        }

        if arg_str == "--session-persist" {
            session_persist.flag = Some(true);
            if matches!(args.peek(), Some(next) if is_value_token(next)) {
                let value = args.next().unwrap();
                session_persist.path = Some(PathBuf::from(value));
            }
            continue;
        }

        if let Some(rest) = arg_str.strip_prefix("--session-persist=") {
            session_persist.flag = Some(true);
            if !rest.is_empty() {
                session_persist.path = Some(PathBuf::from(rest));
            }
            continue;
        }

        if arg_str == "--no-session-persist" {
            session_persist.flag = Some(false);
            session_persist.path = None;
            continue;
        }

        if let Some(rest) = arg_str.strip_prefix("--session-persist-") {
            // Catch accidental misspellings like --session-persist-path.
            return Err(ParseError(format!(
                "unrecognized session persistence flag '--session-persist-{rest}'"
            )));
        }

        // Ignore all other arguments; they are handled elsewhere (e.g. clap)
    }

    Ok(session_persist)
}
