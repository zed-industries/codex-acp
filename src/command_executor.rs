use std::sync::OnceLock;
use std::time::{Duration, Instant};

use agent_client_protocol::{
    Client as _, CreateTerminalRequest, EnvVariable, KillTerminalCommandRequest,
    ReleaseTerminalRequest, SessionId, TerminalOutputRequest, WaitForTerminalExitRequest,
};
use tokio::sync::{mpsc, oneshot};

use crate::ACP_CLIENT;

static COMMAND_DISPATCHER: OnceLock<CommandDispatcher> = OnceLock::new();

pub fn init_command_dispatcher() {
    COMMAND_DISPATCHER.get_or_init(CommandDispatcher::new);
}

pub fn command_dispatcher() -> Option<&'static CommandDispatcher> {
    COMMAND_DISPATCHER.get()
}

pub struct ShellExecutionRequest {
    pub session_id: SessionId,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: std::path::PathBuf,
    pub env: Vec<EnvVariable>,
    pub timeout_ms: Option<u64>,
}

pub struct ShellExecutionResult {
    pub stdout: String,
    pub exit_code: i32,
    pub duration: Duration,
    pub timed_out: bool,
}

type ResponseSender =
    oneshot::Sender<Result<ShellExecutionResult, String>>;

pub struct CommandDispatcher {
    sender: mpsc::UnboundedSender<(ShellExecutionRequest, ResponseSender)>,
}

impl CommandDispatcher {
    pub fn new() -> Self {
        let (sender, mut receiver) =
            mpsc::unbounded_channel::<(ShellExecutionRequest, ResponseSender)>();
        tokio::task::spawn_local(async move {
            while let Some((request, reply)) = receiver.recv().await {
                let result = execute_shell_request(request).await;
                let _unused = reply.send(result);
            }
        });
        Self { sender }
    }

    pub async fn run_shell(
        &self,
        request: ShellExecutionRequest,
    ) -> Result<ShellExecutionResult, String> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send((request, tx))
            .map_err(|_| "command executor shut down".to_string())?;
        rx.await
            .map_err(|_| "command executor dropped response".to_string())?
    }
}

async fn execute_shell_request(
    request: ShellExecutionRequest,
) -> Result<ShellExecutionResult, String> {
    let client = ACP_CLIENT
        .get()
        .ok_or_else(|| "ACP client not initialized".to_string())?
        .clone();

    let ShellExecutionRequest {
        session_id,
        command,
        args,
        cwd,
        env,
        timeout_ms,
    } = request;

    let create_req = CreateTerminalRequest {
        session_id: session_id.clone(),
        command,
        args,
        env,
        cwd: Some(cwd),
        output_byte_limit: None,
        meta: None,
    };

    let start = Instant::now();
    let terminal = client
        .create_terminal(create_req)
        .await
        .map_err(|e| e.to_string())?;
    let terminal_id = terminal.terminal_id.clone();
    let mut timed_out = false;

    let wait_future = client.wait_for_terminal_exit(WaitForTerminalExitRequest {
        session_id: session_id.clone(),
        terminal_id: terminal_id.clone(),
        meta: None,
    });

    let exit_status = if let Some(timeout_ms) = timeout_ms {
        match tokio::time::timeout(Duration::from_millis(timeout_ms), wait_future).await {
            Ok(result) => result.map_err(|e| e.to_string())?.exit_status,
            Err(_) => {
                timed_out = true;
                let _unused = client
                    .kill_terminal_command(KillTerminalCommandRequest {
                        session_id: session_id.clone(),
                        terminal_id: terminal_id.clone(),
                        meta: None,
                    })
                    .await;
                client
                    .wait_for_terminal_exit(WaitForTerminalExitRequest {
                        session_id: session_id.clone(),
                        terminal_id: terminal_id.clone(),
                        meta: None,
                    })
                    .await
                    .map_err(|e| e.to_string())?
                    .exit_status
            }
        }
    } else {
        wait_future.await.map_err(|e| e.to_string())?.exit_status
    };

    let output_response = client
        .terminal_output(TerminalOutputRequest {
            session_id: session_id.clone(),
            terminal_id: terminal_id.clone(),
            meta: None,
        })
        .await
        .map_err(|e| e.to_string())?;

    let _unused = client
        .release_terminal(ReleaseTerminalRequest {
            session_id,
            terminal_id,
            meta: None,
        })
        .await;

    let mut stdout_text = output_response.output;
    if output_response.truncated {
        stdout_text.push_str("\n[output truncated by client]");
    }
    if timed_out {
        stdout_text.push_str("\n[command timed out]");
    }

    let exit_code = exit_status.exit_code.map(|c| c as i32).unwrap_or(-1);

    Ok(ShellExecutionResult {
        stdout: stdout_text,
        exit_code,
        duration: start.elapsed(),
        timed_out,
    })
}
