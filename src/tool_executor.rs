use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use codex_core::exec::{ExecToolCallOutput, StreamOutput};
use codex_core::{
    ApplyPatchRequest, DynToolExecutor, SandboxAttempt, ShellRequest, ToolCtx, ToolError,
    ToolExecutor,
};

#[derive(Clone, Default)]
pub struct HelloWorldExecutor;

impl HelloWorldExecutor {
    pub fn shared() -> DynToolExecutor {
        Arc::new(Self::default())
    }

    fn hello_world_output(&self) -> ExecToolCallOutput {
        let stdout = "Hello from codex-acp tool executor!\n".to_string();
        ExecToolCallOutput {
            exit_code: 0,
            stdout: StreamOutput::new(stdout.clone()),
            stderr: StreamOutput::new(String::new()),
            aggregated_output: StreamOutput::new(stdout),
            duration: Duration::from_millis(0),
            timed_out: false,
        }
    }
}

#[async_trait]
impl ToolExecutor for HelloWorldExecutor {
    async fn run_shell(
        &self,
        _req: &ShellRequest,
        _attempt: &SandboxAttempt<'_>,
        _ctx: &ToolCtx<'_>,
    ) -> Result<ExecToolCallOutput, ToolError> {
        Ok(self.hello_world_output())
    }

    async fn run_apply_patch(
        &self,
        _req: &ApplyPatchRequest,
        _attempt: &SandboxAttempt<'_>,
        _ctx: &ToolCtx<'_>,
    ) -> Result<ExecToolCallOutput, ToolError> {
        Err(ToolError::Rejected(
            "apply_patch not yet implemented in codex-acp executor".to_string(),
        ))
    }
}
