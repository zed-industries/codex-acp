use std::sync::Arc;
use async_trait::async_trait;
use agent_client_protocol::{EnvVariable, SessionId};
use codex_core::exec::{ExecToolCallOutput, StreamOutput};
use codex_core::{
    ApplyPatchRequest, DynToolExecutor, SandboxAttempt, ShellRequest, ToolCtx, ToolError,
    ToolExecutor,
};
use crate::command_executor::{command_dispatcher, ShellExecutionRequest};

#[derive(Clone, Default)]
pub struct HelloWorldExecutor;

impl HelloWorldExecutor {
    pub fn shared() -> DynToolExecutor {
        Arc::new(Self::default())
    }
}

#[async_trait]
impl ToolExecutor for HelloWorldExecutor {
    async fn run_shell(
        &self,
        req: &ShellRequest,
        _attempt: &SandboxAttempt<'_>,
        ctx: &ToolCtx<'_>,
    ) -> Result<ExecToolCallOutput, ToolError> {
        let (program, args) = req
            .command
            .split_first()
            .ok_or_else(|| ToolError::Rejected("command args are empty".to_string()))?;

        let dispatcher = command_dispatcher()
            .ok_or_else(|| ToolError::Rejected("command dispatcher not initialized".to_string()))?;

        let session_id = SessionId(ctx.conversation_id().to_string().into());
        let env: Vec<EnvVariable> = req
            .env
            .iter()
            .map(|(name, value)| EnvVariable {
                name: name.clone(),
                value: value.clone(),
                meta: None,
            })
            .collect();

        let request = ShellExecutionRequest {
            session_id,
            command: program.clone(),
            args: args.to_vec(),
            cwd: req.cwd.clone(),
            env,
            timeout_ms: req.timeout_ms,
        };

        let result = dispatcher
            .run_shell(request)
            .await
            .map_err(|e| ToolError::Rejected(e))?;

        Ok(ExecToolCallOutput {
            exit_code: result.exit_code,
            stdout: StreamOutput::new(result.stdout.clone()),
            stderr: StreamOutput::new(String::new()),
            aggregated_output: StreamOutput::new(result.stdout),
            duration: result.duration,
            timed_out: result.timed_out,
        })
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
