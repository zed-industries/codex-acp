use crate::ACP_CLIENT;
use crate::command_executor::{ShellExecutionRequest, command_dispatcher};
use agent_client_protocol::{
    AgentSideConnection, Client as _, ErrorCode, ReadTextFileRequest, SessionId,
    WriteTextFileRequest,
};
use async_trait::async_trait;
use codex_apply_patch::{
    ApplyPatchArgs, ApplyPatchError, Hunk, UpdateFileChunk, apply_chunks_to_contents, parse_patch,
};
use codex_core::exec::{ExecToolCallOutput, StreamOutput};
use codex_core::{
    ApplyPatchRequest, DynToolExecutor, SandboxAttempt, ShellRequest, ToolCtx, ToolError,
    ToolExecutor,
};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Handle;

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
        let env = req
            .env
            .iter()
            .map(|(name, value)| agent_client_protocol::EnvVariable {
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
            .map_err(ToolError::Rejected)?;

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
        req: &ApplyPatchRequest,
        _attempt: &SandboxAttempt<'_>,
        ctx: &ToolCtx<'_>,
    ) -> Result<ExecToolCallOutput, ToolError> {
        let start = Instant::now();
        let session_id = SessionId(ctx.conversation_id().to_string().into());

        let parsed = parse_patch(&req.patch)
            .map_err(|e| ToolError::Rejected(format!("failed to parse patch: {e}")))?;
        let base_dir = resolve_base_directory(&req.cwd, &parsed);

        let mut operations = Vec::new();
        for hunk in parsed.hunks {
            match hunk {
                Hunk::AddFile { path, contents } => {
                    let abs_path = resolve_path(&base_dir, &path);
                    operations.push(FileOperation::Write {
                        path: abs_path,
                        content: contents,
                    });
                }
                Hunk::DeleteFile { path } => {
                    let abs_path = resolve_path(&base_dir, &path);
                    operations.push(FileOperation::Delete { path: abs_path });
                }
                Hunk::UpdateFile {
                    path,
                    move_path,
                    chunks,
                } => {
                    let source_path = resolve_path(&base_dir, &path);
                    let target_path = move_path
                        .map(|new_path| resolve_path(&base_dir, &new_path))
                        .unwrap_or_else(|| source_path.clone());
                    operations.push(FileOperation::Update {
                        source: source_path,
                        target: target_path,
                        chunks,
                    });
                }
            }
        }

        let mut summary: Vec<String> = Vec::new();
        for op in operations {
            match op {
                FileOperation::Write { path, content } => {
                    ensure_parent_directories(&path, req, &session_id).await?;
                    write_remote_file(&path, &content, req, &session_id).await?;
                    summary.push(format!("Wrote {}", display_path(&path)));
                }
                FileOperation::Delete { path } => {
                    delete_remote_path(&path, req, &session_id).await?;
                    summary.push(format!("Deleted {}", display_path(&path)));
                }
                FileOperation::Update {
                    source,
                    target,
                    chunks,
                } => {
                    let current = read_remote_file(&source, req, &session_id).await?;
                    let contents = current.ok_or_else(|| {
                        ToolError::Rejected(format!(
                            "cannot update {} because it does not exist",
                            display_path(&source)
                        ))
                    })?;
                    let new_contents = apply_chunks_to_contents(&contents, &source, &chunks)
                        .map_err(|e| format_apply_error(&source, e))?;
                    ensure_parent_directories(&target, req, &session_id).await?;
                    write_remote_file(&target, &new_contents, req, &session_id).await?;
                    if source != target {
                        delete_remote_path(&source, req, &session_id).await?;
                        summary.push(format!(
                            "Moved {} -> {}",
                            display_path(&source),
                            display_path(&target)
                        ));
                    } else {
                        summary.push(format!("Updated {}", display_path(&target)));
                    }
                }
            }
        }

        let summary_text = if summary.is_empty() {
            "No changes applied".to_string()
        } else {
            summary.join("\n")
        };

        Ok(ExecToolCallOutput {
            exit_code: 0,
            stdout: StreamOutput::new(summary_text.clone()),
            stderr: StreamOutput::new(String::new()),
            aggregated_output: StreamOutput::new(summary_text),
            duration: start.elapsed(),
            timed_out: false,
        })
    }
}

enum FileOperation {
    Write {
        path: PathBuf,
        content: String,
    },
    Delete {
        path: PathBuf,
    },
    Update {
        source: PathBuf,
        target: PathBuf,
        chunks: Vec<UpdateFileChunk>,
    },
}

fn resolve_base_directory(cwd: &Path, args: &ApplyPatchArgs) -> PathBuf {
    match &args.workdir {
        Some(workdir) => {
            let workdir_path = PathBuf::from(workdir);
            if workdir_path.is_absolute() {
                workdir_path
            } else {
                cwd.join(workdir_path)
            }
        }
        None => cwd.to_path_buf(),
    }
}

fn resolve_path(base: &Path, relative: &Path) -> PathBuf {
    if relative.is_absolute() {
        relative.to_path_buf()
    } else {
        base.join(relative)
    }
}

async fn read_remote_file(
    path: &Path,
    _req: &ApplyPatchRequest,
    session_id: &SessionId,
) -> Result<Option<String>, ToolError> {
    let request = ReadTextFileRequest {
        session_id: session_id.clone(),
        path: path.to_path_buf(),
        line: None,
        limit: None,
        meta: None,
    };

    match client_call(move |client| async move { client.read_text_file(request).await }).await? {
        Ok(response) => Ok(Some(response.content)),
        Err(err) if err.code == ErrorCode::RESOURCE_NOT_FOUND.code => Ok(None),
        Err(err) => Err(ToolError::Rejected(format!(
            "failed to read {}: {}",
            display_path(path),
            err.message
        ))),
    }
}

async fn write_remote_file(
    path: &Path,
    content: &str,
    _req: &ApplyPatchRequest,
    session_id: &SessionId,
) -> Result<(), ToolError> {
    let request = WriteTextFileRequest {
        session_id: session_id.clone(),
        path: path.to_path_buf(),
        content: content.to_string(),
        meta: None,
    };

    client_call(move |client| async move { client.write_text_file(request).await })
        .await?
        .map_err(|err| {
            ToolError::Rejected(format!(
                "failed to write {}: {}",
                display_path(path),
                err.message
            ))
        })?;

    Ok(())
}

async fn ensure_parent_directories(
    path: &Path,
    req: &ApplyPatchRequest,
    session_id: &SessionId,
) -> Result<(), ToolError> {
    if let Some(parent) = path.parent() {
        run_command(
            "mkdir",
            vec!["-p".into(), parent.display().to_string()],
            req,
            session_id,
        )
        .await?;
    }
    Ok(())
}

async fn delete_remote_path(
    path: &Path,
    req: &ApplyPatchRequest,
    session_id: &SessionId,
) -> Result<(), ToolError> {
    run_command(
        "rm",
        vec!["-f".into(), path.display().to_string()],
        req,
        session_id,
    )
    .await
}

async fn run_command(
    command: &str,
    args: Vec<String>,
    req: &ApplyPatchRequest,
    session_id: &SessionId,
) -> Result<(), ToolError> {
    let dispatcher = command_dispatcher()
        .ok_or_else(|| ToolError::Rejected("command dispatcher not initialized".to_string()))?;
    dispatcher
        .run_shell(ShellExecutionRequest {
            session_id: session_id.clone(),
            command: command.into(),
            args,
            cwd: req.cwd.clone(),
            env: Vec::new(),
            timeout_ms: None,
        })
        .await
        .map_err(ToolError::Rejected)?;
    Ok(())
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

fn format_apply_error(path: &Path, err: ApplyPatchError) -> ToolError {
    ToolError::Rejected(format!(
        "failed to apply patch to {}: {err}",
        display_path(path)
    ))
}

async fn client_call<R, F, Fut>(f: F) -> Result<Result<R, agent_client_protocol::Error>, ToolError>
where
    R: Send + 'static,
    F: FnOnce(Arc<AgentSideConnection>) -> Fut + Send + 'static,
    Fut: Future<Output = Result<R, agent_client_protocol::Error>> + 'static,
{
    let client = ACP_CLIENT
        .get()
        .ok_or_else(|| ToolError::Rejected("ACP client not initialized".to_string()))?
        .clone();
    let handle = Handle::current();
    tokio::task::spawn_blocking(move || handle.block_on(async move { f(client).await }))
        .await
        .map_err(|err| ToolError::Rejected(format!("ACP client call failed: {err}")))
}
