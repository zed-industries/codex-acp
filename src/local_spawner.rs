use std::{
    collections::HashMap,
    io::Cursor,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use agent_client_protocol::{
    AgentSideConnection, Client, ClientCapabilities, ReadTextFileRequest, SessionId,
    WriteTextFileRequest,
};
use codex_apply_patch::StdFs;
use tokio::sync::mpsc;

use crate::ACP_CLIENT;

#[derive(Debug)]
pub enum FsTask {
    ReadFile {
        session_id: SessionId,
        path: PathBuf,
        tx: std::sync::mpsc::Sender<std::io::Result<String>>,
    },
    ReadFileLimit {
        session_id: SessionId,
        path: PathBuf,
        limit: usize,
        tx: tokio::sync::oneshot::Sender<std::io::Result<String>>,
    },
    WriteFile {
        session_id: SessionId,
        path: PathBuf,
        content: String,
        tx: std::sync::mpsc::Sender<std::io::Result<()>>,
    },
}

impl FsTask {
    async fn run(self) {
        match self {
            FsTask::ReadFile {
                session_id,
                path,
                tx,
            } => {
                let read_text_file =
                    Self::client().read_text_file(ReadTextFileRequest::new(session_id, path));
                let response = read_text_file
                    .await
                    .map(|response| response.content)
                    .map_err(|e| std::io::Error::other(e.to_string()));
                tx.send(response).ok();
            }
            FsTask::ReadFileLimit {
                session_id,
                path,
                limit,
                tx,
            } => {
                let read_text_file = Self::client().read_text_file(
                    ReadTextFileRequest::new(session_id, path)
                        .limit(limit.try_into().unwrap_or(u32::MAX)),
                );
                let response = read_text_file
                    .await
                    .map(|response| response.content)
                    .map_err(|e| std::io::Error::other(e.to_string()));
                tx.send(response).ok();
            }
            FsTask::WriteFile {
                session_id,
                path,
                content,
                tx,
            } => {
                let response = Self::client()
                    .write_text_file(WriteTextFileRequest::new(session_id, path, content))
                    .await
                    .map(|_| ())
                    .map_err(|e| std::io::Error::other(e.to_string()));
                tx.send(response).ok();
            }
        }
    }

    fn client() -> &'static AgentSideConnection {
        ACP_CLIENT.get().expect("Missing ACP client")
    }
}

pub struct AcpFs {
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    local_spawner: LocalSpawner,
    session_id: SessionId,
    session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
}

impl AcpFs {
    pub fn new(
        session_id: SessionId,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        local_spawner: LocalSpawner,
        session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
    ) -> Self {
        Self {
            client_capabilities,
            local_spawner,
            session_id,
            session_roots,
        }
    }

    fn session_root(&self) -> std::io::Result<PathBuf> {
        self.session_roots
            .lock()
            .unwrap()
            .get(&self.session_id)
            .cloned()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "session root not registered",
                )
            })
    }

    fn ensure_within_root(&self, path: &std::path::Path) -> std::io::Result<PathBuf> {
        let root = std::path::absolute(self.session_root()?)?;
        // Fix: Resolve relative paths against session root, not CWD
        // This ensures that relative paths from apply_patch are correctly resolved
        let abs_path = if path.is_absolute() {
            std::path::absolute(path)?
        } else {
            let resolved = root.join(path);
            std::path::absolute(&resolved)?
        };
        if abs_path.starts_with(&root) {
            Ok(abs_path)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "access to {} denied (outside session root {})",
                    abs_path.display(),
                    root.display()
                ),
            ))
        }
    }
}

impl codex_apply_patch::Fs for AcpFs {
    fn read_to_string(&self, path: &std::path::Path) -> std::io::Result<String> {
        if !self.client_capabilities.lock().unwrap().fs.read_text_file {
            return StdFs.read_to_string(path);
        }
        let path = self.ensure_within_root(path)?;
        let (tx, rx) = std::sync::mpsc::channel();
        self.local_spawner.spawn(FsTask::ReadFile {
            session_id: self.session_id.clone(),
            path,
            tx,
        });
        rx.recv()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .flatten()
    }

    fn write(&self, path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
        if !self.client_capabilities.lock().unwrap().fs.write_text_file {
            return StdFs.write(path, contents);
        }
        let path = self.ensure_within_root(path)?;
        let (tx, rx) = std::sync::mpsc::channel();
        self.local_spawner.spawn(FsTask::WriteFile {
            session_id: self.session_id.clone(),
            path,
            content: String::from_utf8(contents.to_vec())
                .map_err(|e| std::io::Error::other(e.to_string()))?,
            tx,
        });
        rx.recv()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .flatten()
    }
}

impl codex_core::codex::Fs for AcpFs {
    fn file_buffer(
        &self,
        path: &std::path::Path,
        limit: usize,
    ) -> std::pin::Pin<
        Box<
            dyn Future<Output = std::io::Result<Box<dyn tokio::io::AsyncBufRead + Unpin + Send>>>
                + Send,
        >,
    > {
        if !self.client_capabilities.lock().unwrap().fs.read_text_file {
            return StdFs.file_buffer(path, limit);
        }
        let path = match self.ensure_within_root(path) {
            Ok(path) => path,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.local_spawner.spawn(FsTask::ReadFileLimit {
            session_id: self.session_id.clone(),
            path,
            limit,
            tx,
        });
        Box::pin(async move {
            let file = rx
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))
                .flatten()?;

            Ok(Box::new(tokio::io::BufReader::new(Cursor::new(file.into_bytes()))) as _)
        })
    }
}

#[derive(Clone)]
pub struct LocalSpawner {
    send: mpsc::UnboundedSender<FsTask>,
}

impl LocalSpawner {
    pub fn new() -> Self {
        let (send, mut recv) = mpsc::unbounded_channel::<FsTask>();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        std::thread::spawn(move || {
            let local = tokio::task::LocalSet::new();

            local.spawn_local(async move {
                while let Some(new_task) = recv.recv().await {
                    tokio::task::spawn_local(new_task.run());
                }
                // If the while loop returns, then all the LocalSpawner
                // objects have been dropped.
            });

            // This will return once all senders are dropped and all
            // spawned tasks have returned.
            rt.block_on(local);
        });

        Self { send }
    }

    pub fn spawn(&self, task: FsTask) {
        self.send
            .send(task)
            .expect("Thread with LocalSet has shut down.");
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        process::Command,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    use agent_client_protocol::{
        Agent, AgentSideConnection, AuthenticateRequest, AuthenticateResponse,
        ClientSideConnection, FileSystemCapabilities, Implementation, InitializeRequest,
        InitializeResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
        NewSessionResponse, PromptRequest, PromptResponse, ReadTextFileResponse,
        SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
        SetSessionModeResponse, StopReason,
    };

    use super::*;

    const DEADLOCK_CHILD_ENV: &str = "CODEX_ACP_APPLY_PATCH_DEADLOCK_CHILD";
    const DEADLOCK_CHILD_TEST: &str = "local_spawner::tests::apply_patch_verification_child";

    #[test]
    fn apply_patch_verification_does_not_deadlock_over_acp_fs() {
        if std::env::var_os(DEADLOCK_CHILD_ENV).is_some() {
            return;
        }

        let current_exe = std::env::current_exe().expect("resolve current test binary");
        let mut child = Command::new(current_exe)
            .arg("--exact")
            .arg(DEADLOCK_CHILD_TEST)
            .arg("--nocapture")
            .env(DEADLOCK_CHILD_ENV, "1")
            .spawn()
            .expect("spawn deadlock child");

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().expect("poll deadlock child") {
                assert!(status.success(), "child exited with {status}");
                return;
            }

            if Instant::now() >= deadline {
                drop(child.kill());
                drop(child.wait());
                panic!("child timed out; apply_patch ACP fs roundtrip deadlocked");
            }

            thread::sleep(Duration::from_millis(25));
        }
    }

    #[test]
    fn apply_patch_verification_child() {
        if std::env::var_os(DEADLOCK_CHILD_ENV).is_none() {
            return;
        }

        reproduce_apply_patch_roundtrip();
    }

    fn reproduce_apply_patch_roundtrip() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build test runtime");
        let local_set = tokio::task::LocalSet::new();

        runtime.block_on(local_set.run_until(async move {
            let client = TestClient::new();
            let agent = TestAgent;
            let root = std::env::temp_dir().join(format!("codex-acp-deadlock-{}", std::process::id()));
            let session_id = SessionId::new("test-session");
            let file_path = root.join("src/client.rs");
            let (client_to_agent_rx, client_to_agent_tx) = piper::pipe(1024);
            let (agent_to_client_rx, agent_to_client_tx) = piper::pipe(1024);
            let (client_ready_tx, client_ready_rx) = std::sync::mpsc::channel();

            std::fs::create_dir_all(file_path.parent().expect("parent dir"))
                .expect("create test dirs");
            client.add_file_content(file_path.clone(), "fn old() {}\n".to_string());

            let _client_thread = thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("build remote client runtime");
                let local_set = tokio::task::LocalSet::new();

                runtime.block_on(local_set.run_until(async move {
                    let (_client_side, client_io_task) = ClientSideConnection::new(
                        client,
                        client_to_agent_tx,
                        agent_to_client_rx,
                        |fut| {
                            tokio::task::spawn_local(fut);
                        },
                    );
                    client_ready_tx.send(()).expect("signal remote client ready");
                    client_io_task.await.expect("run remote client io task");
                }));
            });

            client_ready_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("wait for remote client bootstrap");

            let (agent_side, agent_io_task) = AgentSideConnection::new(
                agent,
                agent_to_client_tx,
                client_to_agent_rx,
                |fut| {
                    tokio::task::spawn_local(fut);
                },
            );

            ACP_CLIENT
                .set(Arc::new(agent_side))
                .expect("install ACP client for child");

            let _agent_io = crate::spawn_acp_io_task("local-spawner-test-agent-io", agent_io_task)
                .expect("spawn agent io thread");

            tokio::task::yield_now().await;

            let capabilities = Arc::new(Mutex::new(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
            ));
            let session_roots = Arc::new(Mutex::new(HashMap::from([(
                session_id.clone(),
                root.clone(),
            )])));
            let fs = AcpFs::new(
                session_id,
                capabilities,
                LocalSpawner::new(),
                session_roots,
            );

            let patch = "*** Begin Patch\n*** Update File: src/client.rs\n@@\n-fn old() {}\n+fn new() {}\n*** End Patch";
            let argv = vec!["apply_patch".to_string(), patch.to_string()];
            let result = codex_apply_patch::maybe_parse_apply_patch_verified(&argv, &root, &fs);

            assert!(
                matches!(result, codex_apply_patch::MaybeApplyPatchVerified::Body(_)),
                "expected verified patch body, got {result:?}"
            );
        }));
    }

    #[derive(Clone)]
    struct TestClient {
        file_contents: Arc<Mutex<HashMap<PathBuf, String>>>,
    }

    impl TestClient {
        fn new() -> Self {
            Self {
                file_contents: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        fn add_file_content(&self, path: PathBuf, content: String) {
            self.file_contents.lock().unwrap().insert(path, content);
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Client for TestClient {
        async fn request_permission(
            &self,
            _arguments: agent_client_protocol::RequestPermissionRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::RequestPermissionResponse>
        {
            unimplemented!()
        }

        async fn write_text_file(
            &self,
            _arguments: WriteTextFileRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::WriteTextFileResponse> {
            unimplemented!()
        }

        async fn read_text_file(
            &self,
            arguments: ReadTextFileRequest,
        ) -> agent_client_protocol::Result<ReadTextFileResponse> {
            let contents = self.file_contents.lock().unwrap();
            let content = contents
                .get(&arguments.path)
                .cloned()
                .unwrap_or_else(|| "default content".to_string());
            Ok(ReadTextFileResponse::new(content))
        }

        async fn session_notification(
            &self,
            _args: agent_client_protocol::SessionNotification,
        ) -> agent_client_protocol::Result<()> {
            Ok(())
        }

        async fn create_terminal(
            &self,
            _args: agent_client_protocol::CreateTerminalRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::CreateTerminalResponse> {
            unimplemented!()
        }

        async fn terminal_output(
            &self,
            _args: agent_client_protocol::TerminalOutputRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::TerminalOutputResponse> {
            unimplemented!()
        }

        async fn kill_terminal(
            &self,
            _args: agent_client_protocol::KillTerminalRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::KillTerminalResponse> {
            unimplemented!()
        }

        async fn release_terminal(
            &self,
            _args: agent_client_protocol::ReleaseTerminalRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::ReleaseTerminalResponse> {
            unimplemented!()
        }

        async fn wait_for_terminal_exit(
            &self,
            _args: agent_client_protocol::WaitForTerminalExitRequest,
        ) -> agent_client_protocol::Result<agent_client_protocol::WaitForTerminalExitResponse>
        {
            unimplemented!()
        }
    }

    #[derive(Clone)]
    struct TestAgent;

    #[async_trait::async_trait(?Send)]
    impl Agent for TestAgent {
        async fn initialize(
            &self,
            arguments: InitializeRequest,
        ) -> agent_client_protocol::Result<InitializeResponse> {
            Ok(InitializeResponse::new(arguments.protocol_version)
                .agent_info(Implementation::new("test-agent", "0.0.0").title("Test Agent")))
        }

        async fn authenticate(
            &self,
            _arguments: AuthenticateRequest,
        ) -> agent_client_protocol::Result<AuthenticateResponse> {
            Ok(AuthenticateResponse::default())
        }

        async fn new_session(
            &self,
            _arguments: NewSessionRequest,
        ) -> agent_client_protocol::Result<NewSessionResponse> {
            Ok(NewSessionResponse::new(SessionId::new("unused")))
        }

        async fn load_session(
            &self,
            _arguments: LoadSessionRequest,
        ) -> agent_client_protocol::Result<LoadSessionResponse> {
            Ok(LoadSessionResponse::new())
        }

        async fn set_session_mode(
            &self,
            _arguments: SetSessionModeRequest,
        ) -> agent_client_protocol::Result<SetSessionModeResponse> {
            Ok(SetSessionModeResponse::new())
        }

        async fn prompt(
            &self,
            _arguments: PromptRequest,
        ) -> agent_client_protocol::Result<PromptResponse> {
            Ok(PromptResponse::new(StopReason::EndTurn))
        }

        async fn cancel(
            &self,
            _arguments: agent_client_protocol::CancelNotification,
        ) -> agent_client_protocol::Result<()> {
            Ok(())
        }

        async fn set_session_config_option(
            &self,
            _args: SetSessionConfigOptionRequest,
        ) -> agent_client_protocol::Result<SetSessionConfigOptionResponse> {
            Ok(SetSessionConfigOptionResponse::new(vec![]))
        }
    }
}
