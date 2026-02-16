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
