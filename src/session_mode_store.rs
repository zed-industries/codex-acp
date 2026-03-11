use agent_client_protocol::{SessionId, SessionModeId};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::warn;

#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedSessionModeStore {
    #[serde(default)]
    session_modes: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub struct SessionModeStore {
    path: PathBuf,
    session_modes: HashMap<String, String>,
}

impl SessionModeStore {
    pub fn load(codex_home: &Path) -> Self {
        let path = codex_home
            .join("acp")
            .join("session-mode-overrides.v1.json");

        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Self {
                    path,
                    session_modes: HashMap::new(),
                };
            }
            Err(error) => {
                warn!(
                    "Failed reading session mode overrides from {}: {}",
                    path.display(),
                    error
                );
                return Self {
                    path,
                    session_modes: HashMap::new(),
                };
            }
        };

        let persisted = match serde_json::from_slice::<PersistedSessionModeStore>(&bytes) {
            Ok(persisted) => persisted,
            Err(error) => {
                warn!(
                    "Failed parsing session mode overrides from {}: {}",
                    path.display(),
                    error
                );
                return Self {
                    path,
                    session_modes: HashMap::new(),
                };
            }
        };

        Self {
            path,
            session_modes: persisted.session_modes,
        }
    }

    pub fn get(&self, session_id: &SessionId) -> Option<SessionModeId> {
        self.session_modes
            .get(session_id.0.as_ref())
            .map(|mode| SessionModeId::new(mode.as_str()))
    }

    pub fn set(&mut self, session_id: &SessionId, mode_id: &SessionModeId) {
        self.session_modes
            .insert(session_id.0.to_string(), mode_id.0.to_string());
        self.persist();
    }

    pub fn remove(&mut self, session_id: &SessionId) {
        if self.session_modes.remove(session_id.0.as_ref()).is_some() {
            self.persist();
        }
    }

    fn persist(&self) {
        if let Some(parent) = self.path.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            warn!(
                "Failed creating parent dir for session mode overrides ({}): {}",
                parent.display(),
                error
            );
            return;
        }

        let payload = match serde_json::to_vec_pretty(&PersistedSessionModeStore {
            session_modes: self.session_modes.clone(),
        }) {
            Ok(payload) => payload,
            Err(error) => {
                warn!(
                    "Failed serializing session mode overrides for {}: {}",
                    self.path.display(),
                    error
                );
                return;
            }
        };

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let tmp_path = self
            .path
            .with_extension(format!("tmp.{}.{}", process::id(), nonce));

        if let Err(error) = fs::write(&tmp_path, payload) {
            warn!(
                "Failed writing temporary session mode overrides file ({}): {}",
                tmp_path.display(),
                error
            );
            return;
        }

        if let Err(error) = fs::rename(&tmp_path, &self.path) {
            warn!(
                "Failed atomically replacing session mode overrides file ({}): {}",
                self.path.display(),
                error
            );
            drop(fs::remove_file(&tmp_path));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SessionModeStore;
    use agent_client_protocol::{SessionId, SessionModeId};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    use uuid::Uuid;

    fn make_temp_codex_home() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        dir.push(format!(
            "codex-acp-session-mode-store-{}-{}",
            Uuid::new_v4(),
            nonce
        ));
        fs::create_dir_all(&dir).expect("create temp codex home");
        dir
    }

    #[test]
    fn persists_and_reloads_modes() {
        let codex_home = make_temp_codex_home();
        let session_id = SessionId::new("session-1");
        let mode_id = SessionModeId::new("plan");

        let mut store = SessionModeStore::load(&codex_home);
        assert!(store.get(&session_id).is_none());

        store.set(&session_id, &mode_id);
        assert_eq!(store.get(&session_id), Some(mode_id.clone()));

        let reloaded = SessionModeStore::load(&codex_home);
        assert_eq!(reloaded.get(&session_id), Some(mode_id));

        fs::remove_dir_all(codex_home).expect("cleanup temp codex home");
    }

    #[test]
    fn tolerates_corrupted_file() {
        let codex_home = make_temp_codex_home();
        let store_file = codex_home
            .join("acp")
            .join("session-mode-overrides.v1.json");
        fs::create_dir_all(store_file.parent().expect("parent dir")).expect("create acp dir");
        fs::write(&store_file, b"{not-json").expect("write corrupt file");

        let store = SessionModeStore::load(&codex_home);
        assert!(store.get(&SessionId::new("missing")).is_none());

        fs::remove_dir_all(codex_home).expect("cleanup temp codex home");
    }
}
