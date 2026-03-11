//! Persistent storage for ACP session mode overrides.
//!
//! Mode overrides are persisted to `$CODEX_HOME/acp/session-mode-overrides.v1.json`
//! so that a client's explicit mode choice (e.g. `"auto"`, `"full-access"`) is
//! replayed automatically when the same project session is reloaded after a
//! server restart.
//!
//! The file format is a JSON object keyed by the project's working-directory
//! path:
//!
//! ```json
//! {
//!   "/home/user/myproject": "auto",
//!   "/home/user/other":     "full-access"
//! }
//! ```
//!
//! Errors are non-fatal: a warning is logged and the session continues with the
//! default mode.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use tracing::warn;

/// Reads and writes per-project session mode overrides.
pub struct ModeOverrideStore {
    path: PathBuf,
}

impl ModeOverrideStore {
    /// Create a new store backed by
    /// `$codex_home/acp/session-mode-overrides.v1.json`.
    pub fn new(codex_home: &Path) -> Self {
        Self {
            path: codex_home.join("acp").join("session-mode-overrides.v1.json"),
        }
    }

    /// Return the persisted mode ID for the given `cwd`, if any.
    pub fn get(&self, cwd: &Path) -> Option<String> {
        let content = std::fs::read_to_string(&self.path).ok()?;
        let map: HashMap<String, String> = serde_json::from_str(&content)
            .map_err(|e| warn!("Failed to parse mode overrides file: {e}"))
            .ok()?;
        map.get(cwd.to_string_lossy().as_ref()).cloned()
    }

    /// Persist the given `mode_id` for `cwd`, creating the backing file if
    /// necessary.
    ///
    /// Non-fatal: logs a warning and returns without panicking on any I/O or
    /// serialisation error.
    pub fn set(&self, cwd: &Path, mode_id: &str) {
        let mut map: HashMap<String, String> = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_default();

        map.insert(cwd.to_string_lossy().into_owned(), mode_id.to_owned());

        if let Some(parent) = self.path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                warn!("Failed to create acp directory for mode overrides: {e}");
                return;
            }
        }

        match serde_json::to_string_pretty(&map) {
            Ok(content) => {
                if let Err(e) = std::fs::write(&self.path, content) {
                    warn!("Failed to write mode overrides: {e}");
                }
            }
            Err(e) => warn!("Failed to serialise mode overrides: {e}"),
        }
    }
}
