use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use agent_client_protocol::{McpServer, ModelId, SessionId, SessionModeId};
use codex_common::approval_presets::{builtin_approval_presets, ApprovalPreset};
use codex_core::config::Config;
use codex_core::protocol::{AskForApproval, SandboxPolicy};
use codex_protocol::config_types::ReasoningEffort;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::cli::SessionPersistCli;

pub(crate) const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    #[serde(default = "manifest_version")]
    pub version: u32,
    pub session_id: String,
    pub conversation_id: String,
    pub rollout_path: PathBuf,
    pub cwd: PathBuf,
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub mode_id: Option<String>,
    #[serde(default)]
    pub approval_policy: Option<AskForApproval>,
    #[serde(default)]
    pub sandbox_policy: Option<SandboxPolicy>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub last_updated: Option<u64>,
}

fn manifest_version() -> u32 {
    MANIFEST_VERSION
}

#[derive(Debug, Clone)]
pub struct SessionPersistenceSettings {
    pub enabled: bool,
    pub directory: PathBuf,
}

impl SessionPersistenceSettings {
    pub fn resolve(config: &Config, cli: &SessionPersistCli) -> Self {
        fn parse_bool(value: &str) -> Option<bool> {
            match value.to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            }
        }

        let env_flag = env::var("CODEX_SESSION_PERSIST")
            .ok()
            .as_deref()
            .and_then(parse_bool);
        let env_dir = env::var("CODEX_SESSION_DIR").ok().map(PathBuf::from);

        let mut enabled = env_flag.unwrap_or(false);
        let mut directory = env_dir.clone();

        if let Some(flag) = cli.flag {
            enabled = flag;
        }

        if let Some(path) = &cli.path {
            directory = Some(path.clone());
            if cli.flag.is_none() {
                enabled = true;
            }
        }

        if directory.is_none() && env_dir.is_some() && env_flag.is_none() && cli.flag.is_none() {
            // Setting CODEX_SESSION_DIR should imply enabling persistence unless explicitly disabled.
            enabled = true;
        }

        let directory = directory.unwrap_or_else(|| config.codex_home.join("sessions"));

        Self { enabled, directory }
    }
}

#[derive(Debug)]
pub struct SessionStore {
    enabled: bool,
    manifest_path: PathBuf,
    records: RefCell<HashMap<String, SessionRecord>>,
}

impl SessionStore {
    pub fn new(settings: SessionPersistenceSettings) -> Self {
        let manifest_path = settings.directory.join("manifest.jsonl");
        let records = if settings.enabled {
            load_manifest(&manifest_path)
        } else {
            HashMap::new()
        };

        if settings.enabled {
            info!(
                "Session persistence enabled; manifest path: {} ({} records)",
                manifest_path.display(),
                records.len()
            );
        } else {
            debug!("Session persistence disabled");
        }

        Self {
            enabled: settings.enabled,
            manifest_path,
            records: RefCell::new(records),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn record_session(&self, record: SessionRecord) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let mut map = self.records.borrow_mut();
        map.insert(record.session_id.clone(), record);
        write_manifest(&self.manifest_path, &map)
    }

    pub fn get(&self, session_id: &SessionId) -> Option<SessionRecord> {
        let map = self.records.borrow();
        map.get(&session_id.to_string()).cloned()
    }

    pub fn update_model(&self, session_id: &SessionId, model_id: &ModelId) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let mut map = self.records.borrow_mut();
        let Some(record) = map.get_mut(&session_id.to_string()) else {
            return Ok(());
        };
        let new_model = Some(model_id.0.as_ref().to_string());
        if record.model_id != new_model {
            record.model_id = new_model;
            record.reasoning_effort = parse_reasoning_effort(record.model_id.clone());
            record.last_updated = current_timestamp();
            write_manifest(&self.manifest_path, &map)?;
        }
        Ok(())
    }

    pub fn update_mode(&self, session_id: &SessionId, mode_id: &SessionModeId) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let mut map = self.records.borrow_mut();
        let Some(record) = map.get_mut(&session_id.to_string()) else {
            return Ok(());
        };
        let new_mode = Some(mode_id.0.as_ref().to_string());
        if record.mode_id != new_mode {
            record.mode_id = new_mode.clone();
            if let Some(mode_id_str) = &record.mode_id {
                if let Some(preset) = resolve_preset(mode_id_str) {
                    record.approval_policy = Some(preset.approval);
                    record.sandbox_policy = Some(preset.sandbox.clone());
                }
            }
            record.last_updated = current_timestamp();
            write_manifest(&self.manifest_path, &map)?;
        }
        Ok(())
    }
}

pub(crate) fn current_timestamp() -> Option<u64> {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

fn resolve_preset(mode_id: &str) -> Option<ApprovalPreset> {
    preset_for_mode(mode_id)
}

fn parse_reasoning_effort(model_id: Option<String>) -> Option<ReasoningEffort> {
    let model_id = model_id?;
    let (_, reasoning) = split_model_id(&model_id).ok()?;
    Some(reasoning)
}

fn load_manifest(path: &Path) -> HashMap<String, SessionRecord> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return HashMap::new(),
        Err(err) => {
            warn!("Failed to read session manifest {}: {err}", path.display());
            return HashMap::new();
        }
    };

    let mut map = HashMap::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        match line {
            Ok(text) if text.trim().is_empty() => continue,
            Ok(text) => match serde_json::from_str::<SessionRecord>(&text) {
                Ok(mut record) => {
                    if record.version != MANIFEST_VERSION {
                        warn!(
                            "Skipping session record with unsupported version {} at {}:{}",
                            record.version,
                            path.display(),
                            idx + 1
                        );
                        continue;
                    }
                    if record.last_updated.is_none() {
                        record.last_updated = current_timestamp();
                    }
                    map.insert(record.session_id.clone(), record);
                }
                Err(err) => warn!(
                    "Failed to parse session record at {}:{}: {err}",
                    path.display(),
                    idx + 1
                ),
            },
            Err(err) => {
                warn!(
                    "Failed to read line {} of manifest {}: {err}",
                    idx + 1,
                    path.display()
                );
            }
        }
    }
    map
}

fn write_manifest(path: &Path, records: &HashMap<String, SessionRecord>) -> io::Result<()> {
    if records.is_empty() {
        if path.exists() {
            fs::remove_file(path)?;
        }
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("jsonl.tmp");
    {
        let mut file = fs::File::create(&tmp_path)?;
        let mut entries: Vec<_> = records.values().cloned().collect();
        entries.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        for record in entries {
            serde_json::to_writer(&mut file, &record)?;
            file.write_all(b"\n")?;
        }
        file.sync_all()?;
    }

    fs::rename(tmp_path, path)?;
    Ok(())
}

pub(crate) fn split_model_id(
    model_id: &str,
) -> Result<(String, ReasoningEffort), serde_json::Error> {
    let (model, reasoning) = model_id
        .split_once('/')
        .ok_or_else(|| {
            serde_json::Error::io(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid model identifier",
            ))
        })?;
    let effort: ReasoningEffort = serde_json::from_value(reasoning.into())?;
    Ok((model.to_string(), effort))
}

pub fn apply_session_config_overrides(
    config: &mut Config,
    record: &SessionRecord,
) {
    config.cwd.clone_from(&record.cwd);

    if let Some(approval) = record.approval_policy {
        config.approval_policy = approval;
    }
    if let Some(policy) = &record.sandbox_policy {
        config.sandbox_policy = policy.clone();
    } else if let Some(mode_id) = &record.mode_id {
        if let Some(preset) = preset_for_mode(mode_id) {
            config.approval_policy = preset.approval;
            config.sandbox_policy = preset.sandbox.clone();
        }
    }

    if let Some(model_id) = &record.model_id {
        if let Ok((model, effort)) = split_model_id(model_id) {
            config.model = model;
            config.model_reasoning_effort = Some(effort);
        }
    } else {
        config.model_reasoning_effort = record.reasoning_effort;
    }
}

fn preset_for_mode(mode_id: &str) -> Option<ApprovalPreset> {
    builtin_approval_presets()
        .into_iter()
        .find(|preset| preset.id == mode_id)
}
