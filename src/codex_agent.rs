use agent_client_protocol::{
    Agent, AgentCapabilities, AuthMethod, AuthMethodId, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, ClientCapabilities, Error, Implementation, InitializeRequest,
    InitializeResponse, ListSessionsRequest, ListSessionsResponse, LoadSessionRequest,
    LoadSessionResponse, McpCapabilities, McpServer, McpServerHttp, McpServerStdio,
    NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse,
    ProtocolVersion, SessionCapabilities, SessionId, SessionInfo, SessionListCapabilities,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
    SetSessionModeResponse, SetSessionModelRequest, SetSessionModelResponse,
};
use codex_core::{
    CodexAuth, NewThread, RolloutRecorder, ThreadManager, ThreadSortKey,
    auth::{AuthManager, read_codex_api_key_from_env, read_openai_api_key_from_env},
    config::{
        Config,
        types::{McpServerConfig, McpServerTransportConfig},
    },
    find_thread_path_by_id_str,
    models_manager::collaboration_mode_presets::CollaborationModesConfig,
    parse_cursor,
};
use codex_login::{CODEX_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR};
use codex_protocol::{
    ThreadId,
    protocol::{InitialHistory, SessionSource},
};
use std::{
    cell::RefCell,
    collections::HashMap,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
};
use tracing::{debug, info};
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    local_spawner::{AcpFs, LocalSpawner},
    thread::Thread,
};

/// The Codex implementation of the ACP Agent trait.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// Handle to the current authentication
    auth_manager: Arc<AuthManager>,
    /// Capabilities of the connected client
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    /// The underlying codex configuration
    config: Config,
    /// Thread manager for handling sessions
    thread_manager: ThreadManager,
    /// Active sessions mapped by `SessionId`
    sessions: Rc<RefCell<HashMap<SessionId, Rc<Thread>>>>,
    /// Session working directories for filesystem sandboxing
    session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
}

const SESSION_LIST_PAGE_SIZE: usize = 25;
const SESSION_TITLE_MAX_GRAPHEMES: usize = 120;

impl CodexAgent {
    /// Create a new `CodexAgent` with the given configuration
    pub fn new(config: Config) -> Self {
        let auth_manager = AuthManager::shared(
            config.codex_home.clone(),
            false,
            config.cli_auth_credentials_store_mode,
        );

        let client_capabilities: Arc<Mutex<ClientCapabilities>> = Arc::default();

        let local_spawner = LocalSpawner::new();
        let capabilities_clone = client_capabilities.clone();
        let session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>> = Arc::default();
        let session_roots_clone = session_roots.clone();
        let thread_manager = ThreadManager::new_with_fs(
            config.codex_home.clone(),
            auth_manager.clone(),
            SessionSource::Unknown,
            config.model_catalog.clone(),
            CollaborationModesConfig {
                // False for now
                default_mode_request_user_input: false,
            },
            Box::new(move |thread_id| {
                Arc::new(AcpFs::new(
                    Self::session_id_from_thread_id(thread_id),
                    capabilities_clone.clone(),
                    local_spawner.clone(),
                    session_roots_clone.clone(),
                ))
            }),
        );
        Self {
            auth_manager,
            client_capabilities,
            config,
            thread_manager,
            sessions: Rc::default(),
            session_roots,
        }
    }

    fn session_id_from_thread_id(thread_id: ThreadId) -> SessionId {
        SessionId::new(thread_id.to_string())
    }

    fn get_thread(&self, session_id: &SessionId) -> Result<Rc<Thread>, Error> {
        Ok(self
            .sessions
            .borrow()
            .get(session_id)
            .ok_or_else(|| Error::resource_not_found(None))?
            .clone())
    }

    async fn check_auth(&self) -> Result<(), Error> {
        if self.config.model_provider_id == "openai" && self.auth_manager.auth().await.is_none() {
            return Err(Error::auth_required());
        }
        Ok(())
    }

    /// Build a session config from base config, working directory, and MCP servers.
    /// This is shared between `new_session` and `load_session`.
    fn build_session_config(
        &self,
        cwd: &PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> Result<Config, Error> {
        let mut config = self.config.clone();
        config.include_apply_patch_tool = true;
        config.cwd.clone_from(cwd);

        // Propagate any client-provided MCP servers that codex-rs supports.
        let mut new_mcp_servers = config.mcp_servers.get().clone();
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex
                McpServer::Sse(..) => {}
                McpServer::Http(McpServerHttp {
                    name, url, headers, ..
                }) => {
                    new_mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::StreamableHttp {
                                url,
                                bearer_token_env_var: None,
                                http_headers: if headers.is_empty() {
                                    None
                                } else {
                                    Some(headers.into_iter().map(|h| (h.name, h.value)).collect())
                                },
                                env_http_headers: None,
                            },
                            required: false,
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                            disabled_reason: None,
                            scopes: None,
                            oauth_resource: None,
                        },
                    );
                }
                McpServer::Stdio(McpServerStdio {
                    name,
                    command,
                    args,
                    env,
                    ..
                }) => {
                    new_mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::Stdio {
                                command: command.display().to_string(),
                                args,
                                env: if env.is_empty() {
                                    None
                                } else {
                                    Some(env.into_iter().map(|env| (env.name, env.value)).collect())
                                },
                                env_vars: vec![],
                                cwd: Some(cwd.clone()),
                            },
                            required: false,
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                            disabled_reason: None,
                            scopes: None,
                            oauth_resource: None,
                        },
                    );
                }
                _ => {}
            }
        }

        config
            .mcp_servers
            .set(new_mcp_servers)
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(config)
    }
}

#[async_trait::async_trait(?Send)]
impl Agent for CodexAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        let InitializeRequest {
            protocol_version,
            client_capabilities,
            client_info: _, // TODO: save and pass into Codex somehow
            ..
        } = request;
        debug!("Received initialize request with protocol version {protocol_version:?}",);
        let protocol_version = ProtocolVersion::V1;

        *self.client_capabilities.lock().unwrap() = client_capabilities;

        let mut agent_capabilities = AgentCapabilities::new()
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .mcp_capabilities(McpCapabilities::new().http(true))
            .load_session(true);

        agent_capabilities.session_capabilities =
            SessionCapabilities::new().list(SessionListCapabilities::new());

        let mut auth_methods = vec![
            CodexAuthMethod::ChatGpt.into(),
            CodexAuthMethod::CodexApiKey.into(),
            CodexAuthMethod::OpenAiApiKey.into(),
        ];
        // Until codex device code auth works, we can't use this in remote ssh projects
        if std::env::var("NO_BROWSER").is_ok() {
            auth_methods.remove(0);
        }

        Ok(InitializeResponse::new(protocol_version)
            .agent_capabilities(agent_capabilities)
            .agent_info(Implementation::new("codex-acp", env!("CARGO_PKG_VERSION")).title("Codex"))
            .auth_methods(auth_methods))
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        let auth_method = CodexAuthMethod::try_from(request.method_id)?;

        // Check before starting login flow if already authenticated with the same method
        if let Some(auth) = self.auth_manager.auth().await {
            match (auth, auth_method) {
                (
                    CodexAuth::ApiKey(..),
                    CodexAuthMethod::CodexApiKey | CodexAuthMethod::OpenAiApiKey,
                )
                | (CodexAuth::Chatgpt(..), CodexAuthMethod::ChatGpt) => {
                    return Ok(AuthenticateResponse::new());
                }
                _ => {}
            }
        }

        match auth_method {
            CodexAuthMethod::ChatGpt => {
                // Perform browser/device login via codex-rs, then report success/failure to the client.
                let opts = codex_login::ServerOptions::new(
                    self.config.codex_home.clone(),
                    codex_core::auth::CLIENT_ID.to_string(),
                    None,
                    self.config.cli_auth_credentials_store_mode,
                );

                let server =
                    codex_login::run_login_server(opts).map_err(Error::into_internal_error)?;

                server
                    .block_until_done()
                    .await
                    .map_err(Error::into_internal_error)?;

                self.auth_manager.reload();
            }
            CodexAuthMethod::CodexApiKey => {
                let api_key = read_codex_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().data(format!("{CODEX_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(
                    &self.config.codex_home,
                    &api_key,
                    self.config.cli_auth_credentials_store_mode,
                )
                .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::OpenAiApiKey => {
                let api_key = read_openai_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().data(format!("{OPENAI_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(
                    &self.config.codex_home,
                    &api_key,
                    self.config.cli_auth_credentials_store_mode,
                )
                .map_err(Error::into_internal_error)?;
            }
        }

        self.auth_manager.reload();

        Ok(AuthenticateResponse::new())
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let NewSessionRequest {
            cwd, mcp_servers, ..
        } = request;
        info!("Creating new session with cwd: {}", cwd.display());

        let config = self.build_session_config(&cwd, mcp_servers)?;
        let num_mcp_servers = config.mcp_servers.len();

        let NewThread {
            thread_id,
            thread,
            session_configured: _,
        } = Box::pin(self.thread_manager.start_thread(config.clone()))
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_id = Self::session_id_from_thread_id(thread_id);
        // Record the session root for filesystem sandboxing.
        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), config.cwd.clone());
        let thread = Rc::new(Thread::new(
            session_id.clone(),
            thread,
            self.auth_manager.clone(),
            self.thread_manager.get_models_manager(),
            self.client_capabilities.clone(),
            config.clone(),
        ));
        let load = thread.load().await?;

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), thread);

        debug!("Created new session with {} MCP servers", num_mcp_servers);

        Ok(NewSessionResponse::new(session_id)
            .modes(load.modes)
            .models(load.models)
            .config_options(load.config_options))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let LoadSessionRequest {
            session_id,
            cwd,
            mcp_servers,
            ..
        } = request;

        let rollout_path =
            find_thread_path_by_id_str(&self.config.codex_home, session_id.0.as_ref())
                .await
                .map_err(|e| Error::internal_error().data(e.to_string()))?
                .ok_or_else(|| Error::resource_not_found(None))?;

        let history = RolloutRecorder::get_rollout_history(&rollout_path)
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        let rollout_items = match &history {
            InitialHistory::Resumed(resumed) => resumed.history.clone(),
            InitialHistory::Forked(items) => items.clone(),
            InitialHistory::New => Vec::new(),
        };

        let config = self.build_session_config(&cwd, mcp_servers)?;

        let NewThread {
            thread_id: _,
            thread,
            session_configured: _,
        } = Box::pin(self.thread_manager.resume_thread_from_rollout(
            config.clone(),
            rollout_path,
            self.auth_manager.clone(),
        ))
        .await
        .map_err(|e| Error::internal_error().data(e.to_string()))?;

        let thread = Rc::new(Thread::new(
            session_id.clone(),
            thread,
            self.auth_manager.clone(),
            self.thread_manager.get_models_manager(),
            self.client_capabilities.clone(),
            config.clone(),
        ));

        thread.replay_history(rollout_items).await?;

        let load = thread.load().await?;

        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), config.cwd);
        self.sessions.borrow_mut().insert(session_id, thread);

        Ok(LoadSessionResponse::new()
            .modes(load.modes)
            .models(load.models)
            .config_options(load.config_options))
    }

    async fn list_sessions(
        &self,
        request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, Error> {
        self.check_auth().await?;

        let ListSessionsRequest { cwd, cursor, .. } = request;
        let cursor_obj = cursor.as_deref().and_then(parse_cursor);

        let page = RolloutRecorder::list_threads(
            &self.config,
            SESSION_LIST_PAGE_SIZE,
            cursor_obj.as_ref(),
            ThreadSortKey::UpdatedAt,
            &[
                SessionSource::Cli,
                SessionSource::VSCode,
                SessionSource::Unknown,
            ],
            None,
            self.config.model_provider_id.as_str(),
            None,
        )
        .await
        .map_err(|err| Error::internal_error().data(format!("failed to list sessions: {err}")))?;

        let sessions = page
            .items
            .into_iter()
            .filter_map(|item| {
                let thread_id = item.thread_id?;
                let item_cwd = item.cwd?;

                if let Some(filter_cwd) = cwd.as_ref()
                    && item_cwd != *filter_cwd
                {
                    return None;
                }

                let title = item
                    .first_user_message
                    .as_deref()
                    .and_then(format_session_title);
                let updated_at = item.updated_at.or(item.created_at);

                Some(
                    SessionInfo::new(SessionId::new(thread_id.to_string()), item_cwd)
                        .title(title)
                        .updated_at(updated_at),
                )
            })
            .collect::<Vec<_>>();

        let next_cursor = page
            .next_cursor
            .as_ref()
            .and_then(|next_cursor| serde_json::to_value(next_cursor).ok())
            .and_then(|value| value.as_str().map(str::to_owned));

        Ok(ListSessionsResponse::new(sessions).next_cursor(next_cursor))
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        // Get the session state
        let thread = self.get_thread(&request.session_id)?;
        let stop_reason = thread.prompt(request).await?;

        Ok(PromptResponse::new(stop_reason))
    }

    async fn cancel(&self, args: CancelNotification) -> Result<(), Error> {
        info!("Cancelling operations for session: {}", args.session_id);
        self.get_thread(&args.session_id)?.cancel().await?;
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting session mode for session: {}", args.session_id);
        self.get_thread(&args.session_id)?
            .set_mode(args.mode_id)
            .await?;
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        args: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!("Setting session model for session: {}", args.session_id);

        self.get_thread(&args.session_id)?
            .set_model(args.model_id)
            .await?;

        Ok(SetSessionModelResponse::default())
    }

    async fn set_session_config_option(
        &self,
        args: SetSessionConfigOptionRequest,
    ) -> Result<SetSessionConfigOptionResponse, Error> {
        info!(
            "Setting session config option for session: {} (config_id: {}, value: {})",
            args.session_id, args.config_id.0, args.value.0
        );

        let thread = self.get_thread(&args.session_id)?;

        thread.set_config_option(args.config_id, args.value).await?;

        let config_options = thread.config_options().await?;

        Ok(SetSessionConfigOptionResponse::new(config_options))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodexAuthMethod {
    ChatGpt,
    CodexApiKey,
    OpenAiApiKey,
}

impl From<CodexAuthMethod> for AuthMethodId {
    fn from(method: CodexAuthMethod) -> Self {
        Self::new(match method {
            CodexAuthMethod::ChatGpt => "chatgpt",
            CodexAuthMethod::CodexApiKey => "codex-api-key",
            CodexAuthMethod::OpenAiApiKey => "openai-api-key",
        })
    }
}

impl From<CodexAuthMethod> for AuthMethod {
    fn from(method: CodexAuthMethod) -> Self {
        match method {
            CodexAuthMethod::ChatGpt => Self::new(method, "Login with ChatGPT").description(
                "Use your ChatGPT login with Codex CLI (requires a paid ChatGPT subscription)",
            ),
            CodexAuthMethod::CodexApiKey => {
                Self::new(method, format!("Use {CODEX_API_KEY_ENV_VAR}")).description(format!(
                    "Requires setting the `{CODEX_API_KEY_ENV_VAR}` environment variable."
                ))
            }
            CodexAuthMethod::OpenAiApiKey => {
                Self::new(method, format!("Use {OPENAI_API_KEY_ENV_VAR}")).description(format!(
                    "Requires setting the `{OPENAI_API_KEY_ENV_VAR}` environment variable."
                ))
            }
        }
    }
}

impl TryFrom<AuthMethodId> for CodexAuthMethod {
    type Error = Error;

    fn try_from(value: AuthMethodId) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "chatgpt" => Ok(CodexAuthMethod::ChatGpt),
            "codex-api-key" => Ok(CodexAuthMethod::CodexApiKey),
            "openai-api-key" => Ok(CodexAuthMethod::OpenAiApiKey),
            _ => Err(Error::invalid_params().data("unsupported authentication method")),
        }
    }
}

fn truncate_graphemes(text: &str, max_graphemes: usize) -> String {
    let mut graphemes = text.grapheme_indices(true);

    if let Some((byte_index, _)) = graphemes.nth(max_graphemes) {
        if max_graphemes >= 3 {
            let mut truncate_graphemes = text.grapheme_indices(true);
            if let Some((truncate_byte_index, _)) = truncate_graphemes.nth(max_graphemes - 3) {
                let truncated = &text[..truncate_byte_index];
                format!("{truncated}...")
            } else {
                text.to_string()
            }
        } else {
            let truncated = &text[..byte_index];
            truncated.to_string()
        }
    } else {
        text.to_string()
    }
}

fn format_session_title(message: &str) -> Option<String> {
    let normalized = message.replace(['\r', '\n'], " ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(truncate_graphemes(trimmed, SESSION_TITLE_MAX_GRAPHEMES))
    }
}
