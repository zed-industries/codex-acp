use agent_client_protocol::{
    Agent, AgentCapabilities, AuthMethod, AuthMethodId, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, ClientCapabilities, Error, Implementation, InitializeRequest,
    InitializeResponse, LoadSessionRequest, LoadSessionResponse, McpCapabilities, McpServer,
    ModelId, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    PromptResponse, SessionId, SessionModeId, SetSessionModeRequest, SetSessionModeResponse,
    SetSessionModelRequest, SetSessionModelResponse, V1,
};
use codex_core::{
    ConversationManager, NewConversation,
    auth::{AuthManager, read_codex_api_key_from_env, read_openai_api_key_from_env},
    config::{
        Config,
        types::{McpServerConfig, McpServerTransportConfig},
    },
    protocol::SessionSource,
};
use codex_login::{AuthMode, CODEX_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR};
use codex_protocol::ConversationId;
use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex},
};
use tracing::{debug, info, warn};

use crate::{
    conversation::Conversation,
    local_spawner::{AcpFs, LocalSpawner},
    session_store::{self, SessionRecord, SessionStore},
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
    /// Conversation manager for handling sessions
    conversation_manager: ConversationManager,
    /// Active sessions mapped by `SessionId`
    sessions: Rc<RefCell<HashMap<SessionId, Rc<Conversation>>>>,
    /// Persistent session manifest store
    session_store: SessionStore,
}

impl CodexAgent {
    /// Create a new `CodexAgent` with the given configuration
    pub fn new(config: Config, session_store: SessionStore) -> Self {
        let auth_manager = AuthManager::shared(
            config.codex_home.clone(),
            false,
            config.cli_auth_credentials_store_mode,
        );

        let client_capabilities: Arc<Mutex<ClientCapabilities>> = Arc::default();

        let local_spawner = LocalSpawner::new();
        let capabilities_clone = client_capabilities.clone();
        let conversation_manager =
            ConversationManager::new(auth_manager.clone(), SessionSource::Unknown).with_fs(
                Box::new(move |conversation_id| {
                    Arc::new(AcpFs::new(
                        Self::session_id_from_conversation_id(conversation_id),
                        capabilities_clone.clone(),
                        local_spawner.clone(),
                    ))
                }),
            );
        Self {
            auth_manager,
            client_capabilities,
            config,
            conversation_manager,
            sessions: Rc::default(),
            session_store,
        }
    }

    fn apply_session_request(config: &mut Config, cwd: &Path, mcp_servers: &[McpServer]) {
        // Allows us to support HTTP MCP servers
        config.use_experimental_use_rmcp_client = true;
        // Make sure we are going through the `apply_patch` code path
        config.include_apply_patch_tool = true;
        config.cwd = cwd.to_path_buf();

        for server in mcp_servers {
            match server {
                McpServer::Sse { .. } => {}
                McpServer::Http { name, url, headers } => {
                    let http_headers = if headers.is_empty() {
                        None
                    } else {
                        Some(
                            headers
                                .iter()
                                .map(|h| (h.name.clone(), h.value.clone()))
                                .collect(),
                        )
                    };
                    config.mcp_servers.insert(
                        name.clone(),
                        McpServerConfig {
                            transport: McpServerTransportConfig::StreamableHttp {
                                url: url.clone(),
                                bearer_token_env_var: None,
                                http_headers,
                                env_http_headers: None,
                            },
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                        },
                    );
                }
                McpServer::Stdio {
                    name,
                    command,
                    args,
                    env,
                } => {
                    let env_map = if env.is_empty() {
                        None
                    } else {
                        Some(
                            env.iter()
                                .map(|entry| (entry.name.clone(), entry.value.clone()))
                                .collect(),
                        )
                    };
                    config.mcp_servers.insert(
                        name.clone(),
                        McpServerConfig {
                            transport: McpServerTransportConfig::Stdio {
                                command: command.display().to_string(),
                                args: args.clone(),
                                env: env_map,
                                env_vars: vec![],
                                cwd: Some(cwd.to_path_buf()),
                            },
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                        },
                    );
                }
            }
        }
    }

    fn persist_record(&self, record: SessionRecord) {
        if let Err(err) = self.session_store.record_session(record) {
            warn!("failed to persist session manifest: {err}");
        }
    }

    fn persist_mode_update(&self, session_id: &SessionId, mode_id: &SessionModeId) {
        if let Err(err) = self.session_store.update_mode(session_id, mode_id) {
            warn!("failed to update persisted mode for {}: {err}", session_id);
        }
    }

    fn persist_model_update(&self, session_id: &SessionId, model_id: &ModelId) {
        if let Err(err) = self.session_store.update_model(session_id, model_id) {
            warn!("failed to update persisted model for {}: {err}", session_id);
        }
    }

    fn build_session_record(
        session_id: &SessionId,
        conversation_id: ConversationId,
        rollout_path: PathBuf,
        config: &Config,
        mcp_servers: Vec<McpServer>,
        load: &LoadSessionResponse,
    ) -> SessionRecord {
        SessionRecord {
            version: session_store::MANIFEST_VERSION,
            session_id: session_id.to_string(),
            conversation_id: conversation_id.to_string(),
            rollout_path,
            cwd: config.cwd.clone(),
            mcp_servers,
            model_id: load
                .models
                .as_ref()
                .map(|state| state.current_model_id.0.as_ref().to_string()),
            mode_id: load
                .modes
                .as_ref()
                .map(|state| state.current_mode_id.0.as_ref().to_string()),
            approval_policy: Some(config.approval_policy),
            sandbox_policy: Some(config.sandbox_policy.clone()),
            reasoning_effort: config.model_reasoning_effort,
            last_updated: session_store::current_timestamp(),
        }
    }

    fn session_id_from_conversation_id(conversation_id: ConversationId) -> SessionId {
        SessionId(conversation_id.to_string().into())
    }

    fn get_conversation(&self, session_id: &SessionId) -> Result<Rc<Conversation>, Error> {
        Ok(self
            .sessions
            .borrow()
            .get(session_id)
            .ok_or_else(Error::invalid_request)?
            .clone())
    }
}

#[async_trait::async_trait(?Send)]
impl Agent for CodexAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        let InitializeRequest {
            protocol_version,
            client_capabilities,
            client_info: _, // TODO: save and pass into Codex somehow
            meta: _,
        } = request;
        debug!("Received initialize request with protocol version {protocol_version:?}",);
        let protocol_version = V1;

        *self.client_capabilities.lock().unwrap() = client_capabilities;

        let agent_capabilities = AgentCapabilities {
            load_session: self.session_store.is_enabled(),
            prompt_capabilities: PromptCapabilities {
                audio: false,
                embedded_context: true,
                image: true,
                meta: None,
            },
            mcp_capabilities: McpCapabilities {
                http: true,
                sse: false,
                meta: None,
            },
            meta: None,
        };

        let mut auth_methods = vec![
            CodexAuthMethod::ChatGpt.into(),
            CodexAuthMethod::CodexApiKey.into(),
            CodexAuthMethod::OpenAiApiKey.into(),
        ];
        // Until codex device code auth works, we can't use this in remote ssh projects
        if std::env::var("NO_BROWSER").is_ok() {
            auth_methods.remove(0);
        }

        Ok(InitializeResponse {
            protocol_version,
            agent_capabilities,
            agent_info: Some(Implementation {
                name: "codex-acp".into(),
                title: Some("Codex".into()),
                version: env!("CARGO_PKG_VERSION").into(),
            }),
            auth_methods,
            meta: None,
        })
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        let auth_method = CodexAuthMethod::try_from(request.method_id)?;

        // Check before starting login flow if already authenticated with the same method
        if let Some(auth) = self.auth_manager.auth() {
            match (auth.mode, auth_method) {
                (
                    AuthMode::ApiKey,
                    CodexAuthMethod::CodexApiKey | CodexAuthMethod::OpenAiApiKey,
                )
                | (AuthMode::ChatGPT, CodexAuthMethod::ChatGpt) => {
                    return Ok(AuthenticateResponse { meta: None });
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
                    Error::internal_error().with_data(format!("{CODEX_API_KEY_ENV_VAR} is not set"))
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
                    Error::internal_error()
                        .with_data(format!("{OPENAI_API_KEY_ENV_VAR} is not set"))
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

        Ok(AuthenticateResponse { meta: None })
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        // Check before sending if authentication was successful or not
        if self.auth_manager.auth().is_none() {
            return Err(Error::auth_required());
        }
        let NewSessionRequest {
            cwd,
            mcp_servers,
            meta: _meta,
        } = request;
        info!("Creating new session with cwd: {}", cwd.display());

        let persisted_mcp_servers = mcp_servers.clone();
        let mut config = self.config.clone();
        Self::apply_session_request(&mut config, &cwd, &mcp_servers);

        let num_mcp_servers = config.mcp_servers.len();

        let NewConversation {
            conversation_id,
            conversation,
            session_configured,
        } = Box::pin(self.conversation_manager.new_conversation(config.clone()))
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_id = Self::session_id_from_conversation_id(conversation_id);
        let conversation = Rc::new(Conversation::new(
            session_id.clone(),
            conversation,
            self.auth_manager.clone(),
            self.client_capabilities.clone(),
            config.clone(),
        ));
        let load = conversation.load().await?;

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), conversation);

        if self.session_store.is_enabled() {
            let record = Self::build_session_record(
                &session_id,
                session_configured.session_id,
                session_configured.rollout_path.clone(),
                &config,
                persisted_mcp_servers,
                &load,
            );
            self.persist_record(record);
        }

        debug!("Created new session with {} MCP servers", num_mcp_servers);

        Ok(NewSessionResponse {
            session_id,
            modes: load.modes,
            models: load.models,
            meta: None,
        })
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        if self.auth_manager.auth().is_none() {
            return Err(Error::auth_required());
        }

        let conversation = {
            let sessions = self.sessions.borrow();
            sessions.get(&request.session_id).cloned()
        };
        if let Some(conversation) = conversation {
            return conversation.load().await;
        }

        if !self.session_store.is_enabled() {
            return Err(Error::invalid_request());
        }

        let Some(record) = self.session_store.get(&request.session_id) else {
            return Err(Error::invalid_request());
        };

        let persisted_servers = record.mcp_servers.clone();
        let mut resume_config = self.config.clone();
        Self::apply_session_request(&mut resume_config, &record.cwd, &persisted_servers);
        session_store::apply_session_config_overrides(&mut resume_config, &record);

        let config_for_actor = resume_config.clone();
        let record_config = resume_config.clone();

        let NewConversation {
            conversation_id,
            conversation,
            session_configured,
        } = self
            .conversation_manager
            .resume_conversation_from_rollout(
                resume_config,
                record.rollout_path.clone(),
                self.auth_manager.clone(),
            )
            .await
            .map_err(|e| {
                Error::internal_error().with_data(format!("failed to resume session: {e}"))
            })?;

        let session_id = Self::session_id_from_conversation_id(conversation_id);
        if session_id != request.session_id {
            warn!(
                "Session ID mismatch when resuming: requested={}, resumed={}",
                request.session_id, session_id
            );
        }

        let conversation = Rc::new(Conversation::new(
            session_id.clone(),
            conversation,
            self.auth_manager.clone(),
            self.client_capabilities.clone(),
            config_for_actor,
        ));
        let load = conversation.load().await?;

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), conversation);

        let updated_record = Self::build_session_record(
            &session_id,
            session_configured.session_id,
            session_configured.rollout_path.clone(),
            &record_config,
            persisted_servers,
            &load,
        );
        self.persist_record(updated_record);

        Ok(load)
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        if self.auth_manager.auth().is_none() {
            return Err(Error::auth_required());
        }

        // Get the session state
        let conversation = self.get_conversation(&request.session_id)?;
        let stop_reason = conversation.prompt(request).await?;

        Ok(PromptResponse {
            stop_reason,
            meta: None,
        })
    }

    async fn cancel(&self, args: CancelNotification) -> Result<(), Error> {
        info!("Cancelling operations for session: {}", args.session_id);
        self.get_conversation(&args.session_id)?.cancel().await?;
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting session mode for session: {}", args.session_id);
        let session_id = args.session_id;
        let mode_id = args.mode_id;
        self.get_conversation(&session_id)?
            .set_mode(mode_id.clone())
            .await?;
        self.persist_mode_update(&session_id, &mode_id);
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        args: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!("Setting session model for session: {}", args.session_id);

        let session_id = args.session_id;
        let model_id = args.model_id;

        self.get_conversation(&session_id)?
            .set_model(model_id.clone())
            .await?;

        self.persist_model_update(&session_id, &model_id);

        Ok(SetSessionModelResponse::default())
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
        Self(
            match method {
                CodexAuthMethod::ChatGpt => "chatgpt",
                CodexAuthMethod::CodexApiKey => "codex-api-key",
                CodexAuthMethod::OpenAiApiKey => "openai-api-key",
            }
            .into(),
        )
    }
}

impl From<CodexAuthMethod> for AuthMethod {
    fn from(method: CodexAuthMethod) -> Self {
        match method {
            CodexAuthMethod::ChatGpt => Self {
                id: method.into(),
                name: "Login with ChatGPT".into(),
                description: Some(
                    "Use your ChatGPT login with Codex CLI (requires a paid ChatGPT subscription)"
                        .into(),
                ),
                meta: None,
            },
            CodexAuthMethod::CodexApiKey => Self {
                id: method.into(),
                name: format!("Use {CODEX_API_KEY_ENV_VAR}"),
                description: Some(format!(
                    "Requires setting the `{CODEX_API_KEY_ENV_VAR}` environment variable."
                )),
                meta: None,
            },
            CodexAuthMethod::OpenAiApiKey => Self {
                id: method.into(),
                name: format!("Use {OPENAI_API_KEY_ENV_VAR}"),
                description: Some(format!(
                    "Requires setting the `{OPENAI_API_KEY_ENV_VAR}` environment variable."
                )),
                meta: None,
            },
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
            _ => Err(Error::invalid_params().with_data("unsupported authentication method")),
        }
    }
}
