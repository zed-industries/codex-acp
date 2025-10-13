use agent_client_protocol::{
    Agent, AgentCapabilities, AuthMethod, AuthMethodId, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, ClientCapabilities, Error, InitializeRequest, InitializeResponse,
    LoadSessionRequest, LoadSessionResponse, McpCapabilities, McpServer, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, SessionId,
    SetSessionModeRequest, SetSessionModeResponse, SetSessionModelRequest, SetSessionModelResponse,
    V1,
};
use codex_common::model_presets::{ModelPreset, builtin_model_presets};
use codex_core::{
    ConversationManager, NewConversation,
    auth::{AuthManager, read_codex_api_key_from_env, read_openai_api_key_from_env},
    config::Config,
    config_types::{McpServerConfig, McpServerTransportConfig},
    protocol::SessionSource,
};
use codex_login::{CODEX_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR};
use codex_protocol::ConversationId;
use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};
use tracing::{debug, info};

use crate::{
    conversation::Conversation,
    local_spawner::{AcpFs, LocalSpawner},
};

/// The Codex implementation of the ACP Agent trait.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// Handle to the current authentication
    auth_manager: Arc<AuthManager>,
    /// Capabilities of the connected client
    client_capabilities: RefCell<ClientCapabilities>,
    /// The underlying codex configuration
    config: Config,
    /// Conversation manager for handling sessions
    conversation_manager: ConversationManager,
    /// Active sessions mapped by `SessionId`
    sessions: Rc<RefCell<HashMap<SessionId, Rc<Conversation>>>>,
    /// Default model presets for a given auth mode
    model_presets: Rc<Vec<ModelPreset>>,
}

impl CodexAgent {
    /// Create a new `CodexAgent` with the given configuration
    pub fn new(config: Config) -> Self {
        let auth_manager = AuthManager::shared(config.codex_home.clone(), false);

        let model_presets = Rc::new(builtin_model_presets(
            auth_manager.auth().map(|auth| auth.mode),
        ));
        let local_spawner = LocalSpawner::new();
        let conversation_manager =
            ConversationManager::new(auth_manager.clone(), SessionSource::Unknown).with_fs(
                Box::new(move |conversation_id| {
                    Arc::new(AcpFs::new(
                        Self::session_id_from_conversation_id(conversation_id),
                        local_spawner.clone(),
                    ))
                }),
            );
        Self {
            auth_manager,
            client_capabilities: RefCell::default(),
            config,
            conversation_manager,
            sessions: Rc::default(),
            model_presets,
        }
    }

    fn session_id_from_conversation_id(conversation_id: ConversationId) -> SessionId {
        SessionId(conversation_id.to_string().into())
    }

    async fn get_conversation(&self, session_id: &SessionId) -> Result<Rc<Conversation>, Error> {
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
        debug!(
            "Received initialize request with protocol version {:?}",
            request.protocol_version
        );
        let protocol_version = V1;

        self.client_capabilities
            .replace(request.client_capabilities);

        let agent_capabilities = AgentCapabilities {
            load_session: false, // Currently only able to do in-memory... which doesn't help us at the moment
            prompt_capabilities: PromptCapabilities {
                audio: false,
                embedded_context: true,
                image: true,
                meta: None,
            },
            mcp_capabilities: McpCapabilities {
                http: false,
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
            auth_methods,
            meta: None,
        })
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        let auth_method = CodexAuthMethod::try_from(request.method_id)?;

        match auth_method {
            CodexAuthMethod::ChatGpt => {
                // Perform browser/device login via codex-rs, then report success/failure to the client.
                let opts = codex_login::ServerOptions::new(
                    self.config.codex_home.clone(),
                    codex_core::auth::CLIENT_ID.to_string(),
                );

                let server =
                    codex_login::run_login_server(opts).map_err(Error::into_internal_error)?;

                server
                    .block_until_done()
                    .await
                    .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::CodexApiKey => {
                let api_key = read_codex_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().with_data(format!("{CODEX_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(&self.config.codex_home, &api_key)
                    .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::OpenAiApiKey => {
                let api_key = read_openai_api_key_from_env().ok_or_else(|| {
                    Error::internal_error()
                        .with_data(format!("{OPENAI_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(&self.config.codex_home, &api_key)
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

        let mut config = self.config.clone();
        // Allows us to support HTTP MCP servers
        config.use_experimental_use_rmcp_client = true;
        // Make sure we are going through the `apply_patch` code path
        config.include_apply_patch_tool = true;
        config.include_plan_tool = true;
        config.cwd.clone_from(&cwd);

        // Propagate any client-provided MCP servers that codex-rs supports.
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex
                McpServer::Sse { .. } => {}
                McpServer::Http {
                    name,
                    url,
                    headers: _,
                } => {
                    config.mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::StreamableHttp {
                                url,
                                bearer_token_env_var: None,
                            },
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                        },
                    );
                }
                McpServer::Stdio {
                    name,
                    command,
                    args,
                    env,
                } => {
                    config.mcp_servers.insert(
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
                            },
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                        },
                    );
                }
            }
        }

        let num_mcp_servers = config.mcp_servers.len();

        let NewConversation {
            conversation_id,
            conversation,
            session_configured: _,
        } = self
            .conversation_manager
            .new_conversation(config.clone())
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_id = Self::session_id_from_conversation_id(conversation_id);
        let conversation = Rc::new(Conversation::new(
            session_id.clone(),
            conversation,
            self.auth_manager.clone(),
            self.client_capabilities.borrow().clone(),
            config.clone(),
            self.model_presets.clone(),
        ));
        let load = conversation.load().await?;

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), conversation);

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

        // Check if we have this session already
        let Some(conversation) = self.sessions.borrow().get(&request.session_id).cloned() else {
            // For now, we can't actually load sessions from disk
            // The conversation manager doesn't have a direct load method
            // We would need to use resume_conversation_from_rollout with a rollout path
            return Err(Error::invalid_request());
        };

        Ok(conversation.load().await?)
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        if self.auth_manager.auth().is_none() {
            return Err(Error::auth_required());
        }

        // Get the session state
        let conversation = self.get_conversation(&request.session_id).await?;
        let stop_reason = conversation.prompt(request).await?;

        Ok(PromptResponse {
            stop_reason,
            meta: None,
        })
    }

    async fn cancel(&self, args: CancelNotification) -> Result<(), Error> {
        info!("Cancelling operations for session: {}", args.session_id);
        self.get_conversation(&args.session_id)
            .await?
            .cancel()
            .await?;
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting session mode for session: {}", args.session_id);
        self.get_conversation(&args.session_id)
            .await?
            .set_mode(args.mode_id)
            .await?;
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        args: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!("Setting session model for session: {}", args.session_id);

        self.get_conversation(&args.session_id)
            .await?
            .set_model(args.model_id)
            .await?;

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
