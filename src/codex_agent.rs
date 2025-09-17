use agent_client_protocol::{
    Agent, AgentCapabilities, AuthenticateRequest, AuthenticateResponse, CancelNotification, Error,
    InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse,
    McpCapabilities, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    PromptResponse, ProtocolVersion, SessionId, StopReason,
};
use codex_core::ConversationManager;
use codex_core::auth::AuthManager;
use codex_core::config::Config;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::mcp_protocol::ConversationId;
use codex_protocol::protocol::{AskForApproval, InputItem, Op, SandboxPolicy};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// The Codex implementation of the ACP Agent trait.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// The underlying codex configuration
    config: Arc<Config>,

    /// Authentication manager for handling credentials
    auth_manager: Arc<AuthManager>,

    /// Conversation manager for handling sessions
    conversation_manager: Arc<Mutex<ConversationManager>>,

    /// Active sessions mapped by SessionId
    sessions: Arc<Mutex<HashMap<SessionId, SessionState>>>,
}

/// State for an individual session
struct SessionState {
    /// The working directory for this session
    cwd: String,

    /// The conversation ID in the conversation manager
    conversation_id: ConversationId,

    /// The model being used for this session
    model: String,
}

impl CodexAgent {
    /// Create a new CodexAgent with the given configuration
    pub fn new(config: Arc<Config>) -> Self {
        // Initialize AuthManager
        // Initialize AuthManager - it needs a path, not config
        let codex_home = config.codex_home.clone();
        let auth_manager = AuthManager::shared(codex_home);

        // Initialize ConversationManager with auth manager
        let conversation_manager =
            Arc::new(Mutex::new(ConversationManager::new(auth_manager.clone())));

        Self {
            config,
            auth_manager,
            conversation_manager,
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Agent for CodexAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        debug!(
            "Received initialize request with protocol version {:?}",
            request.protocol_version
        );

        // For now, we support protocol version 1
        let protocol_version = ProtocolVersion::default();

        // Build list of available models from codex configuration

        // Define our agent capabilities
        let agent_capabilities = AgentCapabilities {
            load_session: true,
            prompt_capabilities: PromptCapabilities {
                audio: false,
                embedded_context: true,
                image: true,
                meta: None,
            },
            mcp_capabilities: McpCapabilities {
                http: false, // TODO: Revisit
                sse: false,
                meta: None,
            },
            meta: None,
        };

        // For now, we don't require authentication
        let auth_methods = vec![];

        Ok(InitializeResponse {
            protocol_version,
            agent_capabilities,
            auth_methods,
            meta: None,
        })
    }

    async fn authenticate(
        &self,
        _request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        // We don't currently require authentication
        Ok(AuthenticateResponse { meta: None })
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        info!("Creating new session with cwd: {}", request.cwd.display());

        // Generate a unique session ID
        let session_id = SessionId(Arc::from(format!("sess_{}", uuid::Uuid::new_v4())));

        // Create a new conversation in the conversation manager
        let conv_manager = self.conversation_manager.lock().await;

        // Create config for the new conversation
        // TODO: Set working directory and MCP servers in the config
        let conv_config = (*self.config).clone();

        // Create the conversation and get back the NewConversation struct
        let new_conversation = conv_manager
            .new_conversation(conv_config)
            .await
            .map_err(|_e| Error::internal_error())?;

        // Store the session state
        let session_state = SessionState {
            cwd: request.cwd.to_string_lossy().to_string(),
            conversation_id: new_conversation.conversation_id,
            model: self.config.model.clone(),
        };

        let mut sessions = self.sessions.lock().await;
        sessions.insert(session_id.clone(), session_state);

        // The conversation manager will handle connecting to MCP servers
        debug!(
            "Created new session with {} MCP servers",
            request.mcp_servers.len()
        );

        Ok(NewSessionResponse {
            session_id,
            modes: None,
            meta: None,
        })
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);

        // Check if we have this session already
        let sessions = self.sessions.lock().await;
        if sessions.contains_key(&request.session_id) {
            // Session already loaded
            return Ok(LoadSessionResponse {
                modes: None,
                meta: None,
            });
        }

        // For now, we can't actually load sessions from disk
        // The conversation manager doesn't have a direct load method
        // We would need to use resume_conversation_from_rollout with a rollout path
        return Err(Error::invalid_request());
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);

        // Get the session state
        let (conversation_id, cwd, model) = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .ok_or_else(Error::invalid_request)?;

            (
                session.conversation_id.clone(),
                session.cwd.clone(),
                session.model.clone(),
            )
        };

        // Get the conversation from the manager
        let conv_manager = self.conversation_manager.lock().await;
        let conversation = conv_manager
            .get_conversation(conversation_id)
            .await
            .map_err(|_e| Error::invalid_request())?;

        // Convert ACP prompt format to codex format
        let mut input_items = Vec::new();
        for block in &request.prompt {
            // TODO make this a ::collect() instead of a `for` loop
            use agent_client_protocol::ContentBlock;
            match block {
                ContentBlock::Text(text_block) => {
                    input_items.push(InputItem::Text {
                        text: text_block.text.clone(),
                    });
                }
                ContentBlock::Image(image_block) => {
                    // Convert to data URI if needed
                    if let Some(uri) = &image_block.uri {
                        input_items.push(InputItem::Image {
                            image_url: uri.clone(),
                        });
                    } else {
                        // Base64 data
                        let data_uri = format!(
                            "data:{};base64,{}",
                            image_block.mime_type.clone(),
                            image_block.data.clone()
                        );
                        input_items.push(InputItem::Image {
                            image_url: data_uri,
                        });
                    }
                }
                _ => {
                    // Skip other content types for now
                }
            }
        }

        let cwd = PathBuf::from(cwd);
        let approval_policy = AskForApproval::Never; // TODO get this from outside
        let sandbox_policy = SandboxPolicy::WorkspaceWrite {
            // TODO get this from outside
            writable_roots: vec![],
            network_access: false,
            exclude_tmpdir_env_var: false,
            exclude_slash_tmp: false,
        };
        let effort = None; // TODO get this from outside
        let summary = ReasoningSummaryConfig::Auto; // TODO get this from outside
        let submission_id = conversation
            .submit(Op::UserTurn {
                items: input_items,
                cwd,
                approval_policy,
                sandbox_policy,
                model,
                effort,
                summary,
            })
            .await
            .map_err(|_e| Error::internal_error())?;

        debug!("Submitted prompt with submission_id: {}", submission_id);

        // TODO: Stream updates back via session notifications
        // This would involve:
        // 1. Setting up a stream from the conversation manager
        // 2. Converting events to SessionNotification messages
        // 3. Sending them via the Client handle
        // 4. Handling tool calls through MCP

        // For now, just return a basic completion
        Ok(PromptResponse {
            stop_reason: StopReason::EndTurn,
            meta: None,
        })
    }

    async fn cancel(&self, notification: CancelNotification) -> Result<(), Error> {
        info!(
            "Cancelling operations for session: {}",
            notification.session_id
        );

        // Get the session to find the conversation ID
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(&notification.session_id)
            .ok_or_else(Error::invalid_request)?;

        let conversation_id = session.conversation_id.clone();
        drop(sessions);

        // Get the conversation and cancel it
        let conv_manager = self.conversation_manager.lock().await;
        if let Ok(_conversation) = conv_manager.get_conversation(conversation_id).await {
            // TODO: Call conversation.cancel() or similar method
            debug!("Would cancel conversation");
        }

        Ok(())
    }

    async fn set_session_mode(
        &self,
        _args: agent_client_protocol::SetSessionModeRequest,
    ) -> Result<agent_client_protocol::SetSessionModeResponse, Error> {
        todo!()
    }

    async fn ext_method(
        &self,
        _args: agent_client_protocol::ExtRequest,
    ) -> Result<agent_client_protocol::ExtResponse, Error> {
        todo!()
    }

    async fn ext_notification(
        &self,
        _args: agent_client_protocol::ExtNotification,
    ) -> Result<(), Error> {
        todo!()
    }
}
