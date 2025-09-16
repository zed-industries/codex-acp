use agent_client_protocol::{
    Agent, AgentCapabilities, AuthenticateRequest, CancelNotification, Error, InitializeRequest,
    InitializeResponse, LoadSessionRequest, NewSessionRequest, NewSessionResponse,
    PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion, SessionId, StopReason,
};
use codex_core::auth::AuthManager;
use codex_core::config::Config;
use codex_core::{ConversationManager, protocol::Op};
use codex_protocol::mcp_protocol::ConversationId;
use std::collections::HashMap;
use std::future::Future;
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
    fn initialize(
        &self,
        request: InitializeRequest,
    ) -> impl Future<Output = Result<InitializeResponse, Error>> {
        async move {
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
                },
            };

            // For now, we don't require authentication
            let auth_methods = vec![];

            Ok(InitializeResponse {
                protocol_version,
                agent_capabilities,
                auth_methods,
            })
        }
    }

    fn authenticate(
        &self,
        _request: AuthenticateRequest,
    ) -> impl Future<Output = Result<(), Error>> {
        async move {
            // We don't currently require authentication
            Ok(())
        }
    }

    fn new_session(
        &self,
        request: NewSessionRequest,
    ) -> impl Future<Output = Result<NewSessionResponse, Error>> {
        let sessions = self.sessions.clone();
        let conversation_manager = self.conversation_manager.clone();
        let config = self.config.clone();

        async move {
            info!("Creating new session with cwd: {}", request.cwd.display());

            // Generate a unique session ID
            let session_id = SessionId(Arc::from(format!("sess_{}", uuid::Uuid::new_v4())));

            // Create a new conversation in the conversation manager
            let conv_manager = conversation_manager.lock().await;

            // Create config for the new conversation
            // TODO: Set working directory and MCP servers in the config
            let conv_config = (*config).clone();

            // Create the conversation and get back the NewConversation struct
            let new_conversation = conv_manager
                .new_conversation(conv_config)
                .await
                .map_err(|_e| Error::internal_error())?;

            // Store the session state
            let session_state = SessionState {
                cwd: request.cwd.to_string_lossy().to_string(),
                conversation_id: new_conversation.conversation_id,
            };

            let mut sessions = sessions.lock().await;
            sessions.insert(session_id.clone(), session_state);

            // The conversation manager will handle connecting to MCP servers
            debug!(
                "Created new session with {} MCP servers",
                request.mcp_servers.len()
            );

            Ok(NewSessionResponse { session_id })
        }
    }

    fn load_session(&self, request: LoadSessionRequest) -> impl Future<Output = Result<(), Error>> {
        let sessions = self.sessions.clone();
        let conversation_manager = self.conversation_manager.clone();

        async move {
            info!("Loading session: {}", request.session_id);

            // Check if we have this session already
            let mut sessions = sessions.lock().await;
            if sessions.contains_key(&request.session_id) {
                // Session already loaded
                return Ok(());
            }

            // For now, we can't actually load sessions from disk
            // The conversation manager doesn't have a direct load method
            // We would need to use resume_conversation_from_rollout with a rollout path
            return Err(Error::invalid_request());
        }
    }

    fn prompt(
        &self,
        request: PromptRequest,
    ) -> impl Future<Output = Result<PromptResponse, Error>> {
        let sessions = self.sessions.clone();
        let conversation_manager = self.conversation_manager.clone();

        async move {
            info!("Processing prompt for session: {}", request.session_id);

            // Get the session state
            let sessions = sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .ok_or_else(Error::invalid_request)?;

            let conversation_id = session.conversation_id.clone();
            drop(sessions);

            // Get the conversation from the manager
            let conv_manager = conversation_manager.lock().await;
            let conversation = conv_manager
                .get_conversation(conversation_id)
                .await
                .map_err(|_e| Error::invalid_request())?;

            // Convert ACP prompt format to codex format
            // TODO: Properly convert ContentBlocks to codex's input format
            let codex_prompt = request
                .prompt
                .iter()
                .filter_map(|block| {
                    // Extract text content for now
                    use agent_client_protocol::ContentBlock;
                    match block {
                        ContentBlock::Text(text_block) => Some(text_block.text.clone()),
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            let submission_id = conversation
                .submit(Op::UserTurn {
                    items: (),
                    cwd: (),
                    approval_policy: (),
                    sandbox_policy: (),
                    model: (),
                    effort: (),
                    summary: (),
                })
                .await?;

            // TODO: Stream updates back via session notifications
            // This would involve:
            // 1. Setting up a stream from the conversation manager
            // 2. Converting events to SessionNotification messages
            // 3. Sending them via the Client handle
            // 4. Handling tool calls through MCP

            // For now, just return a basic completion
            Ok(PromptResponse {
                stop_reason: StopReason::EndTurn,
            })
        }
    }

    fn cancel(&self, notification: CancelNotification) -> impl Future<Output = Result<(), Error>> {
        let sessions = self.sessions.clone();
        let conversation_manager = self.conversation_manager.clone();

        async move {
            info!(
                "Cancelling operations for session: {}",
                notification.session_id
            );

            // Get the session to find the conversation ID
            let sessions = sessions.lock().await;
            let session = sessions
                .get(&notification.session_id)
                .ok_or_else(Error::invalid_request)?;

            let conversation_id = session.conversation_id.clone();
            drop(sessions);

            // Get the conversation and cancel it
            let conv_manager = conversation_manager.lock().await;
            if let Ok(_conversation) = conv_manager.get_conversation(conversation_id).await {
                // TODO: Call conversation.cancel() or similar method
                debug!("Would cancel conversation");
            }

            Ok(())
        }
    }
}
