use agent_client_protocol::{
    Agent, AgentCapabilities, AuthenticateRequest, CancelNotification, Error, InitializeRequest,
    InitializeResponse, LoadSessionRequest, NewSessionRequest, NewSessionResponse,
    PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion, SessionId, StopReason,
};
use codex_core::config::Config;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// The Codex implementation of the ACP Agent trait.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// The underlying codex configuration
    config: Arc<Config>,

    /// Active sessions mapped by SessionId
    sessions: Arc<Mutex<HashMap<SessionId, SessionState>>>,
}

/// State for an individual session
struct SessionState {
    /// The working directory for this session
    cwd: String,

    /// The conversation ID in the conversation manager
    conversation_id: String,
}

impl CodexAgent {
    /// Create a new CodexAgent with the given configuration
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            config,
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

        async move {
            info!("Creating new session with cwd: {}", request.cwd.display());

            // Generate a unique session ID
            let session_id = SessionId(Arc::from(format!("sess_{}", uuid::Uuid::new_v4())));

            // Create a new conversation ID
            let conversation_id = format!("conv_{}", uuid::Uuid::new_v4());

            // Store the session state
            let session_state = SessionState {
                cwd: request.cwd.to_string_lossy().to_string(),
                conversation_id,
            };

            let mut sessions = sessions.lock().await;
            sessions.insert(session_id.clone(), session_state);

            // TODO: Connect to MCP servers specified in request.mcp_servers
            for server in &request.mcp_servers {
                debug!(
                    "Would connect to MCP server: {} at {}",
                    server.name,
                    server.command.display()
                );
            }

            Ok(NewSessionResponse { session_id })
        }
    }

    fn load_session(&self, request: LoadSessionRequest) -> impl Future<Output = Result<(), Error>> {
        let sessions = self.sessions.clone();

        async move {
            info!("Loading session: {}", request.session_id);

            // Check if we have this session
            let sessions = sessions.lock().await;
            if !sessions.contains_key(&request.session_id) {
                return Err(Error::invalid_request());
            }

            // TODO: Actually restore the session state from storage
            // This would involve loading conversation history, reconnecting to MCP servers, etc.

            Ok(())
        }
    }

    fn prompt(
        &self,
        request: PromptRequest,
    ) -> impl Future<Output = Result<PromptResponse, Error>> {
        let sessions = self.sessions.clone();

        async move {
            info!("Processing prompt for session: {}", request.session_id);

            // Get the session state
            let sessions = sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .ok_or_else(Error::invalid_request)?;

            let _conversation_id = session.conversation_id.clone();
            drop(sessions);

            // TODO: Convert ACP prompt format to codex format and process
            // This would involve:
            // 1. Converting ContentBlocks to codex's input format
            // 2. Sending to the conversation manager
            // 3. Streaming updates back via session notifications
            // 4. Handling tool calls through MCP

            // For now, just return a placeholder response
            warn!("Prompt processing not yet fully implemented");

            Ok(PromptResponse {
                stop_reason: StopReason::EndTurn,
            })
        }
    }

    fn cancel(&self, notification: CancelNotification) -> impl Future<Output = Result<(), Error>> {
        async move {
            info!(
                "Cancelling operations for session: {}",
                notification.session_id
            );

            // TODO: Implement cancellation logic
            // This would stop any ongoing prompt processing for the specified session

            Ok(())
        }
    }
}
