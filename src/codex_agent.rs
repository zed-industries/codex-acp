use agent_client_protocol::{
    Agent, AgentCapabilities, AuthenticateRequest, AuthenticateResponse, CancelNotification,
    ContentBlock, Error, InitializeRequest, InitializeResponse, LoadSessionRequest,
    LoadSessionResponse, McpCapabilities, NewSessionRequest, NewSessionResponse,
    PromptCapabilities, PromptRequest, PromptResponse, SessionId, SessionMode, SessionModeId,
    SessionModeState, SessionNotification, SessionUpdate, SetSessionModeRequest,
    SetSessionModeResponse, StopReason, TextContent, V1,
};
use codex_common::approval_presets::{ApprovalPreset, builtin_approval_presets};
use codex_core::auth::{AuthManager, CodexAuth, read_openai_api_key_from_env};
use codex_core::config::Config;
use codex_core::config_types::McpServerConfig;
use codex_core::{CodexConversation, ConversationManager};
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::mcp_protocol::ConversationId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::{AskForApproval, InputItem, Op, SandboxPolicy};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);

/// The Codex implementation of the ACP Agent trait.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// The underlying codex configuration
    config: Config,
    /// Conversation manager for handling sessions
    conversation_manager: ConversationManager,
    /// Active sessions mapped by SessionId
    sessions: Rc<RefCell<HashMap<SessionId, SessionState>>>,
    /// Channel for sending notifications back to the client
    notification_tx: mpsc::UnboundedSender<SessionNotification>,
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
    pub fn new(
        config: Config,
        notification_tx: mpsc::UnboundedSender<SessionNotification>, // TODO maybe make it bounded
    ) -> Self {
        let auth_manager = AuthManager::shared(config.codex_home.clone());

        let auth_manager = if auth_manager.auth().is_none() {
            // No auth.json found, try environment variable
            if let Some(api_key) = read_openai_api_key_from_env() {
                // TODO obviously this is "for testing" - let's try to find a more robust way!
                AuthManager::from_auth_for_testing(CodexAuth::from_api_key(&api_key))
            } else {
                // TODO report this to end user
                warn!(
                    "No authentication configured: neither auth.json nor OPENAI_API_KEY environment variable found"
                );
                auth_manager
            }
        } else {
            auth_manager
        };

        let conversation_manager = ConversationManager::new(auth_manager.clone());

        Self {
            config,
            conversation_manager,
            sessions: Rc::new(RefCell::new(HashMap::new())),
            notification_tx,
        }
    }

    fn modes(config: &Config) -> Option<SessionModeState> {
        let current_mode_id = APPROVAL_PRESETS
            .iter()
            .find(|preset| {
                preset.approval == config.approval_policy && preset.sandbox == config.sandbox_policy
            })
            .map(|preset| SessionModeId(preset.id.into()))?;

        Some(SessionModeState {
            current_mode_id,
            available_modes: APPROVAL_PRESETS
                .iter()
                .map(|preset| SessionMode {
                    id: SessionModeId(preset.id.into()),
                    name: preset.label.to_owned(),
                    description: Some(preset.description.to_owned()),
                    meta: None,
                })
                .collect(),
            meta: None,
        })
    }

    async fn get_conversation(
        &self,
        session_id: &SessionId,
    ) -> Result<Arc<CodexConversation>, Error> {
        // Get the session to find the conversation ID
        let conversation_id = self
            .sessions
            .borrow()
            .get(session_id)
            .ok_or_else(Error::invalid_request)?
            .conversation_id;

        self.conversation_manager
            .get_conversation(conversation_id)
            .await
            .map_err(|e| anyhow::anyhow!(e).into())
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
                http: false,
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
        let NewSessionRequest {
            cwd,
            mcp_servers,
            meta: _meta,
        } = request;
        info!("Creating new session with cwd: {}", cwd.display());

        // Generate a unique session ID
        let session_id = SessionId(Arc::from(format!("sess_{}", uuid::Uuid::new_v4())));

        // Create config for the new conversation
        // TODO: Set working directory and MCP servers in the config
        let mut config = self.config.clone();
        config.cwd = cwd.clone();
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex yet
                agent_client_protocol::McpServer::Http { .. }
                | agent_client_protocol::McpServer::Sse { .. } => {}
                agent_client_protocol::McpServer::Stdio {
                    name,
                    command,
                    args,
                    env,
                } => {
                    config.mcp_servers.insert(
                        name.clone(),
                        McpServerConfig {
                            command: command.display().to_string(),
                            args,
                            env: if env.is_empty() {
                                None
                            } else {
                                Some(env.into_iter().map(|env| (env.name, env.value)).collect())
                            },
                            startup_timeout_ms: None,
                        },
                    );
                }
            }
        }
        let num_mcp_servers = config.mcp_servers.len();

        let modes = Self::modes(&config);

        let new_conversation = self
            .conversation_manager
            .new_conversation(config)
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_state = SessionState {
            cwd: cwd.to_string_lossy().to_string(),
            conversation_id: new_conversation.conversation_id,
            model: self.config.model.clone(),
        };

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), session_state);

        debug!("Created new session with {} MCP servers", num_mcp_servers);

        Ok(NewSessionResponse {
            session_id,
            modes,
            meta: None,
        })
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);

        // Check if we have this session already
        if self.sessions.borrow().contains_key(&request.session_id) {
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
            let sessions = self.sessions.borrow();
            let session = sessions
                .get(&request.session_id)
                .ok_or_else(Error::invalid_request)?;

            (
                session.conversation_id,
                session.cwd.clone(),
                session.model.clone(),
            )
        };

        // Get the conversation from the manager
        let conversation = self
            .conversation_manager
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
                items: input_items.clone(),
                cwd: cwd.clone(),
                approval_policy,
                sandbox_policy,
                model: model.clone(),
                effort,
                summary,
            })
            .await
            .map_err(|e| {
                error!("Failed to submit prompt: {:?}", e);
                Error::internal_error()
            })?;

        info!(
            "Submitted prompt with submission_id: {}, model: {}, {} input items",
            submission_id,
            model,
            input_items.len()
        );

        // Wait for the conversation to complete (TaskComplete or TurnAborted)
        let stop_reason;

        info!(
            "Starting to wait for conversation events for submission_id: {}",
            submission_id
        );

        let mut event_count = 0;
        loop {
            event_count += 1;
            match conversation.next_event().await {
                Ok(event) => {
                    info!(
                        "Received event #{}: {:?} (id: {})",
                        event_count, event.msg, event.id
                    );

                    match event.msg {
                        EventMsg::TaskComplete(complete_event) => {
                            info!(
                                "Task completed successfully after {} events. Last agent message: {:?}",
                                event_count, complete_event.last_agent_message
                            );
                            stop_reason = StopReason::EndTurn;
                            break;
                        }
                        EventMsg::TurnAborted(abort_event) => {
                            info!("Turn aborted: {:?}", abort_event.reason);
                            stop_reason = StopReason::Cancelled;
                            break;
                        }
                        EventMsg::Error(error_event) => {
                            error!("Error during turn: {}", error_event.message);
                            return Err(Error::internal_error());
                        }
                        EventMsg::AgentMessage(msg_event) => {
                            // Send this to the client via session/update notification
                            info!("Agent message received: {:?}", msg_event.message);

                            let notification = SessionNotification {
                                session_id: request.session_id.clone(),
                                update: SessionUpdate::AgentMessageChunk {
                                    content: ContentBlock::Text(TextContent {
                                        text: msg_event.message.clone(),
                                        annotations: None,
                                        meta: None,
                                    }),
                                },
                                meta: None,
                            };

                            if let Err(e) = self.notification_tx.send(notification) {
                                error!("Failed to send session notification: {:?}", e);
                            }
                        }
                        EventMsg::UserMessage(msg_event) => {
                            info!("User message echoed: {:?}", msg_event.message);
                        }
                        _ => {
                            // TODO: handle others, many of which should become
                            // session/update notifications sent to the client
                        }
                    }
                }
                Err(e) => {
                    error!("Error getting next event: {:?}", e);
                    return Err(Error::internal_error());
                }
            }
        }

        // TODO: Stream updates back via session notifications
        // This would involve:
        // 1. Setting up a stream from the conversation manager
        // 2. Converting events to SessionNotification messages
        // 3. Sending them via the Client handle
        // 4. Handling tool calls through MCP

        Ok(PromptResponse {
            stop_reason,
            meta: None,
        })
    }

    async fn cancel(&self, notification: CancelNotification) -> Result<(), Error> {
        info!(
            "Cancelling operations for session: {}",
            notification.session_id
        );

        // Get the session to find the conversation ID
        let conversation_id = self
            .sessions
            .borrow()
            .get(&notification.session_id)
            .ok_or_else(Error::invalid_request)?
            .conversation_id;

        // Get the conversation and cancel it
        if let Ok(_conversation) = self
            .conversation_manager
            .get_conversation(conversation_id)
            .await
        {
            // TODO: Call conversation.cancel() or similar method
            debug!("Would cancel conversation");
        }

        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting session mode for session: {}", args.session_id);

        let preset = APPROVAL_PRESETS
            .iter()
            .find(|preset| args.mode_id.0.as_ref() == preset.id)
            .ok_or_else(Error::invalid_params)?;

        let conversation = self.get_conversation(&args.session_id).await?;

        conversation
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: Some(preset.approval),
                sandbox_policy: Some(preset.sandbox.clone()),
                model: None,
                effort: None,
                summary: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        Ok(SetSessionModeResponse::default())
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
