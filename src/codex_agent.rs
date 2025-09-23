use agent_client_protocol::{
    Agent, AgentCapabilities, AgentSideConnection, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, Client, ContentBlock, Error, ExtNotification, ExtRequest, ExtResponse,
    InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse,
    McpCapabilities, McpServer, ModelId, ModelInfo, NewSessionRequest, NewSessionResponse, Plan,
    PlanEntry, PlanEntryPriority, PlanEntryStatus, PromptCapabilities, PromptRequest,
    PromptResponse, SessionId, SessionMode, SessionModeId, SessionModeState, SessionModelState,
    SessionNotification, SessionUpdate, SetSessionModeRequest, SetSessionModeResponse,
    SetSessionModelRequest, SetSessionModelResponse, StopReason, TextContent, ToolCall,
    ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind, V1,
};
use codex_common::approval_presets::{ApprovalPreset, builtin_approval_presets};
use codex_common::model_presets::{ModelPreset, builtin_model_presets};
use codex_core::auth::{AuthManager, CodexAuth, read_openai_api_key_from_env};
use codex_core::config::Config;
use codex_core::config_types::McpServerConfig;
use codex_core::plan_tool::{StepStatus, UpdatePlanArgs};
use codex_core::protocol::{
    AgentMessageDeltaEvent, AgentMessageEvent, AgentReasoningDeltaEvent, AgentReasoningEvent,
    AgentReasoningRawContentDeltaEvent, AgentReasoningRawContentEvent,
    AgentReasoningSectionBreakEvent, ErrorEvent, StreamErrorEvent,
};
use codex_core::{CodexConversation, ConversationManager};
use codex_protocol::config_types::ReasoningEffort;
use codex_protocol::mcp_protocol::ConversationId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::{InputItem, Op};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};
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
    /// Default model presets for a given auth mode
    model_presets: Vec<ModelPreset>,
    /// Channel for communication with the client
    client: Rc<AgentSideConnection>,
}

/// State for an individual session
struct SessionState {
    /// The conversation ID in the conversation manager
    conversation_id: ConversationId,
    /// The config used for this session
    config: Config,
}

impl CodexAgent {
    /// Create a new `CodexAgent` with the given configuration
    pub fn new(config: Config, client: Rc<AgentSideConnection>) -> Self {
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

        let model_presets = builtin_model_presets(auth_manager.auth().map(|auth| auth.mode));

        Self {
            config,
            conversation_manager: ConversationManager::new(auth_manager),
            sessions: Rc::new(RefCell::new(HashMap::new())),
            model_presets,
            client,
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

    fn find_model_preset(&self, config: &Config) -> Option<&ModelPreset> {
        if let Some(preset) = self.model_presets.iter().find(|preset| {
            preset.model == config.model && preset.effort == config.model_reasoning_effort
        }) {
            return Some(preset);
        }

        // If we didn't find it, and it is set to none, see if we can find one with the default value
        if config.model_reasoning_effort.is_none()
            && let Some(preset) = self.model_presets.iter().find(|preset| {
                preset.model == config.model && preset.effort == Some(ReasoningEffort::default())
            })
        {
            return Some(preset);
        }

        None
    }

    fn models(&self, config: &Config) -> Result<SessionModelState, Error> {
        let current_model_id = self
            .find_model_preset(config)
            .map(|preset| ModelId(preset.id.into()))
            .ok_or_else(|| anyhow::anyhow!("No valid model preset for model {}", config.model))?;

        let available_models = self
            .model_presets
            .iter()
            .map(|preset| ModelInfo {
                model_id: ModelId(preset.id.into()),
                name: preset.label.into(),
                description: Some(
                    preset
                        .description
                        .strip_prefix("— ")
                        .unwrap_or(preset.description)
                        .into(),
                ),
                meta: None,
            })
            .collect();

        Ok(SessionModelState {
            current_model_id,
            available_models,
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

    async fn send_notification(&self, session_id: SessionId, update: SessionUpdate) {
        let notification = SessionNotification {
            session_id,
            update,
            meta: None,
        };

        if let Err(e) = self.client.session_notification(notification).await {
            error!("Failed to send session notification: {:?}", e);
        }
    }

    /// Complete an active web search by sending a completion notification
    async fn complete_web_search(
        &self,
        session_id: SessionId,
        active_web_search: &mut Option<String>,
    ) {
        if let Some(call_id) = active_web_search.take() {
            self.send_notification(
                session_id,
                SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    id: ToolCallId(call_id.into()),
                    fields: ToolCallUpdateFields {
                        status: Some(ToolCallStatus::Completed),
                        ..Default::default()
                    },
                    meta: None,
                }),
            )
            .await;
        }
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
            load_session: false, // Currently only able to do in-memory... which doens't help us at the moment
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
        config.cwd.clone_from(&cwd);
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex yet
                McpServer::Http { .. } | McpServer::Sse { .. } => {}
                McpServer::Stdio {
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
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                        },
                    );
                }
            }
        }
        let num_mcp_servers = config.mcp_servers.len();

        let modes = Self::modes(&config);
        let models = self.models(&config)?;

        let new_conversation = self
            .conversation_manager
            .new_conversation(config.clone())
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_state = SessionState {
            conversation_id: new_conversation.conversation_id,
            config,
        };

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), session_state);

        debug!("Created new session with {} MCP servers", num_mcp_servers);

        Ok(NewSessionResponse {
            session_id,
            modes,
            models: Some(models),
            meta: None,
        })
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);

        // Check if we have this session already
        if let Some(session_state) = self.sessions.borrow().get(&request.session_id) {
            // Session already loaded
            return Ok(LoadSessionResponse {
                modes: Self::modes(&session_state.config),
                models: Some(self.models(&session_state.config)?),
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
        let conversation = self.get_conversation(&request.session_id).await?;

        // Convert ACP prompt format to codex format
        let mut input_items = Vec::new();
        for block in &request.prompt {
            // TODO make this a ::collect() instead of a `for` loop
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
                ContentBlock::Audio(..)
                | ContentBlock::Resource(..)
                | ContentBlock::ResourceLink(..) => {
                    // Skip other content types for now
                }
            }
        }

        let submission_id = conversation
            .submit(Op::UserInput {
                items: input_items.clone(),
            })
            .await
            .map_err(|e| {
                error!("Failed to submit prompt: {:?}", e);
                Error::internal_error()
            })?;

        info!(
            "Submitted prompt with submission_id: {}, {} input items",
            submission_id,
            input_items.len()
        );

        // Wait for the conversation to complete (TaskComplete or TurnAborted)
        let stop_reason;

        info!(
            "Starting to wait for conversation events for submission_id: {}",
            submission_id
        );

        let mut event_count = 0;
        let mut active_web_search: Option<String> = None;
        let mut active_command: Option<(String, ToolCallId)> = None;
        let mut command_output: Vec<String> = Vec::new();
        loop {
            event_count += 1;
            match conversation.next_event().await {
                Ok(event) => {
                    info!(
                        "Received event #{}: {:?} (id: {})",
                        event_count, event.msg, event.id
                    );

                    match event.msg {
                        EventMsg::TaskStarted(event) => {
                            info!("Task started with context window of {:?}", event.model_context_window);
                        }
                        EventMsg::UserMessage(msg_event) => {
                            info!("User message echoed: {:?}", msg_event.message);
                        }
                        // Since we are getting the deltas, we can ignore these events
                        EventMsg::AgentReasoning(AgentReasoningEvent { .. })
                        | EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent {
                            ..
                        })
                        | EventMsg::AgentMessage(AgentMessageEvent { .. }) => {}
                        EventMsg::AgentMessageDelta(AgentMessageDeltaEvent { delta: message }) => {
                            // Send this to the client via session/update notification
                            info!("Agent message received: {:?}", message);

                            self.send_notification(
                                request.session_id.clone(),
                                SessionUpdate::AgentMessageChunk {
                                    content: ContentBlock::Text(TextContent {
                                        text: message,
                                        annotations: None,
                                        meta: None,
                                    }),
                                },
                            ).await;
                        }
                        EventMsg::AgentReasoningDelta(AgentReasoningDeltaEvent { delta: text })
                        | EventMsg::AgentReasoningRawContentDelta(
                            AgentReasoningRawContentDeltaEvent { delta: text },
                        ) => {
                            // Send this to the client via session/update notification
                            info!("Agent reasoning message received: {:?}", text);

                            self.send_notification(
                                request.session_id.clone(),
                                SessionUpdate::AgentThoughtChunk {
                                    content: ContentBlock::Text(TextContent {
                                        text,
                                        annotations: None,
                                        meta: None,
                                    }),
                                },
                            ).await;
                        }
                        EventMsg::AgentReasoningSectionBreak(
                            AgentReasoningSectionBreakEvent {},
                        ) => {
                            // Make sure the section heading actually get spacing
                            self.send_notification(
                                request.session_id.clone(),
                                SessionUpdate::AgentThoughtChunk {
                                    content: ContentBlock::Text(TextContent {
                                        text: "\n\n".to_owned(),
                                        annotations: None,
                                        meta: None,
                                    }),
                                },
                            ).await;
                        }
                        EventMsg::PlanUpdate(UpdatePlanArgs { explanation, plan }) => {
                            // Send this to the client via session/update notification
                            info!("Agent plan updated. Explanation: {:?}", explanation);

                            self.send_notification(
                                request.session_id.clone(),
                                SessionUpdate::Plan(Plan {
                                    entries: plan
                                        .into_iter()
                                        .map(|entry| PlanEntry {
                                            content: entry.step,
                                            priority: PlanEntryPriority::Medium,
                                            status: match entry.status {
                                                StepStatus::Pending => PlanEntryStatus::Pending,
                                                StepStatus::InProgress => {
                                                    PlanEntryStatus::InProgress
                                                }
                                                StepStatus::Completed => PlanEntryStatus::Completed,
                                            },
                                            meta: None,
                                        })
                                        .collect(),
                                    meta: None,
                                }),
                            ).await;
                        }
                        EventMsg::WebSearchBegin(search_event) => {
                            info!("Web search started: call_id={}", search_event.call_id);

                            // Complete any previous web search before starting a new one
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;
                            active_web_search = Some(search_event.call_id.clone());

                            // Create a ToolCall notification for the search beginning
                            self.send_notification(request.session_id.clone(), SessionUpdate::ToolCall(ToolCall {
                                id: ToolCallId(search_event.call_id.clone().into()),
                                title: "Searching the Web".to_string(),
                                kind: ToolKind::Fetch,
                                status: ToolCallStatus::Pending,
                                content: vec![],
                                locations: vec![],
                                raw_input: None,
                                raw_output: None,
                                meta: None,
                            })).await;
                        }
                        EventMsg::WebSearchEnd(search_event) => {
                            info!(
                                "Web search query received: call_id={}, query={}",
                                search_event.call_id, search_event.query
                            );

                            // Send update that the search is in progress with the query
                            // (WebSearchEnd just means we have the query, not that results are ready)
                            self.send_notification(
                                request.session_id.clone(),
                                SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                    id: ToolCallId(search_event.call_id.clone().into()),
                                    fields: ToolCallUpdateFields {
                                        status: Some(ToolCallStatus::InProgress),
                                        title: Some(format!(
                                            "Searching for: {}",
                                            search_event.query
                                        )),
                                        raw_input: Some(serde_json::json!({
                                            "query": search_event.query.clone()
                                        })),
                                        ..Default::default()
                                    },
                                    meta: None,
                                }),
                            ).await;

                            // The actual search results will come through AgentMessage events
                            // We mark as completed when a new tool call begins
                        }
                        EventMsg::ExecCommandBegin(exec_event) => {
                            info!(
                                "Command execution started: call_id={}, command={:?}",
                                exec_event.call_id, exec_event.command
                            );

                            // Complete any active web search when a command starts
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;

                            // Create a new tool call for the command execution
                            let tool_call_id = ToolCallId(exec_event.call_id.clone().into());
                            active_command =
                                Some((exec_event.call_id.clone(), tool_call_id.clone()));

                            self.send_notification(request.session_id.clone(), SessionUpdate::ToolCall(ToolCall {
                                id: tool_call_id,
                                title: format!("Running: {}", exec_event.command.join(" ")),
                                kind: ToolKind::Execute,
                                status: ToolCallStatus::InProgress,
                                content: vec![],
                                locations: if exec_event.cwd != std::path::PathBuf::from(".") {
                                    vec![ToolCallLocation {
                                        path: exec_event.cwd.clone(),
                                        line: None,
                                        meta: None,
                                    }]
                                } else {
                                    vec![]
                                },
                                raw_input: Some(serde_json::json!({
                                    "command": exec_event.command,
                                    "cwd": exec_event.cwd,
                                })),
                                raw_output: None,
                                meta: None,
                            })).await;
                        }
                        EventMsg::ExecCommandOutputDelta(delta_event) => {
                            // Accumulate command output and send the full content
                            if let Some((ref call_id, ref tool_call_id)) = active_command
                                && call_id == &delta_event.call_id
                            {
                                // Convert the output chunk to a string (best effort)
                                let output_text = String::from_utf8_lossy(&delta_event.chunk);

                                // Accumulate the output
                                command_output.push(output_text.to_string());

                                // Send the full accumulated output (content is replaced, not appended)
                                let accumulated_output = command_output.join("");

                                self.send_notification(request.session_id.clone(), SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                    id: tool_call_id.clone(),
                                    fields: ToolCallUpdateFields {
                                        // Send the full accumulated content
                                        content: Some(vec![ToolCallContent::Content {
                                            content: ContentBlock::Text(TextContent {
                                                text: accumulated_output,
                                                annotations: None,
                                                meta: Some(serde_json::json!({
                                                    "stream": format!("{:?}", delta_event.stream),
                                                    "streaming": true,
                                                })),
                                            }),
                                        }]),
                                        ..Default::default()
                                    },
                                    meta: None,
                                })).await;
                            }
                        }
                        EventMsg::ExecCommandEnd(end_event) => {
                            info!(
                                "Command execution ended: call_id={}, exit_code={}",
                                end_event.call_id, end_event.exit_code
                            );

                            if let Some((call_id, tool_call_id)) = active_command.take()
                                && call_id == end_event.call_id
                            {
                                let is_success = end_event.exit_code == 0;

                                let completion_update = SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                        id: tool_call_id,
                                        fields: ToolCallUpdateFields {
                                            status: Some(if is_success {
                                                ToolCallStatus::Completed
                                            } else {
                                                ToolCallStatus::Failed
                                            }),
                                            // Send final aggregated output
                                            content: Some(vec![ToolCallContent::Content {
                                                content: ContentBlock::Text(TextContent {
                                                    text: if !end_event.formatted_output.is_empty()
                                                    {
                                                        end_event.formatted_output.clone()
                                                    } else if !end_event
                                                        .aggregated_output
                                                        .is_empty()
                                                    {
                                                        end_event.aggregated_output.clone()
                                                    } else {
                                                        format!(
                                                            "stdout:\n{}\n\nstderr:\n{}",
                                                            end_event.stdout, end_event.stderr
                                                        )
                                                    },
                                                    annotations: None,
                                                    meta: None,
                                                }),
                                            }]),
                                            raw_output: Some(serde_json::json!({
                                                "exit_code": end_event.exit_code,
                                                "stdout": end_event.stdout,
                                                "stderr": end_event.stderr,
                                                "duration": end_event.duration.as_secs_f64(),
                                            })),
                                            ..Default::default()
                                        },
                                        meta: None,
                                    });

                                // Clear accumulated output since we're done
                                command_output.clear();

                                self.send_notification(request.session_id.clone(), completion_update).await;
                            }
                        }
                        EventMsg::McpToolCallBegin(_) | EventMsg::PatchApplyBegin(_) => {
                            // Complete any active web search when other tools start
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;
                            // TODO: handle these tool call events properly
                        }
                        EventMsg::TaskComplete(complete_event) => {
                            info!(
                                "Task completed successfully after {} events. Last agent message: {:?}",
                                event_count, complete_event.last_agent_message
                            );

                            // Complete any remaining active web search
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;

                            stop_reason = StopReason::EndTurn;
                            break;
                        }
                        EventMsg::Error(ErrorEvent { message })
                        | EventMsg::StreamError(StreamErrorEvent { message }) => {
                            error!("Error during turn: {}", message);
                            return Err(Error::internal_error().with_data(message));
                        }
                        EventMsg::TurnAborted(abort_event) => {
                            info!("Turn aborted: {:?}", abort_event.reason);

                            // Complete any remaining active web search
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;
                            stop_reason = StopReason::Cancelled;
                            break;
                        }
                        EventMsg::ShutdownComplete => {
                            info!("Agent shutting down");

                            // Complete any remaining active web search
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;
                            stop_reason = StopReason::Cancelled;
                            break;
                        }
                        // In the future we can use this to update usage stats
                        EventMsg::TokenCount(..)=> {
                            // Complete any remaining active web search
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;
                        }
                        // we already have a way to diff the turn, so ignore
                        EventMsg::TurnDiff(..) =>  {
                            // Complete any remaining active web search
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            ).await;
                        }
                        EventMsg::ApplyPatchApprovalRequest(..)
                        | EventMsg::PatchApplyEnd(..)
                        | EventMsg::ExecApprovalRequest(..)
                        | EventMsg::McpToolCallEnd(..)
                        | EventMsg::ListCustomPromptsResponse(..) // Get slash commands
                        | EventMsg::ConversationPath(..) // Used for loading history, not needed for prompt
                        | EventMsg::SessionConfigured(..) // use for loading session and replay
                        | EventMsg::GetHistoryEntryResponse(..) // use for loading session?
                        | EventMsg::EnteredReviewMode(..) // Figure out how to handle this..
                        | EventMsg::ExitedReviewMode(..) // Figure out how to handle this..
                        | EventMsg::BackgroundEvent(..) // Revisit when we can emit status updates
                        | EventMsg::McpListToolsResponse(..) // use for /mcp?
                         => {}
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

    async fn cancel(&self, args: CancelNotification) -> Result<(), Error> {
        info!("Cancelling operations for session: {}", args.session_id);

        self.get_conversation(&args.session_id)
            .await?
            .submit(Op::Interrupt)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

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

        if let Some(session_state) = self.sessions.borrow_mut().get_mut(&args.session_id) {
            session_state.config.approval_policy = preset.approval;
            session_state.config.sandbox_policy = preset.sandbox.clone();
        }

        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        args: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!("Setting session model for session: {}", args.session_id);

        let conversation = self.get_conversation(&args.session_id).await?;

        let preset = self
            .model_presets
            .iter()
            .find(|p| p.id == args.model_id.0.as_ref())
            .ok_or_else(|| Error::invalid_params().with_data("Model not found"))?;

        conversation
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(preset.model.into()),
                effort: Some(preset.effort),
                summary: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        if let Some(session_state) = self.sessions.borrow_mut().get_mut(&args.session_id) {
            session_state.config.model = preset.model.into();
            session_state.config.model_reasoning_effort = preset.effort;
        }

        Ok(SetSessionModelResponse::default())
    }

    async fn ext_method(&self, _args: ExtRequest) -> Result<ExtResponse, Error> {
        Err(Error::method_not_found())
    }

    async fn ext_notification(&self, _args: ExtNotification) -> Result<(), Error> {
        Err(Error::method_not_found())
    }
}
