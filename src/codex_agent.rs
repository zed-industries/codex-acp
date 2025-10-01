use agent_client_protocol::{
    Agent, AgentCapabilities, AgentSideConnection, Annotations, AudioContent, AuthenticateRequest,
    AuthenticateResponse, BlobResourceContents, CancelNotification, Client, ContentBlock, Diff,
    EmbeddedResource, EmbeddedResourceResource, Error, ExtNotification, ExtRequest, ExtResponse,
    ImageContent, InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse,
    McpCapabilities, McpServer, ModelId, ModelInfo, NewSessionRequest, NewSessionResponse,
    PermissionOption, PermissionOptionId, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority,
    PlanEntryStatus, PromptCapabilities, PromptRequest, PromptResponse, ReadTextFileRequest,
    RequestPermissionOutcome, RequestPermissionRequest, ResourceLink, SessionId, SessionMode,
    SessionModeId, SessionModeState, SessionModelState, SessionNotification, SessionUpdate,
    SetSessionModeRequest, SetSessionModeResponse, SetSessionModelRequest, SetSessionModelResponse,
    StopReason, TextContent, TextResourceContents, ToolCall, ToolCallContent, ToolCallId,
    ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind, V1,
    WriteTextFileRequest,
};
use codex_common::{
    approval_presets::{ApprovalPreset, builtin_approval_presets},
    model_presets::{ModelPreset, builtin_model_presets},
};
use codex_core::{
    CodexConversation, ConversationManager,
    auth::{AuthManager, CodexAuth, login_with_api_key},
    config::Config,
    config_types::{McpServerConfig, McpServerTransportConfig},
    plan_tool::{PlanItemArg, StepStatus, UpdatePlanArgs},
    protocol::{
        AgentMessageDeltaEvent, AgentMessageEvent, AgentReasoningDeltaEvent, AgentReasoningEvent,
        AgentReasoningRawContentDeltaEvent, AgentReasoningRawContentEvent,
        AgentReasoningSectionBreakEvent, ApplyPatchApprovalRequestEvent, ErrorEvent,
        ExecApprovalRequestEvent, ExecCommandBeginEvent, FileChange, McpInvocation,
        McpToolCallBeginEvent, McpToolCallEndEvent, PatchApplyBeginEvent, PatchApplyEndEvent,
        ReviewDecision, StreamErrorEvent, TaskCompleteEvent, TaskStartedEvent, TurnAbortedEvent,
        UserMessageEvent, WebSearchBeginEvent, WebSearchEndEvent,
    },
};
use codex_protocol::{
    ConversationId,
    config_types::ReasoningEffort,
    protocol::{EventMsg, InputItem, Op},
};
use itertools::Itertools;
use mcp_types::CallToolResult;
use std::{
    cell::RefCell,
    collections::HashMap,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, LazyLock, OnceLock},
};
use tracing::{debug, error, info};

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);
pub static ACP_CLIENT: OnceLock<AgentSideConnection> = OnceLock::new();

/// The Codex implementation of the ACP Agent trait.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// The underlying codex configuration
    config: Config,
    /// Conversation manager for handling sessions
    conversation_manager: ConversationManager,
    /// Active sessions mapped by `SessionId`
    sessions: Rc<RefCell<HashMap<SessionId, SessionState>>>,
    /// Default model presets for a given auth mode
    model_presets: Vec<ModelPreset>,
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
    pub fn new(config: Config) -> Self {
        // Pre-auth from OPENAI_API_KEY before creating the AuthManager, so restored sessions
        // have API-key auth state ready without surfacing any UI or errors.
        if CodexAuth::from_codex_home(&config.codex_home)
            .ok()
            .flatten()
            .is_none()
            && let Ok(api_key) = std::env::var("OPENAI_API_KEY")
            && !api_key.is_empty()
        {
            let _ = login_with_api_key(&config.codex_home, &api_key);
        }
        let auth_manager = AuthManager::shared(config.codex_home.clone());

        let model_presets = builtin_model_presets(auth_manager.auth().map(|auth| auth.mode));
        let local_spawner = LocalSpawner::new();
        let conversation_manager =
            ConversationManager::new(auth_manager).with_fs(Box::new(move |conversation_id| {
                Box::new(AcpFs::new(
                    Self::session_id_from_conversation_id(conversation_id),
                    local_spawner.clone(),
                ))
            }));

        Self {
            config,
            conversation_manager,
            sessions: Rc::new(RefCell::new(HashMap::new())),
            model_presets,
        }
    }

    fn client(&self) -> &'static AgentSideConnection {
        ACP_CLIENT.get().expect("Client should be set")
    }

    fn session_id_from_conversation_id(conversation_id: ConversationId) -> SessionId {
        SessionId(conversation_id.to_string().into())
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

        if let Err(e) = self.client().session_notification(notification).await {
            error!("Failed to send session notification: {:?}", e);
        }
    }

    async fn start_mcp_tool_call(
        &self,
        session_id: SessionId,
        call_id: String,
        invocation: McpInvocation,
    ) {
        // Create a ToolCall so subsequent ToolCallUpdate (e.g. terminal embedding) can attach by id.
        let tool_call_id = ToolCallId(call_id.clone().into());
        let title = format!("Tool: {}/{}", invocation.server, invocation.tool);
        self.send_notification(
            session_id.clone(),
            SessionUpdate::ToolCall(ToolCall {
                id: tool_call_id,
                title,
                kind: ToolKind::Other,
                status: ToolCallStatus::InProgress,
                content: vec![],
                locations: vec![],
                raw_input: Some(serde_json::json!(&invocation)),
                raw_output: None,
                meta: None,
            }),
        )
        .await;
    }

    async fn end_mcp_tool_call(
        &self,
        session_id: SessionId,
        call_id: String,
        result: Result<CallToolResult, String>,
    ) {
        let is_error = match result.as_ref() {
            Ok(result) => result.is_error.unwrap_or_default(),
            Err(_) => true,
        };
        let raw_output = match result.as_ref() {
            Ok(result) => serde_json::json!(result),
            Err(err) => serde_json::json!(err),
        };
        self.send_notification(
            session_id,
            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                id: ToolCallId(call_id.into()),
                fields: ToolCallUpdateFields {
                    status: Some(if is_error {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    }),
                    content: result.ok().filter(|result| !result.content.is_empty()).map(
                        |result| {
                            result
                                .content
                                .into_iter()
                                .map(Self::codex_content_to_acp_content)
                                .collect()
                        },
                    ),
                    raw_output: Some(raw_output),
                    ..Default::default()
                },
                meta: None,
            }),
        )
        .await;
    }

    fn codex_content_to_acp_content(content: mcp_types::ContentBlock) -> ToolCallContent {
        ToolCallContent::Content {
            content: match content {
                mcp_types::ContentBlock::TextContent(text_content) => {
                    ContentBlock::Text(TextContent {
                        annotations: text_content.annotations.map(convert_annotations),
                        text: text_content.text,
                        meta: None,
                    })
                }
                mcp_types::ContentBlock::ImageContent(image_content) => {
                    ContentBlock::Image(ImageContent {
                        annotations: image_content.annotations.map(convert_annotations),
                        data: image_content.data,
                        mime_type: image_content.mime_type,
                        uri: None,
                        meta: None,
                    })
                }
                mcp_types::ContentBlock::AudioContent(audio_content) => {
                    ContentBlock::Audio(AudioContent {
                        annotations: audio_content.annotations.map(convert_annotations),
                        data: audio_content.data,
                        mime_type: audio_content.mime_type,
                        meta: None,
                    })
                }
                mcp_types::ContentBlock::ResourceLink(resource_link) => {
                    ContentBlock::ResourceLink(ResourceLink {
                        annotations: resource_link.annotations.map(convert_annotations),
                        description: resource_link.description,
                        mime_type: resource_link.mime_type,
                        name: resource_link.name,
                        size: resource_link.size,
                        title: resource_link.title,
                        uri: resource_link.uri,
                        meta: None,
                    })
                }
                mcp_types::ContentBlock::EmbeddedResource(embedded_resource) => {
                    ContentBlock::Resource(EmbeddedResource {
                        annotations: embedded_resource.annotations.map(convert_annotations),
                        resource: match embedded_resource.resource {
                            mcp_types::EmbeddedResourceResource::TextResourceContents(
                                text_resource_contents,
                            ) => EmbeddedResourceResource::TextResourceContents(
                                TextResourceContents {
                                    mime_type: text_resource_contents.mime_type,
                                    text: text_resource_contents.text,
                                    uri: text_resource_contents.uri,
                                    meta: None,
                                },
                            ),
                            mcp_types::EmbeddedResourceResource::BlobResourceContents(
                                blob_resource_contents,
                            ) => EmbeddedResourceResource::BlobResourceContents(
                                BlobResourceContents {
                                    blob: blob_resource_contents.blob,
                                    mime_type: blob_resource_contents.mime_type,
                                    uri: blob_resource_contents.uri,
                                    meta: None,
                                },
                            ),
                        },
                        meta: None,
                    })
                }
            },
        }
    }

    async fn start_web_search(
        &self,
        session_id: SessionId,
        call_id: String,
        active_web_search: &mut Option<String>,
    ) {
        *active_web_search = Some(call_id.clone());
        self.send_notification(
            session_id,
            SessionUpdate::ToolCall(ToolCall {
                id: ToolCallId(call_id.into()),
                title: "Searching the Web".to_string(),
                kind: ToolKind::Fetch,
                status: ToolCallStatus::Pending,
                content: vec![],
                locations: vec![],
                raw_input: None,
                raw_output: None,
                meta: None,
            }),
        )
        .await;
    }

    async fn update_web_search_query(&self, session_id: SessionId, call_id: String, query: String) {
        self.send_notification(
            session_id,
            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                id: ToolCallId(call_id.into()),
                fields: ToolCallUpdateFields {
                    status: Some(ToolCallStatus::InProgress),
                    title: Some(format!("Searching for: {query}")),
                    raw_input: Some(serde_json::json!({
                        "query": query
                    })),
                    ..Default::default()
                },
                meta: None,
            }),
        )
        .await;
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

    async fn send_agent_text(&self, session_id: SessionId, text: impl Into<String>) {
        let update = SessionUpdate::AgentMessageChunk {
            content: ContentBlock::Text(TextContent {
                text: text.into(),
                annotations: None,
                meta: None,
            }),
        };
        self.send_notification(session_id, update).await;
    }

    async fn send_agent_thought(&self, session_id: SessionId, text: impl Into<String>) {
        let update = SessionUpdate::AgentThoughtChunk {
            content: ContentBlock::Text(TextContent {
                text: text.into(),
                annotations: None,
                meta: None,
            }),
        };
        self.send_notification(session_id, update).await;
    }

    async fn update_plan(&self, session_id: SessionId, plan: Vec<PlanItemArg>) {
        let update = SessionUpdate::Plan(Plan {
            entries: plan
                .into_iter()
                .map(|entry| PlanEntry {
                    content: entry.step,
                    priority: PlanEntryPriority::Medium,
                    status: match entry.status {
                        StepStatus::Pending => PlanEntryStatus::Pending,
                        StepStatus::InProgress => PlanEntryStatus::InProgress,
                        StepStatus::Completed => PlanEntryStatus::Completed,
                    },
                    meta: None,
                })
                .collect(),
            meta: None,
        });
        self.send_notification(session_id, update).await;
    }

    async fn exec_approval(
        &self,
        session_id: SessionId,
        submission_id: String,
        event: ExecApprovalRequestEvent,
        active_command: &mut Option<(String, ToolCallId)>,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let ExecApprovalRequestEvent {
            call_id,
            command,
            cwd,
            reason,
        } = event;
        let conversation = self.get_conversation(&session_id).await?;

        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId(call_id.clone().into());
        *active_command = Some((call_id, tool_call_id.clone()));

        let response = self
            .client()
            .request_permission(RequestPermissionRequest {
                session_id,
                tool_call: ToolCallUpdate {
                    id: tool_call_id,
                    fields: ToolCallUpdateFields {
                        kind: Some(ToolKind::Execute),
                        status: Some(ToolCallStatus::Pending),
                        title: Some(format!("Running: {}", command.join(" "))),
                        content: reason.map(|r| vec![r.into()]),
                        locations: if cwd == std::path::PathBuf::from(".") {
                            None
                        } else {
                            Some(vec![ToolCallLocation {
                                path: cwd.clone(),
                                line: None,
                                meta: None,
                            }])
                        },
                        raw_input: Some(raw_input),
                        raw_output: None,
                    },
                    meta: None,
                },
                options: vec![
                    PermissionOption {
                        id: PermissionOptionId("approved-for-session".into()),
                        name: "Always".into(),
                        kind: PermissionOptionKind::AllowAlways,
                        meta: None,
                    },
                    PermissionOption {
                        id: PermissionOptionId("approved".into()),
                        name: "Yes".into(),
                        kind: PermissionOptionKind::AllowOnce,
                        meta: None,
                    },
                    PermissionOption {
                        id: PermissionOptionId("abort".into()),
                        name: "No, provide feedback".into(),
                        kind: PermissionOptionKind::RejectOnce,
                        meta: None,
                    },
                ],
                meta: None,
            })
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Cancelled => ReviewDecision::Abort,
            RequestPermissionOutcome::Selected { option_id } => match option_id.0.as_ref() {
                "approved-for-session" => ReviewDecision::ApprovedForSession,
                "approved" => ReviewDecision::Approved,
                _ => ReviewDecision::Abort,
            },
        };

        conversation
            .submit(Op::ExecApproval {
                id: submission_id,
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn patch_approval(
        &self,
        session_id: SessionId,
        submission_id: String,
        event: ApplyPatchApprovalRequestEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let ApplyPatchApprovalRequestEvent {
            call_id,
            changes,
            reason,
            // grant_root doesn't seem to be set anywhere on the codex side
            grant_root: _,
        } = event;
        let conversation = self.get_conversation(&session_id).await?;
        let (title, locations, content) = Self::extract_tool_call_content_from_changes(changes);
        let response = self
            .client()
            .request_permission(RequestPermissionRequest {
                session_id,
                tool_call: ToolCallUpdate {
                    id: ToolCallId(call_id.into()),
                    fields: ToolCallUpdateFields {
                        kind: Some(ToolKind::Edit),
                        status: Some(ToolCallStatus::Pending),
                        title: Some(title),
                        locations: Some(locations),
                        content: Some(
                            content
                                .chain(
                                    reason.map(|r| ToolCallContent::Content { content: r.into() }),
                                )
                                .collect(),
                        ),
                        raw_input: Some(raw_input),
                        ..Default::default()
                    },
                    meta: None,
                },
                options: vec![
                    PermissionOption {
                        id: PermissionOptionId("approved".into()),
                        name: "Yes".into(),
                        kind: PermissionOptionKind::AllowOnce,
                        meta: None,
                    },
                    PermissionOption {
                        id: PermissionOptionId("abort".into()),
                        name: "No, provide feedback".into(),
                        kind: PermissionOptionKind::RejectOnce,
                        meta: None,
                    },
                ],
                meta: None,
            })
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Cancelled => ReviewDecision::Abort,
            RequestPermissionOutcome::Selected { option_id } => match option_id.0.as_ref() {
                "approved" => ReviewDecision::Approved,
                _ => ReviewDecision::Abort,
            },
        };
        conversation
            .submit(Op::PatchApproval {
                id: submission_id,
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn start_patch_apply(&self, session_id: SessionId, event: PatchApplyBeginEvent) {
        let raw_input = serde_json::json!(&event);
        let PatchApplyBeginEvent {
            call_id,
            auto_approved: _,
            changes,
        } = event;

        let (title, locations, content) = Self::extract_tool_call_content_from_changes(changes);

        let update = SessionUpdate::ToolCall(ToolCall {
            id: ToolCallId(call_id.into()),
            title,
            kind: ToolKind::Edit,
            status: ToolCallStatus::InProgress,
            locations,
            content: content.collect(),
            raw_input: Some(raw_input),
            raw_output: None,
            meta: None,
        });
        self.send_notification(session_id, update).await;
    }

    async fn end_patch_apply(&self, session_id: SessionId, event: PatchApplyEndEvent) {
        let raw_output = serde_json::json!(&event);
        let PatchApplyEndEvent {
            call_id,
            stdout: _,
            stderr: _,
            success,
        } = event;

        let update = SessionUpdate::ToolCallUpdate(ToolCallUpdate {
            id: ToolCallId(call_id.into()),
            fields: ToolCallUpdateFields {
                status: Some(if success {
                    ToolCallStatus::Completed
                } else {
                    ToolCallStatus::Failed
                }),
                raw_output: Some(raw_output),
                ..Default::default()
            },
            meta: None,
        });
        self.send_notification(session_id, update).await;
    }

    fn extract_tool_call_content_from_changes(
        changes: HashMap<PathBuf, FileChange>,
    ) -> (
        String,
        Vec<ToolCallLocation>,
        impl Iterator<Item = ToolCallContent>,
    ) {
        (
            format!(
                "Edit {}",
                changes.keys().map(|p| p.display().to_string()).join(", ")
            ),
            changes
                .keys()
                .map(|p| ToolCallLocation {
                    path: p.clone(),
                    line: None,
                    meta: None,
                })
                .collect(),
            changes
                .into_iter()
                .map(|(path, change)| ToolCallContent::Diff {
                    diff: match change {
                        codex_core::protocol::FileChange::Add { content } => Diff {
                            path,
                            old_text: None,
                            new_text: content,
                            meta: None,
                        },
                        codex_core::protocol::FileChange::Delete { content } => Diff {
                            path,
                            old_text: Some(content),
                            new_text: String::new(),
                            meta: None,
                        },
                        codex_core::protocol::FileChange::Update {
                            unified_diff: _,
                            move_path,
                            old_content,
                            new_content,
                        } => Diff {
                            path: move_path.unwrap_or(path),
                            old_text: Some(old_content),
                            new_text: new_content,
                            meta: None,
                        },
                    },
                }),
        )
    }

    fn build_prompt_items(prompt: Vec<ContentBlock>) -> Vec<InputItem> {
        prompt
            .into_iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text_block) => Some(InputItem::Text {
                    text: text_block.text.clone(),
                }),
                ContentBlock::Image(image_block) => {
                    // Convert to data URI if needed
                    if let Some(uri) = &image_block.uri {
                        Some(InputItem::Image {
                            image_url: uri.clone(),
                        })
                    } else {
                        // Base64 data
                        let data_uri = format!(
                            "data:{};base64,{}",
                            image_block.mime_type.clone(),
                            image_block.data.clone()
                        );
                        Some(InputItem::Image {
                            image_url: data_uri,
                        })
                    }
                }
                ContentBlock::Audio(..)
                | ContentBlock::Resource(..)
                | ContentBlock::ResourceLink(..) => {
                    // Skip other content types for now
                    None
                }
            })
            .collect()
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

        // Hide auth methods when already authenticated, otherwise advertise a single login method.
        let is_authenticated = matches!(
            CodexAuth::from_codex_home(&self.config.codex_home),
            Ok(Some(_))
        );
        let auth_methods = if is_authenticated {
            vec![]
        } else {
            vec![agent_client_protocol::AuthMethod {
                id: agent_client_protocol::AuthMethodId("codex-login".into()),
                name: "Codex Login".into(),
                description: Some("Authenticate Codex from the terminal".into()),
                meta: None,
            }]
        };

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
        if request.method_id.0.as_ref() != "codex-login" {
            return Err(Error::invalid_params().with_data("unsupported authentication method"));
        }

        // Perform browser/device login via codex-rs, then report success/failure to the client.
        let opts = codex_login::ServerOptions::new(
            self.config.codex_home.clone(),
            codex_core::auth::CLIENT_ID.to_string(),
        );

        let server = codex_login::run_login_server(opts).map_err(Error::into_internal_error)?;

        server
            .block_until_done()
            .await
            .map_err(Error::into_internal_error)?;

        Ok(AuthenticateResponse { meta: None })
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
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
        config.cwd.clone_from(&cwd);

        // Propagate any client-provided MCP servers that codex-rs supports.
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex
                McpServer::Sse { .. } => {}
                McpServer::Http { name, url, headers } => {
                    config.mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::StreamableHttp {
                                url,
                                bearer_token: headers
                                    .into_iter()
                                    .find(|header| header.name == "Authorization")
                                    .and_then(|header| {
                                        header
                                            .value
                                            .strip_prefix("Bearer ")
                                            .map(std::borrow::ToOwned::to_owned)
                                    }),
                            },
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
        let session_id = Self::session_id_from_conversation_id(new_conversation.conversation_id);

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
        let items = Self::build_prompt_items(request.prompt);
        let items_len = items.len();

        let submission_id = conversation
            .submit(Op::UserInput { items })
            .await
            .map_err(|e| {
                error!("Failed to submit prompt: {e:?}");
                Error::internal_error()
            })?;

        info!("Submitted prompt with submission_id: {submission_id}, {items_len} input items");

        // Wait for the conversation to complete (TaskComplete or TurnAborted)
        let stop_reason;

        info!("Starting to wait for conversation events for submission_id: {submission_id}");

        let mut event_count = 0;
        let mut active_web_search: Option<String> = None;
        let mut active_command: Option<(String, ToolCallId)> = None;
        let mut command_output: Vec<String> = Vec::new();
        loop {
            event_count += 1;
            match conversation.next_event().await {
                Ok(event) => {
                    info!(
                        "Received event #{event_count}: {:?} (id: {})",
                        event.msg, event.id
                    );

                    // Complete any previous web search before starting a new one
                    match &event.msg {
                        EventMsg::Error(..)
                        | EventMsg::StreamError(..)
                        | EventMsg::WebSearchBegin(..)
                        | EventMsg::UserMessage(..)
                        | EventMsg::ExecApprovalRequest(..)
                        | EventMsg::ExecCommandBegin(..)
                        | EventMsg::ExecCommandOutputDelta(..)
                        | EventMsg::ExecCommandEnd(..)
                        | EventMsg::McpToolCallBegin(..)
                        | EventMsg::McpToolCallEnd(..)
                        | EventMsg::ApplyPatchApprovalRequest(..)
                        | EventMsg::PatchApplyBegin(..)
                        | EventMsg::PatchApplyEnd(..)
                        | EventMsg::TaskComplete(..)
                        | EventMsg::TokenCount(..)
                        | EventMsg::TurnDiff(..)
                        | EventMsg::TurnAborted(..)
                        | EventMsg::ShutdownComplete => {
                            self.complete_web_search(
                                request.session_id.clone(),
                                &mut active_web_search,
                            )
                            .await;
                        }
                        _ => {}
                    }

                    match event.msg {
                        EventMsg::TaskStarted(TaskStartedEvent { model_context_window }) => {
                            info!("Task started with context window of {model_context_window:?}");
                        }
                        EventMsg::UserMessage(UserMessageEvent { message, .. }) => {
                            info!("User message echoed: {message:?}");
                        }
                        EventMsg::AgentMessageDelta(AgentMessageDeltaEvent { delta }) => {
                            // Send this to the client via session/update notification
                            info!("Agent message received: {delta:?}");
                            self.send_agent_text(request.session_id.clone(), delta).await;
                        }
                        EventMsg::AgentReasoningDelta(AgentReasoningDeltaEvent { delta })
                        | EventMsg::AgentReasoningRawContentDelta(
                            AgentReasoningRawContentDeltaEvent { delta },
                        ) => {
                            // Send this to the client via session/update notification
                            info!("Agent reasoning message received: {:?}", delta);
                            self.send_agent_thought(request.session_id.clone(), delta).await;
                        }
                        EventMsg::AgentReasoningSectionBreak(
                            AgentReasoningSectionBreakEvent {},
                        ) => {
                            // Make sure the section heading actually get spacing
                            self.send_agent_thought(request.session_id.clone(), "\n\n").await;
                        }
                        EventMsg::PlanUpdate(UpdatePlanArgs { explanation, plan }) => {
                            // Send this to the client via session/update notification
                            info!("Agent plan updated. Explanation: {:?}", explanation);
                            self.update_plan(request.session_id.clone(), plan).await;
                        }
                        EventMsg::WebSearchBegin(WebSearchBeginEvent { call_id }) => {
                            info!("Web search started: call_id={}", call_id);
                            // Create a ToolCall notification for the search beginning
                            self.start_web_search(request.session_id.clone(), call_id, &mut active_web_search).await;
                        }
                        EventMsg::WebSearchEnd(WebSearchEndEvent { call_id, query }) => {
                            info!("Web search query received: call_id={call_id}, query={query}");
                            // Send update that the search is in progress with the query
                            // (WebSearchEnd just means we have the query, not that results are ready)
                            self.update_web_search_query(request.session_id.clone(), call_id, query).await;
                            // The actual search results will come through AgentMessage events
                            // We mark as completed when a new tool call begins
                        }
                        EventMsg::ExecApprovalRequest(event) => {
                            info!("Command execution started: call_id={}, command={:?}", event.call_id, event.command);
                            self.exec_approval(request.session_id.clone(), submission_id.clone(), event, &mut active_command).await?;
                        }
                        EventMsg::ExecCommandBegin(event) => {
                            info!(
                                "Command execution started: call_id={}, command={:?}",
                                event.call_id, event.command
                            );

                            let raw_input = serde_json::json!(&event);
                            let ExecCommandBeginEvent {
                                call_id,
                                command,
                                cwd,
                                parsed_cmd: _,
                            } = event;
                            // Create a new tool call for the command execution
                            let tool_call_id = ToolCallId(call_id.clone().into());
                            active_command = Some((call_id, tool_call_id.clone()));

                            self.send_notification(
                                request.session_id.clone(),
                                SessionUpdate::ToolCall(ToolCall {
                                    id: tool_call_id,
                                    title: format!("Running: {}", command.join(" ")),
                                    kind: ToolKind::Execute,
                                    status: ToolCallStatus::InProgress,
                                    content: vec![],
                                    locations: if cwd == std::path::PathBuf::from(".") {
                                        vec![]
                                    } else {
                                        vec![ToolCallLocation {
                                            path: cwd,
                                            line: None,
                                            meta: None,
                                        }]
                                    },
                                    raw_input: Some(raw_input),
                                    raw_output: None,
                                    meta: None,
                                }),
                            )
                            .await;
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
                            let raw_output = serde_json::json!(&end_event);
                            info!(
                                "Command execution ended: call_id={}, exit_code={}",
                                end_event.call_id, end_event.exit_code
                            );

                            if let Some((call_id, tool_call_id)) = active_command.take()
                                && call_id == end_event.call_id
                            {
                                let is_success = end_event.exit_code == 0;

                                let update = SessionUpdate::ToolCallUpdate(ToolCallUpdate {
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
                                        raw_output: Some(raw_output),
                                        ..Default::default()
                                    },
                                    meta: None,
                                });
                                self.send_notification(request.session_id.clone(), update).await;

                                // Clear accumulated output since we're done
                                command_output.clear();
                            }
                        }
                        EventMsg::McpToolCallBegin(McpToolCallBeginEvent { call_id, invocation }) => {
                            info!("MCP tool call begin: call_id={call_id}, invocation={} {}", invocation.server, invocation.tool);
                            self.start_mcp_tool_call(request.session_id.clone(), call_id, invocation).await;
                        }
                        EventMsg::McpToolCallEnd(McpToolCallEndEvent { call_id, invocation, duration, result }) => {
                            info!("MCP tool call ended: call_id={call_id}, invocation={} {}, duration={duration:?}", invocation.server, invocation.tool);
                            self.end_mcp_tool_call(request.session_id.clone(), call_id, result).await;
                        }
                        // File based events
                        EventMsg::ApplyPatchApprovalRequest(event) => {
                            info!("Apply patch approval request: call_id={}, reason={:?}", event.call_id, event.reason);
                            self.patch_approval(request.session_id.clone(), submission_id.clone(), event).await?;
                        }
                        EventMsg::PatchApplyBegin(event) => {
                            info!("Patch apply begin: call_id={}, auto_approved={}", event.call_id,event.auto_approved);
                            self.start_patch_apply(request.session_id.clone(), event).await;
                        }
                        EventMsg::PatchApplyEnd(event) => {
                            info!("Patch apply end: call_id={}, success={}", event.call_id, event.success);
                            self.end_patch_apply(request.session_id.clone(), event).await;
                        }
                        EventMsg::TaskComplete(TaskCompleteEvent { last_agent_message}) => {
                            info!(
                                "Task completed successfully after {event_count} events. Last agent message: {last_agent_message:?}",
                            );
                            stop_reason = StopReason::EndTurn;
                            break;
                        }
                        EventMsg::Error(ErrorEvent { message })
                        | EventMsg::StreamError(StreamErrorEvent { message }) => {
                            error!("Error during turn: {}", message);
                            return Err(Error::internal_error().with_data(message));
                        }
                        EventMsg::TurnAborted(TurnAbortedEvent { reason }) => {
                            info!("Turn aborted: {reason:?}");
                            stop_reason = StopReason::Cancelled;
                            break;
                        }
                        EventMsg::ShutdownComplete => {
                            info!("Agent shutting down");
                            stop_reason = StopReason::Cancelled;
                            break;
                        }

                        // Since we are getting the deltas, we can ignore these events
                        EventMsg::AgentReasoning(AgentReasoningEvent { .. })
                        | EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent {
                            ..
                        })
                        | EventMsg::AgentMessage(AgentMessageEvent { .. })
                        // In the future we can use this to update usage stats
                        | EventMsg::TokenCount(..)
                        // we already have a way to diff the turn, so ignore
                        | EventMsg::TurnDiff(..)
                        // returned from Op::ListMcpTools, ignore
                        | EventMsg::McpListToolsResponse(..)
                        // returned from Op::ListCustomPrompts, ignore
                        | EventMsg::ListCustomPromptsResponse(..)
                        // returned from Op::GetPath, ignore
                        | EventMsg::ConversationPath(..)
                        // Used for returning a single history entry
                        | EventMsg::GetHistoryEntryResponse(..)
                        // Used for session loading and replay
                        | EventMsg::SessionConfigured(..)
                        // used when requesting a code review, ignore for now
                        | EventMsg::EnteredReviewMode(..)
                        | EventMsg::ExitedReviewMode(..)
                        // Revisit when we can emit status updates
                        | EventMsg::BackgroundEvent(..) => {}
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

fn convert_annotations(annotations: mcp_types::Annotations) -> Annotations {
    Annotations {
        audience: annotations.audience.map(|audience| {
            audience
                .into_iter()
                .map(|audience| match audience {
                    mcp_types::Role::Assistant => agent_client_protocol::Role::Assistant,
                    mcp_types::Role::User => agent_client_protocol::Role::User,
                })
                .collect()
        }),
        last_modified: annotations.last_modified,
        priority: annotations.priority,
        meta: None,
    }
}

#[derive(Debug)]
pub enum FsTask {
    ReadFile {
        session_id: SessionId,
        path: PathBuf,
        tx: std::sync::mpsc::Sender<std::io::Result<String>>,
    },
    WriteFile {
        session_id: SessionId,
        path: PathBuf,
        content: String,
        tx: std::sync::mpsc::Sender<std::io::Result<()>>,
    },
}

impl FsTask {
    async fn run(self) {
        match self {
            FsTask::ReadFile {
                session_id,
                path,
                tx,
            } => {
                let read_text_file = Self::client().read_text_file(ReadTextFileRequest {
                    session_id,
                    path,
                    line: None,
                    limit: None,
                    meta: None,
                });
                let response = read_text_file
                    .await
                    .map(|response| response.content)
                    .map_err(|e| std::io::Error::other(e.to_string()));
                tx.send(response).ok();
            }
            FsTask::WriteFile {
                session_id,
                path,
                content,
                tx,
            } => {
                let response = Self::client()
                    .write_text_file(WriteTextFileRequest {
                        session_id,
                        path,
                        content,
                        meta: None,
                    })
                    .await
                    .map(|_| ())
                    .map_err(|e| std::io::Error::other(e.to_string()));
                tx.send(response).ok();
            }
        }
    }

    fn client() -> &'static AgentSideConnection {
        ACP_CLIENT.get().expect("Missing ACP client")
    }
}

struct AcpFs {
    session_id: SessionId,
    local_spawner: LocalSpawner,
}

impl AcpFs {
    fn new(session_id: SessionId, local_spawner: LocalSpawner) -> Self {
        Self {
            session_id,
            local_spawner,
        }
    }
}

impl codex_apply_patch::Fs for AcpFs {
    fn read_to_string(&self, path: &std::path::Path) -> std::io::Result<String> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.local_spawner.spawn(FsTask::ReadFile {
            session_id: self.session_id.clone(),
            path: std::path::absolute(path)?,
            tx,
        });
        rx.recv()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .flatten()
    }

    fn write(&self, path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.local_spawner.spawn(FsTask::WriteFile {
            session_id: self.session_id.clone(),
            path: std::path::absolute(path)?,
            content: String::from_utf8(contents.to_vec())
                .map_err(|e| std::io::Error::other(e.to_string()))?,
            tx,
        });
        rx.recv()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .flatten()
    }
}

#[derive(Clone)]
struct LocalSpawner {
    send: tokio::sync::mpsc::UnboundedSender<FsTask>,
}

impl LocalSpawner {
    pub fn new() -> Self {
        let (send, mut recv) = tokio::sync::mpsc::unbounded_channel::<FsTask>();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        std::thread::spawn(move || {
            let local = tokio::task::LocalSet::new();

            local.spawn_local(async move {
                while let Some(new_task) = recv.recv().await {
                    tokio::task::spawn_local(new_task.run());
                }
                // If the while loop returns, then all the LocalSpawner
                // objects have been dropped.
            });

            // This will return once all senders are dropped and all
            // spawned tasks have returned.
            rt.block_on(local);
        });

        Self { send }
    }

    pub fn spawn(&self, task: FsTask) {
        self.send
            .send(task)
            .expect("Thread with LocalSet has shut down.");
    }
}
