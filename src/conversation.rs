use std::{
    cell::RefCell,
    collections::HashMap,
    ops::DerefMut,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, LazyLock, Mutex},
};

use agent_client_protocol::{
    Annotations, AudioContent, AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate,
    BlobResourceContents, Client, ClientCapabilities, ContentBlock, ContentChunk, Diff,
    EmbeddedResource, EmbeddedResourceResource, Error, ImageContent, LoadSessionResponse, ModelId,
    ModelInfo, PermissionOption, PermissionOptionId, PermissionOptionKind, Plan, PlanEntry,
    PlanEntryPriority, PlanEntryStatus, PromptRequest, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, ResourceLink, SessionId, SessionMode,
    SessionModeId, SessionModeState, SessionModelState, SessionNotification, SessionUpdate,
    StopReason, TerminalId, TextContent, TextResourceContents, ToolCall, ToolCallContent,
    ToolCallId, ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use codex_common::{
    approval_presets::{ApprovalPreset, builtin_approval_presets},
    model_presets::ModelPreset,
};
use codex_core::{
    AuthManager, CodexConversation,
    config::Config,
    error::CodexErr,
    protocol::{
        AgentMessageDeltaEvent, AgentMessageEvent, AgentReasoningDeltaEvent, AgentReasoningEvent,
        AgentReasoningRawContentDeltaEvent, AgentReasoningSectionBreakEvent,
        ApplyPatchApprovalRequestEvent, ErrorEvent, Event, EventMsg, ExecApprovalRequestEvent,
        ExecCommandBeginEvent, ExecCommandEndEvent, ExecCommandOutputDeltaEvent,
        ExitedReviewModeEvent, FileChange, ItemCompletedEvent, ItemStartedEvent,
        ListCustomPromptsResponseEvent, McpInvocation, McpToolCallBeginEvent, McpToolCallEndEvent,
        Op, PatchApplyBeginEvent, PatchApplyEndEvent, ReviewDecision, ReviewOutputEvent,
        ReviewRequest, StreamErrorEvent, TaskCompleteEvent, TaskStartedEvent, TurnAbortedEvent,
        UserMessageEvent, ViewImageToolCallEvent, WebSearchBeginEvent, WebSearchEndEvent,
    },
    review_format::format_review_findings_block,
};
use codex_protocol::{
    config_types::ReasoningEffort,
    custom_prompts::CustomPrompt,
    parse_command::ParsedCommand,
    plan_tool::{PlanItemArg, StepStatus, UpdatePlanArgs},
    user_input::UserInput,
};
use itertools::Itertools;
use mcp_types::CallToolResult;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

use crate::{
    ACP_CLIENT,
    prompt_args::{expand_custom_prompt, parse_slash_name},
};

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);
const INIT_COMMAND_PROMPT: &str = include_str!("./prompt_for_init_command.md");

/// Trait for abstracting over the CodexConversation to make testing easier.
#[async_trait::async_trait]
pub trait CodexConversationImpl {
    async fn submit(&self, op: Op) -> Result<String, CodexErr>;
    async fn next_event(&self) -> Result<Event, CodexErr>;
}

#[async_trait::async_trait]
impl CodexConversationImpl for CodexConversation {
    async fn submit(&self, op: Op) -> Result<String, CodexErr> {
        self.submit(op).await
    }

    async fn next_event(&self) -> Result<Event, CodexErr> {
        self.next_event().await
    }
}

pub trait Auth {
    fn logout(&self) -> Result<bool, Error>;
}

impl Auth for Arc<AuthManager> {
    fn logout(&self) -> Result<bool, Error> {
        self.as_ref()
            .logout()
            .map_err(|e| Error::internal_error().with_data(e.to_string()))
    }
}

enum ConversationMessage {
    Load {
        response_tx: oneshot::Sender<Result<LoadSessionResponse, Error>>,
    },
    Prompt {
        request: PromptRequest,
        response_tx: oneshot::Sender<Result<oneshot::Receiver<Result<StopReason, Error>>, Error>>,
    },
    SetMode {
        mode: SessionModeId,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    SetModel {
        model: ModelId,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    Cancel {
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
}

pub struct Conversation {
    /// A sender for interacting with the conversation.
    message_tx: mpsc::UnboundedSender<ConversationMessage>,
    /// A handle to the spawned task.
    _handle: tokio::task::JoinHandle<()>,
}

impl Conversation {
    pub fn new(
        session_id: SessionId,
        conversation: Arc<dyn CodexConversationImpl>,
        auth: Arc<AuthManager>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        config: Config,
        model_presets: Rc<Vec<ModelPreset>>,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        let actor = ConversationActor::new(
            auth,
            SessionClient::new(session_id, client_capabilities),
            conversation.clone(),
            config,
            model_presets,
            message_rx,
        );
        let handle = tokio::task::spawn_local(actor.spawn());

        Self {
            message_tx,
            _handle: handle,
        }
    }

    pub async fn load(&self) -> Result<LoadSessionResponse, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::Load { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }

    pub async fn prompt(&self, request: PromptRequest) -> Result<StopReason, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::Prompt {
            request,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))??
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }

    pub async fn set_mode(&self, mode: SessionModeId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::SetMode { mode, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }

    pub async fn set_model(&self, model: ModelId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::SetModel { model, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }

    pub async fn cancel(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::Cancel { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }
}

enum SubmissionState {
    /// Loading custom prompts from the project
    CustomPrompts(CustomPromptsState),
    /// User prompts + some slash commands like /init or /review
    Prompt(PromptState),
    /// Subtask, like /compact
    Task(TaskState),
}

impl SubmissionState {
    fn is_active(&self) -> bool {
        match self {
            Self::CustomPrompts(state) => state.is_active(),
            Self::Prompt(state) => state.is_active(),
            Self::Task(state) => state.is_active(),
        }
    }

    async fn handle_event(&mut self, client: &SessionClient, event: EventMsg) {
        match self {
            Self::CustomPrompts(state) => state.handle_event(event),
            Self::Prompt(state) => state.handle_event(client, event).await,
            Self::Task(state) => state.handle_event(client, event).await,
        }
    }
}

struct CustomPromptsState {
    response_tx: Option<oneshot::Sender<Result<Vec<CustomPrompt>, Error>>>,
}

impl CustomPromptsState {
    fn new(response_tx: oneshot::Sender<Result<Vec<CustomPrompt>, Error>>) -> Self {
        Self {
            response_tx: Some(response_tx),
        }
    }

    fn is_active(&self) -> bool {
        let Some(response_tx) = &self.response_tx else {
            return false;
        };
        !response_tx.is_closed()
    }

    fn handle_event(&mut self, event: EventMsg) {
        match event {
            EventMsg::ListCustomPromptsResponse(ListCustomPromptsResponseEvent {
                custom_prompts,
            }) => {
                if let Some(tx) = self.response_tx.take() {
                    let _ = tx.send(Ok(custom_prompts));
                }
            }
            e => {
                warn!("Unexpected event: {e:?}");
            }
        }
    }
}

struct ActiveCommand {
    call_id: String,
    tool_call_id: ToolCallId,
    terminal_output: bool,
    output: String,
    file_extension: Option<String>,
}

struct PromptState {
    active_command: Option<ActiveCommand>,
    active_web_search: Option<String>,
    conversation: Arc<dyn CodexConversationImpl>,
    event_count: usize,
    response_tx: Option<oneshot::Sender<Result<StopReason, Error>>>,
    submission_id: String,
}

impl PromptState {
    fn new(
        conversation: Arc<dyn CodexConversationImpl>,
        response_tx: oneshot::Sender<Result<StopReason, Error>>,
        submission_id: String,
    ) -> Self {
        Self {
            active_command: None,
            active_web_search: None,
            conversation,
            event_count: 0,
            response_tx: Some(response_tx),
            submission_id,
        }
    }

    fn is_active(&self) -> bool {
        let Some(response_tx) = &self.response_tx else {
            return false;
        };
        !response_tx.is_closed()
    }

    async fn handle_event(&mut self, client: &SessionClient, event: EventMsg) {
        self.event_count += 1;

        // Complete any previous web search before starting a new one
        match &event {
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
            | EventMsg::TaskStarted(..)
            | EventMsg::TaskComplete(..)
            | EventMsg::TokenCount(..)
            | EventMsg::TurnDiff(..)
            | EventMsg::TurnAborted(..)
            | EventMsg::EnteredReviewMode(..)
            | EventMsg::ExitedReviewMode(..)
            | EventMsg::ShutdownComplete => {
                self.complete_web_search(client).await;
            }
            _ => {}
        }

        match event {
            EventMsg::TaskStarted(TaskStartedEvent {
                model_context_window,
            }) => {
                info!("Task started with context window of {model_context_window:?}");
            }
            EventMsg::ItemStarted(ItemStartedEvent { thread_id, turn_id, item }) => {

                info!("Item started with thread_id: {thread_id}, turn_id: {turn_id}, item: {item:?}");
            }
            EventMsg::UserMessage(UserMessageEvent {
                message,
                images: _,
            }) => {
                info!("User message: {message:?}");
            }
            EventMsg::AgentMessageDelta(AgentMessageDeltaEvent { delta }) => {
                // Send this to the client via session/update notification
                info!("Agent message received: {delta:?}");
                client.send_agent_text(delta).await;
            }
            EventMsg::AgentReasoningDelta(AgentReasoningDeltaEvent { delta })
            | EventMsg::AgentReasoningRawContentDelta(AgentReasoningRawContentDeltaEvent {
                delta,
            }) => {
                // Send this to the client via session/update notification
                info!("Agent reasoning message received: {:?}", delta);
                client.send_agent_thought(delta).await;
            }
            EventMsg::AgentReasoningSectionBreak(AgentReasoningSectionBreakEvent {}) => {
                // Make sure the section heading actually get spacing
                client.send_agent_thought("\n\n").await;
            }
            EventMsg::PlanUpdate(UpdatePlanArgs { explanation, plan }) => {
                // Send this to the client via session/update notification
                info!("Agent plan updated. Explanation: {:?}", explanation);
                client.update_plan(plan).await;
            }
            EventMsg::WebSearchBegin(WebSearchBeginEvent { call_id }) => {
                info!("Web search started: call_id={}", call_id);
                // Create a ToolCall notification for the search beginning
                self.start_web_search(client, call_id).await;
            }
            EventMsg::WebSearchEnd(WebSearchEndEvent { call_id, query }) => {
                info!("Web search query received: call_id={call_id}, query={query}");
                // Send update that the search is in progress with the query
                // (WebSearchEnd just means we have the query, not that results are ready)
                self.update_web_search_query(client, call_id, query).await;
                // The actual search results will come through AgentMessage events
                // We mark as completed when a new tool call begins
            }
            EventMsg::ExecApprovalRequest(event) => {
                info!("Command execution started: call_id={}, command={:?}", event.call_id, event.command);
                if let Err(err) = self.exec_approval(client, event).await && let Some(response_tx) = self.response_tx.take() {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::ExecCommandBegin(event) => {
                info!(
                    "Command execution started: call_id={}, command={:?}",
                    event.call_id, event.command
                );
                self.exec_command_begin(client, event).await;
            }
            EventMsg::ExecCommandOutputDelta(delta_event) => {
                self.exec_command_output_delta(client, delta_event).await;
            }
            EventMsg::ExecCommandEnd(end_event) => {
                info!(
                    "Command execution ended: call_id={}, exit_code={}",
                    end_event.call_id, end_event.exit_code
                );
                self.exec_command_end(client, end_event).await;
            }
            EventMsg::McpToolCallBegin(McpToolCallBeginEvent { call_id, invocation }) => {
                info!("MCP tool call begin: call_id={call_id}, invocation={} {}", invocation.server, invocation.tool);
                self.start_mcp_tool_call(client, call_id, invocation).await;
            }
            EventMsg::McpToolCallEnd(McpToolCallEndEvent { call_id, invocation, duration, result }) => {
                info!("MCP tool call ended: call_id={call_id}, invocation={} {}, duration={duration:?}", invocation.server, invocation.tool);
                self.end_mcp_tool_call(client, call_id, result).await;
            }
            EventMsg::ApplyPatchApprovalRequest(event) => {
                info!("Apply patch approval request: call_id={}, reason={:?}", event.call_id, event.reason);
                if let Err(err) = self.patch_approval(client, event).await && let Some(response_tx) = self.response_tx.take() {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::PatchApplyBegin(event) => {
                info!("Patch apply begin: call_id={}, auto_approved={}", event.call_id,event.auto_approved);
                self.start_patch_apply(client, event).await;
            }
            EventMsg::PatchApplyEnd(event) => {
                info!("Patch apply end: call_id={}, success={}", event.call_id, event.success);
                self.end_patch_apply(client, event).await;
            }
            EventMsg::ItemCompleted(ItemCompletedEvent { thread_id, turn_id, item }) => {
                info!("Item completed: thread_id={}, turn_id={}, item={:?}", thread_id, turn_id, item);
            }
            EventMsg::TaskComplete(TaskCompleteEvent { last_agent_message}) => {
                info!(
                    "Task completed successfully after {} events. Last agent message: {last_agent_message:?}", self.event_count
                );
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::EndTurn)).ok();
                }
            }
            EventMsg::Error(ErrorEvent { message })
            | EventMsg::StreamError(StreamErrorEvent { message }) => {
                error!("Error during turn: {}", message);
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Err(Error::internal_error().with_data(message))).ok();
                }
            }
            EventMsg::TurnAborted(TurnAbortedEvent { reason }) => {
                info!("Turn aborted: {reason:?}");
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            EventMsg::ShutdownComplete => {
                info!("Agent shutting down");
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            EventMsg::ViewImageToolCall(ViewImageToolCallEvent { call_id, path }) => {
                info!("ViewImageToolCallEvent received");
                let display_path = path.display().to_string();
                client
                    .send_notification(SessionUpdate::ToolCall(ToolCall {
                        id: ToolCallId(call_id.into()),
                        title: format!("View Image {display_path}"),
                        kind: ToolKind::Read,
                        status: ToolCallStatus::Completed,
                        content: vec![ToolCallContent::Content {
                            content: ContentBlock::ResourceLink(ResourceLink {
                                annotations: None,
                                description: None,
                                mime_type: None,
                                name: display_path.clone(),
                                size: None,
                                title: None,
                                uri: display_path.clone(),
                                meta: None,
                            }),
                        }],
                        locations: vec![ToolCallLocation {
                            path,
                            line: None,
                            meta: None,
                        }],
                        raw_input: None,
                        raw_output: None,
                        meta: None,
                    }))
                    .await;
            }
            EventMsg::EnteredReviewMode(review_request) => {
                info!("Review begin: request={review_request:?}");
            }
            EventMsg::ExitedReviewMode(event) => {
                info!("Review end: output={event:?}");
                if let Err(err) = self.review_mode_exit(client, event).await && let Some(response_tx) = self.response_tx.take() {
                    drop(response_tx.send(Err(err)));
                }
            }

            // Since we are getting the deltas, we can ignore these events
            EventMsg::AgentReasoning(..)
            | EventMsg::AgentReasoningRawContent(..)
            | EventMsg::AgentMessage(..)
            // In the future we can use this to update usage stats
            | EventMsg::TokenCount(..)
            // we already have a way to diff the turn, so ignore
            | EventMsg::TurnDiff(..)
            // Revisit when we can emit status updates
            | EventMsg::BackgroundEvent(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::SessionConfigured(..) => {}

            // Unexpected events for this submission
            e @ (EventMsg::McpListToolsResponse(..)
            // returned from Op::ListCustomPrompts, ignore
            | EventMsg::ListCustomPromptsResponse(..)
            // Used for returning a single history entry
            | EventMsg::GetHistoryEntryResponse(..)) => {
                warn!("Unexpected event: {:?}", e);
            }
        }
    }

    async fn review_mode_exit(
        &self,
        client: &SessionClient,
        event: ExitedReviewModeEvent,
    ) -> Result<(), Error> {
        let ExitedReviewModeEvent { review_output } = event;
        let Some(ReviewOutputEvent {
            findings,
            overall_correctness: _,
            overall_explanation,
            overall_confidence_score: _,
        }) = review_output
        else {
            return Ok(());
        };

        let text = if findings.is_empty() {
            let explanation = overall_explanation.trim();
            if explanation.is_empty() {
                "Reviewer failed to output a response"
            } else {
                explanation
            }
            .to_string()
        } else {
            format_review_findings_block(&findings, None)
        };

        client.send_agent_text(&text).await;
        Ok(())
    }

    async fn patch_approval(
        &self,
        client: &SessionClient,
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
        let (title, locations, content) = extract_tool_call_content_from_changes(changes);
        let response = client
            .request_permission(
                ToolCallUpdate {
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
                vec![
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
            )
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Cancelled => ReviewDecision::Abort,
            RequestPermissionOutcome::Selected { option_id } => match option_id.0.as_ref() {
                "approved" => ReviewDecision::Approved,
                _ => ReviewDecision::Abort,
            },
        };

        self.conversation
            .submit(Op::PatchApproval {
                id: self.submission_id.clone(),
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn start_patch_apply(&self, client: &SessionClient, event: PatchApplyBeginEvent) {
        let raw_input = serde_json::json!(&event);
        let PatchApplyBeginEvent {
            call_id,
            auto_approved: _,
            changes,
        } = event;

        let (title, locations, content) = extract_tool_call_content_from_changes(changes);

        client
            .send_notification(SessionUpdate::ToolCall(ToolCall {
                id: ToolCallId(call_id.into()),
                title,
                kind: ToolKind::Edit,
                status: ToolCallStatus::InProgress,
                locations,
                content: content.collect(),
                raw_input: Some(raw_input),
                raw_output: None,
                meta: None,
            }))
            .await;
    }

    async fn end_patch_apply(&self, client: &SessionClient, event: PatchApplyEndEvent) {
        let raw_output = serde_json::json!(&event);
        let PatchApplyEndEvent {
            call_id,
            stdout: _,
            stderr: _,
            success,
        } = event;

        client
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
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
            }))
            .await;
    }

    async fn start_mcp_tool_call(
        &self,
        client: &SessionClient,
        call_id: String,
        invocation: McpInvocation,
    ) {
        let tool_call_id = ToolCallId(call_id.clone().into());
        let title = format!("Tool: {}/{}", invocation.server, invocation.tool);
        client
            .send_notification(SessionUpdate::ToolCall(ToolCall {
                id: tool_call_id,
                title,
                kind: ToolKind::Other,
                status: ToolCallStatus::InProgress,
                content: vec![],
                locations: vec![],
                raw_input: Some(serde_json::json!(&invocation)),
                raw_output: None,
                meta: None,
            }))
            .await;
    }

    async fn end_mcp_tool_call(
        &self,
        client: &SessionClient,
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
        client
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
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
                                .map(codex_content_to_acp_content)
                                .collect()
                        },
                    ),
                    raw_output: Some(raw_output),
                    ..Default::default()
                },
                meta: None,
            }))
            .await;
    }

    async fn exec_approval(
        &mut self,
        client: &SessionClient,
        event: ExecApprovalRequestEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let ExecApprovalRequestEvent {
            call_id,
            command: _,
            cwd,
            reason,
            parsed_cmd,
            risk,
        } = event;

        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId(call_id.clone().into());
        let ParseCommandToolCall {
            title,
            terminal_output,
            file_extension,
            locations,
            kind,
        } = parse_command_tool_call(parsed_cmd, &cwd);
        self.active_command = Some(ActiveCommand {
            call_id,
            terminal_output,
            tool_call_id: tool_call_id.clone(),
            output: String::new(),
            file_extension,
        });

        let risk = risk.map(|risk| {
            let mut str = format!(
                "Risk Assessment: {}\nRisk Level: {}",
                risk.description,
                risk.risk_level.as_str()
            );
            if !risk.risk_categories.is_empty() {
                str.push_str("\nRisk Categories:");
                str.push_str(&risk.risk_categories.iter().map(|c| c.as_str()).join(", "));
            }
            str
        });

        let content = match (reason, risk) {
            (Some(reason), Some(risk)) => Some(vec![[reason, risk].join("\n").into()]),
            (Some(reason), None) => Some(vec![reason.into()]),
            (None, Some(risk)) => Some(vec![risk.into()]),
            (None, None) => None,
        };

        let response = client
            .request_permission(
                ToolCallUpdate {
                    id: tool_call_id,
                    fields: ToolCallUpdateFields {
                        kind: Some(kind),
                        status: Some(ToolCallStatus::Pending),
                        title: Some(title),
                        content,
                        locations: if locations.is_empty() {
                            None
                        } else {
                            Some(locations)
                        },
                        raw_input: Some(raw_input),
                        raw_output: None,
                    },
                    meta: None,
                },
                vec![
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
            )
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Cancelled => ReviewDecision::Abort,
            RequestPermissionOutcome::Selected { option_id } => match option_id.0.as_ref() {
                "approved-for-session" => ReviewDecision::ApprovedForSession,
                "approved" => ReviewDecision::Approved,
                _ => ReviewDecision::Abort,
            },
        };

        self.conversation
            .submit(Op::ExecApproval {
                id: self.submission_id.clone(),
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        Ok(())
    }

    async fn exec_command_begin(&mut self, client: &SessionClient, event: ExecCommandBeginEvent) {
        let raw_input = serde_json::json!(&event);
        let ExecCommandBeginEvent {
            call_id,
            command: _,
            cwd,
            parsed_cmd,
        } = event;
        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId(call_id.clone().into());
        let ParseCommandToolCall {
            title,
            file_extension,
            locations,
            terminal_output,
            kind,
        } = parse_command_tool_call(parsed_cmd, &cwd);

        let active_command = ActiveCommand {
            call_id: call_id.clone(),
            tool_call_id: tool_call_id.clone(),
            output: String::new(),
            file_extension,
            terminal_output,
        };

        let (content, meta) = if client.supports_terminal_output(&active_command) {
            let content = vec![ToolCallContent::Terminal {
                terminal_id: TerminalId(call_id.clone().into()),
            }];
            let meta = Some(serde_json::json!({
                "terminal_info": {
                    "terminal_id": call_id,
                    "cwd": cwd
                }
            }));
            (content, meta)
        } else {
            (vec![], None)
        };
        self.active_command = Some(active_command);

        client
            .send_notification(SessionUpdate::ToolCall(ToolCall {
                id: tool_call_id,
                title,
                kind,
                status: ToolCallStatus::InProgress,
                content,
                locations,
                raw_input: Some(raw_input),
                raw_output: None,
                meta,
            }))
            .await;
    }

    async fn exec_command_output_delta(
        &mut self,
        client: &SessionClient,
        event: ExecCommandOutputDeltaEvent,
    ) {
        let ExecCommandOutputDeltaEvent {
            call_id,
            chunk,
            stream: _,
        } = event;
        // Stream output bytes to the display-only terminal via ToolCallUpdate meta.
        if let Some(active_command) = &mut self.active_command
            && *active_command.call_id == call_id
        {
            let data_str = String::from_utf8_lossy(&chunk).to_string();

            let (fields, meta) = if client.supports_terminal_output(active_command) {
                let meta = Some(serde_json::json!({
                    "terminal_output": {
                        "terminal_id": call_id,
                        "data": data_str
                    }
                }));
                (ToolCallUpdateFields::default(), meta)
            } else {
                active_command.output.push_str(&data_str);
                let content = match active_command.file_extension.as_deref() {
                    Some("md") => active_command.output.clone(),
                    Some(ext) => format!(
                        "```{ext}\n{}\n```\n",
                        active_command.output.trim_end_matches('\n')
                    ),
                    None => format!(
                        "```sh\n{}\n```\n",
                        active_command.output.trim_end_matches('\n')
                    ),
                };
                (
                    ToolCallUpdateFields {
                        content: Some(vec![content.into()]),
                        ..Default::default()
                    },
                    None,
                )
            };

            client
                .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    id: active_command.tool_call_id.clone(),
                    fields,
                    meta,
                }))
                .await;
        }
    }

    async fn exec_command_end(&mut self, client: &SessionClient, event: ExecCommandEndEvent) {
        let raw_output = serde_json::json!(&event);
        let ExecCommandEndEvent {
            call_id,
            exit_code,
            stdout: _,
            stderr: _,
            aggregated_output: _,
            duration: _,
            formatted_output: _,
        } = event;
        if let Some(active_command) = self.active_command.take()
            && active_command.call_id == call_id
        {
            let is_success = exit_code == 0;

            let meta = (client.supports_terminal_output(&active_command)).then(|| {
                serde_json::json!({
                    "terminal_exit": {
                        "terminal_id": call_id,
                        "exit_code": exit_code,
                        "signal": null
                    }
                })
            });

            client
                .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    id: active_command.tool_call_id,
                    fields: ToolCallUpdateFields {
                        status: Some(if is_success {
                            ToolCallStatus::Completed
                        } else {
                            ToolCallStatus::Failed
                        }),
                        raw_output: Some(raw_output),
                        ..Default::default()
                    },
                    meta,
                }))
                .await;
        }
    }

    async fn start_web_search(&mut self, client: &SessionClient, call_id: String) {
        self.active_web_search = Some(call_id.clone());
        client
            .send_notification(SessionUpdate::ToolCall(ToolCall {
                id: ToolCallId(call_id.into()),
                title: "Searching the Web".to_string(),
                kind: ToolKind::Fetch,
                status: ToolCallStatus::Pending,
                content: vec![],
                locations: vec![],
                raw_input: None,
                raw_output: None,
                meta: None,
            }))
            .await;
    }

    async fn update_web_search_query(
        &self,
        client: &SessionClient,
        call_id: String,
        query: String,
    ) {
        client
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
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
            }))
            .await;
    }

    async fn complete_web_search(&mut self, client: &SessionClient) {
        if let Some(call_id) = self.active_web_search.take() {
            client
                .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    id: ToolCallId(call_id.into()),
                    fields: ToolCallUpdateFields {
                        status: Some(ToolCallStatus::Completed),
                        ..Default::default()
                    },
                    meta: None,
                }))
                .await;
        }
    }
}

struct ParseCommandToolCall {
    title: String,
    file_extension: Option<String>,
    terminal_output: bool,
    locations: Vec<ToolCallLocation>,
    kind: ToolKind,
}

fn parse_command_tool_call(parsed_cmd: Vec<ParsedCommand>, cwd: &Path) -> ParseCommandToolCall {
    let mut titles = Vec::new();
    let mut locations = Vec::new();
    let mut file_extension = None;
    let mut terminal_output = false;
    let mut kind = ToolKind::Execute;

    for cmd in parsed_cmd {
        let mut cmd_path = None;
        match cmd {
            ParsedCommand::Read { cmd: _, name, path } => {
                titles.push(format!("Read {name}"));
                file_extension = path
                    .extension()
                    .map(|ext| ext.to_string_lossy().to_string());
                cmd_path = Some(path);
                kind = ToolKind::Read;
            }
            ParsedCommand::ListFiles { cmd: _, path } => {
                let dir = if let Some(path) = path.as_ref() {
                    &cwd.join(path)
                } else {
                    cwd
                };
                titles.push(format!("List {}", dir.display()));
                cmd_path = path.map(PathBuf::from);
                kind = ToolKind::Search;
            }
            ParsedCommand::Search { cmd, query, path } => {
                titles.push(match (query, path.as_ref()) {
                    (Some(query), Some(path)) => format!("Search {query} in {path}"),
                    (Some(query), None) => format!("Search {query}"),
                    _ => format!("Search {}", cmd),
                });
                cmd_path = path.map(PathBuf::from);
                kind = ToolKind::Search;
            }
            ParsedCommand::Unknown { cmd } => {
                titles.push(format!("Run {cmd}"));
                terminal_output = true;
            }
        }

        if let Some(path) = cmd_path {
            locations.push(ToolCallLocation {
                path: if path.is_relative() {
                    cwd.join(&path)
                } else {
                    path
                },
                line: None,
                meta: None,
            });
        }
    }

    ParseCommandToolCall {
        title: titles.join(", "),
        file_extension,
        terminal_output,
        locations,
        kind,
    }
}

struct TaskState {
    response_tx: Option<oneshot::Sender<Result<StopReason, Error>>>,
}

impl TaskState {
    fn new(response_tx: oneshot::Sender<Result<StopReason, Error>>) -> Self {
        Self {
            response_tx: Some(response_tx),
        }
    }

    fn is_active(&self) -> bool {
        self.response_tx.is_some()
    }

    async fn handle_event(&mut self, client: &SessionClient, event: EventMsg) {
        match event {
            EventMsg::TaskComplete(..) => {
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::EndTurn)).ok();
                }
            }
            // Safer to grab the non-streaming version of the events so we don't duplicate
            // and it is likely these are synthetic events, not from the model
            EventMsg::AgentMessage(AgentMessageEvent { message }) => {
                client.send_agent_text(message).await;
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                client.send_agent_thought(text).await;
            }
            EventMsg::Error(ErrorEvent { message })
            | EventMsg::StreamError(StreamErrorEvent { message }) => {
                error!("Error during turn: {}", message);
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx
                        .send(Err(Error::internal_error().with_data(message)))
                        .ok();
                }
            }
            EventMsg::TurnAborted(TurnAbortedEvent { reason }) => {
                info!("Turn aborted: {reason:?}");
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            EventMsg::ShutdownComplete => {
                info!("Agent shutting down");
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            // Expected but ignored
            EventMsg::TaskStarted(..)
            | EventMsg::ItemStarted(..)
            | EventMsg::ItemCompleted(..)
            | EventMsg::TokenCount(..)
            | EventMsg::AgentMessageDelta(..)
            | EventMsg::AgentReasoningDelta(..)
            | EventMsg::AgentReasoningRawContent(..)
            | EventMsg::AgentReasoningRawContentDelta(..)
            | EventMsg::AgentReasoningSectionBreak(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::BackgroundEvent(..) => {}
            // Unexpected events for this submission
            e @ (EventMsg::UserMessage(..)
            | EventMsg::SessionConfigured(..)
            | EventMsg::McpToolCallBegin(..)
            | EventMsg::McpToolCallEnd(..)
            | EventMsg::WebSearchBegin(..)
            | EventMsg::WebSearchEnd(..)
            | EventMsg::ExecCommandBegin(..)
            | EventMsg::ExecCommandOutputDelta(..)
            | EventMsg::ExecCommandEnd(..)
            | EventMsg::ViewImageToolCall(..)
            | EventMsg::ExecApprovalRequest(..)
            | EventMsg::ApplyPatchApprovalRequest(..)
            | EventMsg::PatchApplyBegin(..)
            | EventMsg::PatchApplyEnd(..)
            | EventMsg::TurnDiff(..)
            | EventMsg::GetHistoryEntryResponse(..)
            | EventMsg::McpListToolsResponse(..)
            | EventMsg::ListCustomPromptsResponse(..)
            | EventMsg::PlanUpdate(..)
            | EventMsg::EnteredReviewMode(..)
            | EventMsg::ExitedReviewMode(..)) => {
                warn!("Unexpected event: {:?}", e);
            }
        }
    }
}

#[derive(Clone)]
struct SessionClient {
    session_id: SessionId,
    client: Arc<dyn Client>,
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
}

impl SessionClient {
    fn new(session_id: SessionId, client_capabilities: Arc<Mutex<ClientCapabilities>>) -> Self {
        Self {
            session_id,
            client: ACP_CLIENT.get().expect("Client should be set").clone(),
            client_capabilities,
        }
    }

    #[cfg(test)]
    fn with_client(
        session_id: SessionId,
        client: Arc<dyn Client>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
    ) -> Self {
        Self {
            session_id,
            client,
            client_capabilities,
        }
    }

    fn supports_terminal_output(&self, active_command: &ActiveCommand) -> bool {
        active_command.terminal_output
            && self
                .client_capabilities
                .lock()
                .unwrap()
                .meta
                .as_ref()
                .is_some_and(|v| {
                    v.get("terminal_output")
                        .is_some_and(|v| v.as_bool().unwrap_or_default())
                })
    }

    async fn send_notification(&self, update: SessionUpdate) {
        let notification = SessionNotification {
            session_id: self.session_id.clone(),
            update,
            meta: None,
        };

        if let Err(e) = self.client.session_notification(notification).await {
            error!("Failed to send session notification: {:?}", e);
        }
    }

    async fn send_agent_text(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentMessageChunk(ContentChunk {
            content: ContentBlock::Text(TextContent {
                text: text.into(),
                annotations: None,
                meta: None,
            }),
            meta: None,
        }))
        .await;
    }

    async fn send_agent_thought(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentThoughtChunk(ContentChunk {
            content: ContentBlock::Text(TextContent {
                text: text.into(),
                annotations: None,
                meta: None,
            }),
            meta: None,
        }))
        .await;
    }

    async fn update_plan(&self, plan: Vec<PlanItemArg>) {
        self.send_notification(SessionUpdate::Plan(Plan {
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
        }))
        .await;
    }

    async fn request_permission(
        &self,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) -> Result<RequestPermissionResponse, Error> {
        self.client
            .request_permission(RequestPermissionRequest {
                session_id: self.session_id.clone(),
                tool_call,
                options,
                meta: None,
            })
            .await
    }
}

struct ConversationActor<A> {
    /// Allows for logging out from slash commands
    auth: A,
    /// Used for sending messages back to the client.
    client: SessionClient,
    /// The conversation associated with this task.
    conversation: Arc<dyn CodexConversationImpl>,
    /// The configuration for the conversation.
    config: Config,
    /// The custom prompts loaded for this workspace.
    custom_prompts: Rc<RefCell<Vec<CustomPrompt>>>,
    /// The model presets for the conversation.
    model_presets: Rc<Vec<ModelPreset>>,
    /// A sender for each interested `Op` submission that needs events routed.
    submissions: HashMap<String, SubmissionState>,
    /// A receiver for incoming conversation messages.
    message_rx: mpsc::UnboundedReceiver<ConversationMessage>,
}

impl<A: Auth> ConversationActor<A> {
    fn new(
        auth: A,
        client: SessionClient,
        conversation: Arc<dyn CodexConversationImpl>,
        config: Config,
        model_presets: Rc<Vec<ModelPreset>>,
        message_rx: mpsc::UnboundedReceiver<ConversationMessage>,
    ) -> Self {
        Self {
            auth,
            client,
            conversation,
            config,
            custom_prompts: Rc::default(),
            model_presets,
            submissions: HashMap::new(),
            message_rx,
        }
    }

    async fn spawn(mut self) {
        loop {
            tokio::select! {
                biased;
                message = self.message_rx.recv() => match message {
                    Some(message) => self.handle_message(message).await,
                    None => break,
                },
                event = self.conversation.next_event() => match event {
                    Ok(event) => self.handle_event(event).await,
                    Err(e) => {
                        error!("Error getting next event: {:?}", e);
                        break;
                    }
                }
            }
            // Litter collection of senders with no receivers
            self.submissions
                .retain(|_, submission| submission.is_active());
        }
    }

    async fn handle_message(&mut self, message: ConversationMessage) {
        match message {
            ConversationMessage::Load { response_tx } => {
                let result = self.handle_load().await;
                drop(response_tx.send(result));
                let client = self.client.clone();
                let mut available_commands = self.builtin_commands();
                let load_custom_prompts = self.load_custom_prompts().await;
                let custom_prompts = self.custom_prompts.clone();

                // Have this happen after the session is loaded by putting it
                // in a separate task
                tokio::task::spawn_local(async move {
                    let mut new_custom_prompts = load_custom_prompts
                        .await
                        .map_err(|_| Error::internal_error())
                        .flatten()
                        .inspect_err(|e| error!("Failed to load custom prompts {e:?}"))
                        .unwrap_or_default();

                    for prompt in &new_custom_prompts {
                        available_commands.push(AvailableCommand {
                            name: prompt.name.clone(),
                            description: prompt.description.clone().unwrap_or_default(),
                            input: prompt
                                .argument_hint
                                .clone()
                                .map(|hint| AvailableCommandInput::Unstructured { hint }),
                            meta: None,
                        });
                    }
                    std::mem::swap(
                        custom_prompts.borrow_mut().deref_mut(),
                        &mut new_custom_prompts,
                    );

                    client
                        .send_notification(SessionUpdate::AvailableCommandsUpdate(
                            AvailableCommandsUpdate {
                                available_commands,
                                meta: None,
                            },
                        ))
                        .await;
                });
            }
            ConversationMessage::Prompt {
                request,
                response_tx,
            } => {
                let result = self.handle_prompt(request).await;
                drop(response_tx.send(result));
            }
            ConversationMessage::SetMode { mode, response_tx } => {
                let result = self.handle_set_mode(mode).await;
                drop(response_tx.send(result));
            }
            ConversationMessage::SetModel { model, response_tx } => {
                let result = self.handle_set_model(model).await;
                drop(response_tx.send(result));
            }
            ConversationMessage::Cancel { response_tx } => {
                let result = self.handle_cancel().await;
                drop(response_tx.send(result));
            }
        }
    }

    fn builtin_commands(&mut self) -> Vec<AvailableCommand> {
        vec![
            AvailableCommand {
                name: "review".to_string(),
                description: "Review my current changes and find issues".into(),
                input: Some(AvailableCommandInput::Unstructured {
                    hint: "optional custom review instructions".into(),
                }),
                meta: None,
            },
            AvailableCommand {
                name: "review-branch".to_string(),
                description: "Review the code changes against a specific branch".into(),
                input: Some(AvailableCommandInput::Unstructured {
                    hint: "branch name".into(),
                }),
                meta: None,
            },
            AvailableCommand {
                name: "review-commit".to_string(),
                description: "Review the code changes introduced by a commit".into(),
                input: Some(AvailableCommandInput::Unstructured {
                    hint: "commit sha".into(),
                }),
                meta: None,
            },
            AvailableCommand {
                name: "init".to_string(),
                description: "create an AGENTS.md file with instructions for Codex".into(),
                input: None,
                meta: None,
            },
            AvailableCommand {
                name: "compact".to_string(),
                description: "summarize conversation to prevent hitting the context limit".into(),
                input: None,
                meta: None,
            },
            AvailableCommand {
                name: "logout".to_string(),
                description: "logout of Codex".into(),
                input: None,
                meta: None,
            },
        ]
    }

    async fn load_custom_prompts(&mut self) -> oneshot::Receiver<Result<Vec<CustomPrompt>, Error>> {
        let (response_tx, response_rx) = oneshot::channel();
        let submission_id = match self.conversation.submit(Op::ListCustomPrompts).await {
            Ok(id) => id,
            Err(e) => {
                drop(response_tx.send(Err(Error::internal_error().with_data(e.to_string()))));
                return response_rx;
            }
        };

        self.submissions.insert(
            submission_id,
            SubmissionState::CustomPrompts(CustomPromptsState::new(response_tx)),
        );

        response_rx
    }

    fn modes(&self) -> Option<SessionModeState> {
        let current_mode_id = APPROVAL_PRESETS
            .iter()
            .find(|preset| {
                preset.approval == self.config.approval_policy
                    && preset.sandbox == self.config.sandbox_policy
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

    fn find_current_model(&self) -> Option<ModelId> {
        let preset = self
            .model_presets
            .iter()
            .find(|preset| preset.model == self.config.model)?;

        let effort = self
            .config
            .model_reasoning_effort
            .and_then(|effort| {
                preset
                    .supported_reasoning_efforts
                    .iter()
                    .find_map(|e| (e.effort == effort).then_some(effort))
            })
            .unwrap_or(preset.default_reasoning_effort);

        Some(Self::model_id(preset.id, effort))
    }

    fn model_id(id: &'static str, effort: ReasoningEffort) -> ModelId {
        ModelId(format!("{id}/{effort}").into())
    }

    fn parse_model_id(id: ModelId) -> Result<(String, ReasoningEffort), Error> {
        let Some((model, reasoning)) = id.0.split_once('/') else {
            return Err(Error::internal_error().with_data(format!("Invalid model ID: {id}")));
        };
        let reasoning = serde_json::from_value(reasoning.into()).map_err(|_| {
            Error::internal_error().with_data(format!("Invalid reasoning effort: {reasoning}"))
        })?;
        Ok((model.to_owned(), reasoning))
    }

    fn models(&self) -> Result<SessionModelState, Error> {
        let current_model_id = self.find_current_model().ok_or_else(|| {
            anyhow::anyhow!("No valid model preset for model {}", self.config.model)
        })?;

        let available_models = self
            .model_presets
            .iter()
            .flat_map(|preset| {
                preset
                    .supported_reasoning_efforts
                    .iter()
                    .map(|effort| ModelInfo {
                        model_id: Self::model_id(preset.id, effort.effort),
                        name: format!("{} ({})", preset.display_name, effort.effort),
                        description: Some(format!("{} {}", preset.description, effort.description)),
                        meta: None,
                    })
            })
            .collect();

        Ok(SessionModelState {
            current_model_id,
            available_models,
            meta: None,
        })
    }

    async fn handle_load(&mut self) -> Result<LoadSessionResponse, Error> {
        Ok(LoadSessionResponse {
            modes: self.modes(),
            models: Some(self.models()?),
            meta: None,
        })
    }

    async fn handle_prompt(
        &mut self,
        request: PromptRequest,
    ) -> Result<oneshot::Receiver<Result<StopReason, Error>>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let items = build_prompt_items(request.prompt);
        let op;
        if let Some((name, rest)) = extract_slash_command(&items) {
            match name {
                "compact" => op = Op::Compact,
                "init" => {
                    op = Op::UserInput {
                        items: vec![UserInput::Text {
                            text: INIT_COMMAND_PROMPT.into(),
                        }],
                    }
                }
                "review" => {
                    let (prompt, user_facing_hint) = if rest.is_empty() {
                        (
                            "Review the current code changes (staged, unstaged, and untracked files) and provide prioritized findings.",
                            "current changes",
                        )
                    } else {
                        let trimmed = rest.trim();
                        (trimmed, trimmed)
                    };

                    op = Op::Review {
                        review_request: ReviewRequest {
                            prompt: prompt.to_string(),
                            user_facing_hint: user_facing_hint.to_string(),
                        },
                    }
                }
                "review-branch" if !rest.is_empty() => {
                    let branch = rest.trim();
                    // https://github.com/zed-industries/codex/blob/9baf30493dd9f531af1e4dc49a781654b1b2c966/codex-rs/tui/src/chatwidget.rs#L1995-L2002
                    op = Op::Review {
                        review_request: ReviewRequest {
                            prompt: format!(
                                "Review the code changes against the base branch '{branch}'. Start by finding the merge diff between the current branch and {branch}'s upstream e.g. (`git merge-base HEAD \"$(git rev-parse --abbrev-ref \"{branch}@{{upstream}}\")\"`), then run `git diff` against that SHA to see what changes we would merge into the {branch} branch. Provide prioritized, actionable findings."
                            ),
                            user_facing_hint: format!("changes against '{branch}'"),
                        },
                    }
                }
                "review-commit" if !rest.is_empty() => {
                    let sha = rest.trim();
                    // https://github.com/zed-industries/codex/blob/9baf30493dd9f531af1e4dc49a781654b1b2c966/codex-rs/tui/src/chatwidget.rs#L2033-L2042
                    op = Op::Review {
                        review_request: ReviewRequest {
                            prompt: format!(
                                "Review the code changes introduced by commit {sha}. Provide prioritized, actionable findings."
                            ),
                            user_facing_hint: format!("commit {sha}"),
                        },
                    }
                }
                "logout" => {
                    self.auth.logout()?;
                    return Err(Error::auth_required());
                }
                _ => {
                    if let Some(prompt) =
                        expand_custom_prompt(name, rest, self.custom_prompts.borrow().as_ref())
                            .map_err(|e| Error::invalid_params().with_data(e.user_message()))?
                    {
                        op = Op::UserInput {
                            items: vec![UserInput::Text { text: prompt }],
                        }
                    } else {
                        op = Op::UserInput { items }
                    }
                }
            }
        } else {
            op = Op::UserInput { items }
        }

        let submission_id = self
            .conversation
            .submit(op.clone())
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?;

        info!("Submitted prompt with submission_id: {submission_id}");
        info!("Starting to wait for conversation events for submission_id: {submission_id}");

        let state = match op {
            Op::Compact => SubmissionState::Task(TaskState::new(response_tx)),
            _ => SubmissionState::Prompt(PromptState::new(
                self.conversation.clone(),
                response_tx,
                submission_id.clone(),
            )),
        };

        self.submissions.insert(submission_id, state);

        Ok(response_rx)
    }

    async fn handle_set_mode(&mut self, mode: SessionModeId) -> Result<(), Error> {
        let preset = APPROVAL_PRESETS
            .iter()
            .find(|preset| mode.0.as_ref() == preset.id)
            .ok_or_else(Error::invalid_params)?;

        self.conversation
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

        self.config.approval_policy = preset.approval;
        self.config.sandbox_policy = preset.sandbox.clone();

        Ok(())
    }

    async fn handle_set_model(&mut self, model: ModelId) -> Result<(), Error> {
        let (model, effort) = Self::parse_model_id(model)?;

        self.conversation
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(model.clone()),
                effort: Some(Some(effort)),
                summary: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = model;
        self.config.model_reasoning_effort = Some(effort);

        Ok(())
    }

    async fn handle_cancel(&mut self) -> Result<(), Error> {
        self.conversation
            .submit(Op::Interrupt)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn handle_event(&mut self, Event { id, msg }: Event) {
        if let Some(submission) = self.submissions.get_mut(&id) {
            submission.handle_event(&self.client, msg).await;
        } else {
            error!("Received event for unknown submission ID: {}", id);
        }
    }
}

fn build_prompt_items(prompt: Vec<ContentBlock>) -> Vec<UserInput> {
    prompt
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text_block) => Some(UserInput::Text {
                text: text_block.text,
            }),
            ContentBlock::Image(image_block) => Some(UserInput::Image {
                image_url: format!("data:{};base64,{}", image_block.mime_type, image_block.data),
            }),
            ContentBlock::ResourceLink(ResourceLink {
                annotations: _,
                description: _,
                mime_type: _,
                name,
                size: _,
                title: _,
                uri,
                meta: _,
            }) => Some(UserInput::Text {
                text: format_uri_as_link(Some(name), uri),
            }),
            ContentBlock::Resource(EmbeddedResource {
                annotations: _,
                resource:
                    EmbeddedResourceResource::TextResourceContents(TextResourceContents {
                        mime_type: _,
                        text,
                        uri,
                        meta: _,
                    }),
                meta: _,
            }) => Some(UserInput::Text {
                text: format!(
                    "{}\n<context ref=\"{uri}\">\n${text}\n</context>",
                    format_uri_as_link(None, uri.clone())
                ),
            }),
            // Skip other content types for now
            ContentBlock::Audio(..) | ContentBlock::Resource(..) => None,
        })
        .collect()
}

fn format_uri_as_link(name: Option<String>, uri: String) -> String {
    if let Some(name) = name
        && !name.is_empty()
    {
        format!("[@{name}]({uri})")
    } else if let Some(path) = uri.strip_prefix("file://") {
        let name = path.split('/').next_back().unwrap_or(path);
        format!("[@{name}]({uri})")
    } else if uri.starts_with("zed://") {
        let name = uri.split('/').next_back().unwrap_or(&uri);
        format!("[@{name}]({uri})")
    } else {
        uri
    }
}

fn codex_content_to_acp_content(content: mcp_types::ContentBlock) -> ToolCallContent {
    ToolCallContent::Content {
        content: match content {
            mcp_types::ContentBlock::TextContent(text_content) => ContentBlock::Text(TextContent {
                annotations: text_content.annotations.map(convert_annotations),
                text: text_content.text,
                meta: None,
            }),
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
                        ) => EmbeddedResourceResource::TextResourceContents(TextResourceContents {
                            mime_type: text_resource_contents.mime_type,
                            text: text_resource_contents.text,
                            uri: text_resource_contents.uri,
                            meta: None,
                        }),
                        mcp_types::EmbeddedResourceResource::BlobResourceContents(
                            blob_resource_contents,
                        ) => EmbeddedResourceResource::BlobResourceContents(BlobResourceContents {
                            blob: blob_resource_contents.blob,
                            mime_type: blob_resource_contents.mime_type,
                            uri: blob_resource_contents.uri,
                            meta: None,
                        }),
                    },
                    meta: None,
                })
            }
        },
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

/// Checks if a prompt is slash command
fn extract_slash_command(content: &[UserInput]) -> Option<(&str, &str)> {
    let line = content.first().and_then(|block| match block {
        UserInput::Text { text, .. } => Some(text),
        _ => None,
    })?;

    parse_slash_name(line)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use codex_core::{config::ConfigOverrides, protocol::AgentMessageEvent};
    use tokio::{
        sync::{Mutex, mpsc::UnboundedSender},
        task::LocalSet,
    };

    use super::*;

    #[tokio::test]
    async fn test_prompt() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["Hi".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Hi"
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_compact() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/compact".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Compact task completed"
        ));
        let ops = conversation.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::Compact]);

        Ok(())
    }

    #[tokio::test]
    async fn test_init() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/init".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }), ..
                }) if text == INIT_COMMAND_PROMPT // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );
        let ops = conversation.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::UserInput {
                items: vec![UserInput::Text {
                    text: INIT_COMMAND_PROMPT.to_string()
                }]
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_review() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/review".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "current changes" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = conversation.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    prompt: "Review the current code changes (staged, unstaged, and untracked files) and provide prioritized findings.".into(),
                    user_facing_hint: "current changes".into()
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_custom_review() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/review Review what we did in agents.md".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Review what we did in agents.md" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = conversation.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    prompt: "Review what we did in agents.md".into(),
                    user_facing_hint: "Review what we did in agents.md".into()
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_commit_review() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/review-commit 123456".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "commit 123456" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = conversation.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    prompt: "Review the code changes introduced by commit 123456. Provide prioritized, actionable findings.".into(),
                    user_facing_hint: "commit 123456".into()
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_branch_review() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/review-branch feature".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "changes against 'feature'" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = conversation.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    prompt: "Review the code changes against the base branch 'feature'. Start by finding the merge diff between the current branch and feature's upstream e.g. (`git merge-base HEAD \"$(git rev-parse --abbrev-ref \"feature@{upstream}\")\"`), then run `git diff` against that SHA to see what changes we would merge into the feature branch. Provide prioritized, actionable findings.".into(),
                    user_facing_hint: "changes against 'feature'".into()
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_custom_prompts() -> anyhow::Result<()> {
        let custom_prompts = vec![CustomPrompt {
            name: "custom".to_string(),
            path: "/tmp/custom.md".into(),
            content: "Custom prompt with $1 arg.".into(),
            description: None,
            argument_hint: None,
        }];
        let (session_id, client, conversation, message_tx, local_set) =
            setup(custom_prompts).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest {
                session_id: session_id.clone(),
                prompt: vec!["/custom foo".into()],
                meta: None,
            },
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Custom prompt with foo arg."
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = conversation.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::UserInput {
                items: vec![UserInput::Text {
                    text: "Custom prompt with foo arg.".into()
                }]
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    async fn setup(
        custom_prompts: Vec<CustomPrompt>,
    ) -> anyhow::Result<(
        SessionId,
        Arc<StubClient>,
        Arc<StubCodexConversation>,
        UnboundedSender<ConversationMessage>,
        LocalSet,
    )> {
        let session_id = SessionId("test".into());
        let client = Arc::new(StubClient::new());
        let session_client =
            SessionClient::with_client(session_id.clone(), client.clone(), Arc::default());
        let conversation = Arc::new(StubCodexConversation::new());
        let config = Config::load_with_cli_overrides(vec![], ConfigOverrides::default()).await?;
        let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut actor = ConversationActor::new(
            StubAuth,
            session_client,
            conversation.clone(),
            config,
            Default::default(),
            message_rx,
        );
        actor.custom_prompts = Rc::new(RefCell::new(custom_prompts));

        let local_set = LocalSet::new();
        local_set.spawn_local(actor.spawn());
        Ok((session_id, client, conversation, message_tx, local_set))
    }

    struct StubAuth;

    impl Auth for StubAuth {
        fn logout(&self) -> Result<bool, Error> {
            Ok(true)
        }
    }

    struct StubCodexConversation {
        current_id: AtomicUsize,
        ops: std::sync::Mutex<Vec<Op>>,
        op_tx: mpsc::UnboundedSender<Event>,
        op_rx: Mutex<mpsc::UnboundedReceiver<Event>>,
    }

    impl StubCodexConversation {
        fn new() -> Self {
            let (op_tx, op_rx) = mpsc::unbounded_channel();
            StubCodexConversation {
                current_id: AtomicUsize::new(0),
                ops: std::sync::Mutex::default(),
                op_tx,
                op_rx: Mutex::new(op_rx),
            }
        }
    }

    #[async_trait::async_trait]
    impl CodexConversationImpl for StubCodexConversation {
        async fn submit(&self, op: Op) -> Result<String, CodexErr> {
            let id = self
                .current_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            self.ops.lock().unwrap().push(op.clone());

            match op {
                Op::UserInput { items } => {
                    let prompt = items
                        .into_iter()
                        .map(|i| match i {
                            UserInput::Text { text } => text,
                            _ => unimplemented!(),
                        })
                        .join("\n");

                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::AgentMessageDelta(AgentMessageDeltaEvent {
                                delta: prompt,
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TaskComplete(TaskCompleteEvent {
                                last_agent_message: None,
                            }),
                        })
                        .unwrap();
                }
                Op::Compact => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TaskStarted(TaskStartedEvent {
                                model_context_window: None,
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::AgentMessage(AgentMessageEvent {
                                message: "Compact task completed".to_string(),
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TaskComplete(TaskCompleteEvent {
                                last_agent_message: None,
                            }),
                        })
                        .unwrap();
                }
                Op::Review { review_request } => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::EnteredReviewMode(review_request.clone()),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::ExitedReviewMode(ExitedReviewModeEvent {
                                review_output: Some(ReviewOutputEvent {
                                    findings: vec![],
                                    overall_correctness: String::new(),
                                    overall_explanation: review_request.user_facing_hint.clone(),
                                    overall_confidence_score: 1.,
                                }),
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TaskComplete(TaskCompleteEvent {
                                last_agent_message: None,
                            }),
                        })
                        .unwrap();
                }
                _ => {
                    unimplemented!()
                }
            }
            Ok(id.to_string())
        }

        async fn next_event(&self) -> Result<Event, CodexErr> {
            let Some(event) = self.op_rx.lock().await.recv().await else {
                return Err(CodexErr::InternalAgentDied);
            };
            Ok(event)
        }
    }

    struct StubClient {
        notifications: std::sync::Mutex<Vec<SessionNotification>>,
    }

    impl StubClient {
        fn new() -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Client for StubClient {
        async fn request_permission(
            &self,
            _args: RequestPermissionRequest,
        ) -> Result<RequestPermissionResponse, Error> {
            unimplemented!()
        }

        async fn session_notification(&self, args: SessionNotification) -> Result<(), Error> {
            self.notifications.lock().unwrap().push(args);
            Ok(())
        }
    }
}
