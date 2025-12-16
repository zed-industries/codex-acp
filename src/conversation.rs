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
    BlobResourceContents, Client, ClientCapabilities, Content, ContentBlock, ContentChunk, Diff,
    EmbeddedResource, EmbeddedResourceResource, Error, ImageContent, LoadSessionResponse, Meta,
    ModelId, ModelInfo, PermissionOption, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority,
    PlanEntryStatus, PromptRequest, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, ResourceLink, SelectedPermissionOutcome, SessionId, SessionMode,
    SessionModeId, SessionModeState, SessionModelState, SessionNotification, SessionUpdate,
    StopReason, Terminal, TextContent, TextResourceContents, ToolCall, ToolCallContent, ToolCallId,
    ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
    UnstructuredCommandInput,
};
use codex_common::approval_presets::{ApprovalPreset, builtin_approval_presets};
use codex_core::{
    AuthManager, CodexConversation,
    config::{Config, set_project_trust_level},
    error::CodexErr,
    openai_models::models_manager::ModelsManager,
    protocol::{
        AgentMessageContentDeltaEvent, AgentMessageEvent, AgentReasoningEvent,
        AgentReasoningSectionBreakEvent, ApplyPatchApprovalRequestEvent, ElicitationAction,
        ErrorEvent, Event, EventMsg, ExecApprovalRequestEvent, ExecCommandBeginEvent,
        ExecCommandEndEvent, ExecCommandOutputDeltaEvent, ExitedReviewModeEvent, FileChange,
        ItemCompletedEvent, ItemStartedEvent, ListCustomPromptsResponseEvent, McpInvocation,
        McpStartupCompleteEvent, McpStartupUpdateEvent, McpToolCallBeginEvent, McpToolCallEndEvent,
        Op, PatchApplyBeginEvent, PatchApplyEndEvent, ReasoningContentDeltaEvent,
        ReasoningRawContentDeltaEvent, ReviewDecision, ReviewOutputEvent, ReviewRequest,
        ReviewTarget, SandboxPolicy, StreamErrorEvent, TaskCompleteEvent, TaskStartedEvent,
        TerminalInteractionEvent, TurnAbortedEvent, UserMessageEvent, ViewImageToolCallEvent,
        WarningEvent, WebSearchBeginEvent, WebSearchEndEvent,
    },
    review_format::format_review_findings_block,
    review_prompts::user_facing_hint,
};
use codex_protocol::{
    approvals::ElicitationRequestEvent,
    config_types::TrustLevel,
    custom_prompts::CustomPrompt,
    openai_models::{ModelPreset, ReasoningEffort},
    parse_command::ParsedCommand,
    plan_tool::{PlanItemArg, StepStatus, UpdatePlanArgs},
    user_input::UserInput,
};
use itertools::Itertools;
use mcp_types::{CallToolResult, RequestId};
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

use crate::{
    ACP_CLIENT,
    prompt_args::{expand_custom_prompt, parse_slash_name},
};

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);
const INIT_COMMAND_PROMPT: &str = include_str!("./prompt_for_init_command.md");

/// Trait for abstracting over the `CodexConversation` to make testing easier.
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

#[async_trait::async_trait]
pub trait ModelsManagerImpl {
    async fn get_model(&self, model_id: &Option<String>, config: &Config) -> String;
    async fn list_models(&self, config: &Config) -> Vec<ModelPreset>;
}

#[async_trait::async_trait]
impl ModelsManagerImpl for ModelsManager {
    async fn get_model(&self, model_id: &Option<String>, config: &Config) -> String {
        self.get_model(model_id, config).await
    }

    async fn list_models(&self, config: &Config) -> Vec<ModelPreset> {
        self.list_models(config).await
    }
}

pub trait Auth {
    fn logout(&self) -> Result<bool, Error>;
}

impl Auth for Arc<AuthManager> {
    fn logout(&self) -> Result<bool, Error> {
        self.as_ref()
            .logout()
            .map_err(|e| Error::internal_error().data(e.to_string()))
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
        models_manager: Arc<dyn ModelsManagerImpl>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        config: Config,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        let actor = ConversationActor::new(
            auth,
            SessionClient::new(session_id, client_capabilities),
            conversation,
            models_manager,
            config,
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
            .map_err(|e| Error::internal_error().data(e.to_string()))?
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
            .map_err(|e| Error::internal_error().data(e.to_string()))??
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_mode(&self, mode: SessionModeId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::SetMode { mode, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_model(&self, model: ModelId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::SetModel { model, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn cancel(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::Cancel { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
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
                    drop(tx.send(Ok(custom_prompts)));
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
    seen_message_deltas: bool,
    seen_reasoning_deltas: bool,
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
            seen_message_deltas: false,
            seen_reasoning_deltas: false,
        }
    }

    fn is_active(&self) -> bool {
        let Some(response_tx) = &self.response_tx else {
            return false;
        };
        !response_tx.is_closed()
    }

    #[expect(clippy::too_many_lines)]
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
            EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent { thread_id, turn_id, item_id, delta }) => {
                info!("Agent message content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, delta: {delta:?}");
                self.seen_message_deltas = true;
                client.send_agent_text(delta).await;
            }
            EventMsg::ReasoningContentDelta(ReasoningContentDeltaEvent { thread_id, turn_id, item_id, delta, summary_index: index })
            | EventMsg::ReasoningRawContentDelta(ReasoningRawContentDeltaEvent { thread_id, turn_id, item_id, delta, content_index: index }) => {
                info!("Agent reasoning content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, index: {index}, delta: {delta:?}");
                self.seen_reasoning_deltas = true;
                client.send_agent_thought(delta).await;
            }
            EventMsg::AgentReasoningSectionBreak(AgentReasoningSectionBreakEvent { item_id, summary_index}) => {
                info!("Agent reasoning section break received:  item_id: {item_id}, index: {summary_index}");
                // Make sure the section heading actually get spacing
                self.seen_reasoning_deltas = true;
                client.send_agent_thought("\n\n").await;
            }
            EventMsg::AgentMessage(AgentMessageEvent { message }) => {
                info!("Agent message (non-delta) received: {message:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_message_deltas) {
                    client.send_agent_text(message).await;
                }
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                info!("Agent reasoning (non-delta) received: {text:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_reasoning_deltas) {
                    client.send_agent_thought(text).await;
                }
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
            EventMsg::TerminalInteraction(event) => {
                info!(
                    "Terminal interaction: call_id={}, process_id={}, stdin={}",
                    event.call_id, event.process_id, event.stdin
                );
                self.terminal_interaction(client, event).await;
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
            EventMsg::UndoStarted(event) => {
                client
                    .send_agent_text(
                        event
                            .message
                            .unwrap_or_else(|| "Undo in progress...".to_string()),
                    )
                    .await;
            }
            EventMsg::UndoCompleted(event) => {
                let fallback = if event.success {
                    "Undo completed.".to_string()
                } else {
                    "Undo failed.".to_string()
                };
                client.send_agent_text(event.message.unwrap_or(fallback)).await;
            }
            EventMsg::StreamError(StreamErrorEvent { message , codex_error_info}) => {
                error!("Handled error during turn: {message} {codex_error_info:?}");
            }
            EventMsg::Error(ErrorEvent { message, codex_error_info }) => {
                error!("Unhandled error during turn: {message} {codex_error_info:?}");
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Err(Error::internal_error().data(json!({ "message": message, "codex_error_info": codex_error_info })))).ok();
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
                    .send_notification(
                        SessionUpdate::ToolCall(
                            ToolCall::new(call_id, format!("View Image {display_path}"))
                                .kind(ToolKind::Read).status(ToolCallStatus::Completed)
                                .content(vec![ToolCallContent::Content(Content::new(ContentBlock::ResourceLink(ResourceLink::new(display_path.clone(), display_path.clone())
                            )
                        )
                    )]).locations(vec![ToolCallLocation::new(path)])))
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
            EventMsg::Warning(WarningEvent { message }) => {
                warn!("Warning: {message}");
            }
            EventMsg::McpStartupUpdate(McpStartupUpdateEvent { server, status }) => {
                info!("MCP startup update: server={server}, status={status:?}");
            }
            EventMsg::McpStartupComplete(McpStartupCompleteEvent {
                ready,
                failed,
                cancelled,
            }) => {
                info!(
                    "MCP startup complete: ready={ready:?}, failed={failed:?}, cancelled={cancelled:?}"
                );
            }
            EventMsg::ElicitationRequest(event) => {
                info!("Elicitation request: server={}, id={:?}, message={}", event.server_name, event.id, event.message);
                if let Err(err) = self.mcp_elicitation(client, event).await && let Some(response_tx) = self.response_tx.take() {
                    drop(response_tx.send(Err(err)));
                }
            }

            // Ignore these events
            EventMsg::AgentReasoningRawContent(..)
            // In the future we can use this to update usage stats
            | EventMsg::TokenCount(..)
            // we already have a way to diff the turn, so ignore
            | EventMsg::TurnDiff(..)
            // Revisit when we can emit status updates
            | EventMsg::BackgroundEvent(..)
            | EventMsg::ContextCompacted(..)
            // Old events
            | EventMsg::AgentMessageDelta(..) | EventMsg::AgentReasoningDelta(..) | EventMsg::AgentReasoningRawContentDelta(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::SessionConfigured(..) => {}

            // Unexpected events for this submission
            e @ (EventMsg::McpListToolsResponse(..)
            // returned from Op::ListCustomPrompts, ignore
            | EventMsg::ListCustomPromptsResponse(..)
            | EventMsg::ListSkillsResponse(..)
            // Used for returning a single history entry
            | EventMsg::GetHistoryEntryResponse(..)
            | EventMsg::DeprecationNotice(..)
            ) => {
                warn!("Unexpected event: {:?}", e);
            }
        }
    }

    async fn mcp_elicitation(
        &self,
        client: &SessionClient,
        event: ElicitationRequestEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let ElicitationRequestEvent {
            server_name,
            id,
            message,
        } = event;
        let tool_call_id = ToolCallId::new(match &id {
            RequestId::String(s) => s.clone(),
            RequestId::Integer(i) => i.to_string(),
        });
        let response = client
            .request_permission(
                ToolCallUpdate::new(
                    tool_call_id.clone(),
                    ToolCallUpdateFields::new()
                        .title(server_name.clone())
                        .status(ToolCallStatus::Pending)
                        .content(vec![message.into()])
                        .raw_input(raw_input),
                ),
                vec![
                    PermissionOption::new(
                        "approved",
                        "Yes, provide the requested info",
                        PermissionOptionKind::AllowOnce,
                    ),
                    PermissionOption::new(
                        "abort",
                        "No, but continue without it",
                        PermissionOptionKind::RejectOnce,
                    ),
                    PermissionOption::new(
                        "cancel",
                        "Cancel this request",
                        PermissionOptionKind::RejectOnce,
                    ),
                ],
            )
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome { option_id, .. }) => {
                match option_id.0.as_ref() {
                    "approved" => ElicitationAction::Accept,
                    "abort" => ElicitationAction::Decline,
                    _ => ElicitationAction::Cancel,
                }
            }
            RequestPermissionOutcome::Cancelled | _ => ElicitationAction::Cancel,
        };

        self.conversation
            .submit(Op::ResolveElicitation {
                server_name,
                request_id: id,
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        client
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                tool_call_id,
                ToolCallUpdateFields::new().status(if decision == ElicitationAction::Accept {
                    ToolCallStatus::Completed
                } else {
                    ToolCallStatus::Failed
                }),
            )))
            .await;

        Ok(())
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
            turn_id: _,
        } = event;
        let (title, locations, content) = extract_tool_call_content_from_changes(changes);
        let response = client
            .request_permission(
                ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new()
                        .kind(ToolKind::Edit)
                        .status(ToolCallStatus::Pending)
                        .title(title)
                        .locations(locations)
                        .content(content.chain(reason.map(|r| r.into())).collect::<Vec<_>>())
                        .raw_input(raw_input),
                ),
                vec![
                    PermissionOption::new("approved", "Yes", PermissionOptionKind::AllowOnce),
                    PermissionOption::new(
                        "abort",
                        "No, provide feedback",
                        PermissionOptionKind::RejectOnce,
                    ),
                ],
            )
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome { option_id, .. }) => {
                match option_id.0.as_ref() {
                    "approved" => ReviewDecision::Approved,
                    _ => ReviewDecision::Abort,
                }
            }
            RequestPermissionOutcome::Cancelled | _ => ReviewDecision::Abort,
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
            turn_id: _,
        } = event;

        let (title, locations, content) = extract_tool_call_content_from_changes(changes);

        client
            .send_notification(SessionUpdate::ToolCall(
                ToolCall::new(call_id, title)
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .content(content.collect())
                    .raw_input(raw_input),
            ))
            .await;
    }

    async fn end_patch_apply(&self, client: &SessionClient, event: PatchApplyEndEvent) {
        let raw_output = serde_json::json!(&event);
        let PatchApplyEndEvent {
            call_id,
            stdout: _,
            stderr: _,
            success,
            changes,
            turn_id: _,
        } = event;

        let (title, locations, content) = if !changes.is_empty() {
            let (title, locations, content) = extract_tool_call_content_from_changes(changes);
            (Some(title), Some(locations), Some(content.collect()))
        } else {
            (None, None, None)
        };

        client
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(if success {
                        ToolCallStatus::Completed
                    } else {
                        ToolCallStatus::Failed
                    })
                    .raw_output(raw_output)
                    .title(title)
                    .locations(locations)
                    .content(content),
            )))
            .await;
    }

    async fn start_mcp_tool_call(
        &self,
        client: &SessionClient,
        call_id: String,
        invocation: McpInvocation,
    ) {
        let title = format!("Tool: {}/{}", invocation.server, invocation.tool);
        client
            .send_notification(SessionUpdate::ToolCall(
                ToolCall::new(call_id, title)
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!(&invocation)),
            ))
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
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(if is_error {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    })
                    .raw_output(raw_output)
                    .content(result.ok().filter(|result| !result.content.is_empty()).map(
                        |result| {
                            result
                                .content
                                .into_iter()
                                .map(codex_content_to_acp_content)
                                .collect()
                        },
                    )),
            )))
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
            turn_id: _,
            cwd,
            reason,
            parsed_cmd,
            proposed_execpolicy_amendment,
        } = event;

        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId::new(call_id.clone());
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

        let mut content = vec![];

        if let Some(reason) = reason {
            content.push(reason);
        }
        if let Some(amendment) = proposed_execpolicy_amendment {
            content.push(format!(
                "Proposed Amendment: {}",
                amendment.command().join("\n")
            ));
        }

        let content = if content.is_empty() {
            None
        } else {
            Some(vec![content.join("\n").into()])
        };

        let response = client
            .request_permission(
                ToolCallUpdate::new(
                    tool_call_id,
                    ToolCallUpdateFields::new()
                        .kind(kind)
                        .status(ToolCallStatus::Pending)
                        .title(title)
                        .raw_input(raw_input)
                        .content(content)
                        .locations(if locations.is_empty() {
                            None
                        } else {
                            Some(locations)
                        }),
                ),
                vec![
                    PermissionOption::new(
                        "approved-for-session",
                        "Always",
                        PermissionOptionKind::AllowAlways,
                    ),
                    PermissionOption::new("approved", "Yes", PermissionOptionKind::AllowOnce),
                    PermissionOption::new(
                        "abort",
                        "No, provide feedback",
                        PermissionOptionKind::RejectOnce,
                    ),
                ],
            )
            .await?;

        let decision = match response.outcome {
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome { option_id, .. }) => {
                match option_id.0.as_ref() {
                    "approved-for-session" => ReviewDecision::ApprovedForSession,
                    "approved" => ReviewDecision::Approved,
                    _ => ReviewDecision::Abort,
                }
            }
            RequestPermissionOutcome::Cancelled | _ => ReviewDecision::Abort,
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
            turn_id: _,
            source: _,
            interaction_input: _,
            call_id,
            command: _,
            cwd,
            parsed_cmd,
            process_id: _,
        } = event;
        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId::new(call_id.clone());
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
            let content = vec![ToolCallContent::Terminal(Terminal::new(call_id.clone()))];
            let meta = Some(Meta::from_iter([(
                "terminal_info".to_owned(),
                serde_json::json!({
                    "terminal_id": call_id,
                    "cwd": cwd
                }),
            )]));
            (content, meta)
        } else {
            (vec![], None)
        };

        self.active_command = Some(active_command);

        client
            .send_notification(SessionUpdate::ToolCall(
                ToolCall::new(tool_call_id, title)
                    .kind(kind)
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(raw_input)
                    .content(content)
                    .meta(meta),
            ))
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

            let update = if client.supports_terminal_output(active_command) {
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new(),
                )
                .meta(Meta::from_iter([(
                    "terminal_output".to_owned(),
                    serde_json::json!({
                        "terminal_id": call_id,
                        "data": data_str
                    }),
                )]))
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
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new().content(vec![content.into()]),
                )
            };

            client
                .send_notification(SessionUpdate::ToolCallUpdate(update))
                .await;
        }
    }

    async fn exec_command_end(&mut self, client: &SessionClient, event: ExecCommandEndEvent) {
        let raw_output = serde_json::json!(&event);
        let ExecCommandEndEvent {
            turn_id: _,
            command: _,
            cwd: _,
            parsed_cmd: _,
            source: _,
            interaction_input: _,
            call_id,
            exit_code,
            stdout: _,
            stderr: _,
            aggregated_output: _,
            duration: _,
            formatted_output: _,
            process_id: _,
        } = event;
        if let Some(active_command) = self.active_command.take()
            && active_command.call_id == call_id
        {
            let is_success = exit_code == 0;

            client
                .send_notification(SessionUpdate::ToolCallUpdate(
                    ToolCallUpdate::new(
                        active_command.tool_call_id.clone(),
                        ToolCallUpdateFields::new()
                            .status(if is_success {
                                ToolCallStatus::Completed
                            } else {
                                ToolCallStatus::Failed
                            })
                            .raw_output(raw_output),
                    )
                    .meta(
                        client.supports_terminal_output(&active_command).then(|| {
                            Meta::from_iter([(
                                "terminal_exit".into(),
                                serde_json::json!({
                                    "terminal_id": call_id,
                                    "exit_code": exit_code,
                                    "signal": null
                                }),
                            )])
                        }),
                    ),
                ))
                .await;
        }
    }

    async fn terminal_interaction(
        &mut self,
        client: &SessionClient,
        event: TerminalInteractionEvent,
    ) {
        let TerminalInteractionEvent {
            call_id,
            process_id: _,
            stdin,
        } = event;

        let stdin = format!("\n{stdin}\n");
        // Stream output bytes to the display-only terminal via ToolCallUpdate meta.
        if let Some(active_command) = &mut self.active_command
            && *active_command.call_id == call_id
        {
            let update = if client.supports_terminal_output(active_command) {
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new(),
                )
                .meta(Meta::from_iter([(
                    "terminal_output".to_owned(),
                    serde_json::json!({
                        "terminal_id": call_id,
                        "data": stdin
                    }),
                )]))
            } else {
                active_command.output.push_str(&stdin);
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
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new().content(vec![content.into()]),
                )
            };

            client
                .send_notification(SessionUpdate::ToolCallUpdate(update))
                .await;
        }
    }

    async fn start_web_search(&mut self, client: &SessionClient, call_id: String) {
        self.active_web_search = Some(call_id.clone());
        client
            .send_notification(SessionUpdate::ToolCall(
                ToolCall::new(call_id, "Searching the Web").kind(ToolKind::Fetch),
            ))
            .await;
    }

    async fn update_web_search_query(
        &self,
        client: &SessionClient,
        call_id: String,
        query: String,
    ) {
        client
            .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::InProgress)
                    .title(format!("Searching for: {query}"))
                    .raw_input(serde_json::json!({
                        "query": query
                    })),
            )))
            .await;
    }

    async fn complete_web_search(&mut self, client: &SessionClient) {
        if let Some(call_id) = self.active_web_search.take() {
            client
                .send_notification(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
                )))
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
                    _ => format!("Search {cmd}"),
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
            locations.push(ToolCallLocation::new(if path.is_relative() {
                cwd.join(&path)
            } else {
                path
            }));
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
            EventMsg::UndoStarted(event) => {
                client
                    .send_agent_text(
                        event
                            .message
                            .unwrap_or_else(|| "Undo in progress...".to_string()),
                    )
                    .await;
            }
            EventMsg::UndoCompleted(event) => {
                let fallback = if event.success {
                    "Undo completed.".to_string()
                } else {
                    "Undo failed.".to_string()
                };
                client
                    .send_agent_text(event.message.unwrap_or(fallback))
                    .await;
            }
            EventMsg::StreamError(StreamErrorEvent {
                message,
                codex_error_info,
            }) => {
                error!("Handled error during turn: {message} {codex_error_info:?}");
            }
            EventMsg::Error(ErrorEvent {
                message,
                codex_error_info,
            }) => {
                error!("Unhandled error during turn: {message} {codex_error_info:?}");
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx
                        .send(Err(Error::internal_error().data(
                            json!({ "message": message, "codex_error_info": codex_error_info }),
                        )))
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
            EventMsg::Warning(WarningEvent { message }) => {
                warn!("Warning: {message}");
            }
            EventMsg::McpStartupUpdate(McpStartupUpdateEvent { server, status }) => {
                info!("MCP startup update: server={server}, status={status:?}");
            }
            EventMsg::McpStartupComplete(McpStartupCompleteEvent {
                ready,
                failed,
                cancelled,
            }) => {
                info!(
                    "MCP startup complete: ready={ready:?}, failed={failed:?}, cancelled={cancelled:?}"
                );
            }
            // Expected but ignore
            EventMsg::TaskStarted(..)
            | EventMsg::ItemStarted(..)
            | EventMsg::ItemCompleted(..)
            | EventMsg::TokenCount(..)
            | EventMsg::AgentMessageDelta(..)
            | EventMsg::AgentReasoningDelta(..)
            | EventMsg::AgentMessageContentDelta(..)
            | EventMsg::AgentReasoningRawContent(..)
            | EventMsg::AgentReasoningRawContentDelta(..)
            | EventMsg::ReasoningContentDelta(..)
            | EventMsg::ReasoningRawContentDelta(..)
            | EventMsg::AgentReasoningSectionBreak(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::BackgroundEvent(..)
            | EventMsg::ContextCompacted(..) => {}
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
            | EventMsg::TerminalInteraction(..)
            | EventMsg::ViewImageToolCall(..)
            | EventMsg::ExecApprovalRequest(..)
            | EventMsg::ApplyPatchApprovalRequest(..)
            | EventMsg::PatchApplyBegin(..)
            | EventMsg::PatchApplyEnd(..)
            | EventMsg::TurnDiff(..)
            | EventMsg::GetHistoryEntryResponse(..)
            | EventMsg::McpListToolsResponse(..)
            | EventMsg::ListCustomPromptsResponse(..)
            | EventMsg::ListSkillsResponse(..)
            | EventMsg::PlanUpdate(..)
            | EventMsg::EnteredReviewMode(..)
            | EventMsg::ExitedReviewMode(..)
            | EventMsg::DeprecationNotice(..)
            | EventMsg::ElicitationRequest(..)) => {
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
        if let Err(e) = self
            .client
            .session_notification(SessionNotification::new(self.session_id.clone(), update))
            .await
        {
            error!("Failed to send session notification: {:?}", e);
        }
    }

    async fn send_agent_text(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            text.into().into(),
        )))
        .await;
    }

    async fn send_agent_thought(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentThoughtChunk(ContentChunk::new(
            text.into().into(),
        )))
        .await;
    }

    async fn update_plan(&self, plan: Vec<PlanItemArg>) {
        self.send_notification(SessionUpdate::Plan(Plan::new(
            plan.into_iter()
                .map(|entry| {
                    PlanEntry::new(
                        entry.step,
                        PlanEntryPriority::Medium,
                        match entry.status {
                            StepStatus::Pending => PlanEntryStatus::Pending,
                            StepStatus::InProgress => PlanEntryStatus::InProgress,
                            StepStatus::Completed => PlanEntryStatus::Completed,
                        },
                    )
                })
                .collect(),
        )))
        .await;
    }

    async fn request_permission(
        &self,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) -> Result<RequestPermissionResponse, Error> {
        self.client
            .request_permission(RequestPermissionRequest::new(
                self.session_id.clone(),
                tool_call,
                options,
            ))
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
    /// The models available for this conversation.
    models_manager: Arc<dyn ModelsManagerImpl>,
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
        models_manager: Arc<dyn ModelsManagerImpl>,
        config: Config,
        message_rx: mpsc::UnboundedReceiver<ConversationMessage>,
    ) -> Self {
        Self {
            auth,
            client,
            conversation,
            config,
            custom_prompts: Rc::default(),
            models_manager,
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
                let mut available_commands = Self::builtin_commands();
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
                        available_commands.push(
                            AvailableCommand::new(
                                prompt.name.clone(),
                                prompt.description.clone().unwrap_or_default(),
                            )
                            .input(prompt.argument_hint.as_ref().map(
                                |hint| {
                                    AvailableCommandInput::Unstructured(
                                        UnstructuredCommandInput::new(hint.clone()),
                                    )
                                },
                            )),
                        );
                    }
                    std::mem::swap(
                        custom_prompts.borrow_mut().deref_mut(),
                        &mut new_custom_prompts,
                    );

                    client
                        .send_notification(SessionUpdate::AvailableCommandsUpdate(
                            AvailableCommandsUpdate::new(available_commands),
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

    fn builtin_commands() -> Vec<AvailableCommand> {
        vec![
            AvailableCommand::new("review", "Review my current changes and find issues").input(
                AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                    "optional custom review instructions",
                )),
            ),
            AvailableCommand::new(
                "review-branch",
                "Review the code changes against a specific branch",
            )
            .input(AvailableCommandInput::Unstructured(
                UnstructuredCommandInput::new("branch name"),
            )),
            AvailableCommand::new(
                "review-commit",
                "Review the code changes introduced by a commit",
            )
            .input(AvailableCommandInput::Unstructured(
                UnstructuredCommandInput::new("commit sha"),
            )),
            AvailableCommand::new(
                "init",
                "create an AGENTS.md file with instructions for Codex",
            ),
            AvailableCommand::new(
                "compact",
                "summarize conversation to prevent hitting the context limit",
            ),
            AvailableCommand::new("undo", "undo Codex’s most recent turn"),
            AvailableCommand::new("logout", "logout of Codex"),
        ]
    }

    async fn load_custom_prompts(&mut self) -> oneshot::Receiver<Result<Vec<CustomPrompt>, Error>> {
        let (response_tx, response_rx) = oneshot::channel();
        let submission_id = match self.conversation.submit(Op::ListCustomPrompts).await {
            Ok(id) => id,
            Err(e) => {
                drop(response_tx.send(Err(Error::internal_error().data(e.to_string()))));
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
            .map(|preset| SessionModeId::new(preset.id))?;

        Some(SessionModeState::new(
            current_mode_id,
            APPROVAL_PRESETS
                .iter()
                .map(|preset| {
                    SessionMode::new(preset.id, preset.label).description(preset.description)
                })
                .collect(),
        ))
    }

    async fn find_current_model(&self) -> Option<ModelId> {
        let model_presets = self.models_manager.list_models(&self.config).await;
        let config_model = self.get_current_model().await;
        let preset = model_presets
            .iter()
            .find(|preset| preset.model == config_model)?;

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

        Some(Self::model_id(&preset.id, effort))
    }

    fn model_id(id: &str, effort: ReasoningEffort) -> ModelId {
        ModelId::new(format!("{id}/{effort}"))
    }

    fn parse_model_id(id: &ModelId) -> Option<(String, ReasoningEffort)> {
        let (model, reasoning) = id.0.split_once('/')?;
        let reasoning = serde_json::from_value(reasoning.into()).ok()?;
        Some((model.to_owned(), reasoning))
    }

    async fn models(&self) -> Result<SessionModelState, Error> {
        let mut available_models = Vec::new();

        let current_model_id = if let Some(model_id) = self.find_current_model().await {
            model_id
        } else {
            // If no preset found, return the current model string as-is
            let model_id = ModelId::new(self.get_current_model().await);
            available_models.push(ModelInfo::new(
                model_id.clone(),
                if self.config.model_provider_id == "openai" {
                    model_id.to_string()
                } else {
                    format!("{model_id} ({})", self.config.model_provider.name)
                },
            ));
            model_id
        };

        // If the user is using a custom provider, don't return the list
        if self.config.model_provider_id != "openai" {
            return Ok(SessionModelState::new(
                current_model_id.clone(),
                available_models,
            ));
        }

        available_models.extend(
            self.models_manager
                .list_models(&self.config)
                .await
                .iter()
                .flat_map(|preset| {
                    preset.supported_reasoning_efforts.iter().map(|effort| {
                        ModelInfo::new(
                            Self::model_id(&preset.id, effort.effort),
                            format!("{} ({})", preset.display_name, effort.effort),
                        )
                        .description(format!("{} {}", preset.description, effort.description))
                    })
                }),
        );

        Ok(SessionModelState::new(current_model_id, available_models))
    }

    async fn handle_load(&mut self) -> Result<LoadSessionResponse, Error> {
        Ok(LoadSessionResponse::new()
            .models(self.models().await?)
            .modes(self.modes()))
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
                "undo" => op = Op::Undo,
                "init" => {
                    op = Op::UserInput {
                        items: vec![UserInput::Text {
                            text: INIT_COMMAND_PROMPT.into(),
                        }],
                    }
                }
                "review" => {
                    let instructions = rest.trim();
                    let target = if instructions.is_empty() {
                        ReviewTarget::UncommittedChanges
                    } else {
                        ReviewTarget::Custom {
                            instructions: instructions.to_owned(),
                        }
                    };

                    op = Op::Review {
                        review_request: ReviewRequest {
                            user_facing_hint: Some(user_facing_hint(&target)),
                            target,
                        },
                    }
                }
                "review-branch" if !rest.is_empty() => {
                    let target = ReviewTarget::BaseBranch {
                        branch: rest.trim().to_owned(),
                    };
                    op = Op::Review {
                        review_request: ReviewRequest {
                            user_facing_hint: Some(user_facing_hint(&target)),
                            target,
                        },
                    }
                }
                "review-commit" if !rest.is_empty() => {
                    let target = ReviewTarget::Commit {
                        sha: rest.trim().to_owned(),
                        title: None,
                    };
                    op = Op::Review {
                        review_request: ReviewRequest {
                            user_facing_hint: Some(user_facing_hint(&target)),
                            target,
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
                            .map_err(|e| Error::invalid_params().data(e.user_message()))?
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
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        info!("Submitted prompt with submission_id: {submission_id}");
        info!("Starting to wait for conversation events for submission_id: {submission_id}");

        let state = match op {
            Op::Compact | Op::Undo => SubmissionState::Task(TaskState::new(response_tx)),
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

        match preset.sandbox {
            // Treat this user action as a trusted dir
            SandboxPolicy::DangerFullAccess | SandboxPolicy::WorkspaceWrite { .. } => {
                set_project_trust_level(
                    &self.config.codex_home,
                    &self.config.cwd,
                    TrustLevel::Trusted,
                )?;
            }
            SandboxPolicy::ReadOnly => {}
        }

        Ok(())
    }

    async fn get_current_model(&self) -> String {
        self.models_manager
            .get_model(&self.config.model, &self.config)
            .await
    }

    async fn handle_set_model(&mut self, model: ModelId) -> Result<(), Error> {
        // Try parsing as preset format, otherwise use as-is, fallback to config
        let (model_to_use, effort_to_use) = if let Some((m, e)) = Self::parse_model_id(&model) {
            (m, Some(e))
        } else {
            let model_str = model.0.to_string();
            let fallback = if !model_str.is_empty() {
                model_str
            } else {
                self.get_current_model().await
            };
            (fallback, self.config.model_reasoning_effort)
        };

        if model_to_use.is_empty() {
            return Err(Error::invalid_params().data("No model parsed or configured"));
        }

        self.conversation
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(model_to_use.clone()),
                effort: Some(effort_to_use),
                summary: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;

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
            warn!("Received event for unknown submission ID: {id} {msg:?}");
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
            ContentBlock::ResourceLink(ResourceLink { name, uri, .. }) => Some(UserInput::Text {
                text: format_uri_as_link(Some(name), uri),
            }),
            ContentBlock::Resource(EmbeddedResource {
                resource:
                    EmbeddedResourceResource::TextResourceContents(TextResourceContents {
                        text,
                        uri,
                        ..
                    }),
                ..
            }) => Some(UserInput::Text {
                text: format!(
                    "{}\n<context ref=\"{uri}\">\n{text}\n</context>",
                    format_uri_as_link(None, uri.clone())
                ),
            }),
            // Skip other content types for now
            ContentBlock::Audio(..) | ContentBlock::Resource(..) | _ => None,
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
    ToolCallContent::Content(Content::new(match content {
        mcp_types::ContentBlock::TextContent(mcp_types::TextContent {
            annotations, text, ..
        }) => ContentBlock::Text(
            TextContent::new(text).annotations(annotations.map(convert_annotations)),
        ),
        mcp_types::ContentBlock::ImageContent(mcp_types::ImageContent {
            annotations,
            data,
            mime_type,
            ..
        }) => ContentBlock::Image(
            ImageContent::new(data, mime_type).annotations(annotations.map(convert_annotations)),
        ),
        mcp_types::ContentBlock::AudioContent(mcp_types::AudioContent {
            annotations,
            data,
            mime_type,
            ..
        }) => ContentBlock::Audio(
            AudioContent::new(data, mime_type).annotations(annotations.map(convert_annotations)),
        ),
        mcp_types::ContentBlock::ResourceLink(mcp_types::ResourceLink {
            annotations,
            description,
            mime_type,
            name,
            size,
            title,
            uri,
            ..
        }) => ContentBlock::ResourceLink(
            ResourceLink::new(name, uri)
                .annotations(annotations.map(convert_annotations))
                .description(description)
                .mime_type(mime_type)
                .size(size)
                .title(title),
        ),
        mcp_types::ContentBlock::EmbeddedResource(mcp_types::EmbeddedResource {
            annotations,
            resource,
            ..
        }) => {
            let resource = match resource {
                mcp_types::EmbeddedResourceResource::TextResourceContents(
                    mcp_types::TextResourceContents {
                        mime_type,
                        text,
                        uri,
                    },
                ) => EmbeddedResourceResource::TextResourceContents(
                    TextResourceContents::new(text, uri).mime_type(mime_type),
                ),
                mcp_types::EmbeddedResourceResource::BlobResourceContents(
                    mcp_types::BlobResourceContents {
                        blob,
                        mime_type,
                        uri,
                    },
                ) => EmbeddedResourceResource::BlobResourceContents(
                    BlobResourceContents::new(blob, uri).mime_type(mime_type),
                ),
            };
            ContentBlock::Resource(
                EmbeddedResource::new(resource).annotations(annotations.map(convert_annotations)),
            )
        }
    }))
}

fn convert_annotations(
    mcp_types::Annotations {
        audience,
        last_modified,
        priority,
    }: mcp_types::Annotations,
) -> Annotations {
    Annotations::new()
        .audience(audience.map(|a| {
            a.into_iter()
                .map(|audience| match audience {
                    mcp_types::Role::Assistant => agent_client_protocol::Role::Assistant,
                    mcp_types::Role::User => agent_client_protocol::Role::User,
                })
                .collect::<Vec<_>>()
        }))
        .last_modified(last_modified)
        .priority(priority)
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
        changes.keys().map(ToolCallLocation::new).collect(),
        changes.into_iter().map(|(path, change)| {
            ToolCallContent::Diff(match change {
                codex_core::protocol::FileChange::Add { content } => Diff::new(path, content),
                codex_core::protocol::FileChange::Delete { content } => {
                    Diff::new(path, String::new()).old_text(content)
                }
                codex_core::protocol::FileChange::Update {
                    unified_diff: _,
                    move_path,
                    old_content,
                    new_content,
                } => Diff::new(move_path.unwrap_or(path), new_content).old_text(old_content),
            })
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

    use codex_core::{
        config::ConfigOverrides, openai_models::model_presets::all_model_presets,
        protocol::AgentMessageEvent,
    };
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
            request: PromptRequest::new(session_id.clone(), vec!["Hi".into()]),
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
            request: PromptRequest::new(session_id.clone(), vec!["/compact".into()]),
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
    async fn test_undo() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/undo".into()]),
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
        assert_eq!(
            notifications.len(),
            2,
            "notifications don't match {notifications:?}"
        );
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Undo in progress..."
        ));
        assert!(matches!(
            &notifications[1].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Undo completed."
        ));

        let ops = conversation.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::Undo]);

        Ok(())
    }

    #[tokio::test]
    async fn test_init() -> anyhow::Result<()> {
        let (session_id, client, conversation, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/init".into()]),
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
            request: PromptRequest::new(session_id.clone(), vec!["/review".into()]),
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
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::UncommittedChanges)),
                    target: ReviewTarget::UncommittedChanges,
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
        let instructions = "Review what we did in agents.md";

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest::new(
                session_id.clone(),
                vec![format!("/review {instructions}").into()],
            ),
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
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::Custom {
                        instructions: instructions.to_owned()
                    })),
                    target: ReviewTarget::Custom {
                        instructions: instructions.to_owned()
                    },
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
            request: PromptRequest::new(session_id.clone(), vec!["/review-commit 123456".into()]),
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
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::Commit {
                        sha: "123456".to_owned(),
                        title: None
                    })),
                    target: ReviewTarget::Commit {
                        sha: "123456".to_owned(),
                        title: None
                    },
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
            request: PromptRequest::new(session_id.clone(), vec!["/review-branch feature".into()]),
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
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::BaseBranch {
                        branch: "feature".to_owned()
                    })),
                    target: ReviewTarget::BaseBranch {
                        branch: "feature".to_owned()
                    },
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
            request: PromptRequest::new(session_id.clone(), vec!["/custom foo".into()]),
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

    #[tokio::test]
    async fn test_delta_deduplication() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ConversationMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["test delta".into()]),
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

        // We should only get ONE notification, not duplicates from both delta and non-delta
        let notifications = client.notifications.lock().unwrap();
        assert_eq!(
            notifications.len(),
            1,
            "Should only receive delta event, not duplicate non-delta. Got: {notifications:?}"
        );
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "test delta"
        ));

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
        let session_id = SessionId::new("test");
        let client = Arc::new(StubClient::new());
        let session_client =
            SessionClient::with_client(session_id.clone(), client.clone(), Arc::default());
        let conversation = Arc::new(StubCodexConversation::new());
        let models_manager = Arc::new(StubModelsManager);
        let config = Config::load_with_cli_overrides(vec![], ConfigOverrides::default()).await?;
        let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut actor = ConversationActor::new(
            StubAuth,
            session_client,
            conversation.clone(),
            models_manager,
            config,
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

    struct StubModelsManager;

    #[async_trait::async_trait]
    impl ModelsManagerImpl for StubModelsManager {
        async fn get_model(&self, _model_id: &Option<String>, _config: &Config) -> String {
            all_model_presets()[0].to_owned().id
        }

        async fn list_models(&self, _config: &Config) -> Vec<ModelPreset> {
            all_model_presets().to_owned()
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
                            msg: EventMsg::AgentMessageContentDelta(
                                AgentMessageContentDeltaEvent {
                                    thread_id: id.to_string(),
                                    turn_id: id.to_string(),
                                    item_id: id.to_string(),
                                    delta: prompt.clone(),
                                },
                            ),
                        })
                        .unwrap();
                    // Send non-delta event (should be deduplicated, but handled by deduplication)
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::AgentMessage(AgentMessageEvent { message: prompt }),
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
                Op::Undo => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::UndoStarted(codex_core::protocol::UndoStartedEvent {
                                message: Some("Undo in progress...".to_string()),
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::UndoCompleted(
                                codex_core::protocol::UndoCompletedEvent {
                                    success: true,
                                    message: Some("Undo completed.".to_string()),
                                },
                            ),
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
                                    overall_explanation: review_request
                                        .user_facing_hint
                                        .clone()
                                        .unwrap_or_default(),
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
