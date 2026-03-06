use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ops::DerefMut,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, LazyLock, Mutex},
};

use agent_client_protocol::{
    AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate, Client, ClientCapabilities,
    ConfigOptionUpdate, Content, ContentBlock, ContentChunk, Diff, EmbeddedResource,
    EmbeddedResourceResource, Error, LoadSessionResponse, Meta, ModelId, ModelInfo,
    PermissionOption, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    ResourceLink, SelectedPermissionOutcome, SessionConfigId, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOption, SessionConfigValueId, SessionId,
    SessionInfoUpdate, SessionMode, SessionModeId, SessionModeState, SessionModelState,
    SessionNotification, SessionUpdate, StopReason, Terminal, TextResourceContents, ToolCall,
    ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind, UnstructuredCommandInput, UsageUpdate,
};
use codex_apply_patch::parse_patch;
use codex_core::{
    AuthManager, CodexThread,
    config::{Config, set_project_trust_level},
    error::CodexErr,
    models_manager::manager::{ModelsManager, RefreshStrategy},
    review_format::format_review_findings_block,
    review_prompts::user_facing_hint,
};
use codex_protocol::protocol::{
    AgentMessageContentDeltaEvent, AgentMessageEvent, AgentReasoningEvent,
    AgentReasoningRawContentEvent, AgentReasoningSectionBreakEvent, ApplyPatchApprovalRequestEvent,
    ElicitationAction, ErrorEvent, Event, EventMsg, ExecApprovalRequestEvent,
    ExecCommandBeginEvent, ExecCommandEndEvent, ExecCommandOutputDeltaEvent, ExecCommandStatus,
    ExitedReviewModeEvent, FileChange, ItemCompletedEvent, ItemStartedEvent,
    ListCustomPromptsResponseEvent, McpInvocation, McpStartupCompleteEvent, McpStartupStatus,
    McpStartupUpdateEvent, McpToolCallBeginEvent, McpToolCallEndEvent, ModelRerouteEvent,
    ModelRerouteReason, Op, PatchApplyBeginEvent, PatchApplyEndEvent, PatchApplyStatus,
    ReasoningContentDeltaEvent, ReasoningRawContentDeltaEvent, ReviewDecision, ReviewOutputEvent,
    ReviewRequest, ReviewTarget, SandboxPolicy, StreamErrorEvent, TerminalInteractionEvent,
    TokenCountEvent, TurnAbortedEvent, TurnCompleteEvent, TurnStartedEvent, UserMessageEvent,
    ViewImageToolCallEvent, WarningEvent, WebSearchBeginEvent, WebSearchEndEvent,
};
use codex_protocol::{
    approvals::ElicitationRequestEvent,
    config_types::TrustLevel,
    custom_prompts::CustomPrompt,
    dynamic_tools::{DynamicToolCallOutputContentItem, DynamicToolCallRequest},
    items::TurnItem,
    mcp::CallToolResult,
    models::{ResponseItem, WebSearchAction},
    openai_models::{ModelPreset, ReasoningEffort},
    parse_command::ParsedCommand,
    plan_tool::{PlanItemArg, StepStatus, UpdatePlanArgs},
    protocol::{
        DynamicToolCallResponseEvent, NetworkApprovalContext, NetworkPolicyAmendment, RolloutItem,
    },
    user_input::UserInput,
};
use codex_shell_command::parse_command::parse_command;
use codex_utils_approval_presets::{ApprovalPreset, builtin_approval_presets};
use heck::ToTitleCase;
use itertools::Itertools;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    ACP_CLIENT,
    prompt_args::{expand_custom_prompt, parse_slash_name},
};

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);
const INIT_COMMAND_PROMPT: &str = include_str!("./prompt_for_init_command.md");

/// Trait for abstracting over the `CodexThread` to make testing easier.
#[async_trait::async_trait]
pub trait CodexThreadImpl {
    async fn submit(&self, op: Op) -> Result<String, CodexErr>;
    async fn next_event(&self) -> Result<Event, CodexErr>;
}

#[async_trait::async_trait]
impl CodexThreadImpl for CodexThread {
    async fn submit(&self, op: Op) -> Result<String, CodexErr> {
        self.submit(op).await
    }

    async fn next_event(&self) -> Result<Event, CodexErr> {
        self.next_event().await
    }
}

#[async_trait::async_trait]
pub trait ModelsManagerImpl {
    async fn get_model(&self, model_id: &Option<String>) -> String;
    async fn list_models(&self) -> Vec<ModelPreset>;
}

#[async_trait::async_trait]
impl ModelsManagerImpl for ModelsManager {
    async fn get_model(&self, model_id: &Option<String>) -> String {
        self.get_default_model(model_id, RefreshStrategy::OnlineIfUncached)
            .await
    }

    async fn list_models(&self) -> Vec<ModelPreset> {
        self.list_models(RefreshStrategy::OnlineIfUncached).await
    }
}

pub trait Auth {
    fn logout(&self) -> Result<bool, Box<Error>>;
}

impl Auth for Arc<AuthManager> {
    fn logout(&self) -> Result<bool, Box<Error>> {
        self.as_ref()
            .logout()
            .map_err(|e| Box::new(Error::internal_error().data(e.to_string())))
    }
}

enum ThreadMessage {
    Load {
        response_tx: oneshot::Sender<Result<LoadSessionResponse, Error>>,
    },
    GetConfigOptions {
        response_tx: oneshot::Sender<Result<Vec<SessionConfigOption>, Error>>,
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
    SetConfigOption {
        config_id: SessionConfigId,
        value: SessionConfigValueId,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    Cancel {
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    ReplayHistory {
        history: Vec<RolloutItem>,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
}

pub struct Thread {
    /// A sender for interacting with the thread.
    message_tx: mpsc::UnboundedSender<ThreadMessage>,
    /// A handle to the spawned task.
    _handle: tokio::task::JoinHandle<()>,
}

impl Thread {
    pub fn new(
        session_id: SessionId,
        thread: Arc<dyn CodexThreadImpl>,
        auth: Arc<AuthManager>,
        models_manager: Arc<dyn ModelsManagerImpl>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        config: Config,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        let actor = ThreadActor::new(
            auth,
            SessionClient::new(session_id, client_capabilities),
            thread,
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

        let message = ThreadMessage::Load { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn config_options(&self) -> Result<Vec<SessionConfigOption>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::GetConfigOptions { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn prompt(&self, request: PromptRequest) -> Result<StopReason, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Prompt {
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

        let message = ThreadMessage::SetMode { mode, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_model(&self, model: ModelId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetModel { model, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_config_option(
        &self,
        config_id: SessionConfigId,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetConfigOption {
            config_id,
            value,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn cancel(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Cancel { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn replay_history(&self, history: Vec<RolloutItem>) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::ReplayHistory {
            history,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }
}

enum SubmissionState {
    /// Loading custom prompts from the project
    CustomPrompts(CustomPromptsState),
    /// User prompts, including slash commands like /init, /review, /compact, /undo.
    Prompt(Box<PromptState>),
}

impl SubmissionState {
    fn is_active(&self) -> bool {
        match self {
            Self::CustomPrompts(state) => state.is_active(),
            Self::Prompt(state) => state.is_active(),
        }
    }

    async fn handle_event(&mut self, client: &SessionClient, event: EventMsg) {
        match self {
            Self::CustomPrompts(state) => state.handle_event(event),
            Self::Prompt(state) => state.handle_event(client, event).await,
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
    tool_call_id: ToolCallId,
    terminal_output: bool,
    output: String,
    file_extension: Option<String>,
}

struct PromptState {
    active_commands: HashMap<String, ActiveCommand>,
    active_mcp_startup_servers: HashSet<String>,
    active_web_search: Option<String>,
    thread: Arc<dyn CodexThreadImpl>,
    event_count: usize,
    response_tx: Option<oneshot::Sender<Result<StopReason, Error>>>,
    pending_user_message_id: Option<String>,
    message_ids: ContentMessageIds,
    seen_message_deltas: bool,
    seen_reasoning_deltas: bool,
}

#[derive(Debug, Default)]
struct ActiveMessageId {
    logical_key: Option<String>,
    message_id: Option<String>,
}

impl ActiveMessageId {
    fn for_logical_key(&mut self, logical_key: &str) -> String {
        if self.logical_key.as_deref() == Some(logical_key)
            && let Some(message_id) = &self.message_id
        {
            return message_id.clone();
        }

        let message_id = generate_message_id();
        self.logical_key = Some(logical_key.to_owned());
        self.message_id = Some(message_id.clone());
        message_id
    }

    fn next_message_id(&mut self) -> String {
        let message_id = generate_message_id();
        self.logical_key = None;
        self.message_id = Some(message_id.clone());
        message_id
    }

    fn clear(&mut self) {
        self.logical_key = None;
        self.message_id = None;
    }
}

#[derive(Debug, Default)]
struct ContentMessageIds {
    user: ActiveMessageId,
    assistant: ActiveMessageId,
    reasoning: ActiveMessageId,
}

impl ContentMessageIds {
    fn next_user_message_id(&mut self) -> String {
        self.user.next_message_id()
    }

    fn assistant_message_id_for_item(&mut self, item_id: &str) -> String {
        self.assistant.for_logical_key(item_id)
    }

    fn next_assistant_message_id(&mut self) -> String {
        self.assistant.next_message_id()
    }

    fn clear_assistant(&mut self) {
        self.assistant.clear();
    }

    fn reasoning_message_id_for_item(&mut self, item_id: &str) -> String {
        self.reasoning.for_logical_key(item_id)
    }

    fn next_reasoning_message_id(&mut self) -> String {
        self.reasoning.next_message_id()
    }

    fn clear_reasoning(&mut self) {
        self.reasoning.clear();
    }

    fn clear_all(&mut self) {
        self.user.clear();
        self.assistant.clear();
        self.reasoning.clear();
    }
}

fn model_reroute_reason_text(reason: &ModelRerouteReason) -> &'static str {
    match reason {
        ModelRerouteReason::HighRiskCyberActivity => "high-risk cyber activity",
    }
}

fn model_reroute_notice(from_model: &str, to_model: &str, reason: &ModelRerouteReason) -> String {
    format!(
        "Model rerouted from {from_model} to {to_model} due to {}.",
        model_reroute_reason_text(reason)
    )
}

async fn send_review_mode_exit(
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

fn mcp_startup_tool_call_id(server: &str) -> String {
    format!("mcp_startup:{server}")
}

fn mcp_startup_tool_call_title(server: &str) -> String {
    format!("MCP Startup: {server}")
}

fn mcp_startup_summary_tool_call_id() -> String {
    "mcp_startup".to_string()
}

fn mcp_startup_tool_status(status: &McpStartupStatus) -> ToolCallStatus {
    match status {
        McpStartupStatus::Starting => ToolCallStatus::InProgress,
        McpStartupStatus::Ready => ToolCallStatus::Completed,
        // ACP has no cancelled tool-call status, so preserve the backend payload verbatim
        // in raw_output and map non-ready terminal states to failed.
        McpStartupStatus::Failed { .. } | McpStartupStatus::Cancelled => ToolCallStatus::Failed,
    }
}

fn mcp_startup_summary_status(event: &McpStartupCompleteEvent) -> ToolCallStatus {
    if event.failed.is_empty() && event.cancelled.is_empty() {
        ToolCallStatus::Completed
    } else {
        ToolCallStatus::Failed
    }
}

fn mcp_startup_server_status_from_summary(
    event: &McpStartupCompleteEvent,
    server: &str,
) -> Option<ToolCallStatus> {
    if event.ready.iter().any(|ready| ready == server) {
        Some(ToolCallStatus::Completed)
    } else if event.failed.iter().any(|failed| failed.server == server)
        || event.cancelled.iter().any(|cancelled| cancelled == server)
    {
        Some(ToolCallStatus::Failed)
    } else {
        None
    }
}

async fn send_mcp_startup_update(
    client: &SessionClient,
    active_mcp_startup_servers: &mut HashSet<String>,
    event: McpStartupUpdateEvent,
) {
    let raw_event = serde_json::json!(&event);
    let McpStartupUpdateEvent { server, status } = event;
    let call_id = mcp_startup_tool_call_id(&server);
    let title = mcp_startup_tool_call_title(&server);

    match status {
        McpStartupStatus::Starting => {
            if active_mcp_startup_servers.insert(server) {
                client
                    .send_tool_call(
                        ToolCall::new(call_id, title)
                            .status(ToolCallStatus::InProgress)
                            .raw_input(raw_event),
                    )
                    .await;
            } else {
                client
                    .send_tool_call_update(ToolCallUpdate::new(
                        call_id,
                        ToolCallUpdateFields::new()
                            .status(ToolCallStatus::InProgress)
                            .title(title)
                            .raw_output(raw_event),
                    ))
                    .await;
            }
        }
        status => {
            let tool_status = mcp_startup_tool_status(&status);
            if active_mcp_startup_servers.remove(&server) {
                client
                    .send_tool_call_update(ToolCallUpdate::new(
                        call_id,
                        ToolCallUpdateFields::new()
                            .status(tool_status)
                            .title(title)
                            .raw_output(raw_event),
                    ))
                    .await;
            } else {
                client
                    .send_tool_call(
                        ToolCall::new(call_id, title)
                            .status(tool_status)
                            .raw_output(raw_event),
                    )
                    .await;
            }
        }
    }
}

async fn send_mcp_startup_complete(
    client: &SessionClient,
    active_mcp_startup_servers: &mut HashSet<String>,
    event: McpStartupCompleteEvent,
) {
    let raw_output = serde_json::json!(&event);
    let active_servers = active_mcp_startup_servers
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    for server in active_servers {
        let Some(status) = mcp_startup_server_status_from_summary(&event, &server) else {
            continue;
        };

        client
            .send_tool_call_update(ToolCallUpdate::new(
                mcp_startup_tool_call_id(&server),
                ToolCallUpdateFields::new()
                    .status(status)
                    .title(mcp_startup_tool_call_title(&server))
                    .raw_output(raw_output.clone()),
            ))
            .await;
        active_mcp_startup_servers.remove(&server);
    }

    client
        .send_tool_call(
            ToolCall::new(mcp_startup_summary_tool_call_id(), "MCP Startup")
                .status(mcp_startup_summary_status(&event))
                .raw_output(raw_output),
        )
        .await;
}

impl PromptState {
    fn new(
        thread: Arc<dyn CodexThreadImpl>,
        response_tx: oneshot::Sender<Result<StopReason, Error>>,
        pending_user_message_id: Option<String>,
    ) -> Self {
        Self {
            active_commands: HashMap::new(),
            active_mcp_startup_servers: HashSet::new(),
            active_web_search: None,
            thread,
            event_count: 0,
            response_tx: Some(response_tx),
            pending_user_message_id,
            message_ids: ContentMessageIds::default(),
            seen_message_deltas: false,
            seen_reasoning_deltas: false,
        }
    }

    fn next_user_message_id(&mut self) -> String {
        self.pending_user_message_id
            .take()
            .unwrap_or_else(|| self.message_ids.next_user_message_id())
    }

    fn clear_content_message_ids(&mut self) {
        self.pending_user_message_id = None;
        self.message_ids.clear_all();
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
            | EventMsg::TurnStarted(..)
            | EventMsg::TurnComplete(..)
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
            EventMsg::TurnStarted(TurnStartedEvent {
                model_context_window,
                collaboration_mode_kind,
                turn_id,
            }) => {
                info!("Task started with context window of {turn_id} {model_context_window:?} {collaboration_mode_kind:?}");
            }
            EventMsg::TokenCount(TokenCountEvent { info, .. }) => {
                if let Some(info) = info
                    && let Some(size) = info.model_context_window {
                        let used = info.last_token_usage.tokens_in_context_window().max(0) as u64;
                        client
                            .send_notification(SessionUpdate::UsageUpdate(UsageUpdate::new(
                                used,
                                size as u64,
                            )))
                            .await;
                    }
            }
            EventMsg::ItemStarted(ItemStartedEvent { thread_id, turn_id, item }) => {
                info!("Item started with thread_id: {thread_id}, turn_id: {turn_id}, item: {item:?}");
            }
            EventMsg::UserMessage(UserMessageEvent {
                message,
                images: _,
                text_elements: _,
                local_images: _,
            }) => {
                info!("User message: {message:?}");
                client
                    .send_user_message_with_id(message, self.next_user_message_id())
                    .await;
            }
            EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
            }) => {
                info!("Agent message content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, delta: {delta:?}");
                self.seen_message_deltas = true;
                client
                    .send_agent_text_with_id(
                        delta,
                        self.message_ids.assistant_message_id_for_item(&item_id),
                    )
                    .await;
            }
            EventMsg::ReasoningContentDelta(ReasoningContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
                summary_index: index,
            })
            | EventMsg::ReasoningRawContentDelta(ReasoningRawContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
                content_index: index,
            }) => {
                info!("Agent reasoning content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, index: {index}, delta: {delta:?}");
                self.seen_reasoning_deltas = true;
                client
                    .send_agent_thought_with_id(
                        delta,
                        self.message_ids.reasoning_message_id_for_item(&item_id),
                    )
                    .await;
            }
            EventMsg::AgentReasoningSectionBreak(AgentReasoningSectionBreakEvent {
                item_id,
                summary_index,
            }) => {
                info!("Agent reasoning section break received:  item_id: {item_id}, index: {summary_index}");
                // Make sure the section heading actually get spacing
                self.seen_reasoning_deltas = true;
                client
                    .send_agent_thought_with_id(
                        "\n\n",
                        self.message_ids.reasoning_message_id_for_item(&item_id),
                    )
                    .await;
            }
            EventMsg::AgentMessage(AgentMessageEvent { message , phase: _ }) => {
                info!("Agent message (non-delta) received: {message:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_message_deltas) {
                    client
                        .send_agent_text_with_id(
                            message,
                            self.message_ids.next_assistant_message_id(),
                        )
                        .await;
                } else {
                    self.message_ids.clear_assistant();
                }
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                info!("Agent reasoning (non-delta) received: {text:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_reasoning_deltas) {
                    client
                        .send_agent_thought_with_id(
                            text,
                            self.message_ids.next_reasoning_message_id(),
                        )
                        .await;
                } else {
                    self.message_ids.clear_reasoning();
                }
            }
            EventMsg::ThreadNameUpdated(event) => {
                info!("Thread name updated: {:?}", event.thread_name);
                if let Some(title) = event.thread_name {
                    client
                        .send_notification(SessionUpdate::SessionInfoUpdate(
                            SessionInfoUpdate::new().title(title),
                        ))
                        .await;
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
            EventMsg::WebSearchEnd(WebSearchEndEvent {
                call_id,
                query,
                action,
            }) => {
                info!("Web search query received: call_id={call_id}, query={query}");
                // Send update that the search is in progress with the query
                // (WebSearchEnd just means we have the query, not that results are ready)
                self.update_web_search_query(client, call_id, query, action)
                    .await;
                // The actual search results will come through AgentMessage events
                // We mark as completed when a new tool call begins
            }
            EventMsg::ExecApprovalRequest(event) => {
                info!(
                    "Command execution started: call_id={}, command={:?}",
                    event.call_id, event.command
                );
                if let Err(err) = self.exec_approval(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
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
            EventMsg::DynamicToolCallRequest(DynamicToolCallRequest { call_id, turn_id, tool, arguments }) => {
                info!("Dynamic tool call request: call_id={call_id}, turn_id={turn_id}, tool={tool}");
                self.start_dynamic_tool_call(client, call_id, tool, arguments).await;
            }
            EventMsg::DynamicToolCallResponse(event) => {
                info!(
                    "Dynamic tool call response: call_id={}, turn_id={}, tool={}",
                    event.call_id, event.turn_id, event.tool
                );
                self.end_dynamic_tool_call(client, event).await;
            }
            EventMsg::McpToolCallBegin(McpToolCallBeginEvent {
                call_id,
                invocation,
            }) => {
                info!(
                    "MCP tool call begin: call_id={call_id}, invocation={} {}",
                    invocation.server, invocation.tool
                );
                self.start_mcp_tool_call(client, call_id, invocation).await;
            }
            EventMsg::McpToolCallEnd(McpToolCallEndEvent {
                call_id,
                invocation,
                duration,
                result,
            }) => {
                info!(
                    "MCP tool call ended: call_id={call_id}, invocation={} {}, duration={duration:?}",
                    invocation.server, invocation.tool
                );
                self.end_mcp_tool_call(client, call_id, result).await;
            }
            EventMsg::ApplyPatchApprovalRequest(event) => {
                info!(
                    "Apply patch approval request: call_id={}, reason={:?}",
                    event.call_id, event.reason
                );
                if let Err(err) = self.patch_approval(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::PatchApplyBegin(event) => {
                info!(
                    "Patch apply begin: call_id={}, auto_approved={}",
                    event.call_id, event.auto_approved
                );
                self.start_patch_apply(client, event).await;
            }
            EventMsg::PatchApplyEnd(event) => {
                info!(
                    "Patch apply end: call_id={}, success={}",
                    event.call_id, event.success
                );
                self.end_patch_apply(client, event).await;
            }
            EventMsg::ItemCompleted(ItemCompletedEvent {
                thread_id,
                turn_id,
                item,
            }) => {
                info!("Item completed: thread_id={}, turn_id={}, item={:?}", thread_id, turn_id, item);
                // Notify the client when context compaction completes so users see
                // a status message rather than silence during /compact.
                if matches!(item, TurnItem::ContextCompaction(..)) {
                    client.send_agent_text("Context compacted".to_string()).await;
                }
            }
            EventMsg::TurnComplete(TurnCompleteEvent { last_agent_message, turn_id }) => {
                info!(
                    "Task {turn_id} completed successfully after {} events. Last agent message: {last_agent_message:?}",
                    self.event_count
                );
                self.clear_content_message_ids();
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
            EventMsg::StreamError(StreamErrorEvent {
                message,
                codex_error_info,
                additional_details,
            }) => {
                error!(
                    "Handled error during turn: {message} {codex_error_info:?} {additional_details:?}"
                );
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
            EventMsg::TurnAborted(TurnAbortedEvent { reason, turn_id }) => {
                info!("Turn {turn_id:?} aborted: {reason:?}");
                self.clear_content_message_ids();
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            EventMsg::ShutdownComplete => {
                info!("Agent shutting down");
                self.clear_content_message_ids();
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
                if let Err(err) = self.review_mode_exit(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::Warning(WarningEvent { message }) => {
                warn!("Warning: {message}");
                // Forward warnings to the client as agent messages so users see
                // informational notices (e.g., the post-compact advisory message).
                client.send_agent_text(message).await;
            }
            EventMsg::McpStartupUpdate(McpStartupUpdateEvent { server, status }) => {
                info!("MCP startup update: server={server}, status={status:?}");
                self.mcp_startup_update(client, McpStartupUpdateEvent { server, status })
                    .await;
            }
            EventMsg::McpStartupComplete(McpStartupCompleteEvent {
                ready,
                failed,
                cancelled,
            }) => {
                info!(
                    "MCP startup complete: ready={ready:?}, failed={failed:?}, cancelled={cancelled:?}"
                );
                self.mcp_startup_complete(
                    client,
                    McpStartupCompleteEvent {
                        ready,
                        failed,
                        cancelled,
                    },
                )
                .await;
            }
            EventMsg::ElicitationRequest(event) => {
                info!("Elicitation request: server={}, id={:?}, message={}", event.server_name, event.id, event.message);
                if let Err(err) = self.mcp_elicitation(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::ModelReroute(ModelRerouteEvent { from_model, to_model, reason }) => {
                info!("Model reroute: from={from_model}, to={to_model}, reason={reason:?}");
                // Surface reroutes as standalone notices without touching grouped content IDs.
                client
                    .send_agent_text(model_reroute_notice(&from_model, &to_model, &reason))
                    .await;
            }

            EventMsg::ContextCompacted(..) => {
                info!("Context compacted");
                client.send_agent_text("Context compacted".to_string()).await;
            }

            // Ignore these events
            EventMsg::AgentReasoningRawContent(..)
            | EventMsg::ThreadRolledBack(..)
            // we already have a way to diff the turn, so ignore
            | EventMsg::TurnDiff(..)
            // Revisit when we can emit status updates
            | EventMsg::BackgroundEvent(..)
            | EventMsg::SkillsUpdateAvailable
            // Old events
            | EventMsg::AgentMessageDelta(..)
            | EventMsg::AgentReasoningDelta(..)
            | EventMsg::AgentReasoningRawContentDelta(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::SessionConfigured(..)
            // TODO: Subagent UI?
            | EventMsg::CollabAgentSpawnBegin(..)
            | EventMsg::CollabAgentSpawnEnd(..)
            | EventMsg::CollabAgentInteractionBegin(..)
            | EventMsg::CollabAgentInteractionEnd(..)
            | EventMsg::RealtimeConversationStarted(..)
            | EventMsg::RealtimeConversationRealtime(..)
            | EventMsg::RealtimeConversationClosed(..)
            | EventMsg::CollabWaitingBegin(..)
            | EventMsg::CollabWaitingEnd(..)
            | EventMsg::CollabResumeBegin(..)
            | EventMsg::CollabResumeEnd(..)
            | EventMsg::CollabCloseBegin(..)
            | EventMsg::CollabCloseEnd(..)
            | EventMsg::PlanDelta(..) => {}
            e @ (EventMsg::McpListToolsResponse(..)
            // returned from Op::ListCustomPrompts, ignore
            | EventMsg::ListCustomPromptsResponse(..)
            | EventMsg::ListSkillsResponse(..)
            // Used for returning a single history entry
            | EventMsg::GetHistoryEntryResponse(..)
            | EventMsg::DeprecationNotice(..)
            | EventMsg::RequestUserInput(..)
            | EventMsg::ListRemoteSkillsResponse(..)
            | EventMsg::RemoteSkillDownloaded(..)) => {
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
            codex_protocol::mcp::RequestId::String(s) => s.clone(),
            codex_protocol::mcp::RequestId::Integer(i) => i.to_string(),
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

        self.thread
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
        send_review_mode_exit(client, event).await
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
                    call_id.clone(),
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

        self.thread
            .submit(Op::PatchApproval {
                id: call_id,
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
            .send_tool_call(
                ToolCall::new(call_id, title)
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .content(content.collect())
                    .raw_input(raw_input),
            )
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
            status,
        } = event;

        let (title, locations, content) = if !changes.is_empty() {
            let (title, locations, content) = extract_tool_call_content_from_changes(changes);
            (Some(title), Some(locations), Some(content.collect()))
        } else {
            (None, None, None)
        };

        let status = match status {
            PatchApplyStatus::Completed => ToolCallStatus::Completed,
            _ if success => ToolCallStatus::Completed,
            PatchApplyStatus::Failed | PatchApplyStatus::Declined => ToolCallStatus::Failed,
        };

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(status)
                    .raw_output(raw_output)
                    .title(title)
                    .locations(locations)
                    .content(content),
            ))
            .await;
    }

    async fn start_dynamic_tool_call(
        &self,
        client: &SessionClient,
        call_id: String,
        tool: String,
        arguments: serde_json::Value,
    ) {
        client
            .send_tool_call(
                ToolCall::new(call_id, format!("Tool: {tool}"))
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!(&arguments)),
            )
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
            .send_tool_call(
                ToolCall::new(call_id, title)
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!(&invocation)),
            )
            .await;
    }

    async fn mcp_startup_update(&mut self, client: &SessionClient, event: McpStartupUpdateEvent) {
        send_mcp_startup_update(client, &mut self.active_mcp_startup_servers, event).await;
    }

    async fn mcp_startup_complete(
        &mut self,
        client: &SessionClient,
        event: McpStartupCompleteEvent,
    ) {
        send_mcp_startup_complete(client, &mut self.active_mcp_startup_servers, event).await;
    }

    async fn end_dynamic_tool_call(
        &self,
        client: &SessionClient,
        event: DynamicToolCallResponseEvent,
    ) {
        let raw_output = serde_json::json!(event);
        let DynamicToolCallResponseEvent {
            call_id,
            turn_id: _,
            tool: _,
            arguments: _,
            content_items,
            success,
            error,
            duration: _,
        } = event;

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(if success {
                        ToolCallStatus::Completed
                    } else {
                        ToolCallStatus::Failed
                    })
                    .raw_output(raw_output)
                    .content(
                        content_items
                            .into_iter()
                            .map(|item| match item {
                                DynamicToolCallOutputContentItem::InputText { text } => {
                                    ToolCallContent::Content(Content::new(text))
                                }
                                DynamicToolCallOutputContentItem::InputImage { image_url } => {
                                    ToolCallContent::Content(Content::new(
                                        ContentBlock::ResourceLink(ResourceLink::new(
                                            image_url.clone(),
                                            image_url,
                                        )),
                                    ))
                                }
                            })
                            .chain(error.map(|e| ToolCallContent::Content(Content::new(e))))
                            .collect::<Vec<_>>(),
                    ),
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
            .send_tool_call_update(ToolCallUpdate::new(
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
                                .filter_map(|content| {
                                    serde_json::from_value::<ContentBlock>(content).ok()
                                })
                                .map(|content| ToolCallContent::Content(Content::new(content)))
                                .collect()
                        },
                    )),
            ))
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
            turn_id,
            cwd,
            reason,
            parsed_cmd,
            proposed_execpolicy_amendment,
            approval_id,
            network_approval_context,
            additional_permissions,
            available_decisions,
            proposed_network_policy_amendments,
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
        self.active_commands.insert(
            call_id.clone(),
            ActiveCommand {
                terminal_output,
                tool_call_id: tool_call_id.clone(),
                output: String::new(),
                file_extension,
            },
        );

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
        if let Some(policy) = network_approval_context {
            let NetworkApprovalContext { host, protocol } = policy;
            content.push(format!("Network Approval Context: {:?} {}", protocol, host));
        }
        if let Some(permissions) = additional_permissions {
            content.push(format!(
                "Additional Permissions: {}",
                serde_json::to_string_pretty(&permissions)?
            ));
        }
        if let Some(decisions) = available_decisions {
            content.push(format!(
                "Available Decisions: {}",
                decisions.into_iter().map(|d| d.to_string()).join("\n")
            ));
        }
        if let Some(amendments) = proposed_network_policy_amendments {
            content.push(format!(
                "Proposed Network Policy Amendments: {}",
                amendments
                    .into_iter()
                    .map(|NetworkPolicyAmendment { host, action }| format!(
                        "{:?} {:?}",
                        action, host
                    ))
                    .join("\n")
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

        self.thread
            .submit(Op::ExecApproval {
                id: approval_id.unwrap_or(call_id),
                turn_id: Some(turn_id),
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

        self.active_commands.insert(call_id.clone(), active_command);

        client
            .send_tool_call(
                ToolCall::new(tool_call_id, title)
                    .kind(kind)
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(raw_input)
                    .content(content)
                    .meta(meta),
            )
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
        if let Some(active_command) = self.active_commands.get_mut(&call_id) {
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

            client.send_tool_call_update(update).await;
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
            status,
        } = event;
        if let Some(active_command) = self.active_commands.remove(&call_id) {
            let is_success = exit_code == 0;

            let status = match status {
                ExecCommandStatus::Completed => ToolCallStatus::Completed,
                _ if is_success => ToolCallStatus::Completed,
                ExecCommandStatus::Failed | ExecCommandStatus::Declined => ToolCallStatus::Failed,
            };

            client
                .send_tool_call_update(
                    ToolCallUpdate::new(
                        active_command.tool_call_id.clone(),
                        ToolCallUpdateFields::new()
                            .status(status)
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
                )
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
        if let Some(active_command) = self.active_commands.get_mut(&call_id) {
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

            client.send_tool_call_update(update).await;
        }
    }

    async fn start_web_search(&mut self, client: &SessionClient, call_id: String) {
        self.active_web_search = Some(call_id.clone());
        client
            .send_tool_call(ToolCall::new(call_id, "Searching the Web").kind(ToolKind::Fetch))
            .await;
    }

    async fn update_web_search_query(
        &self,
        client: &SessionClient,
        call_id: String,
        query: String,
        action: WebSearchAction,
    ) {
        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::InProgress)
                    .title(web_search_action_to_title(&action))
                    .raw_input(serde_json::json!({
                        "query": query,
                        "action": action
                    })),
            ))
            .await;
    }

    async fn complete_web_search(&mut self, client: &SessionClient) {
        if let Some(call_id) = self.active_web_search.take() {
            client
                .send_tool_call_update(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
                ))
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

    fn text_chunk(text: impl Into<String>, message_id: impl Into<String>) -> ContentChunk {
        ContentChunk::new(text.into().into()).message_id(message_id.into())
    }

    async fn send_user_message_with_id(
        &self,
        text: impl Into<String>,
        message_id: impl Into<String>,
    ) {
        self.send_notification(SessionUpdate::UserMessageChunk(Self::text_chunk(
            text, message_id,
        )))
        .await;
    }

    async fn send_agent_text(&self, text: impl Into<String>) {
        self.send_agent_text_with_id(text, generate_message_id())
            .await;
    }

    async fn send_agent_text_with_id(
        &self,
        text: impl Into<String>,
        message_id: impl Into<String>,
    ) {
        self.send_notification(SessionUpdate::AgentMessageChunk(Self::text_chunk(
            text, message_id,
        )))
        .await;
    }

    async fn send_agent_thought_with_id(
        &self,
        text: impl Into<String>,
        message_id: impl Into<String>,
    ) {
        self.send_notification(SessionUpdate::AgentThoughtChunk(Self::text_chunk(
            text, message_id,
        )))
        .await;
    }

    async fn send_tool_call(&self, tool_call: ToolCall) {
        self.send_notification(SessionUpdate::ToolCall(tool_call))
            .await;
    }

    async fn send_tool_call_update(&self, update: ToolCallUpdate) {
        self.send_notification(SessionUpdate::ToolCallUpdate(update))
            .await;
    }

    /// Send a replayed tool call with an explicit starting status.
    async fn send_replayed_tool_call(
        &self,
        call_id: impl Into<ToolCallId>,
        title: impl Into<String>,
        kind: ToolKind,
        status: ToolCallStatus,
        raw_input: Option<serde_json::Value>,
    ) {
        let mut tool_call = ToolCall::new(call_id, title).kind(kind).status(status);
        if let Some(input) = raw_input {
            tool_call = tool_call.raw_input(input);
        }
        self.send_tool_call(tool_call).await;
    }

    /// Send a tool call completion update (used for replay)
    async fn send_tool_call_completed(
        &self,
        call_id: impl Into<ToolCallId>,
        raw_output: Option<serde_json::Value>,
    ) {
        let mut fields = ToolCallUpdateFields::new().status(ToolCallStatus::Completed);
        if let Some(output) = raw_output {
            fields = fields.raw_output(output);
        }
        self.send_tool_call_update(ToolCallUpdate::new(call_id, fields))
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

struct ThreadActor<A> {
    /// Allows for logging out from slash commands
    auth: A,
    /// Used for sending messages back to the client.
    client: SessionClient,
    /// The thread associated with this task.
    thread: Arc<dyn CodexThreadImpl>,
    /// The configuration for the thread.
    config: Config,
    /// The custom prompts loaded for this workspace.
    custom_prompts: Rc<RefCell<Vec<CustomPrompt>>>,
    /// The models available for this thread.
    models_manager: Arc<dyn ModelsManagerImpl>,
    /// A sender for each interested `Op` submission that needs events routed.
    submissions: HashMap<String, SubmissionState>,
    /// A receiver for incoming thread messages.
    message_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    /// Last config options state we emitted to the client, used for deduping updates.
    last_sent_config_options: Option<Vec<SessionConfigOption>>,
}

impl<A: Auth> ThreadActor<A> {
    fn new(
        auth: A,
        client: SessionClient,
        thread: Arc<dyn CodexThreadImpl>,
        models_manager: Arc<dyn ModelsManagerImpl>,
        config: Config,
        message_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    ) -> Self {
        Self {
            auth,
            client,
            thread,
            config,
            custom_prompts: Rc::default(),
            models_manager,
            submissions: HashMap::new(),
            message_rx,
            last_sent_config_options: None,
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
                event = self.thread.next_event() => match event {
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

    async fn handle_message(&mut self, message: ThreadMessage) {
        match message {
            ThreadMessage::Load { response_tx } => {
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
            ThreadMessage::GetConfigOptions { response_tx } => {
                let result = self.config_options().await;
                drop(response_tx.send(result));
            }
            ThreadMessage::Prompt {
                request,
                response_tx,
            } => {
                let result = self.handle_prompt(request).await;
                drop(response_tx.send(result));
            }
            ThreadMessage::SetMode { mode, response_tx } => {
                let result = self.handle_set_mode(mode).await;
                drop(response_tx.send(result));
                self.maybe_emit_config_options_update().await;
            }
            ThreadMessage::SetModel { model, response_tx } => {
                let result = self.handle_set_model(model).await;
                drop(response_tx.send(result));
                self.maybe_emit_config_options_update().await;
            }
            ThreadMessage::SetConfigOption {
                config_id,
                value,
                response_tx,
            } => {
                let result = self.handle_set_config_option(config_id, value).await;
                drop(response_tx.send(result));
            }
            ThreadMessage::Cancel { response_tx } => {
                let result = self.handle_cancel().await;
                drop(response_tx.send(result));
            }
            ThreadMessage::ReplayHistory {
                history,
                response_tx,
            } => {
                let result = self.handle_replay_history(history).await;
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
        let submission_id = match self.thread.submit(Op::ListCustomPrompts).await {
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
                &preset.approval == self.config.permissions.approval_policy.get()
                    && &preset.sandbox == self.config.permissions.sandbox_policy.get()
            })
            .or_else(|| {
                // When the project is untrusted, the above code won't match
                // since AskForApproval::UnlessTrusted is not part of the
                // default presets. However, in this case we still want to show
                // the mode selector, which allows the user to choose a
                // different mode (which will set the project to be trusted)
                // See https://github.com/zed-industries/zed/issues/48132
                if self.config.active_project.is_untrusted() {
                    APPROVAL_PRESETS
                        .iter()
                        .find(|preset| preset.id == "read-only")
                } else {
                    None
                }
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
        let model_presets = self.models_manager.list_models().await;
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

    async fn config_options(&self) -> Result<Vec<SessionConfigOption>, Error> {
        let mut options = Vec::new();

        if let Some(modes) = self.modes() {
            let select_options = modes
                .available_modes
                .into_iter()
                .map(|m| SessionConfigSelectOption::new(m.id.0, m.name).description(m.description))
                .collect::<Vec<_>>();

            options.push(
                SessionConfigOption::select(
                    "mode",
                    "Approval Preset",
                    modes.current_mode_id.0,
                    select_options,
                )
                .category(SessionConfigOptionCategory::Mode)
                .description("Choose an approval and sandboxing preset for your session"),
            );
        }

        let presets = self.models_manager.list_models().await;

        let current_model = self.get_current_model().await;
        let current_preset = presets.iter().find(|p| p.model == current_model).cloned();

        let mut model_select_options = Vec::new();

        if current_preset.is_none() {
            // If no preset found, return the current model string as-is
            model_select_options.push(SessionConfigSelectOption::new(
                current_model.clone(),
                current_model.clone(),
            ));
        };

        model_select_options.extend(
            presets
                .into_iter()
                .filter(|model| model.show_in_picker || model.model == current_model)
                .map(|preset| {
                    SessionConfigSelectOption::new(preset.id, preset.display_name)
                        .description(preset.description)
                }),
        );

        options.push(
            SessionConfigOption::select("model", "Model", current_model, model_select_options)
                .category(SessionConfigOptionCategory::Model)
                .description("Choose which model Codex should use"),
        );

        // Reasoning effort selector (only if the current preset exists and has >1 supported effort)
        if let Some(preset) = current_preset
            && preset.supported_reasoning_efforts.len() > 1
        {
            let supported = &preset.supported_reasoning_efforts;

            let current_effort = self
                .config
                .model_reasoning_effort
                .and_then(|effort| {
                    supported
                        .iter()
                        .find_map(|e| (e.effort == effort).then_some(effort))
                })
                .unwrap_or(preset.default_reasoning_effort);

            let effort_select_options = supported
                .iter()
                .map(|e| {
                    SessionConfigSelectOption::new(
                        e.effort.to_string(),
                        e.effort.to_string().to_title_case(),
                    )
                    .description(e.description.clone())
                })
                .collect::<Vec<_>>();

            options.push(
                SessionConfigOption::select(
                    "reasoning_effort",
                    "Reasoning Effort",
                    current_effort.to_string(),
                    effort_select_options,
                )
                .category(SessionConfigOptionCategory::ThoughtLevel)
                .description("Choose how much reasoning effort the model should use"),
            );
        }

        Ok(options)
    }

    async fn maybe_emit_config_options_update(&mut self) {
        let config_options = self.config_options().await.unwrap_or_default();

        if self
            .last_sent_config_options
            .as_ref()
            .is_some_and(|prev| prev == &config_options)
        {
            return;
        }

        self.last_sent_config_options = Some(config_options.clone());

        self.client
            .send_notification(SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(
                config_options,
            )))
            .await;
    }

    async fn handle_set_config_option(
        &mut self,
        config_id: SessionConfigId,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        match config_id.0.as_ref() {
            "mode" => self.handle_set_mode(SessionModeId::new(value.0)).await,
            "model" => self.handle_set_config_model(value).await,
            "reasoning_effort" => self.handle_set_config_reasoning_effort(value).await,
            _ => Err(Error::invalid_params().data("Unsupported config option")),
        }
    }

    async fn handle_set_config_model(&mut self, value: SessionConfigValueId) -> Result<(), Error> {
        let model_id = value.0;

        let presets = self.models_manager.list_models().await;
        let preset = presets.iter().find(|p| p.id.as_str() == &*model_id);

        let model_to_use = preset
            .map(|p| p.model.clone())
            .unwrap_or_else(|| model_id.to_string());

        if model_to_use.is_empty() {
            return Err(Error::invalid_params().data("No model selected"));
        }

        let effort_to_use = if let Some(preset) = preset {
            if let Some(effort) = self.config.model_reasoning_effort
                && preset
                    .supported_reasoning_efforts
                    .iter()
                    .any(|e| e.effort == effort)
            {
                Some(effort)
            } else {
                Some(preset.default_reasoning_effort)
            }
        } else {
            // If the user selected a raw model string (not a known preset), don't invent a default.
            // Keep whatever was previously configured (or leave unset) so Codex can decide.
            self.config.model_reasoning_effort
        };

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(model_to_use.clone()),
                effort: Some(effort_to_use),
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;

        Ok(())
    }

    async fn handle_set_config_reasoning_effort(
        &mut self,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        let effort: ReasoningEffort =
            serde_json::from_value(value.0.as_ref().into()).map_err(|_| Error::invalid_params())?;

        let current_model = self.get_current_model().await;
        let presets = self.models_manager.list_models().await;
        let Some(preset) = presets.iter().find(|p| p.model == current_model) else {
            return Err(Error::invalid_params()
                .data("Reasoning effort can only be set for known model presets"));
        };

        if !preset
            .supported_reasoning_efforts
            .iter()
            .any(|e| e.effort == effort)
        {
            return Err(
                Error::invalid_params().data("Unsupported reasoning effort for selected model")
            );
        }

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: None,
                effort: Some(Some(effort)),
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model_reasoning_effort = Some(effort);

        Ok(())
    }

    async fn models(&self) -> Result<SessionModelState, Error> {
        let mut available_models = Vec::new();
        let config_model = self.get_current_model().await;

        let current_model_id = if let Some(model_id) = self.find_current_model().await {
            model_id
        } else {
            // If no preset found, return the current model string as-is
            let model_id = ModelId::new(self.get_current_model().await);
            available_models.push(ModelInfo::new(model_id.clone(), model_id.to_string()));
            model_id
        };

        available_models.extend(
            self.models_manager
                .list_models()
                .await
                .iter()
                .filter(|model| model.show_in_picker || model.model == config_model)
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
            .modes(self.modes())
            .config_options(self.config_options().await?))
    }

    async fn handle_prompt(
        &mut self,
        request: PromptRequest,
    ) -> Result<oneshot::Receiver<Result<StopReason, Error>>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let pending_user_message_id = request.message_id.clone();
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
                            text_elements: vec![],
                        }],
                        final_output_json_schema: None,
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
                    self.auth.logout().map_err(|error| *error)?;
                    return Err(Error::auth_required());
                }
                _ => {
                    if let Some(prompt) =
                        expand_custom_prompt(name, rest, self.custom_prompts.borrow().as_ref())
                            .map_err(|e| Error::invalid_params().data(e.user_message()))?
                    {
                        op = Op::UserInput {
                            items: vec![UserInput::Text {
                                text: prompt,
                                text_elements: vec![],
                            }],
                            final_output_json_schema: None,
                        }
                    } else {
                        op = Op::UserInput {
                            items,
                            final_output_json_schema: None,
                        }
                    }
                }
            }
        } else {
            op = Op::UserInput {
                items,
                final_output_json_schema: None,
            }
        }

        let submission_id = self
            .thread
            .submit(op.clone())
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        info!("Submitted prompt with submission_id: {submission_id}");
        info!("Starting to wait for conversation events for submission_id: {submission_id}");

        let state = SubmissionState::Prompt(Box::new(PromptState::new(
            self.thread.clone(),
            response_tx,
            pending_user_message_id,
        )));

        self.submissions.insert(submission_id, state);

        Ok(response_rx)
    }

    async fn handle_set_mode(&mut self, mode: SessionModeId) -> Result<(), Error> {
        let preset = APPROVAL_PRESETS
            .iter()
            .find(|preset| mode.0.as_ref() == preset.id)
            .ok_or_else(Error::invalid_params)?;

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: Some(preset.approval),
                sandbox_policy: Some(preset.sandbox.clone()),
                model: None,
                effort: None,
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config
            .permissions
            .approval_policy
            .set(preset.approval)
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        self.config
            .permissions
            .sandbox_policy
            .set(preset.sandbox.clone())
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        match preset.sandbox {
            // Treat this user action as a trusted dir
            SandboxPolicy::DangerFullAccess
            | SandboxPolicy::WorkspaceWrite { .. }
            | SandboxPolicy::ExternalSandbox { .. } => {
                set_project_trust_level(
                    &self.config.codex_home,
                    &self.config.cwd,
                    TrustLevel::Trusted,
                )?;
            }
            SandboxPolicy::ReadOnly { .. } => {}
        }

        Ok(())
    }

    async fn get_current_model(&self) -> String {
        self.models_manager.get_model(&self.config.model).await
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

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(model_to_use.clone()),
                effort: Some(effort_to_use),
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;

        Ok(())
    }

    async fn handle_cancel(&mut self) -> Result<(), Error> {
        self.thread
            .submit(Op::Interrupt)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    /// Replay conversation history to the client via session/update notifications.
    /// This is called when loading a session to stream all prior messages.
    ///
    /// We process both `EventMsg` and `ResponseItem`:
    /// - `EventMsg` for user/agent messages, reasoning, direct ACP-mappable notices,
    ///   and replayable MCP startup status events
    /// - `ResponseItem` for tool calls only (not persisted as EventMsg)
    async fn handle_replay_history(&mut self, history: Vec<RolloutItem>) -> Result<(), Error> {
        let mut message_ids = ContentMessageIds::default();
        let mut active_mcp_startup_servers = HashSet::new();

        for item in history {
            match item {
                RolloutItem::EventMsg(event_msg) => {
                    self.replay_event_msg(
                        event_msg,
                        &mut message_ids,
                        &mut active_mcp_startup_servers,
                    )
                    .await;
                }
                RolloutItem::ResponseItem(response_item) => {
                    self.replay_response_item(&response_item).await;
                }
                // Skip SessionMeta, TurnContext, Compacted
                _ => {}
            }
        }
        Ok(())
    }

    /// Convert and send an EventMsg as ACP notification(s) during replay.
    /// Handles messages, reasoning, direct ACP-mappable notices, and replayable MCP startup
    /// status events.
    async fn replay_event_msg(
        &self,
        msg: EventMsg,
        message_ids: &mut ContentMessageIds,
        active_mcp_startup_servers: &mut HashSet<String>,
    ) {
        match msg {
            EventMsg::UserMessage(UserMessageEvent { message, .. }) => {
                self.client
                    .send_user_message_with_id(message, message_ids.next_user_message_id())
                    .await;
            }
            EventMsg::AgentMessage(AgentMessageEvent { message, phase: _ }) => {
                self.client
                    .send_agent_text_with_id(message, message_ids.next_assistant_message_id())
                    .await;
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                self.client
                    .send_agent_thought_with_id(text, message_ids.next_reasoning_message_id())
                    .await;
            }
            EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent { text }) => {
                self.client
                    .send_agent_thought_with_id(text, message_ids.next_reasoning_message_id())
                    .await;
            }
            EventMsg::ThreadNameUpdated(event) => {
                if let Some(title) = event.thread_name {
                    self.client
                        .send_notification(SessionUpdate::SessionInfoUpdate(
                            SessionInfoUpdate::new().title(title),
                        ))
                        .await;
                }
            }
            EventMsg::PlanUpdate(UpdatePlanArgs {
                explanation: _,
                plan,
            }) => {
                self.client.update_plan(plan).await;
            }
            EventMsg::Warning(WarningEvent { message }) => {
                self.client.send_agent_text(message).await;
            }
            EventMsg::ModelReroute(ModelRerouteEvent {
                from_model,
                to_model,
                reason,
            }) => {
                self.client
                    .send_agent_text(model_reroute_notice(&from_model, &to_model, &reason))
                    .await;
            }
            EventMsg::ExitedReviewMode(event) => {
                if let Err(err) = send_review_mode_exit(&self.client, event).await {
                    error!("Failed to replay review mode exit: {err:?}");
                }
            }
            EventMsg::UndoStarted(event) => {
                self.client
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
                self.client
                    .send_agent_text(event.message.unwrap_or(fallback))
                    .await;
            }
            EventMsg::ContextCompacted(..) => {
                self.client
                    .send_agent_text("Context compacted".to_string())
                    .await;
            }
            EventMsg::McpStartupUpdate(event) => {
                send_mcp_startup_update(&self.client, active_mcp_startup_servers, event).await;
            }
            EventMsg::McpStartupComplete(event) => {
                send_mcp_startup_complete(&self.client, active_mcp_startup_servers, event).await;
            }
            EventMsg::ViewImageToolCall(..) => {
                // Intentionally omitted during replay. View-image invocations can already surface
                // through persisted ResponseItem tool reconstruction, and replaying the event-side
                // tool notification here would risk duplicate tool state.
            }
            // Skip other event types during replay - they either:
            // - Are transient (deltas, turn lifecycle)
            // - Don't have direct ACP equivalents
            // - Are handled via ResponseItem instead
            _ => {
                // no-op
            }
        }
    }

    /// Parse apply_patch call input to extract patch content for display.
    /// Returns (title, locations, content) if successful.
    /// For CustomToolCall, the input is the patch string directly.
    fn parse_apply_patch_call(
        &self,
        input: &str,
    ) -> Option<(String, Vec<ToolCallLocation>, Vec<ToolCallContent>)> {
        // Try to parse the patch using codex-apply-patch parser
        let parsed = parse_patch(input).ok()?;

        let mut locations = Vec::new();
        let mut file_names = Vec::new();
        let mut content = Vec::new();

        for hunk in &parsed.hunks {
            match hunk {
                codex_apply_patch::Hunk::AddFile { path, contents } => {
                    let full_path = self.config.cwd.join(path);
                    file_names.push(path.display().to_string());
                    locations.push(ToolCallLocation::new(full_path.clone()));
                    // New file: no old_text, new_text is the contents
                    content.push(ToolCallContent::Diff(Diff::new(
                        full_path,
                        contents.clone(),
                    )));
                }
                codex_apply_patch::Hunk::DeleteFile { path } => {
                    let full_path = self.config.cwd.join(path);
                    file_names.push(path.display().to_string());
                    locations.push(ToolCallLocation::new(full_path.clone()));
                    // Delete file: old_text would be original content, new_text is empty
                    content.push(ToolCallContent::Diff(
                        Diff::new(full_path, "").old_text("[file deleted]"),
                    ));
                }
                codex_apply_patch::Hunk::UpdateFile {
                    path,
                    move_path,
                    chunks,
                } => {
                    let full_path = self.config.cwd.join(path);
                    let dest_path = move_path
                        .as_ref()
                        .map(|p| self.config.cwd.join(p))
                        .unwrap_or_else(|| full_path.clone());
                    file_names.push(path.display().to_string());
                    locations.push(ToolCallLocation::new(dest_path.clone()));

                    // Build old and new text from chunks
                    let old_lines: Vec<String> = chunks
                        .iter()
                        .flat_map(|c| c.old_lines.iter().cloned())
                        .collect();
                    let new_lines: Vec<String> = chunks
                        .iter()
                        .flat_map(|c| c.new_lines.iter().cloned())
                        .collect();

                    content.push(ToolCallContent::Diff(
                        Diff::new(dest_path, new_lines.join("\n")).old_text(old_lines.join("\n")),
                    ));
                }
            }
        }

        let title = if file_names.is_empty() {
            "Apply patch".to_string()
        } else {
            format!("Edit {}", file_names.join(", "))
        };

        Some((title, locations, content))
    }

    /// Parse shell function call arguments to extract command info for rich display.
    /// Returns (title, kind, locations) if successful.
    ///
    /// Handles both:
    /// - `shell` / `container.exec`: `command` is `Vec<String>`
    /// - `shell_command`: `command` is a `String` (shell script)
    fn parse_shell_function_call(
        &self,
        name: &str,
        arguments: &str,
    ) -> Option<(String, ToolKind, Vec<ToolCallLocation>)> {
        // Extract command and workdir based on tool type
        let (command_vec, workdir): (Vec<String>, Option<String>) = if name == "shell_command" {
            // shell_command: command is a string (shell script)
            #[derive(serde::Deserialize)]
            struct ShellCommandArgs {
                command: String,
                #[serde(default)]
                workdir: Option<String>,
            }
            let args: ShellCommandArgs = serde_json::from_str(arguments).ok()?;
            // Wrap in bash -lc for parsing
            (
                vec!["bash".to_string(), "-lc".to_string(), args.command],
                args.workdir,
            )
        } else {
            // shell / container.exec: command is Vec<String>
            #[derive(serde::Deserialize)]
            struct ShellArgs {
                command: Vec<String>,
                #[serde(default)]
                workdir: Option<String>,
            }
            let args: ShellArgs = serde_json::from_str(arguments).ok()?;
            (args.command, args.workdir)
        };

        let cwd = workdir
            .map(PathBuf::from)
            .unwrap_or_else(|| self.config.cwd.clone());

        let parsed_cmd = parse_command(&command_vec);
        let ParseCommandToolCall {
            title,
            file_extension: _,
            terminal_output: _,
            locations,
            kind,
        } = parse_command_tool_call(parsed_cmd, &cwd);

        Some((title, kind, locations))
    }

    /// Convert and send a single ResponseItem as ACP notification(s) during replay.
    /// Only handles tool calls - messages/reasoning are handled via EventMsg.
    async fn replay_response_item(&self, item: &ResponseItem) {
        match item {
            // Skip Message and Reasoning - these are handled via EventMsg
            ResponseItem::Message { .. } | ResponseItem::Reasoning { .. } => {}
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                // Replay the opening lifecycle step here and let the paired output item
                // deliver completion so ACP preserves the persisted two-step tool flow.
                // Check if this is a shell command - parse it like we do for LocalShellCall.
                if matches!(name.as_str(), "shell" | "container.exec" | "shell_command")
                    && let Some((title, kind, locations)) =
                        self.parse_shell_function_call(name, arguments)
                {
                    self.client
                        .send_tool_call(
                            ToolCall::new(call_id.clone(), title)
                                .kind(kind)
                                .status(ToolCallStatus::InProgress)
                                .locations(locations)
                                .raw_input(
                                    serde_json::from_str::<serde_json::Value>(arguments).ok(),
                                ),
                        )
                        .await;
                    return;
                }

                // Fall through to generic function call handling
                self.client
                    .send_replayed_tool_call(
                        call_id.clone(),
                        name.clone(),
                        ToolKind::Other,
                        ToolCallStatus::InProgress,
                        serde_json::from_str(arguments).ok(),
                    )
                    .await;
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                self.client
                    .send_tool_call_completed(call_id.clone(), serde_json::to_value(output).ok())
                    .await;
            }
            ResponseItem::LocalShellCall {
                call_id: Some(call_id),
                action,
                status,
                ..
            } => {
                let codex_protocol::models::LocalShellAction::Exec(exec) = action;
                let cwd = exec
                    .working_directory
                    .as_ref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.config.cwd.clone());

                // Parse the command to get rich info like the live event handler does
                let parsed_cmd = parse_command(&exec.command);
                let ParseCommandToolCall {
                    title,
                    file_extension: _,
                    terminal_output: _,
                    locations,
                    kind,
                } = parse_command_tool_call(parsed_cmd, &cwd);

                let tool_status = match status {
                    codex_protocol::models::LocalShellStatus::Completed => {
                        ToolCallStatus::Completed
                    }
                    codex_protocol::models::LocalShellStatus::InProgress => {
                        ToolCallStatus::InProgress
                    }
                    codex_protocol::models::LocalShellStatus::Incomplete => ToolCallStatus::Failed,
                };
                self.client
                    .send_tool_call(
                        ToolCall::new(call_id.clone(), title)
                            .kind(kind)
                            .status(tool_status)
                            .locations(locations),
                    )
                    .await;
            }
            ResponseItem::CustomToolCall {
                name,
                input,
                call_id,
                ..
            } => {
                // Replay the opening lifecycle step here and let the paired output item
                // deliver completion so ACP preserves the persisted two-step tool flow.
                // Check if this is an apply_patch call - show the patch content.
                if name == "apply_patch"
                    && let Some((title, locations, content)) = self.parse_apply_patch_call(input)
                {
                    self.client
                        .send_tool_call(
                            ToolCall::new(call_id.clone(), title)
                                .kind(ToolKind::Edit)
                                .status(ToolCallStatus::InProgress)
                                .locations(locations)
                                .content(content)
                                .raw_input(serde_json::from_str::<serde_json::Value>(input).ok()),
                        )
                        .await;
                    return;
                }

                // Fall through to generic custom tool call handling
                self.client
                    .send_replayed_tool_call(
                        call_id.clone(),
                        name.clone(),
                        ToolKind::Other,
                        ToolCallStatus::InProgress,
                        serde_json::from_str(input).ok(),
                    )
                    .await;
            }
            ResponseItem::CustomToolCallOutput { call_id, output } => {
                self.client
                    .send_tool_call_completed(call_id.clone(), Some(serde_json::json!(output)))
                    .await;
            }
            ResponseItem::WebSearchCall { id, status, action } => {
                let call_id = web_search_call_id(id, action.as_ref());
                let status = web_search_status_to_tool_call_status(status.as_deref());
                let title = action
                    .as_ref()
                    .map(web_search_action_to_title)
                    .unwrap_or_else(|| default_web_search_title(status));

                let mut tool_call = ToolCall::new(call_id, title)
                    .kind(ToolKind::Fetch)
                    .status(status);
                if let Some(raw_input) = action.as_ref().map(web_search_action_to_replay_raw_input)
                {
                    tool_call = tool_call.raw_input(raw_input);
                }

                self.client.send_tool_call(tool_call).await;
            }
            // Skip GhostSnapshot, Compaction, Other, LocalShellCall without call_id
            _ => {}
        }
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
                text_elements: vec![],
            }),
            ContentBlock::Image(image_block) => Some(UserInput::Image {
                image_url: format!("data:{};base64,{}", image_block.mime_type, image_block.data),
            }),
            ContentBlock::ResourceLink(ResourceLink { name, uri, .. }) => Some(UserInput::Text {
                text: format_uri_as_link(Some(name), uri),
                text_elements: vec![],
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
                text_elements: vec![],
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
                codex_protocol::protocol::FileChange::Add { content } => Diff::new(path, content),
                codex_protocol::protocol::FileChange::Delete { content } => {
                    Diff::new(path, String::new()).old_text(content)
                }
                codex_protocol::protocol::FileChange::Update {
                    unified_diff: _,
                    move_path,
                    old_content,
                    new_content,
                } => Diff::new(move_path.unwrap_or(path), new_content).old_text(old_content),
            })
        }),
    )
}

/// Shared title mapping for live web-search updates and replayed web-search items.
fn web_search_action_to_title(action: &WebSearchAction) -> String {
    match action {
        WebSearchAction::Search { query, queries } => queries
            .as_ref()
            .map(|q| format!("Searching for: {}", q.join(", ")))
            .or_else(|| query.as_ref().map(|q| format!("Searching for: {q}")))
            .unwrap_or_else(|| "Web search".to_string()),
        WebSearchAction::OpenPage { url } => url
            .as_ref()
            .map(|u| format!("Opening: {u}"))
            .unwrap_or_else(|| "Open page".to_string()),
        WebSearchAction::FindInPage { pattern, url } => match (pattern, url) {
            (Some(p), Some(u)) => format!("Finding: {p} in {u}"),
            (Some(p), None) => format!("Finding: {p}"),
            (None, Some(u)) => format!("Find in page: {u}"),
            (None, None) => "Find in page".to_string(),
        },
        WebSearchAction::Other => "Web search".to_string(),
    }
}

fn web_search_action_to_replay_raw_input(action: &WebSearchAction) -> serde_json::Value {
    let mut raw_input = serde_json::json!({
        "action": action,
    });

    if let WebSearchAction::Search {
        query: Some(query), ..
    } = action
    {
        raw_input["query"] = serde_json::Value::String(query.clone());
    }

    raw_input
}

fn web_search_call_id(id: &Option<String>, action: Option<&WebSearchAction>) -> String {
    if let Some(id) = id {
        return id.clone();
    }

    match action {
        Some(WebSearchAction::OpenPage { .. }) => generate_fallback_id("web_open"),
        Some(WebSearchAction::FindInPage { .. }) => generate_fallback_id("web_find"),
        _ => generate_fallback_id("web_search"),
    }
}

fn web_search_status_to_tool_call_status(status: Option<&str>) -> ToolCallStatus {
    match status {
        Some("completed") => ToolCallStatus::Completed,
        Some("open" | "in_progress") => ToolCallStatus::InProgress,
        Some("failed" | "incomplete") => ToolCallStatus::Failed,
        _ => ToolCallStatus::Completed,
    }
}

fn default_web_search_title(status: ToolCallStatus) -> String {
    match status {
        ToolCallStatus::InProgress | ToolCallStatus::Pending => "Searching the Web".to_string(),
        ToolCallStatus::Completed | ToolCallStatus::Failed | _ => "Web search".to_string(),
    }
}

/// Generate a UUID-format ACP message ID.
fn generate_message_id() -> String {
    Uuid::new_v4().to_string()
}

/// Generate a fallback ID using UUID (used when id is missing)
fn generate_fallback_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
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

    use agent_client_protocol::TextContent;
    use codex_core::{config::ConfigOverrides, test_support::all_model_presets};
    use codex_protocol::config_types::ModeKind;
    use tokio::{
        sync::{Mutex, mpsc::UnboundedSender},
        task::LocalSet,
    };
    use uuid::Uuid;

    use super::*;

    fn message_chunk_id(update: &SessionUpdate) -> &str {
        match update {
            SessionUpdate::UserMessageChunk(ContentChunk { message_id, .. })
            | SessionUpdate::AgentMessageChunk(ContentChunk { message_id, .. })
            | SessionUpdate::AgentThoughtChunk(ContentChunk { message_id, .. }) => message_id
                .as_deref()
                .unwrap_or_else(|| panic!("missing message_id on {update:?}")),
            _ => panic!("expected content chunk update, got {update:?}"),
        }
    }

    fn message_chunk_text(update: &SessionUpdate) -> &str {
        match update {
            SessionUpdate::UserMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            })
            | SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            })
            | SessionUpdate::AgentThoughtChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) => text,
            _ => panic!("expected text content chunk update, got {update:?}"),
        }
    }

    fn assert_uuid_message_id(message_id: &str) {
        assert!(
            Uuid::parse_str(message_id).is_ok(),
            "expected UUID message_id, got {message_id}"
        );
    }

    fn is_content_update(update: &SessionUpdate) -> bool {
        matches!(
            update,
            SessionUpdate::UserMessageChunk(..)
                | SessionUpdate::AgentThoughtChunk(..)
                | SessionUpdate::AgentMessageChunk(..)
        )
    }

    fn tool_calls(notifications: &[SessionNotification]) -> Vec<ToolCall> {
        notifications
            .iter()
            .filter_map(|notification| match &notification.update {
                SessionUpdate::ToolCall(tool_call) => Some(tool_call.clone()),
                _ => None,
            })
            .collect()
    }

    fn tool_call_updates(notifications: &[SessionNotification]) -> Vec<ToolCallUpdate> {
        notifications
            .iter()
            .filter_map(|notification| match &notification.update {
                SessionUpdate::ToolCallUpdate(update) => Some(update.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn test_prompt() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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
        let ops = thread.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::Compact]);

        Ok(())
    }

    #[tokio::test]
    async fn test_undo() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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

        let ops = thread.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::Undo]);

        Ok(())
    }

    #[tokio::test]
    async fn test_init() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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
        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::UserInput {
                items: vec![UserInput::Text {
                    text: INIT_COMMAND_PROMPT.to_string(),
                    text_elements: vec![]
                }],
                final_output_json_schema: None,
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_review() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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

        let ops = thread.ops.lock().unwrap();
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
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
        let instructions = "Review what we did in agents.md";

        message_tx.send(ThreadMessage::Prompt {
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

        let ops = thread.ops.lock().unwrap();
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
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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

        let ops = thread.ops.lock().unwrap();
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
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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

        let ops = thread.ops.lock().unwrap();
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
        let (session_id, client, thread, message_tx, local_set) = setup(custom_prompts).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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

        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::UserInput {
                items: vec![UserInput::Text {
                    text: "Custom prompt with foo arg.".into(),
                    text_elements: vec![]
                }],
                final_output_json_schema: None,
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_delta_deduplication() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
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

    #[tokio::test]
    async fn test_live_message_id_grouping() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
        let user_message_id = Uuid::new_v4().to_string();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["grouped-live".into()])
                .message_id(user_message_id.clone()),
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
            8,
            "unexpected notifications: {notifications:?}"
        );

        let live_user_id = message_chunk_id(&notifications[0].update);
        let reasoning_id = message_chunk_id(&notifications[1].update);
        let reasoning_follow_up_id = message_chunk_id(&notifications[2].update);
        let reasoning_section_break_id = message_chunk_id(&notifications[3].update);
        let second_reasoning_id = message_chunk_id(&notifications[4].update);
        let assistant_id = message_chunk_id(&notifications[5].update);
        let assistant_follow_up_id = message_chunk_id(&notifications[6].update);
        let second_assistant_id = message_chunk_id(&notifications[7].update);

        for message_id in [
            live_user_id,
            reasoning_id,
            reasoning_follow_up_id,
            reasoning_section_break_id,
            second_reasoning_id,
            assistant_id,
            assistant_follow_up_id,
            second_assistant_id,
        ] {
            assert_uuid_message_id(message_id);
        }

        assert_eq!(live_user_id, user_message_id);
        assert_eq!(reasoning_id, reasoning_follow_up_id);
        assert_eq!(reasoning_id, reasoning_section_break_id);
        assert_ne!(reasoning_id, second_reasoning_id);
        assert_eq!(assistant_id, assistant_follow_up_id);
        assert_ne!(assistant_id, second_assistant_id);
        assert_ne!(live_user_id, reasoning_id);
        assert_ne!(live_user_id, assistant_id);
        assert_ne!(reasoning_id, assistant_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_reroute_is_surfaced_without_breaking_message_grouping() -> anyhow::Result<()>
    {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
        let user_message_id = Uuid::new_v4().to_string();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["reroute-live".into()])
                .message_id(user_message_id.clone()),
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
            7,
            "unexpected notifications: {notifications:?}"
        );

        assert_eq!(message_chunk_text(&notifications[0].update), "reroute-live");
        assert_eq!(message_chunk_text(&notifications[1].update), "thinking ");
        assert_eq!(
            message_chunk_text(&notifications[2].update),
            "Model rerouted from gpt-5.3-codex to gpt-5.2 due to high-risk cyber activity."
        );
        assert_eq!(message_chunk_text(&notifications[3].update), "deeply");
        assert_eq!(message_chunk_text(&notifications[4].update), "Hello");
        assert_eq!(
            message_chunk_text(&notifications[5].update),
            "Fallback warning"
        );
        assert_eq!(message_chunk_text(&notifications[6].update), " world");

        let live_user_id = message_chunk_id(&notifications[0].update);
        let reasoning_id = message_chunk_id(&notifications[1].update);
        let reroute_id = message_chunk_id(&notifications[2].update);
        let reasoning_follow_up_id = message_chunk_id(&notifications[3].update);
        let assistant_id = message_chunk_id(&notifications[4].update);
        let warning_id = message_chunk_id(&notifications[5].update);
        let assistant_follow_up_id = message_chunk_id(&notifications[6].update);

        for message_id in [
            live_user_id,
            reasoning_id,
            reroute_id,
            assistant_id,
            warning_id,
        ] {
            assert_uuid_message_id(message_id);
        }

        assert_eq!(live_user_id, user_message_id);
        assert_eq!(reasoning_id, reasoning_follow_up_id);
        assert_eq!(assistant_id, assistant_follow_up_id);
        assert_ne!(reroute_id, reasoning_id);
        assert_ne!(reroute_id, assistant_id);
        assert_ne!(warning_id, reasoning_id);
        assert_ne!(warning_id, assistant_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_startup_events_are_surfaced_without_breaking_message_grouping()
    -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
        let user_message_id = Uuid::new_v4().to_string();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["mcp-startup-live".into()])
                .message_id(user_message_id.clone()),
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
        let content_notifications: Vec<_> = notifications
            .iter()
            .filter(|notification| is_content_update(&notification.update))
            .collect();
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|notification| match &notification.update {
                SessionUpdate::ToolCall(tool_call) => Some(tool_call.clone()),
                _ => None,
            })
            .collect();
        let tool_call_updates: Vec<_> = notifications
            .iter()
            .filter_map(|notification| match &notification.update {
                SessionUpdate::ToolCallUpdate(update) => Some(update.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            content_notifications.len(),
            5,
            "unexpected content notifications: {notifications:?}"
        );
        assert_eq!(
            content_notifications
                .iter()
                .map(|notification| message_chunk_text(&notification.update))
                .collect::<Vec<_>>(),
            vec!["mcp-startup-live", "thinking ", "deeply", "Hello", " world"]
        );

        let live_user_id = message_chunk_id(&content_notifications[0].update);
        let reasoning_id = message_chunk_id(&content_notifications[1].update);
        let reasoning_follow_up_id = message_chunk_id(&content_notifications[2].update);
        let assistant_id = message_chunk_id(&content_notifications[3].update);
        let assistant_follow_up_id = message_chunk_id(&content_notifications[4].update);

        for message_id in [live_user_id, reasoning_id, assistant_id] {
            assert_uuid_message_id(message_id);
        }

        assert_eq!(live_user_id, user_message_id);
        assert_eq!(reasoning_id, reasoning_follow_up_id);
        assert_eq!(assistant_id, assistant_follow_up_id);
        assert_ne!(live_user_id, reasoning_id);
        assert_ne!(live_user_id, assistant_id);
        assert_ne!(reasoning_id, assistant_id);

        assert_eq!(
            tool_calls.len(),
            4,
            "unexpected ToolCall notifications: {tool_calls:?}"
        );
        assert_eq!(
            tool_call_updates.len(),
            3,
            "unexpected ToolCallUpdate notifications: {tool_call_updates:?}"
        );

        let alpha_start_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "alpha".into(),
            status: McpStartupStatus::Starting,
        })?;
        let beta_start_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "beta".into(),
            status: McpStartupStatus::Starting,
        })?;
        let gamma_start_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "gamma".into(),
            status: McpStartupStatus::Starting,
        })?;
        let alpha_ready_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "alpha".into(),
            status: McpStartupStatus::Ready,
        })?;
        let beta_failed_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "beta".into(),
            status: McpStartupStatus::Failed {
                error: "auth failed".into(),
            },
        })?;
        let gamma_cancelled_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "gamma".into(),
            status: McpStartupStatus::Cancelled,
        })?;
        let summary_raw = serde_json::to_value(&McpStartupCompleteEvent {
            ready: vec!["alpha".into()],
            failed: vec![codex_protocol::protocol::McpStartupFailure {
                server: "beta".into(),
                error: "auth failed".into(),
            }],
            cancelled: vec!["gamma".into()],
        })?;

        let alpha_start_id = ToolCallId::new(mcp_startup_tool_call_id("alpha"));
        let beta_start_id = ToolCallId::new(mcp_startup_tool_call_id("beta"));
        let gamma_start_id = ToolCallId::new(mcp_startup_tool_call_id("gamma"));
        let summary_id = ToolCallId::new(mcp_startup_summary_tool_call_id());

        let alpha_start = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == alpha_start_id)
            .unwrap_or_else(|| panic!("missing alpha startup tool call: {tool_calls:?}"));
        assert_eq!(alpha_start.title, mcp_startup_tool_call_title("alpha"));
        assert_eq!(alpha_start.status, ToolCallStatus::InProgress);
        assert_eq!(alpha_start.raw_input.as_ref(), Some(&alpha_start_raw));

        let beta_start = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == beta_start_id)
            .unwrap_or_else(|| panic!("missing beta startup tool call: {tool_calls:?}"));
        assert_eq!(beta_start.title, mcp_startup_tool_call_title("beta"));
        assert_eq!(beta_start.status, ToolCallStatus::InProgress);
        assert_eq!(beta_start.raw_input.as_ref(), Some(&beta_start_raw));

        let gamma_start = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == gamma_start_id)
            .unwrap_or_else(|| panic!("missing gamma startup tool call: {tool_calls:?}"));
        assert_eq!(gamma_start.title, mcp_startup_tool_call_title("gamma"));
        assert_eq!(gamma_start.status, ToolCallStatus::InProgress);
        assert_eq!(gamma_start.raw_input.as_ref(), Some(&gamma_start_raw));

        let summary = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == summary_id)
            .unwrap_or_else(|| panic!("missing startup summary tool call: {tool_calls:?}"));
        assert_eq!(summary.title, "MCP Startup");
        assert_eq!(summary.status, ToolCallStatus::Failed);
        assert_eq!(summary.raw_output.as_ref(), Some(&summary_raw));

        let alpha_ready = tool_call_updates
            .iter()
            .find(|update| {
                update.tool_call_id == ToolCallId::new(mcp_startup_tool_call_id("alpha"))
            })
            .unwrap_or_else(|| panic!("missing alpha startup update: {tool_call_updates:?}"));
        assert_eq!(alpha_ready.fields.status, Some(ToolCallStatus::Completed));
        assert_eq!(
            alpha_ready.fields.title.as_deref(),
            Some(mcp_startup_tool_call_title("alpha").as_str())
        );
        assert_eq!(
            alpha_ready.fields.raw_output.as_ref(),
            Some(&alpha_ready_raw)
        );

        let beta_failed = tool_call_updates
            .iter()
            .find(|update| update.tool_call_id == ToolCallId::new(mcp_startup_tool_call_id("beta")))
            .unwrap_or_else(|| panic!("missing beta startup update: {tool_call_updates:?}"));
        assert_eq!(beta_failed.fields.status, Some(ToolCallStatus::Failed));
        assert_eq!(
            beta_failed.fields.title.as_deref(),
            Some(mcp_startup_tool_call_title("beta").as_str())
        );
        assert_eq!(
            beta_failed.fields.raw_output.as_ref(),
            Some(&beta_failed_raw)
        );

        let gamma_cancelled = tool_call_updates
            .iter()
            .find(|update| {
                update.tool_call_id == ToolCallId::new(mcp_startup_tool_call_id("gamma"))
            })
            .unwrap_or_else(|| panic!("missing gamma startup update: {tool_call_updates:?}"));
        assert_eq!(gamma_cancelled.fields.status, Some(ToolCallStatus::Failed));
        assert_eq!(
            gamma_cancelled.fields.title.as_deref(),
            Some(mcp_startup_tool_call_title("gamma").as_str())
        );
        assert_eq!(
            gamma_cancelled.fields.raw_output.as_ref(),
            Some(&gamma_cancelled_raw)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_notice_events_are_surfaced_during_replay_history() -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::EventMsg(EventMsg::ThreadNameUpdated(
                    codex_protocol::protocol::ThreadNameUpdatedEvent {
                        thread_id: codex_protocol::ThreadId::new(),
                        thread_name: Some("Replay title".into()),
                    },
                )),
                RolloutItem::EventMsg(EventMsg::PlanUpdate(UpdatePlanArgs {
                    explanation: Some("Replay plan".into()),
                    plan: vec![
                        PlanItemArg {
                            step: "first step".into(),
                            status: StepStatus::InProgress,
                        },
                        PlanItemArg {
                            step: "done step".into(),
                            status: StepStatus::Completed,
                        },
                    ],
                })),
                RolloutItem::EventMsg(EventMsg::Warning(WarningEvent {
                    message: "Persisted warning".into(),
                })),
                RolloutItem::EventMsg(EventMsg::ExitedReviewMode(ExitedReviewModeEvent {
                    review_output: Some(ReviewOutputEvent {
                        findings: vec![],
                        overall_correctness: String::new(),
                        overall_explanation: "Replay review output".into(),
                        overall_confidence_score: 0.75,
                    }),
                })),
                RolloutItem::EventMsg(EventMsg::UndoStarted(
                    codex_protocol::protocol::UndoStartedEvent { message: None },
                )),
                RolloutItem::EventMsg(EventMsg::UndoCompleted(
                    codex_protocol::protocol::UndoCompletedEvent {
                        success: false,
                        message: None,
                    },
                )),
                RolloutItem::EventMsg(EventMsg::ContextCompacted(
                    codex_protocol::protocol::ContextCompactedEvent,
                )),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
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
            7,
            "unexpected notifications: {notifications:?}"
        );

        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::SessionInfoUpdate(update)
                if update.title.as_opt_deref() == Some(Some("Replay title"))
        ));
        assert!(matches!(
            &notifications[1].update,
            SessionUpdate::Plan(plan)
                if plan.entries
                    == vec![
                        PlanEntry::new(
                            "first step",
                            PlanEntryPriority::Medium,
                            PlanEntryStatus::InProgress,
                        ),
                        PlanEntry::new(
                            "done step",
                            PlanEntryPriority::Medium,
                            PlanEntryStatus::Completed,
                        ),
                    ]
        ));

        assert_eq!(
            message_chunk_text(&notifications[2].update),
            "Persisted warning"
        );
        assert_eq!(
            message_chunk_text(&notifications[3].update),
            "Replay review output"
        );
        assert_eq!(
            message_chunk_text(&notifications[4].update),
            "Undo in progress..."
        );
        assert_eq!(message_chunk_text(&notifications[5].update), "Undo failed.");
        assert_eq!(
            message_chunk_text(&notifications[6].update),
            "Context compacted"
        );

        let content_message_ids: Vec<_> = notifications[2..]
            .iter()
            .map(|notification| message_chunk_id(&notification.update).to_string())
            .collect();
        for message_id in &content_message_ids {
            assert_uuid_message_id(message_id);
        }
        assert_eq!(
            content_message_ids.len(),
            content_message_ids
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            "expected unique message ids for replayed notices: {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_message_id_grouping() -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
                    message: "replayed user".into(),
                    images: None,
                    text_elements: vec![],
                    local_images: vec![],
                })),
                RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                    message: "assistant one".into(),
                    phase: None,
                })),
                RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                    message: "assistant two".into(),
                    phase: None,
                })),
                RolloutItem::EventMsg(EventMsg::AgentReasoning(AgentReasoningEvent {
                    text: "reasoning one".into(),
                })),
                RolloutItem::EventMsg(EventMsg::AgentReasoningRawContent(
                    AgentReasoningRawContentEvent {
                        text: "reasoning two".into(),
                    },
                )),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
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
            5,
            "unexpected notifications: {notifications:?}"
        );

        let user_id = message_chunk_id(&notifications[0].update);
        let first_assistant_id = message_chunk_id(&notifications[1].update);
        let second_assistant_id = message_chunk_id(&notifications[2].update);
        let first_reasoning_id = message_chunk_id(&notifications[3].update);
        let second_reasoning_id = message_chunk_id(&notifications[4].update);

        for message_id in [
            user_id,
            first_assistant_id,
            second_assistant_id,
            first_reasoning_id,
            second_reasoning_id,
        ] {
            assert_uuid_message_id(message_id);
        }

        assert_ne!(user_id, first_assistant_id);
        assert_ne!(first_assistant_id, second_assistant_id);
        assert_ne!(first_reasoning_id, second_reasoning_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_model_reroute_and_notices_preserve_existing_message_grouping()
    -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
                    message: "reroute-replay".into(),
                    images: None,
                    text_elements: vec![],
                    local_images: vec![],
                })),
                RolloutItem::EventMsg(EventMsg::AgentReasoning(AgentReasoningEvent {
                    text: "thinking ".into(),
                })),
                RolloutItem::EventMsg(EventMsg::ModelReroute(ModelRerouteEvent {
                    from_model: "gpt-5.3-codex".into(),
                    to_model: "gpt-5.2".into(),
                    reason: ModelRerouteReason::HighRiskCyberActivity,
                })),
                RolloutItem::EventMsg(EventMsg::Warning(WarningEvent {
                    message: "Fallback warning".into(),
                })),
                RolloutItem::EventMsg(EventMsg::AgentReasoningRawContent(
                    AgentReasoningRawContentEvent {
                        text: "deeply".into(),
                    },
                )),
                RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                    message: "Hello".into(),
                    phase: None,
                })),
                RolloutItem::EventMsg(EventMsg::UndoStarted(
                    codex_protocol::protocol::UndoStartedEvent { message: None },
                )),
                RolloutItem::EventMsg(EventMsg::ContextCompacted(
                    codex_protocol::protocol::ContextCompactedEvent,
                )),
                RolloutItem::EventMsg(EventMsg::UndoCompleted(
                    codex_protocol::protocol::UndoCompletedEvent {
                        success: true,
                        message: None,
                    },
                )),
                RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                    message: " world".into(),
                    phase: None,
                })),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
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
            10,
            "unexpected notifications: {notifications:?}"
        );

        assert_eq!(
            message_chunk_text(&notifications[0].update),
            "reroute-replay"
        );
        assert_eq!(message_chunk_text(&notifications[1].update), "thinking ");
        assert_eq!(
            message_chunk_text(&notifications[2].update),
            "Model rerouted from gpt-5.3-codex to gpt-5.2 due to high-risk cyber activity."
        );
        assert_eq!(
            message_chunk_text(&notifications[3].update),
            "Fallback warning"
        );
        assert_eq!(message_chunk_text(&notifications[4].update), "deeply");
        assert_eq!(message_chunk_text(&notifications[5].update), "Hello");
        assert_eq!(
            message_chunk_text(&notifications[6].update),
            "Undo in progress..."
        );
        assert_eq!(
            message_chunk_text(&notifications[7].update),
            "Context compacted"
        );
        assert_eq!(
            message_chunk_text(&notifications[8].update),
            "Undo completed."
        );
        assert_eq!(message_chunk_text(&notifications[9].update), " world");

        let user_id = message_chunk_id(&notifications[0].update).to_string();
        let reasoning_id = message_chunk_id(&notifications[1].update).to_string();
        let reroute_id = message_chunk_id(&notifications[2].update).to_string();
        let warning_id = message_chunk_id(&notifications[3].update).to_string();
        let reasoning_follow_up_id = message_chunk_id(&notifications[4].update).to_string();
        let assistant_id = message_chunk_id(&notifications[5].update).to_string();
        let undo_started_id = message_chunk_id(&notifications[6].update).to_string();
        let compacted_id = message_chunk_id(&notifications[7].update).to_string();
        let undo_completed_id = message_chunk_id(&notifications[8].update).to_string();
        let assistant_follow_up_id = message_chunk_id(&notifications[9].update).to_string();

        let all_ids = [
            user_id.clone(),
            reasoning_id.clone(),
            reroute_id.clone(),
            warning_id.clone(),
            reasoning_follow_up_id.clone(),
            assistant_id.clone(),
            undo_started_id.clone(),
            compacted_id.clone(),
            undo_completed_id.clone(),
            assistant_follow_up_id.clone(),
        ];
        for message_id in &all_ids {
            assert_uuid_message_id(message_id);
        }

        assert_eq!(
            all_ids.len(),
            all_ids
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            "expected unique replay message ids with notices interleaved: {notifications:?}"
        );

        assert_ne!(user_id, reasoning_id);
        assert_ne!(reasoning_id, reasoning_follow_up_id);
        assert_ne!(assistant_id, assistant_follow_up_id);
        assert_ne!(reroute_id, reasoning_id);
        assert_ne!(reroute_id, assistant_id);
        assert_ne!(warning_id, reasoning_id);
        assert_ne!(warning_id, assistant_id);
        assert_ne!(undo_started_id, assistant_id);
        assert_ne!(compacted_id, assistant_id);
        assert_ne!(undo_completed_id, assistant_follow_up_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_tool_lifecycle_function_and_custom_calls_start_in_progress()
    -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        let function_output =
            codex_protocol::models::FunctionCallOutputPayload::from_text("ok".to_string());
        let custom_output =
            codex_protocol::models::FunctionCallOutputPayload::from_text("done".to_string());

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::ResponseItem(ResponseItem::FunctionCall {
                    id: None,
                    name: "do_thing".into(),
                    arguments: r#"{"value":1}"#.into(),
                    call_id: "function-call-id".into(),
                }),
                RolloutItem::ResponseItem(ResponseItem::FunctionCallOutput {
                    call_id: "function-call-id".into(),
                    output: function_output.clone(),
                }),
                RolloutItem::ResponseItem(ResponseItem::CustomToolCall {
                    id: None,
                    status: Some("completed".into()),
                    call_id: "custom-tool-call-id".into(),
                    name: "custom_tool".into(),
                    input: r#"{"task":"patch"}"#.into(),
                }),
                RolloutItem::ResponseItem(ResponseItem::CustomToolCallOutput {
                    call_id: "custom-tool-call-id".into(),
                    output: custom_output.clone(),
                }),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        let tool_calls = tool_calls(&notifications);
        let tool_call_updates = tool_call_updates(&notifications);

        assert_eq!(
            tool_calls.len(),
            2,
            "unexpected ToolCall notifications: {tool_calls:?}"
        );
        assert_eq!(
            tool_call_updates.len(),
            2,
            "unexpected ToolCallUpdate notifications: {tool_call_updates:?}"
        );

        let function_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("function-call-id"))
            .unwrap_or_else(|| panic!("missing function tool call: {tool_calls:?}"));
        assert_eq!(function_call.title, "do_thing");
        assert_eq!(function_call.status, ToolCallStatus::InProgress);
        assert_eq!(
            function_call.raw_input.as_ref(),
            Some(&serde_json::json!({ "value": 1 }))
        );

        let function_update = tool_call_updates
            .iter()
            .find(|update| update.tool_call_id == ToolCallId::new("function-call-id"))
            .unwrap_or_else(|| panic!("missing function tool call update: {tool_call_updates:?}"));
        assert_eq!(
            function_update.fields.status,
            Some(ToolCallStatus::Completed)
        );
        assert_eq!(
            function_update.fields.raw_output.as_ref(),
            Some(&serde_json::to_value(&function_output)?)
        );

        let custom_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("custom-tool-call-id"))
            .unwrap_or_else(|| panic!("missing custom tool call: {tool_calls:?}"));
        assert_eq!(custom_call.title, "custom_tool");
        assert_eq!(custom_call.status, ToolCallStatus::InProgress);
        assert_eq!(
            custom_call.raw_input.as_ref(),
            Some(&serde_json::json!({ "task": "patch" }))
        );

        let custom_update = tool_call_updates
            .iter()
            .find(|update| update.tool_call_id == ToolCallId::new("custom-tool-call-id"))
            .unwrap_or_else(|| panic!("missing custom tool call update: {tool_call_updates:?}"));
        assert_eq!(custom_update.fields.status, Some(ToolCallStatus::Completed));
        assert_eq!(
            custom_update.fields.raw_output.as_ref(),
            Some(&serde_json::json!(custom_output))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_tool_lifecycle_shell_and_apply_patch_calls_start_in_progress()
    -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        let shell_output =
            codex_protocol::models::FunctionCallOutputPayload::from_text("shell ok".to_string());
        let patch_output =
            codex_protocol::models::FunctionCallOutputPayload::from_text("patch ok".to_string());
        let shell_arguments = serde_json::json!({
            "command": ["echo", "hello"],
        })
        .to_string();
        let apply_patch_input = [
            "*** Begin Patch",
            "*** Add File: hello.txt",
            "+hello",
            "*** End Patch",
        ]
        .join("\n");

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::ResponseItem(ResponseItem::FunctionCall {
                    id: None,
                    name: "shell".into(),
                    arguments: shell_arguments,
                    call_id: "shell-call-id".into(),
                }),
                RolloutItem::ResponseItem(ResponseItem::FunctionCallOutput {
                    call_id: "shell-call-id".into(),
                    output: shell_output.clone(),
                }),
                RolloutItem::ResponseItem(ResponseItem::CustomToolCall {
                    id: None,
                    status: Some("completed".into()),
                    call_id: "apply-patch-call-id".into(),
                    name: "apply_patch".into(),
                    input: apply_patch_input,
                }),
                RolloutItem::ResponseItem(ResponseItem::CustomToolCallOutput {
                    call_id: "apply-patch-call-id".into(),
                    output: patch_output.clone(),
                }),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        let tool_calls = tool_calls(&notifications);
        let tool_call_updates = tool_call_updates(&notifications);

        assert_eq!(
            tool_calls.len(),
            2,
            "unexpected ToolCall notifications: {tool_calls:?}"
        );
        assert_eq!(
            tool_call_updates.len(),
            2,
            "unexpected ToolCallUpdate notifications: {tool_call_updates:?}"
        );

        let shell_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("shell-call-id"))
            .unwrap_or_else(|| panic!("missing shell tool call: {tool_calls:?}"));
        assert_eq!(shell_call.status, ToolCallStatus::InProgress);

        let shell_update = tool_call_updates
            .iter()
            .find(|update| update.tool_call_id == ToolCallId::new("shell-call-id"))
            .unwrap_or_else(|| panic!("missing shell tool call update: {tool_call_updates:?}"));
        assert_eq!(shell_update.fields.status, Some(ToolCallStatus::Completed));
        assert_eq!(
            shell_update.fields.raw_output.as_ref(),
            Some(&serde_json::to_value(&shell_output)?)
        );

        let apply_patch_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("apply-patch-call-id"))
            .unwrap_or_else(|| panic!("missing apply_patch tool call: {tool_calls:?}"));
        assert_eq!(apply_patch_call.title, "Edit hello.txt");
        assert_eq!(apply_patch_call.status, ToolCallStatus::InProgress);

        let apply_patch_update = tool_call_updates
            .iter()
            .find(|update| update.tool_call_id == ToolCallId::new("apply-patch-call-id"))
            .unwrap_or_else(|| {
                panic!("missing apply_patch tool call update: {tool_call_updates:?}")
            });
        assert_eq!(
            apply_patch_update.fields.status,
            Some(ToolCallStatus::Completed)
        );
        assert_eq!(
            apply_patch_update.fields.raw_output.as_ref(),
            Some(&serde_json::json!(patch_output))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_tool_lifecycle_local_shell_in_progress_is_preserved() -> anyhow::Result<()>
    {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        let shell_output =
            codex_protocol::models::FunctionCallOutputPayload::from_text("hello".to_string());

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::ResponseItem(ResponseItem::LocalShellCall {
                    id: None,
                    call_id: Some("local-shell-call-id".into()),
                    status: codex_protocol::models::LocalShellStatus::InProgress,
                    action: codex_protocol::models::LocalShellAction::Exec(
                        codex_protocol::models::LocalShellExecAction {
                            command: vec!["echo".into(), "hello".into()],
                            timeout_ms: None,
                            working_directory: None,
                            env: None,
                            user: None,
                        },
                    ),
                }),
                RolloutItem::ResponseItem(ResponseItem::FunctionCallOutput {
                    call_id: "local-shell-call-id".into(),
                    output: shell_output.clone(),
                }),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        let tool_calls = tool_calls(&notifications);
        let tool_call_updates = tool_call_updates(&notifications);

        assert_eq!(
            tool_calls.len(),
            1,
            "unexpected ToolCall notifications: {tool_calls:?}"
        );
        assert_eq!(
            tool_call_updates.len(),
            1,
            "unexpected ToolCallUpdate notifications: {tool_call_updates:?}"
        );

        assert_eq!(tool_calls[0].status, ToolCallStatus::InProgress);
        assert_eq!(
            tool_call_updates[0].fields.status,
            Some(ToolCallStatus::Completed)
        );
        assert_eq!(
            tool_call_updates[0].fields.raw_output.as_ref(),
            Some(&serde_json::to_value(&shell_output)?)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_tool_lifecycle_web_search_calls_preserve_truthful_status_kind_and_input()
    -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        let search_action = WebSearchAction::Search {
            query: Some("weather seattle".into()),
            queries: Some(vec!["weather seattle".into(), "seattle weather now".into()]),
        };
        let open_page_action = WebSearchAction::OpenPage {
            url: Some("https://example.com".into()),
        };

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::ResponseItem(ResponseItem::WebSearchCall {
                    id: Some("ws_search".into()),
                    status: Some("open".into()),
                    action: Some(search_action.clone()),
                }),
                RolloutItem::ResponseItem(ResponseItem::WebSearchCall {
                    id: Some("ws_open".into()),
                    status: Some("completed".into()),
                    action: Some(open_page_action.clone()),
                }),
                RolloutItem::ResponseItem(ResponseItem::WebSearchCall {
                    id: Some("ws_partial".into()),
                    status: Some("in_progress".into()),
                    action: None,
                }),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        let tool_calls = tool_calls(&notifications);
        let tool_call_updates = tool_call_updates(&notifications);

        assert_eq!(
            tool_calls.len(),
            3,
            "unexpected ToolCall notifications: {tool_calls:?}"
        );
        assert!(
            tool_call_updates.is_empty(),
            "unexpected ToolCallUpdate notifications: {tool_call_updates:?}"
        );

        let search_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("ws_search"))
            .unwrap_or_else(|| panic!("missing web search tool call: {tool_calls:?}"));
        assert_eq!(search_call.kind, ToolKind::Fetch);
        assert_eq!(search_call.status, ToolCallStatus::InProgress);
        assert_eq!(
            search_call.title,
            "Searching for: weather seattle, seattle weather now"
        );
        assert_eq!(
            search_call.raw_input.as_ref(),
            Some(&serde_json::json!({
                "query": "weather seattle",
                "action": search_action.clone(),
            }))
        );

        let open_page_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("ws_open"))
            .unwrap_or_else(|| panic!("missing open-page tool call: {tool_calls:?}"));
        assert_eq!(open_page_call.kind, ToolKind::Fetch);
        assert_eq!(open_page_call.status, ToolCallStatus::Completed);
        assert_eq!(open_page_call.title, "Opening: https://example.com");
        assert_eq!(
            open_page_call.raw_input.as_ref(),
            Some(&serde_json::json!({
                "action": open_page_action.clone(),
            }))
        );

        let partial_call = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == ToolCallId::new("ws_partial"))
            .unwrap_or_else(|| panic!("missing partial web search tool call: {tool_calls:?}"));
        assert_eq!(partial_call.kind, ToolKind::Fetch);
        assert_eq!(partial_call.status, ToolCallStatus::InProgress);
        assert_eq!(partial_call.title, "Searching the Web");
        assert_eq!(partial_call.raw_input.as_ref(), None);

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_mcp_startup_events_preserve_existing_message_grouping()
    -> anyhow::Result<()> {
        let (_, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (replay_response_tx, replay_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::ReplayHistory {
            history: vec![
                RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
                    message: "mcp-startup-replay".into(),
                    images: None,
                    text_elements: vec![],
                    local_images: vec![],
                })),
                RolloutItem::EventMsg(EventMsg::AgentReasoning(AgentReasoningEvent {
                    text: "thinking ".into(),
                })),
                RolloutItem::EventMsg(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                    server: "alpha".into(),
                    status: McpStartupStatus::Starting,
                })),
                RolloutItem::EventMsg(EventMsg::AgentReasoningRawContent(
                    AgentReasoningRawContentEvent {
                        text: "deeply".into(),
                    },
                )),
                RolloutItem::EventMsg(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                    server: "beta".into(),
                    status: McpStartupStatus::Starting,
                })),
                RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                    message: "Hello".into(),
                    phase: None,
                })),
                RolloutItem::EventMsg(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                    server: "gamma".into(),
                    status: McpStartupStatus::Starting,
                })),
                RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                    message: " world".into(),
                    phase: None,
                })),
                RolloutItem::EventMsg(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                    server: "alpha".into(),
                    status: McpStartupStatus::Ready,
                })),
                RolloutItem::EventMsg(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                    server: "beta".into(),
                    status: McpStartupStatus::Failed {
                        error: "auth failed".into(),
                    },
                })),
                RolloutItem::EventMsg(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                    server: "gamma".into(),
                    status: McpStartupStatus::Cancelled,
                })),
                RolloutItem::EventMsg(EventMsg::McpStartupComplete(McpStartupCompleteEvent {
                    ready: vec!["alpha".into()],
                    failed: vec![codex_protocol::protocol::McpStartupFailure {
                        server: "beta".into(),
                        error: "auth failed".into(),
                    }],
                    cancelled: vec!["gamma".into()],
                })),
            ],
            response_tx: replay_response_tx,
        })?;

        tokio::try_join!(
            async {
                replay_response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        let content_notifications: Vec<_> = notifications
            .iter()
            .filter(|notification| is_content_update(&notification.update))
            .collect();
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|notification| match &notification.update {
                SessionUpdate::ToolCall(tool_call) => Some(tool_call.clone()),
                _ => None,
            })
            .collect();
        let tool_call_updates: Vec<_> = notifications
            .iter()
            .filter_map(|notification| match &notification.update {
                SessionUpdate::ToolCallUpdate(update) => Some(update.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            notifications.len(),
            12,
            "unexpected notifications: {notifications:?}"
        );
        assert_eq!(
            content_notifications.len(),
            5,
            "unexpected content notifications: {notifications:?}"
        );
        assert_eq!(
            content_notifications
                .iter()
                .map(|notification| message_chunk_text(&notification.update))
                .collect::<Vec<_>>(),
            vec![
                "mcp-startup-replay",
                "thinking ",
                "deeply",
                "Hello",
                " world"
            ]
        );

        let replay_user_id = message_chunk_id(&content_notifications[0].update);
        let reasoning_id = message_chunk_id(&content_notifications[1].update);
        let reasoning_follow_up_id = message_chunk_id(&content_notifications[2].update);
        let assistant_id = message_chunk_id(&content_notifications[3].update);
        let assistant_follow_up_id = message_chunk_id(&content_notifications[4].update);

        for message_id in [
            replay_user_id,
            reasoning_id,
            reasoning_follow_up_id,
            assistant_id,
            assistant_follow_up_id,
        ] {
            assert_uuid_message_id(message_id);
        }

        assert_ne!(replay_user_id, reasoning_id);
        assert_ne!(replay_user_id, reasoning_follow_up_id);
        assert_ne!(replay_user_id, assistant_id);
        assert_ne!(replay_user_id, assistant_follow_up_id);
        assert_ne!(reasoning_id, reasoning_follow_up_id);
        assert_ne!(reasoning_id, assistant_id);
        assert_ne!(reasoning_follow_up_id, assistant_id);
        assert_ne!(assistant_id, assistant_follow_up_id);

        assert_eq!(
            tool_calls.len(),
            4,
            "unexpected ToolCall notifications: {tool_calls:?}"
        );
        assert_eq!(
            tool_call_updates.len(),
            3,
            "unexpected ToolCallUpdate notifications: {tool_call_updates:?}"
        );

        let alpha_start_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "alpha".into(),
            status: McpStartupStatus::Starting,
        })?;
        let beta_start_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "beta".into(),
            status: McpStartupStatus::Starting,
        })?;
        let gamma_start_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "gamma".into(),
            status: McpStartupStatus::Starting,
        })?;
        let alpha_ready_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "alpha".into(),
            status: McpStartupStatus::Ready,
        })?;
        let beta_failed_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "beta".into(),
            status: McpStartupStatus::Failed {
                error: "auth failed".into(),
            },
        })?;
        let gamma_cancelled_raw = serde_json::to_value(&McpStartupUpdateEvent {
            server: "gamma".into(),
            status: McpStartupStatus::Cancelled,
        })?;
        let summary_raw = serde_json::to_value(&McpStartupCompleteEvent {
            ready: vec!["alpha".into()],
            failed: vec![codex_protocol::protocol::McpStartupFailure {
                server: "beta".into(),
                error: "auth failed".into(),
            }],
            cancelled: vec!["gamma".into()],
        })?;

        let alpha_start_id = ToolCallId::new(mcp_startup_tool_call_id("alpha"));
        let beta_start_id = ToolCallId::new(mcp_startup_tool_call_id("beta"));
        let gamma_start_id = ToolCallId::new(mcp_startup_tool_call_id("gamma"));
        let summary_id = ToolCallId::new(mcp_startup_summary_tool_call_id());

        let alpha_start = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == alpha_start_id)
            .unwrap_or_else(|| panic!("missing alpha startup tool call: {tool_calls:?}"));
        assert_eq!(alpha_start.title, mcp_startup_tool_call_title("alpha"));
        assert_eq!(alpha_start.status, ToolCallStatus::InProgress);
        assert_eq!(alpha_start.raw_input.as_ref(), Some(&alpha_start_raw));

        let beta_start = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == beta_start_id)
            .unwrap_or_else(|| panic!("missing beta startup tool call: {tool_calls:?}"));
        assert_eq!(beta_start.title, mcp_startup_tool_call_title("beta"));
        assert_eq!(beta_start.status, ToolCallStatus::InProgress);
        assert_eq!(beta_start.raw_input.as_ref(), Some(&beta_start_raw));

        let gamma_start = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == gamma_start_id)
            .unwrap_or_else(|| panic!("missing gamma startup tool call: {tool_calls:?}"));
        assert_eq!(gamma_start.title, mcp_startup_tool_call_title("gamma"));
        assert_eq!(gamma_start.status, ToolCallStatus::InProgress);
        assert_eq!(gamma_start.raw_input.as_ref(), Some(&gamma_start_raw));

        let summary = tool_calls
            .iter()
            .find(|tool_call| tool_call.tool_call_id == summary_id)
            .unwrap_or_else(|| panic!("missing startup summary tool call: {tool_calls:?}"));
        assert_eq!(summary.title, "MCP Startup");
        assert_eq!(summary.status, ToolCallStatus::Failed);
        assert_eq!(summary.raw_output.as_ref(), Some(&summary_raw));

        let alpha_ready = tool_call_updates
            .iter()
            .find(|update| {
                update.tool_call_id == ToolCallId::new(mcp_startup_tool_call_id("alpha"))
            })
            .unwrap_or_else(|| panic!("missing alpha startup update: {tool_call_updates:?}"));
        assert_eq!(alpha_ready.fields.status, Some(ToolCallStatus::Completed));
        assert_eq!(
            alpha_ready.fields.title.as_deref(),
            Some(mcp_startup_tool_call_title("alpha").as_str())
        );
        assert_eq!(
            alpha_ready.fields.raw_output.as_ref(),
            Some(&alpha_ready_raw)
        );

        let beta_failed = tool_call_updates
            .iter()
            .find(|update| update.tool_call_id == ToolCallId::new(mcp_startup_tool_call_id("beta")))
            .unwrap_or_else(|| panic!("missing beta startup update: {tool_call_updates:?}"));
        assert_eq!(beta_failed.fields.status, Some(ToolCallStatus::Failed));
        assert_eq!(
            beta_failed.fields.title.as_deref(),
            Some(mcp_startup_tool_call_title("beta").as_str())
        );
        assert_eq!(
            beta_failed.fields.raw_output.as_ref(),
            Some(&beta_failed_raw)
        );

        let gamma_cancelled = tool_call_updates
            .iter()
            .find(|update| {
                update.tool_call_id == ToolCallId::new(mcp_startup_tool_call_id("gamma"))
            })
            .unwrap_or_else(|| panic!("missing gamma startup update: {tool_call_updates:?}"));
        assert_eq!(gamma_cancelled.fields.status, Some(ToolCallStatus::Failed));
        assert_eq!(
            gamma_cancelled.fields.title.as_deref(),
            Some(mcp_startup_tool_call_title("gamma").as_str())
        );
        assert_eq!(
            gamma_cancelled.fields.raw_output.as_ref(),
            Some(&gamma_cancelled_raw)
        );

        Ok(())
    }

    async fn setup(
        custom_prompts: Vec<CustomPrompt>,
    ) -> anyhow::Result<(
        SessionId,
        Arc<StubClient>,
        Arc<StubCodexThread>,
        UnboundedSender<ThreadMessage>,
        LocalSet,
    )> {
        let session_id = SessionId::new("test");
        let client = Arc::new(StubClient::new());
        let session_client =
            SessionClient::with_client(session_id.clone(), client.clone(), Arc::default());
        let conversation = Arc::new(StubCodexThread::new());
        let models_manager = Arc::new(StubModelsManager);
        let config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut actor = ThreadActor::new(
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
        fn logout(&self) -> Result<bool, Box<Error>> {
            Ok(true)
        }
    }

    struct StubModelsManager;

    #[async_trait::async_trait]
    impl ModelsManagerImpl for StubModelsManager {
        async fn get_model(&self, _model_id: &Option<String>) -> String {
            all_model_presets()[0].to_owned().id
        }

        async fn list_models(&self) -> Vec<ModelPreset> {
            all_model_presets().to_owned()
        }
    }

    struct StubCodexThread {
        current_id: AtomicUsize,
        ops: std::sync::Mutex<Vec<Op>>,
        op_tx: mpsc::UnboundedSender<Event>,
        op_rx: Mutex<mpsc::UnboundedReceiver<Event>>,
    }

    impl StubCodexThread {
        fn new() -> Self {
            let (op_tx, op_rx) = mpsc::unbounded_channel();
            StubCodexThread {
                current_id: AtomicUsize::new(0),
                ops: std::sync::Mutex::default(),
                op_tx,
                op_rx: Mutex::new(op_rx),
            }
        }
    }

    #[async_trait::async_trait]
    impl CodexThreadImpl for StubCodexThread {
        async fn submit(&self, op: Op) -> Result<String, CodexErr> {
            let id = self
                .current_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            self.ops.lock().unwrap().push(op.clone());

            match op {
                Op::UserInput { items, .. } => {
                    let prompt = items
                        .into_iter()
                        .map(|i| match i {
                            UserInput::Text { text, .. } => text,
                            _ => unimplemented!(),
                        })
                        .join("\n");

                    if prompt == "parallel-exec" {
                        // Emit interleaved exec events: Begin A, Begin B, End A, End B
                        let turn_id = id.to_string();
                        let cwd = std::env::current_dir().unwrap();
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                            call_id: "call-a".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "a".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo a".into(),
                            }],
                            source: Default::default(),
                            interaction_input: None,
                        }));
                        send(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                            call_id: "call-b".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "b".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo b".into(),
                            }],
                            source: Default::default(),
                            interaction_input: None,
                        }));
                        send(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                            call_id: "call-a".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "a".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![],
                            source: Default::default(),
                            interaction_input: None,
                            stdout: "a\n".into(),
                            stderr: String::new(),
                            aggregated_output: "a\n".into(),
                            exit_code: 0,
                            duration: std::time::Duration::from_millis(10),
                            formatted_output: "a\n".into(),
                            status: ExecCommandStatus::Completed,
                        }));
                        send(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                            call_id: "call-b".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "b".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![],
                            source: Default::default(),
                            interaction_input: None,
                            stdout: "b\n".into(),
                            stderr: String::new(),
                            aggregated_output: "b\n".into(),
                            exit_code: 0,
                            duration: std::time::Duration::from_millis(10),
                            formatted_output: "b\n".into(),
                            status: ExecCommandStatus::Completed,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id,
                        }));
                    } else if prompt == "grouped-live" {
                        let turn_id = id.to_string();
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };

                        send(EventMsg::UserMessage(UserMessageEvent {
                            message: prompt.clone(),
                            images: None,
                            text_elements: vec![],
                            local_images: vec![],
                        }));
                        send(EventMsg::ReasoningContentDelta(
                            ReasoningContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-1".into(),
                                delta: "thinking ".into(),
                                summary_index: 0,
                            },
                        ));
                        send(EventMsg::ReasoningRawContentDelta(
                            ReasoningRawContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-1".into(),
                                delta: "deeply".into(),
                                content_index: 1,
                            },
                        ));
                        send(EventMsg::AgentReasoningSectionBreak(
                            AgentReasoningSectionBreakEvent {
                                item_id: "reasoning-1".into(),
                                summary_index: 1,
                            },
                        ));
                        send(EventMsg::ReasoningContentDelta(
                            ReasoningContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-2".into(),
                                delta: "next thought".into(),
                                summary_index: 2,
                            },
                        ));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-1".into(),
                                delta: "Hello".into(),
                            },
                        ));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-1".into(),
                                delta: " world".into(),
                            },
                        ));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-2".into(),
                                delta: "Second reply".into(),
                            },
                        ));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id,
                        }));
                    } else if prompt == "reroute-live" {
                        let turn_id = id.to_string();
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };

                        send(EventMsg::UserMessage(UserMessageEvent {
                            message: prompt.clone(),
                            images: None,
                            text_elements: vec![],
                            local_images: vec![],
                        }));
                        send(EventMsg::ReasoningContentDelta(
                            ReasoningContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-reroute".into(),
                                delta: "thinking ".into(),
                                summary_index: 0,
                            },
                        ));
                        send(EventMsg::ModelReroute(ModelRerouteEvent {
                            from_model: "gpt-5.3-codex".into(),
                            to_model: "gpt-5.2".into(),
                            reason: ModelRerouteReason::HighRiskCyberActivity,
                        }));
                        send(EventMsg::ReasoningRawContentDelta(
                            ReasoningRawContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-reroute".into(),
                                delta: "deeply".into(),
                                content_index: 1,
                            },
                        ));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-reroute".into(),
                                delta: "Hello".into(),
                            },
                        ));
                        send(EventMsg::Warning(WarningEvent {
                            message: "Fallback warning".into(),
                        }));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-reroute".into(),
                                delta: " world".into(),
                            },
                        ));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id,
                        }));
                    } else if prompt == "mcp-startup-live" {
                        let turn_id = id.to_string();
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };

                        send(EventMsg::UserMessage(UserMessageEvent {
                            message: prompt.clone(),
                            images: None,
                            text_elements: vec![],
                            local_images: vec![],
                        }));
                        send(EventMsg::ReasoningContentDelta(
                            ReasoningContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-mcp".into(),
                                delta: "thinking ".into(),
                                summary_index: 0,
                            },
                        ));
                        send(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                            server: "alpha".into(),
                            status: McpStartupStatus::Starting,
                        }));
                        send(EventMsg::ReasoningRawContentDelta(
                            ReasoningRawContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "reasoning-mcp".into(),
                                delta: "deeply".into(),
                                content_index: 1,
                            },
                        ));
                        send(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                            server: "beta".into(),
                            status: McpStartupStatus::Starting,
                        }));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-mcp".into(),
                                delta: "Hello".into(),
                            },
                        ));
                        send(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                            server: "gamma".into(),
                            status: McpStartupStatus::Starting,
                        }));
                        send(EventMsg::AgentMessageContentDelta(
                            AgentMessageContentDeltaEvent {
                                thread_id: id.to_string(),
                                turn_id: turn_id.clone(),
                                item_id: "assistant-mcp".into(),
                                delta: " world".into(),
                            },
                        ));
                        send(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                            server: "alpha".into(),
                            status: McpStartupStatus::Ready,
                        }));
                        send(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                            server: "beta".into(),
                            status: McpStartupStatus::Failed {
                                error: "auth failed".into(),
                            },
                        }));
                        send(EventMsg::McpStartupUpdate(McpStartupUpdateEvent {
                            server: "gamma".into(),
                            status: McpStartupStatus::Cancelled,
                        }));
                        send(EventMsg::McpStartupComplete(McpStartupCompleteEvent {
                            ready: vec!["alpha".into()],
                            failed: vec![codex_protocol::protocol::McpStartupFailure {
                                server: "beta".into(),
                                error: "auth failed".into(),
                            }],
                            cancelled: vec!["gamma".into()],
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id,
                        }));
                    } else {
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
                                msg: EventMsg::AgentMessage(AgentMessageEvent {
                                    message: prompt,
                                    phase: None,
                                }),
                            })
                            .unwrap();
                        self.op_tx
                            .send(Event {
                                id: id.to_string(),
                                msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                    last_agent_message: None,
                                    turn_id: id.to_string(),
                                }),
                            })
                            .unwrap();
                    }
                }
                Op::Compact => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnStarted(TurnStartedEvent {
                                model_context_window: None,
                                collaboration_mode_kind: ModeKind::default(),
                                turn_id: id.to_string(),
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::AgentMessage(AgentMessageEvent {
                                message: "Compact task completed".to_string(),
                                phase: None,
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
                            }),
                        })
                        .unwrap();
                }
                Op::Undo => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::UndoStarted(
                                codex_protocol::protocol::UndoStartedEvent {
                                    message: Some("Undo in progress...".to_string()),
                                },
                            ),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::UndoCompleted(
                                codex_protocol::protocol::UndoCompletedEvent {
                                    success: true,
                                    message: Some("Undo completed.".to_string()),
                                },
                            ),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
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
                            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
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

    #[tokio::test]
    async fn test_parallel_exec_commands() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["parallel-exec".into()]),
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

        // Collect all ToolCall (begin) notifications keyed by their tool_call_id prefix.
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();

        // Collect all ToolCallUpdate notifications that carry a terminal status.
        let completed_updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(update) => {
                    if update.fields.status == Some(ToolCallStatus::Completed) {
                        Some(update.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        // Both commands A and B should have produced a ToolCall (begin).
        assert_eq!(
            tool_calls.len(),
            2,
            "expected 2 ToolCall begin notifications, got {tool_calls:?}"
        );

        // Both commands A and B should have produced a completed ToolCallUpdate.
        assert_eq!(
            completed_updates.len(),
            2,
            "expected 2 completed ToolCallUpdate notifications, got {completed_updates:?}"
        );

        // The completed updates should reference the same tool_call_ids as the begins.
        let begin_ids: std::collections::HashSet<_> = tool_calls
            .iter()
            .map(|tc| tc.tool_call_id.clone())
            .collect();
        let end_ids: std::collections::HashSet<_> = completed_updates
            .iter()
            .map(|u| u.tool_call_id.clone())
            .collect();
        assert_eq!(
            begin_ids, end_ids,
            "completed update tool_call_ids should match begin tool_call_ids"
        );

        Ok(())
    }
}
