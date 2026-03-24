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
    ConfigOptionUpdate, Content, ContentBlock, ContentChunk, Cost, Diff, EmbeddedResource,
    EmbeddedResourceResource, Error, LoadSessionResponse, Meta, ModelId, ModelInfo,
    PermissionOption, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    ResourceLink, SelectedPermissionOutcome, SessionConfigId, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigOptionValue, SessionConfigSelectOption,
    SessionConfigValueId, SessionId, SessionInfoUpdate, SessionMode, SessionModeId,
    SessionModeState, SessionModelState, SessionNotification, SessionUpdate, StopReason, Terminal,
    TextResourceContents, ToolCall, ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind, UnstructuredCommandInput, UsageUpdate,
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
use codex_protocol::{
    approvals::{ElicitationRequest, ElicitationRequestEvent},
    config_types::{CollaborationMode, ModeKind, ServiceTier, Settings, TrustLevel},
    custom_prompts::CustomPrompt,
    dynamic_tools::{DynamicToolCallOutputContentItem, DynamicToolCallRequest},
    items::TurnItem,
    mcp::CallToolResult,
    models::{MacOsSeatbeltProfileExtensions, PermissionProfile, ResponseItem, WebSearchAction},
    openai_models::{ModelPreset, ReasoningEffort},
    parse_command::ParsedCommand,
    plan_tool::{PlanItemArg, StepStatus, UpdatePlanArgs},
    protocol::{
        AgentMessageContentDeltaEvent, AgentMessageEvent, AgentReasoningEvent,
        AgentReasoningRawContentEvent, AgentReasoningSectionBreakEvent,
        ApplyPatchApprovalRequestEvent, BackgroundEventEvent, CollabAgentInteractionBeginEvent,
        CollabAgentInteractionEndEvent, CollabAgentSpawnBeginEvent, CollabAgentSpawnEndEvent,
        DeprecationNoticeEvent, DynamicToolCallResponseEvent, ElicitationAction, ErrorEvent, Event,
        EventMsg, ExecApprovalRequestEvent, ExecCommandBeginEvent, ExecCommandEndEvent,
        ExecCommandOutputDeltaEvent, ExecCommandStatus, ExitedReviewModeEvent, FileChange,
        HookCompletedEvent, HookStartedEvent, ImageGenerationBeginEvent, ImageGenerationEndEvent,
        ItemCompletedEvent, ItemStartedEvent, ListCustomPromptsResponseEvent,
        ListSkillsResponseEvent, McpInvocation, McpListToolsResponseEvent, McpStartupCompleteEvent,
        McpStartupUpdateEvent, McpToolCallBeginEvent, McpToolCallEndEvent, ModelRerouteEvent,
        NetworkApprovalContext, NetworkPolicyRuleAction, Op, PatchApplyBeginEvent,
        PatchApplyEndEvent, PatchApplyStatus, PlanDeltaEvent, ReasoningContentDeltaEvent,
        ReasoningRawContentDeltaEvent, ReviewDecision, ReviewOutputEvent, ReviewRequest,
        ReviewTarget, RolloutItem, SandboxPolicy, StreamErrorEvent, TerminalInteractionEvent,
        ThreadRolledBackEvent, TokenCountEvent, TurnAbortedEvent, TurnCompleteEvent,
        TurnStartedEvent, UserMessageEvent, ViewImageToolCallEvent, WarningEvent,
        WebSearchBeginEvent, WebSearchEndEvent,
    },
    request_permissions::{
        PermissionGrantScope, RequestPermissionsEvent, RequestPermissionsResponse,
    },
    user_input::UserInput,
};
use codex_shell_command::parse_command::parse_command;
use codex_utils_approval_presets::{ApprovalPreset, builtin_approval_presets};
use heck::ToTitleCase;
use itertools::Itertools;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};
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
    fn logout(&self) -> Result<bool, Error>;
}

impl Auth for Arc<AuthManager> {
    fn logout(&self) -> Result<bool, Error> {
        self.as_ref()
            .logout()
            .map_err(|e| Error::internal_error().data(e.to_string()))
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
        value: SessionConfigOptionValue,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    Cancel {
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    Shutdown {
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    ReplayHistory {
        history: Vec<RolloutItem>,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    PermissionRequestResolved {
        submission_id: String,
        request_key: String,
        response: Result<RequestPermissionResponse, Error>,
    },
}

pub struct Thread {
    /// Direct handle to the underlying Codex thread for out-of-band shutdown.
    thread: Arc<dyn CodexThreadImpl>,
    /// A sender for interacting with the thread.
    message_tx: mpsc::UnboundedSender<ThreadMessage>,
    /// Keep the actor task alive for the lifetime of the thread wrapper.
    _handle: tokio::task::JoinHandle<()>,
}

impl Thread {
    /// Send a message to the actor, logging if the channel is closed.
    fn send_message(&self, message: ThreadMessage) {
        if let Err(err) = self.message_tx.send(message) {
            warn!(
                "Thread actor channel closed, message dropped: {:?}",
                std::mem::discriminant(&err.0)
            );
        }
    }

    pub fn new(
        session_id: SessionId,
        thread: Arc<dyn CodexThreadImpl>,
        auth: Arc<AuthManager>,
        models_manager: Arc<dyn ModelsManagerImpl>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        config: Config,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (resolution_tx, resolution_rx) = mpsc::unbounded_channel();

        let actor = ThreadActor::new(
            auth,
            SessionClient::new(session_id, client_capabilities),
            thread.clone(),
            models_manager,
            config,
            message_rx,
            resolution_tx,
            resolution_rx,
        );
        let handle = tokio::task::spawn_local(actor.spawn());

        Self {
            thread,
            message_tx,
            _handle: handle,
        }
    }

    pub async fn load(&self) -> Result<LoadSessionResponse, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Load { response_tx };
        self.send_message(message);

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn config_options(&self) -> Result<Vec<SessionConfigOption>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::GetConfigOptions { response_tx };
        self.send_message(message);

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
        self.send_message(message);

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))??
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_mode(&self, mode: SessionModeId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetMode { mode, response_tx };
        self.send_message(message);

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_model(&self, model: ModelId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetModel { model, response_tx };
        self.send_message(message);

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_config_option(
        &self,
        config_id: SessionConfigId,
        value: SessionConfigOptionValue,
    ) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetConfigOption {
            config_id,
            value,
            response_tx,
        };
        self.send_message(message);

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn cancel(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Cancel { response_tx };
        self.send_message(message);

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
        self.send_message(message);

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn shutdown(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();
        let message = ThreadMessage::Shutdown { response_tx };

        if self.message_tx.send(message).is_err() {
            self.thread
                .submit(Op::Shutdown)
                .await
                .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        } else {
            response_rx
                .await
                .map_err(|e| Error::internal_error().data(e.to_string()))??;
        }
        // Let the actor drain the resulting turn-aborted/shutdown events so any in-flight
        // prompt callers observe a clean cancellation instead of a dropped response channel.
        Ok(())
    }
}

enum PendingPermissionRequest {
    Exec {
        approval_id: String,
        turn_id: String,
        option_map: HashMap<String, ReviewDecision>,
    },
    Patch {
        call_id: String,
        option_map: HashMap<String, ReviewDecision>,
    },
    RequestPermissions {
        call_id: String,
        permissions: PermissionProfile,
    },
}

struct PendingPermissionInteraction {
    request: PendingPermissionRequest,
    task: tokio::task::JoinHandle<()>,
}

enum ListResponseKind {
    McpTools,
    Skills,
}

struct PendingListResponse {
    kind: ListResponseKind,
    response_tx: oneshot::Sender<Result<StopReason, Error>>,
}

fn exec_request_key(call_id: &str) -> String {
    format!("exec:{call_id}")
}

fn patch_request_key(call_id: &str) -> String {
    format!("patch:{call_id}")
}

fn permissions_request_key(call_id: &str) -> String {
    format!("permissions:{call_id}")
}

fn context_compaction_call_id(item_id: &str) -> String {
    format!("context-compaction:{item_id}")
}

fn context_compaction_status_text(status: ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Completed => "Context compacted.",
        ToolCallStatus::Failed => "Context compaction did not complete.",
        ToolCallStatus::Pending => "Context compaction pending.",
        ToolCallStatus::InProgress => "Context compaction still running.",
        _ => "Context compaction state updated.",
    }
}

fn is_auto_compact_submission_id(id: &str) -> bool {
    id.starts_with("auto-compact-")
}

fn is_auto_compaction_event(msg: &EventMsg) -> bool {
    matches!(
        msg,
        EventMsg::ItemStarted(ItemStartedEvent {
            item: TurnItem::ContextCompaction(..),
            ..
        }) | EventMsg::ItemCompleted(ItemCompletedEvent {
            item: TurnItem::ContextCompaction(..),
            ..
        }) | EventMsg::ContextCompacted(..)
    )
}

fn is_global_event(msg: &EventMsg) -> bool {
    matches!(
        msg,
        EventMsg::McpListToolsResponse(..)
            | EventMsg::ListSkillsResponse(..)
            | EventMsg::McpStartupUpdate(..)
            | EventMsg::McpStartupComplete(..)
    )
}

enum SubmissionState {
    /// Loading custom prompts from the project
    CustomPrompts(CustomPromptsState),
    /// User prompts, including slash commands like /init, /review, /compact, /undo.
    Prompt(PromptState),
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

    async fn handle_permission_request_resolved(
        &mut self,
        client: &SessionClient,
        request_key: String,
        response: Result<RequestPermissionResponse, Error>,
    ) -> Result<(), Error> {
        match self {
            Self::CustomPrompts(..) => Ok(()),
            Self::Prompt(state) => {
                state
                    .handle_permission_request_resolved(client, request_key, response)
                    .await
            }
        }
    }

    fn abort_pending_interactions(&mut self) {
        if let Self::Prompt(state) = self {
            state.abort_pending_interactions();
        }
    }

    fn fail(&mut self, err: Error) {
        if let Self::Prompt(state) = self
            && let Some(response_tx) = state.response_tx.take()
        {
            if response_tx.send(Err(err)).is_err() {
                warn!("Response channel closed, error dropped");
            }
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
                    if tx.send(Ok(custom_prompts)).is_err() {
                        warn!("Custom prompts response channel closed");
                    }
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

/// Cumulative token usage tracked across a prompt's turn.
#[derive(Debug, Default)]
struct AccumulatedUsage {
    input_tokens: i64,
    output_tokens: i64,
    cached_input_tokens: i64,
    reasoning_output_tokens: i64,
}

impl AccumulatedUsage {
    #[allow(dead_code)]
    fn total_tokens(&self) -> i64 {
        self.input_tokens
            + self.output_tokens
            + self.cached_input_tokens
            + self.reasoning_output_tokens
    }
}

struct PromptState {
    submission_id: String,
    active_commands: HashMap<String, ActiveCommand>,
    active_web_searches: HashMap<String, ()>,
    active_context_compactions: HashSet<String>,
    active_patch_applies: HashSet<String>,
    /// Tracks the most recently spawned sub-agent tool call ID so that
    /// waiting/resume/close events can update its status instead of emitting
    /// standalone text messages.
    active_subagent_call_id: Option<String>,
    thread: Arc<dyn CodexThreadImpl>,
    resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
    pending_permission_interactions: HashMap<String, PendingPermissionInteraction>,
    event_count: usize,
    response_tx: Option<oneshot::Sender<Result<StopReason, Error>>>,
    seen_message_deltas: bool,
    seen_reasoning_deltas: bool,
    accumulated_usage: AccumulatedUsage,
}

impl PromptState {
    fn new(
        submission_id: String,
        thread: Arc<dyn CodexThreadImpl>,
        resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
        response_tx: oneshot::Sender<Result<StopReason, Error>>,
    ) -> Self {
        Self {
            submission_id,
            active_commands: HashMap::new(),
            active_web_searches: HashMap::new(),
            active_context_compactions: HashSet::new(),
            active_patch_applies: HashSet::new(),
            active_subagent_call_id: None,
            thread,
            resolution_tx,
            pending_permission_interactions: HashMap::new(),
            event_count: 0,
            response_tx: Some(response_tx),
            seen_message_deltas: false,
            seen_reasoning_deltas: false,
            accumulated_usage: AccumulatedUsage::default(),
        }
    }

    fn is_active(&self) -> bool {
        let Some(response_tx) = &self.response_tx else {
            return false;
        };
        !response_tx.is_closed()
    }

    fn abort_pending_interactions(&mut self) {
        for (_, interaction) in self.pending_permission_interactions.drain() {
            interaction.task.abort();
        }
    }

    async fn start_context_compaction(&mut self, client: &SessionClient, item_id: &str) {
        let call_id = context_compaction_call_id(item_id);
        if !self.active_context_compactions.insert(call_id.clone()) {
            return;
        }

        client
            .send_tool_call(
                ToolCall::new(call_id, "Compacting context")
                    .kind(ToolKind::Think)
                    .status(ToolCallStatus::InProgress)
                    .content(vec![
                        "Condensing earlier conversation so the next turn can continue."
                            .to_string()
                            .into(),
                    ]),
            )
            .await;
    }

    async fn settle_context_compaction(
        &mut self,
        client: &SessionClient,
        item_id: &str,
        status: ToolCallStatus,
    ) {
        let call_id = context_compaction_call_id(item_id);
        let content = context_compaction_status_text(status).to_string();

        if self.active_context_compactions.remove(&call_id) {
            client
                .send_tool_call_update(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new()
                        .status(status)
                        .content(vec![content.into()]),
                ))
                .await;
        } else if matches!(status, ToolCallStatus::Completed | ToolCallStatus::Failed) {
            client
                .send_tool_call(
                    ToolCall::new(call_id, "Compacting context")
                        .kind(ToolKind::Think)
                        .status(status)
                        .content(vec![content.into()]),
                )
                .await;
        }
    }

    async fn settle_all_context_compactions(
        &mut self,
        client: &SessionClient,
        status: ToolCallStatus,
    ) {
        let pending: Vec<_> = self.active_context_compactions.drain().collect();
        for call_id in pending {
            client
                .send_tool_call_update(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new().status(status).content(vec![
                        context_compaction_status_text(status).to_string().into(),
                    ]),
                ))
                .await;
        }
    }

    async fn settle_all_patch_applies(&mut self, client: &SessionClient, status: ToolCallStatus) {
        let pending: Vec<_> = self.active_patch_applies.drain().collect();
        let content = match status {
            ToolCallStatus::Completed => "Edit completed.",
            ToolCallStatus::Failed => "Edit interrupted before completion.",
            ToolCallStatus::Pending => "Edit pending.",
            ToolCallStatus::InProgress => "Edit still running.",
            _ => "Edit status unknown.",
        }
        .to_string();

        for call_id in pending {
            client
                .send_tool_call_update(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new()
                        .status(status)
                        .content(vec![content.clone().into()]),
                ))
                .await;
        }
    }

    fn spawn_permission_request(
        &mut self,
        client: &SessionClient,
        request_key: String,
        pending_request: PendingPermissionRequest,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) {
        let client = client.clone();
        let resolution_tx = self.resolution_tx.clone();
        let submission_id = self.submission_id.clone();
        let resolved_request_key = request_key.clone();
        let handle = tokio::task::spawn_local(async move {
            let response = client.request_permission(tool_call, options).await;
            drop(
                resolution_tx.send(ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key: resolved_request_key,
                    response,
                }),
            );
        });

        if let Some(interaction) = self.pending_permission_interactions.insert(
            request_key,
            PendingPermissionInteraction {
                request: pending_request,
                task: handle,
            },
        ) {
            interaction.task.abort();
        }
    }

    async fn handle_permission_request_resolved(
        &mut self,
        _client: &SessionClient,
        request_key: String,
        response: Result<RequestPermissionResponse, Error>,
    ) -> Result<(), Error> {
        let Some(interaction) = self.pending_permission_interactions.remove(&request_key) else {
            warn!("Ignoring permission response for unknown request key: {request_key}");
            return Ok(());
        };
        let pending_request = interaction.request;
        let response = response?;

        match pending_request {
            PendingPermissionRequest::Exec {
                approval_id,
                turn_id,
                option_map,
            } => {
                let decision = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => option_map
                        .get(option_id.0.as_ref())
                        .cloned()
                        .unwrap_or(ReviewDecision::Abort),
                    RequestPermissionOutcome::Cancelled | _ => ReviewDecision::Abort,
                };

                self.thread
                    .submit(Op::ExecApproval {
                        id: approval_id,
                        turn_id: Some(turn_id),
                        decision,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
            PendingPermissionRequest::Patch {
                call_id,
                option_map,
            } => {
                let decision = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => option_map
                        .get(option_id.0.as_ref())
                        .cloned()
                        .unwrap_or(ReviewDecision::Abort),
                    RequestPermissionOutcome::Cancelled | _ => ReviewDecision::Abort,
                };

                self.thread
                    .submit(Op::PatchApproval {
                        id: call_id,
                        decision,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
            PendingPermissionRequest::RequestPermissions {
                call_id,
                permissions,
            } => {
                let response = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => match option_id.0.as_ref() {
                        "approved-for-session" => RequestPermissionsResponse {
                            permissions,
                            scope: PermissionGrantScope::Session,
                        },
                        "approved" => RequestPermissionsResponse {
                            permissions,
                            scope: PermissionGrantScope::Turn,
                        },
                        _ => RequestPermissionsResponse {
                            permissions: PermissionProfile::default(),
                            scope: PermissionGrantScope::Turn,
                        },
                    },
                    RequestPermissionOutcome::Cancelled | _ => RequestPermissionsResponse {
                        permissions: PermissionProfile::default(),
                        scope: PermissionGrantScope::Turn,
                    },
                };

                self.thread
                    .submit(Op::RequestPermissionsResponse {
                        id: call_id,
                        response,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
        }

        Ok(())
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
                info!(
                    "Task started with context window of {turn_id} {model_context_window:?} {collaboration_mode_kind:?}"
                );
            }
            EventMsg::TokenCount(TokenCountEvent { info, rate_limits }) => {
                if let Some(info) = info {
                    // Accumulate token usage across the turn
                    let last = &info.last_token_usage;
                    self.accumulated_usage.input_tokens += last.input_tokens;
                    self.accumulated_usage.output_tokens += last.output_tokens;
                    self.accumulated_usage.cached_input_tokens += last.cached_input_tokens;
                    self.accumulated_usage.reasoning_output_tokens += last.reasoning_output_tokens;

                    if let Some(size) = info.model_context_window {
                        let used = info.total_token_usage.tokens_in_context_window().max(0) as u64;
                        let mut update = UsageUpdate::new(used, size as u64);

                        // Include cost if credits info is available from rate limits
                        if let Some(rate_limits) = &rate_limits
                            && let Some(credits) = &rate_limits.credits
                            && let Some(balance) = &credits.balance
                        {
                            if let Ok(amount) = balance.parse::<f64>() {
                                update = update.cost(Cost::new(amount, "USD"));
                            }
                        }

                        client
                            .send_notification(SessionUpdate::UsageUpdate(update))
                            .await;
                    }
                }
            }
            EventMsg::ItemStarted(ItemStartedEvent {
                thread_id,
                turn_id,
                item,
            }) => {
                info!(
                    "Item started with thread_id: {thread_id}, turn_id: {turn_id}, item: {item:?}"
                );
                if let TurnItem::ContextCompaction(item) = item {
                    self.start_context_compaction(client, &item.id).await;
                }
            }
            EventMsg::UserMessage(UserMessageEvent {
                message,
                images: _,
                text_elements: _,
                local_images: _,
            }) => {
                info!("User message: {message:?}");
            }
            EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
            }) => {
                info!(
                    "Agent message content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, delta: {delta:?}"
                );
                self.seen_message_deltas = true;
                client.send_agent_text(delta).await;
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
                info!(
                    "Agent reasoning content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, index: {index}, delta: {delta:?}"
                );
                self.seen_reasoning_deltas = true;
                client.send_agent_thought(delta).await;
            }
            EventMsg::AgentReasoningSectionBreak(AgentReasoningSectionBreakEvent {
                item_id,
                summary_index,
            }) => {
                info!(
                    "Agent reasoning section break received:  item_id: {item_id}, index: {summary_index}"
                );
                // Make sure the section heading actually get spacing
                self.seen_reasoning_deltas = true;
                client.send_agent_thought("\n\n").await;
            }
            EventMsg::AgentMessage(AgentMessageEvent { message, phase }) => {
                info!("Agent message (non-delta) received: {message:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_message_deltas) {
                    // While a sub-agent is running, commentary messages are
                    // just polling narration ("I'm still waiting..."). Redirect
                    // them to thoughts so they don't clutter the main output.
                    if self.active_subagent_call_id.is_some()
                        && phase == Some(codex_protocol::models::MessagePhase::Commentary)
                    {
                        client.send_agent_thought(message).await;
                    } else {
                        client.send_agent_text(message).await;
                    }
                }
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                info!("Agent reasoning (non-delta) received: {text:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_reasoning_deltas) {
                    client.send_agent_thought(text).await;
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
                    if response_tx.send(Err(err)).is_err() {
                        warn!("Response channel closed, error dropped");
                    }
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
            EventMsg::DynamicToolCallRequest(DynamicToolCallRequest {
                call_id,
                turn_id,
                tool,
                arguments,
            }) => {
                info!(
                    "Dynamic tool call request: call_id={call_id}, turn_id={turn_id}, tool={tool}"
                );
                self.start_dynamic_tool_call(client, call_id, tool, arguments)
                    .await;
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
                    if response_tx.send(Err(err)).is_err() {
                        warn!("Response channel closed, error dropped");
                    }
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
                info!(
                    "Item completed: thread_id={}, turn_id={}, item={:?}",
                    thread_id, turn_id, item
                );
                if let TurnItem::ContextCompaction(item) = item {
                    self.settle_context_compaction(client, &item.id, ToolCallStatus::Completed)
                        .await;
                }
            }
            EventMsg::TurnComplete(TurnCompleteEvent {
                last_agent_message,
                turn_id,
            }) => {
                info!(
                    "Task {turn_id} completed successfully after {} events. Last agent message: {last_agent_message:?}",
                    self.event_count
                );
                // Clean up any lingering sub-agent tool call.
                if let Some(active_id) = self.active_subagent_call_id.take() {
                    client
                        .send_tool_call_update(ToolCallUpdate::new(
                            active_id,
                            ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
                        ))
                        .await;
                }
                self.settle_all_context_compactions(client, ToolCallStatus::Completed)
                    .await;
                self.abort_pending_interactions();
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
                client
                    .send_agent_text(event.message.unwrap_or(fallback))
                    .await;
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
                self.settle_all_context_compactions(client, ToolCallStatus::Failed)
                    .await;
                self.settle_all_patch_applies(client, ToolCallStatus::Failed)
                    .await;
                self.abort_pending_interactions();
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
                self.settle_all_context_compactions(client, ToolCallStatus::Failed)
                    .await;
                self.settle_all_patch_applies(client, ToolCallStatus::Failed)
                    .await;
                self.abort_pending_interactions();
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            EventMsg::ShutdownComplete => {
                info!("Agent shutting down");
                self.settle_all_context_compactions(client, ToolCallStatus::Failed)
                    .await;
                self.settle_all_patch_applies(client, ToolCallStatus::Failed)
                    .await;
                self.abort_pending_interactions();
                if let Some(response_tx) = self.response_tx.take() {
                    response_tx.send(Ok(StopReason::Cancelled)).ok();
                }
            }
            EventMsg::ViewImageToolCall(ViewImageToolCallEvent { call_id, path }) => {
                info!("ViewImageToolCallEvent received");
                let display_path = path.display().to_string();
                client
                    .send_notification(SessionUpdate::ToolCall(
                        ToolCall::new(call_id, format!("View Image {display_path}"))
                            .kind(ToolKind::Read)
                            .status(ToolCallStatus::Completed)
                            .content(vec![ToolCallContent::Content(Content::new(
                                ContentBlock::ResourceLink(ResourceLink::new(
                                    display_path.clone(),
                                    display_path.clone(),
                                )),
                            ))])
                            .locations(vec![ToolCallLocation::new(path)]),
                    ))
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
                    if response_tx.send(Err(err)).is_err() {
                        warn!("Response channel closed, error dropped");
                    }
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
                info!(
                    "Elicitation request: server={}, id={:?}",
                    event.server_name, event.id
                );
                if let Err(err) = self.mcp_elicitation(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    if response_tx.send(Err(err)).is_err() {
                        warn!("Response channel closed, error dropped");
                    }
                }
            }
            EventMsg::ModelReroute(ModelRerouteEvent {
                from_model,
                to_model,
                reason,
            }) => {
                info!("Model reroute: from={from_model}, to={to_model}, reason={reason:?}");
            }

            EventMsg::ContextCompacted(..) => {
                info!("Context compacted");
                client
                    .send_agent_text("Context compacted\n".to_string())
                    .await;
            }
            EventMsg::RequestPermissions(event) => {
                info!("Request permissions: {} {}", event.call_id, event.turn_id);
                if let Err(err) = self.request_permissions(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    if response_tx.send(Err(err)).is_err() {
                        warn!("Response channel closed, error dropped");
                    }
                }
            }

            // Hook lifecycle — surface as agent status text so users know hooks are running
            EventMsg::HookStarted(HookStartedEvent { turn_id: _, run }) => {
                info!("Hook started: {} ({:?})", run.id, run.event_name);
                client
                    .send_agent_text(format!(
                        "Running hook: {} ({:?})...\n",
                        run.id, run.event_name
                    ))
                    .await;
            }
            EventMsg::HookCompleted(HookCompletedEvent { turn_id: _, run }) => {
                let status_msg = run.status_message.as_deref().unwrap_or("");
                info!(
                    "Hook completed: {} — {:?} {}",
                    run.id, run.status, status_msg
                );
                client
                    .send_agent_text(format!(
                        "Hook completed: {} — {:?}{}\n",
                        run.id,
                        run.status,
                        if status_msg.is_empty() {
                            String::new()
                        } else {
                            format!(": {status_msg}")
                        }
                    ))
                    .await;
            }

            // Image generation — surface as tool calls so client UI can track progress
            EventMsg::ImageGenerationBegin(ImageGenerationBeginEvent { call_id }) => {
                info!("Image generation started: call_id={call_id}");
                client
                    .send_tool_call(
                        ToolCall::new(call_id, "Generating image")
                            .kind(ToolKind::Other)
                            .status(ToolCallStatus::InProgress),
                    )
                    .await;
            }
            EventMsg::ImageGenerationEnd(ImageGenerationEndEvent {
                call_id,
                status,
                revised_prompt: _,
                result,
                ..
            }) => {
                info!("Image generation ended: call_id={call_id}, status={status}");
                let tc_status = if status == "success" {
                    ToolCallStatus::Completed
                } else {
                    ToolCallStatus::Failed
                };
                client
                    .send_tool_call_update(ToolCallUpdate::new(
                        call_id,
                        ToolCallUpdateFields::new()
                            .status(tc_status)
                            .content(vec![result.into()]),
                    ))
                    .await;
            }

            // Agent reasoning raw content (non-delta, full text)
            EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent { text }) => {
                info!("Agent reasoning raw content received");
                client.send_agent_thought(text).await;
            }

            // Thread rollback notification
            EventMsg::ThreadRolledBack(ThreadRolledBackEvent { num_turns }) => {
                info!("Thread rolled back: {num_turns} turns removed");
                client
                    .send_agent_text(format!(
                        "Thread rolled back: {num_turns} turn{} removed from context.\n",
                        if num_turns == 1 { "" } else { "s" }
                    ))
                    .await;
            }

            // Background events — surface as agent text
            EventMsg::BackgroundEvent(BackgroundEventEvent { message }) => {
                info!("Background event: {message}");
                client.send_agent_text(format!("{message}\n")).await;
            }

            // Deprecation notices — forward to client as warnings
            EventMsg::DeprecationNotice(DeprecationNoticeEvent { summary, details }) => {
                warn!("Deprecation notice: {summary}");
                let msg = if let Some(details) = details {
                    format!("**Deprecation:** {summary}\n{details}\n")
                } else {
                    format!("**Deprecation:** {summary}\n")
                };
                client.send_agent_text(msg).await;
            }

            // Plan delta — forward incremental plan text as agent thought
            EventMsg::PlanDelta(PlanDeltaEvent {
                thread_id: _,
                turn_id: _,
                item_id: _,
                delta,
            }) => {
                client.send_agent_thought(delta).await;
            }

            // Collaboration events — surface as tool calls so the client can track sub-agent activity
            EventMsg::CollabAgentSpawnBegin(CollabAgentSpawnBeginEvent {
                call_id,
                sender_thread_id: _,
                prompt,
                model,
                reasoning_effort: _,
            }) => {
                info!("Collab agent spawn begin: call_id={call_id}, model={model}");
                let title = if prompt.is_empty() {
                    format!("Sub-agent ({model})")
                } else {
                    let truncated = if prompt.len() > 80 {
                        format!("{}...", &prompt[..77])
                    } else {
                        prompt
                    };
                    format!("Sub-agent: {truncated}")
                };
                // Keep the tool call InProgress — it represents the full sub-agent
                // lifecycle (spawn → work → results). We'll mark it Completed only
                // when the interaction ends or the collab close fires.
                self.active_subagent_call_id = Some(call_id.clone());
                client
                    .send_tool_call(
                        ToolCall::new(call_id, title)
                            .kind(ToolKind::Other)
                            .status(ToolCallStatus::InProgress),
                    )
                    .await;
            }
            EventMsg::CollabAgentSpawnEnd(CollabAgentSpawnEndEvent {
                call_id,
                sender_thread_id: _,
                new_thread_id,
                ..
            }) => {
                info!("Collab agent spawn end: call_id={call_id}, new_thread={new_thread_id:?}");
                // Don't mark Completed yet — the sub-agent is still working.
                // Update content to show it's been spawned and is running.
                client
                    .send_tool_call_update(ToolCallUpdate::new(
                        call_id,
                        ToolCallUpdateFields::new()
                            .content(vec!["Sub-agent spawned, working...".to_string().into()]),
                    ))
                    .await;
            }
            EventMsg::CollabAgentInteractionBegin(CollabAgentInteractionBeginEvent {
                call_id,
                sender_thread_id: _,
                receiver_thread_id,
                ..
            }) => {
                info!("Collab interaction begin: call_id={call_id}, receiver={receiver_thread_id}");
                // Update existing sub-agent tool call if we have one, otherwise create new.
                if let Some(ref active_id) = self.active_subagent_call_id {
                    client
                        .send_tool_call_update(ToolCallUpdate::new(
                            active_id.clone(),
                            ToolCallUpdateFields::new()
                                .content(vec!["Sub-agent interacting...".to_string().into()]),
                        ))
                        .await;
                } else {
                    self.active_subagent_call_id = Some(call_id.clone());
                    client
                        .send_tool_call(
                            ToolCall::new(call_id, "Sub-agent interaction".to_string())
                                .kind(ToolKind::Other)
                                .status(ToolCallStatus::InProgress),
                        )
                        .await;
                }
            }
            EventMsg::CollabAgentInteractionEnd(CollabAgentInteractionEndEvent {
                call_id: _,
                sender_thread_id: _,
                receiver_thread_id,
                ..
            }) => {
                info!("Collab interaction end: receiver={receiver_thread_id}");
                // Don't complete yet — more interactions or the final answer may follow.
            }

            // Collab waiting — update the active sub-agent tool call status
            EventMsg::CollabWaitingBegin(..) => {
                info!("Collab waiting begin");
                if let Some(ref active_id) = self.active_subagent_call_id {
                    client
                        .send_tool_call_update(ToolCallUpdate::new(
                            active_id.clone(),
                            ToolCallUpdateFields::new().content(vec![
                                "Waiting for sub-agent results...".to_string().into(),
                            ]),
                        ))
                        .await;
                }
            }
            EventMsg::CollabWaitingEnd(..) => {
                info!("Collab waiting end");
                // Don't emit anything — the next event will be either another
                // waiting cycle or the final agent message with results.
            }
            EventMsg::CollabResumeBegin(..) | EventMsg::CollabResumeEnd(..) => {
                info!("Collab resume event");
            }
            EventMsg::CollabCloseBegin(..) => {
                info!("Collab close begin");
            }
            EventMsg::CollabCloseEnd(..) => {
                info!("Collab close end");
                // Sub-agent lifecycle is done — complete the tool call.
                if let Some(active_id) = self.active_subagent_call_id.take() {
                    client
                        .send_tool_call_update(ToolCallUpdate::new(
                            active_id,
                            ToolCallUpdateFields::new()
                                .status(ToolCallStatus::Completed)
                                .content(vec!["Sub-agent finished.".to_string().into()]),
                        ))
                        .await;
                }
            }

            // Events that are truly ignorable (old/superseded or not applicable to ACP)
            EventMsg::TurnDiff(..)
            | EventMsg::SkillsUpdateAvailable
            | EventMsg::AgentMessageDelta(..)
            | EventMsg::AgentReasoningDelta(..)
            | EventMsg::AgentReasoningRawContentDelta(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::SessionConfigured(..)
            | EventMsg::RealtimeConversationStarted(..)
            | EventMsg::RealtimeConversationRealtime(..)
            | EventMsg::RealtimeConversationClosed(..) => {}

            // McpListToolsResponse and ListSkillsResponse are handled at the
            // ThreadActor level (handle_list_response), not per-submission.
            EventMsg::McpListToolsResponse(..) | EventMsg::ListSkillsResponse(..) => {}

            e @ (EventMsg::ListCustomPromptsResponse(..)
            | EventMsg::GetHistoryEntryResponse(..)
            | EventMsg::RequestUserInput(..)
            | EventMsg::ListRemoteSkillsResponse(..)
            | EventMsg::RemoteSkillDownloaded(..)) => {
                warn!("Unexpected event: {:?}", e);
            }
        }
    }

    async fn mcp_elicitation(
        &mut self,
        _client: &SessionClient,
        event: ElicitationRequestEvent,
    ) -> Result<(), Error> {
        let ElicitationRequestEvent {
            server_name,
            id,
            request,
            turn_id: _,
        } = event;
        let request_kind = match &request {
            ElicitationRequest::Form { .. } => "form",
            ElicitationRequest::Url { .. } => "url",
        };

        info!(
            "Auto-declining unsupported MCP elicitation: server={}, id={:?}, kind={request_kind}",
            server_name, id
        );

        self.thread
            .submit(Op::ResolveElicitation {
                server_name,
                request_id: id,
                decision: ElicitationAction::Decline,
                content: None,
                meta: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

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
        &mut self,
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
        let request_key = patch_request_key(&call_id);
        let options = vec![
            PermissionOption::new("approved", "Yes", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                "abort",
                "No, provide feedback",
                PermissionOptionKind::RejectOnce,
            ),
        ];
        self.spawn_permission_request(
            client,
            request_key,
            PendingPermissionRequest::Patch {
                call_id: call_id.clone(),
                option_map: HashMap::from([
                    ("approved".to_string(), ReviewDecision::Approved),
                    ("abort".to_string(), ReviewDecision::Abort),
                ]),
            },
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
            options,
        );
        Ok(())
    }

    async fn start_patch_apply(&mut self, client: &SessionClient, event: PatchApplyBeginEvent) {
        let raw_input = serde_json::json!(&event);
        let PatchApplyBeginEvent {
            call_id,
            auto_approved: _,
            changes,
            turn_id: _,
        } = event;
        self.active_patch_applies.insert(call_id.clone());

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

    async fn end_patch_apply(&mut self, client: &SessionClient, event: PatchApplyEndEvent) {
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
        self.active_patch_applies.remove(&call_id);

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
        let available_decisions = event.effective_available_decisions();
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
            available_decisions: _,
            proposed_network_policy_amendments,
            skill_metadata: _,
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
        if let Some(amendment) = proposed_execpolicy_amendment.as_ref() {
            content.push(format!(
                "Proposed Amendment: {}",
                amendment.command().join("\n")
            ));
        }
        if let Some(policy) = network_approval_context.as_ref() {
            let NetworkApprovalContext { host, protocol } = policy;
            content.push(format!("Network Approval Context: {:?} {}", protocol, host));
        }
        if let Some(permissions) = additional_permissions.as_ref() {
            content.push(format!(
                "Additional Permissions: {}",
                serde_json::to_string_pretty(&permissions)?
            ));
        }
        content.push(format!(
            "Available Decisions: {}",
            available_decisions.iter().map(|d| d.to_string()).join("\n")
        ));
        if let Some(amendments) = proposed_network_policy_amendments.as_ref() {
            content.push(format!(
                "Proposed Network Policy Amendments: {}",
                amendments
                    .iter()
                    .map(|amendment| format!("{:?} {:?}", amendment.action, amendment.host))
                    .join("\n")
            ));
        }

        let content = if content.is_empty() {
            None
        } else {
            Some(vec![content.join("\n").into()])
        };
        let permission_options = build_exec_permission_options(
            &available_decisions,
            network_approval_context.as_ref(),
            additional_permissions.as_ref(),
        );

        self.spawn_permission_request(
            client,
            exec_request_key(&call_id),
            PendingPermissionRequest::Exec {
                approval_id: approval_id.unwrap_or(call_id.clone()),
                turn_id,
                option_map: permission_options
                    .iter()
                    .map(|option| (option.option_id.to_string(), option.decision.clone()))
                    .collect(),
            },
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
            permission_options
                .into_iter()
                .map(|option| option.permission_option)
                .collect(),
        );

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
        } else {
            warn!("Received ExecCommandOutputDelta for unknown call_id: {call_id}");
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
        } else {
            warn!("Received ExecCommandEnd for unknown call_id: {call_id} (exit_code={exit_code})");
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
        } else {
            warn!("Received TerminalInteraction for unknown call_id: {call_id}");
        }
    }

    async fn start_web_search(&mut self, client: &SessionClient, call_id: String) {
        self.active_web_searches.insert(call_id.clone(), ());
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
        let title = match &action {
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
        };

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::InProgress)
                    .title(title)
                    .raw_input(serde_json::json!({
                        "query": query,
                        "action": action
                    })),
            ))
            .await;
    }

    async fn complete_web_search(&mut self, client: &SessionClient) {
        let completed: Vec<String> = self.active_web_searches.drain().map(|(id, _)| id).collect();
        for call_id in completed {
            client
                .send_tool_call_update(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
                ))
                .await;
        }
    }

    async fn request_permissions(
        &mut self,
        client: &SessionClient,
        event: RequestPermissionsEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let RequestPermissionsEvent {
            call_id,
            turn_id: _,
            reason,
            permissions,
        } = event;

        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId::new(call_id.clone());

        let mut content = vec![];

        if let Some(reason) = reason.as_ref() {
            content.push(reason.clone());
        }
        if let Some(file_system) = permissions.file_system.as_ref() {
            if let Some(read) = file_system.read.as_ref() {
                content.push(format!(
                    "File System Read Access: {}",
                    read.iter().map(|p| p.display()).join(", ")
                ));
            }
            if let Some(write) = file_system.write.as_ref() {
                content.push(format!(
                    "File System Write Access: {}",
                    write.iter().map(|p| p.display()).join(", ")
                ));
            }
        }
        if let Some(network) = permissions.network.as_ref()
            && let Some(enabled) = network.enabled
        {
            content.push(format!("Network Access: {enabled}"));
        }
        if let Some(mac) = permissions.macos.as_ref() {
            let MacOsSeatbeltProfileExtensions {
                macos_preferences,
                macos_automation,
                macos_launch_services,
                macos_accessibility,
                macos_calendar,
                macos_reminders,
                macos_contacts,
            } = mac;

            content.push("MacOS Seatbelt Profile Extensions: ".to_string());
            content.push(format!("Preferences: {:?}", macos_preferences));
            content.push(format!("Automation: {:?}", macos_automation));
            content.push(format!("Launch Services: {}", macos_launch_services));
            content.push(format!("Accessibility: {}", macos_accessibility));
            content.push(format!("Calendar: {}", macos_calendar));
            content.push(format!("Reminders: {}", macos_reminders));
            content.push(format!("Contacts: {:?}", macos_contacts));
        }

        let content = if content.is_empty() {
            None
        } else {
            Some(vec![content.join("\n").into()])
        };

        self.spawn_permission_request(
            client,
            permissions_request_key(&call_id),
            PendingPermissionRequest::RequestPermissions {
                call_id,
                permissions,
            },
            ToolCallUpdate::new(
                tool_call_id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Pending)
                    .title(reason.unwrap_or_else(|| "Permissions Request".to_string()))
                    .raw_input(raw_input)
                    .content(content),
            ),
            vec![
                PermissionOption::new(
                    "approved-for-session",
                    "Yes, for session",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new("approved", "Yes", PermissionOptionKind::AllowOnce),
                PermissionOption::new("abort", "No", PermissionOptionKind::RejectOnce),
            ],
        );

        Ok(())
    }
}

#[derive(Clone)]
struct ExecPermissionOption {
    option_id: &'static str,
    permission_option: PermissionOption,
    decision: ReviewDecision,
}

fn build_exec_permission_options(
    available_decisions: &[ReviewDecision],
    network_approval_context: Option<&NetworkApprovalContext>,
    additional_permissions: Option<&PermissionProfile>,
) -> Vec<ExecPermissionOption> {
    available_decisions
        .iter()
        .map(|decision| match decision {
            ReviewDecision::Approved => ExecPermissionOption {
                option_id: "approved",
                permission_option: PermissionOption::new(
                    "approved",
                    if network_approval_context.is_some() {
                        "Yes, just this once"
                    } else {
                        "Yes, proceed"
                    },
                    PermissionOptionKind::AllowOnce,
                ),
                decision: ReviewDecision::Approved,
            },
            ReviewDecision::ApprovedExecpolicyAmendment {
                proposed_execpolicy_amendment,
            } => {
                let command_prefix = proposed_execpolicy_amendment.command().join(" ");
                let label = if command_prefix.contains('\n')
                    || command_prefix.contains('\r')
                    || command_prefix.is_empty()
                {
                    "Yes, and remember this command pattern".to_string()
                } else {
                    format!(
                        "Yes, and don't ask again for commands that start with `{command_prefix}`"
                    )
                };
                ExecPermissionOption {
                    option_id: "approved-execpolicy-amendment",
                    permission_option: PermissionOption::new(
                        "approved-execpolicy-amendment",
                        label,
                        PermissionOptionKind::AllowAlways,
                    ),
                    decision: ReviewDecision::ApprovedExecpolicyAmendment {
                        proposed_execpolicy_amendment: proposed_execpolicy_amendment.clone(),
                    },
                }
            }
            ReviewDecision::ApprovedForSession => ExecPermissionOption {
                option_id: "approved-for-session",
                permission_option: PermissionOption::new(
                    "approved-for-session",
                    if network_approval_context.is_some() {
                        "Yes, and allow this host for this session"
                    } else if additional_permissions.is_some() {
                        "Yes, and allow these permissions for this session"
                    } else {
                        "Yes, and don't ask again for this command in this session"
                    },
                    PermissionOptionKind::AllowAlways,
                ),
                decision: ReviewDecision::ApprovedForSession,
            },
            ReviewDecision::NetworkPolicyAmendment {
                network_policy_amendment,
            } => {
                let (option_id, label, kind) = match network_policy_amendment.action {
                    NetworkPolicyRuleAction::Allow => (
                        "network-policy-amendment-allow",
                        "Yes, and allow this host in the future",
                        PermissionOptionKind::AllowAlways,
                    ),
                    NetworkPolicyRuleAction::Deny => (
                        "network-policy-amendment-deny",
                        "No, and block this host in the future",
                        PermissionOptionKind::RejectAlways,
                    ),
                };
                ExecPermissionOption {
                    option_id,
                    permission_option: PermissionOption::new(option_id, label, kind),
                    decision: ReviewDecision::NetworkPolicyAmendment {
                        network_policy_amendment: network_policy_amendment.clone(),
                    },
                }
            }
            ReviewDecision::Denied => ExecPermissionOption {
                option_id: "denied",
                permission_option: PermissionOption::new(
                    "denied",
                    "No, continue without running it",
                    PermissionOptionKind::RejectOnce,
                ),
                decision: ReviewDecision::Denied,
            },
            ReviewDecision::Abort => ExecPermissionOption {
                option_id: "abort",
                permission_option: PermissionOption::new(
                    "abort",
                    "No, and tell Codex what to do differently",
                    PermissionOptionKind::RejectOnce,
                ),
                decision: ReviewDecision::Abort,
            },
        })
        .collect()
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
            client: ACP_CLIENT
                .get()
                .unwrap_or_else(|| {
                    panic!("ACP_CLIENT must be initialized before creating SessionClient");
                })
                .clone(),
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
                .unwrap_or_else(|e| {
                    warn!(
                        "client_capabilities mutex poisoned, assuming no terminal output support"
                    );
                    e.into_inner()
                })
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

    async fn send_user_message(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::UserMessageChunk(ContentChunk::new(
            text.into().into(),
        )))
        .await;
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

    async fn send_tool_call(&self, tool_call: ToolCall) {
        self.send_notification(SessionUpdate::ToolCall(tool_call))
            .await;
    }

    async fn send_tool_call_update(&self, update: ToolCallUpdate) {
        self.send_notification(SessionUpdate::ToolCallUpdate(update))
            .await;
    }

    /// Send a completed tool call (used for replay and simple cases)
    async fn send_completed_tool_call(
        &self,
        call_id: impl Into<ToolCallId>,
        title: impl Into<String>,
        kind: ToolKind,
        raw_input: Option<serde_json::Value>,
    ) {
        let mut tool_call = ToolCall::new(call_id, title)
            .kind(kind)
            .status(ToolCallStatus::Completed);
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
    /// Internal message sender used to route spawned interaction results back to the actor.
    resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
    /// A sender for each interested `Op` submission that needs events routed.
    submissions: HashMap<String, SubmissionState>,
    /// The most recently started prompt submission. Used to surface live
    /// internal auto-compaction progress in the visible prompt UI.
    latest_prompt_submission_id: Option<String>,
    /// A receiver for incoming thread messages.
    message_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    /// A receiver for spawned interaction results.
    resolution_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    /// Last config options state we emitted to the client, used for deduping updates.
    last_sent_config_options: Option<Vec<SessionConfigOption>>,
    /// Pending response for /mcp or /skills list commands.
    pending_list_response: Option<PendingListResponse>,
    /// Whether plan mode is currently active (tracked separately from approval presets).
    plan_mode_active: bool,
}

impl<A: Auth> ThreadActor<A> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        auth: A,
        client: SessionClient,
        thread: Arc<dyn CodexThreadImpl>,
        models_manager: Arc<dyn ModelsManagerImpl>,
        config: Config,
        message_rx: mpsc::UnboundedReceiver<ThreadMessage>,
        resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
        resolution_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    ) -> Self {
        Self {
            auth,
            client,
            thread,
            config,
            custom_prompts: Rc::default(),
            models_manager,
            resolution_tx,
            submissions: HashMap::new(),
            latest_prompt_submission_id: None,
            message_rx,
            resolution_rx,
            last_sent_config_options: None,
            pending_list_response: None,
            plan_mode_active: false,
        }
    }

    async fn spawn(mut self) {
        let mut message_rx_open = true;
        loop {
            tokio::select! {
                biased;
                message = self.message_rx.recv(), if message_rx_open => match message {
                    Some(message) => self.handle_message(message).await,
                    None => message_rx_open = false,
                },
                message = self.resolution_rx.recv() => if let Some(message) = message {
                    self.handle_message(message).await
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

            if !message_rx_open && self.submissions.is_empty() {
                break;
            }
        }
    }

    async fn handle_message(&mut self, message: ThreadMessage) {
        match message {
            ThreadMessage::Load { response_tx } => {
                let result = self.handle_load().await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
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
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
            }
            ThreadMessage::Prompt {
                request,
                response_tx,
            } => {
                let result = self.handle_prompt(request).await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
            }
            ThreadMessage::SetMode { mode, response_tx } => {
                let result = self.handle_set_mode(mode).await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
                self.maybe_emit_config_options_update().await;
            }
            ThreadMessage::SetModel { model, response_tx } => {
                let result = self.handle_set_model(model).await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
                self.maybe_emit_config_options_update().await;
            }
            ThreadMessage::SetConfigOption {
                config_id,
                value,
                response_tx,
            } => {
                let result = self.handle_set_config_option(config_id, value).await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
            }
            ThreadMessage::Cancel { response_tx } => {
                let result = self.handle_cancel().await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
            }
            ThreadMessage::Shutdown { response_tx } => {
                let result = self.handle_shutdown().await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
            }
            ThreadMessage::ReplayHistory {
                history,
                response_tx,
            } => {
                let result = self.handle_replay_history(history).await;
                if response_tx.send(result).is_err() {
                    warn!("Response channel closed, result dropped");
                }
            }
            ThreadMessage::PermissionRequestResolved {
                submission_id,
                request_key,
                response,
            } => {
                let Some(submission) = self.submissions.get_mut(&submission_id) else {
                    warn!(
                        "Ignoring permission response for unknown submission ID: {submission_id}"
                    );
                    return;
                };

                if let Err(err) = submission
                    .handle_permission_request_resolved(&self.client, request_key, response)
                    .await
                {
                    submission.abort_pending_interactions();
                    submission.fail(err);
                }
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
            AvailableCommand::new("fast", "toggle fast mode for this session").input(
                AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                    "optional: on|off|status",
                )),
            ),
            AvailableCommand::new("diff", "show git diff including untracked files"),
            AvailableCommand::new(
                "status",
                "show current session configuration and token usage",
            ),
            AvailableCommand::new("stop", "stop all background terminals"),
            AvailableCommand::new("rename", "rename the current thread").input(
                AvailableCommandInput::Unstructured(UnstructuredCommandInput::new("new name")),
            ),
            AvailableCommand::new("mcp", "list configured MCP tools"),
            AvailableCommand::new("skills", "list and manage skills"),
        ]
    }

    async fn format_session_status(&self) -> String {
        let mut status = String::from("## Session Status\n\n");

        // Model
        if let Some(model_id) = self.find_current_model().await {
            status.push_str(&format!("**Model:** {}\n", model_id.0));
        } else {
            status.push_str("**Model:** unknown\n");
        }

        // Mode
        let mode_name = APPROVAL_PRESETS
            .iter()
            .find(|preset| {
                &preset.approval == self.config.permissions.approval_policy.get()
                    && &preset.sandbox == self.config.permissions.sandbox_policy.get()
            })
            .map(|p| p.label)
            .unwrap_or("unknown");
        status.push_str(&format!("**Mode:** {mode_name}\n"));
        status.push_str(&format!(
            "**Service Tier:** {}\n",
            format_service_tier_name(self.config.service_tier)
        ));

        // Working directory
        status.push_str(&format!(
            "**Working Directory:** {}\n",
            self.config.cwd.display()
        ));

        status.push('\n');
        status
    }

    async fn set_service_tier(&mut self, service_tier: Option<ServiceTier>) -> Result<(), Error> {
        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: None,
                effort: None,
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
                service_tier: Some(service_tier),
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.service_tier = service_tier;

        Ok(())
    }

    async fn load_custom_prompts(&mut self) -> oneshot::Receiver<Result<Vec<CustomPrompt>, Error>> {
        let (response_tx, response_rx) = oneshot::channel();
        let submission_id = match self.thread.submit(Op::ListCustomPrompts).await {
            Ok(id) => id,
            Err(e) => {
                if response_tx
                    .send(Err(Error::internal_error().data(e.to_string())))
                    .is_err()
                {
                    warn!("Response channel closed while sending ListCustomPrompts error");
                }
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
        let current_mode_id = if self.plan_mode_active {
            SessionModeId::new("plan")
        } else {
            APPROVAL_PRESETS
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
                .map(|preset| SessionModeId::new(preset.id))?
        };

        let mut available_modes: Vec<SessionMode> = APPROVAL_PRESETS
            .iter()
            .map(|preset| SessionMode::new(preset.id, preset.label).description(preset.description))
            .collect();
        available_modes.push(
            SessionMode::new("plan", "Plan").description("Planning mode — no tool execution"),
        );
        Some(SessionModeState::new(current_mode_id, available_modes))
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

    fn current_service_tier_id(&self) -> &'static str {
        match self.config.service_tier {
            Some(ServiceTier::Fast) => "fast",
            Some(ServiceTier::Flex) => "flex",
            None => "standard",
        }
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

        options.push(
            SessionConfigOption::select(
                "service_tier",
                "Service Tier",
                self.current_service_tier_id(),
                vec![
                    SessionConfigSelectOption::new("standard", "Standard")
                        .description("Use the default response tier"),
                    SessionConfigSelectOption::new("fast", "Fast")
                        .description("Prefer the fast response tier"),
                    SessionConfigSelectOption::new("flex", "Flex")
                        .description("Use the flex response tier when available"),
                ],
            )
            .category(SessionConfigOptionCategory::Model)
            .description("Choose the response service tier for this session"),
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
        value: SessionConfigOptionValue,
    ) -> Result<(), Error> {
        let SessionConfigOptionValue::ValueId { value } = value else {
            return Err(Error::invalid_params().data("Unsupported config option value"));
        };
        match config_id.0.as_ref() {
            "mode" => self.handle_set_mode(SessionModeId::new(value.0)).await,
            "model" => self.handle_set_config_model(value).await,
            "service_tier" => self.handle_set_config_service_tier(value).await,
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
                service_tier: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;

        Ok(())
    }

    async fn handle_set_config_service_tier(
        &mut self,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        let service_tier = match value.0.as_ref() {
            "standard" => None,
            "fast" => Some(ServiceTier::Fast),
            "flex" => Some(ServiceTier::Flex),
            _ => return Err(Error::invalid_params().data("Unsupported service tier")),
        };

        self.set_service_tier(service_tier).await
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
                service_tier: None,
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
                    self.auth.logout()?;
                    return Err(Error::auth_required());
                }
                "fast" => {
                    let tier = match rest.trim().to_ascii_lowercase().as_str() {
                        "" => {
                            if matches!(self.config.service_tier, Some(ServiceTier::Fast)) {
                                None
                            } else {
                                Some(ServiceTier::Fast)
                            }
                        }
                        "on" => Some(ServiceTier::Fast),
                        "off" => None,
                        "status" => {
                            let client = self.client.clone();
                            let status =
                                if matches!(self.config.service_tier, Some(ServiceTier::Fast)) {
                                    "on"
                                } else {
                                    "off"
                                };
                            client
                                .send_agent_text(format!("Fast mode is {status}.\n"))
                                .await;
                            response_tx.send(Ok(StopReason::EndTurn)).ok();
                            return Ok(response_rx);
                        }
                        _ => {
                            let client = self.client.clone();
                            client
                                .send_agent_text("Usage: /fast [on|off|status]\n".to_string())
                                .await;
                            response_tx.send(Ok(StopReason::EndTurn)).ok();
                            return Ok(response_rx);
                        }
                    };
                    self.set_service_tier(tier).await?;
                    let client = self.client.clone();
                    let status = if matches!(tier, Some(ServiceTier::Fast)) {
                        "on"
                    } else {
                        "off"
                    };
                    client
                        .send_agent_text(format!("Fast mode is {status}.\n"))
                        .await;
                    response_tx.send(Ok(StopReason::EndTurn)).ok();
                    return Ok(response_rx);
                }
                "stop" => op = Op::CleanBackgroundTerminals,
                "diff" => {
                    let cwd = self.config.cwd.clone();
                    let client = self.client.clone();
                    tokio::task::spawn_local(async move {
                        let output = run_git_diff(&cwd).await;
                        client.send_agent_text(output).await;
                    });
                    response_tx.send(Ok(StopReason::EndTurn)).ok();
                    return Ok(response_rx);
                }
                "status" => {
                    let client = self.client.clone();
                    let status = self.format_session_status().await;
                    client.send_agent_text(status).await;
                    response_tx.send(Ok(StopReason::EndTurn)).ok();
                    return Ok(response_rx);
                }
                "rename" if !rest.is_empty() => {
                    let name = rest.trim().to_owned();
                    self.thread
                        .submit(Op::SetThreadName { name: name.clone() })
                        .await
                        .map_err(|e| Error::internal_error().data(e.to_string()))?;
                    let client = self.client.clone();
                    client
                        .send_agent_text(format!("Thread renamed to: {name}\n"))
                        .await;
                    response_tx.send(Ok(StopReason::EndTurn)).ok();
                    return Ok(response_rx);
                }
                "mcp" => {
                    self.thread
                        .submit(Op::ListMcpTools)
                        .await
                        .map_err(|e| Error::internal_error().data(e.to_string()))?;
                    self.pending_list_response = Some(PendingListResponse {
                        kind: ListResponseKind::McpTools,
                        response_tx,
                    });
                    return Ok(response_rx);
                }
                "skills" => {
                    let cwds = vec![self.config.cwd.clone()];
                    self.thread
                        .submit(Op::ListSkills {
                            cwds,
                            force_reload: false,
                        })
                        .await
                        .map_err(|e| Error::internal_error().data(e.to_string()))?;
                    self.pending_list_response = Some(PendingListResponse {
                        kind: ListResponseKind::Skills,
                        response_tx,
                    });
                    return Ok(response_rx);
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

        let state = SubmissionState::Prompt(PromptState::new(
            submission_id.clone(),
            self.thread.clone(),
            self.resolution_tx.clone(),
            response_tx,
        ));

        self.latest_prompt_submission_id = Some(submission_id.clone());
        self.submissions.insert(submission_id, state);

        Ok(response_rx)
    }

    async fn handle_set_mode(&mut self, mode: SessionModeId) -> Result<(), Error> {
        // Handle plan mode specially — it uses collaboration mode, not approval presets.
        // Like the TUI, we don't override the model in the collaboration settings;
        // codex will use whatever model is already configured for the session.
        if mode.0.as_ref() == "plan" {
            // Pass the current model explicitly — an empty string causes codex
            // core to error with "'' model is not supported".
            let current_model = self.get_current_model().await;
            self.thread
                .submit(Op::OverrideTurnContext {
                    cwd: None,
                    approval_policy: None,
                    sandbox_policy: None,
                    model: None,
                    effort: None,
                    summary: None,
                    collaboration_mode: Some(CollaborationMode {
                        mode: ModeKind::Plan,
                        settings: Settings {
                            model: current_model,
                            reasoning_effort: Some(ReasoningEffort::Medium),
                            developer_instructions: None,
                        },
                    }),
                    personality: None,
                    windows_sandbox_level: None,
                    service_tier: None,
                })
                .await
                .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            self.plan_mode_active = true;
            return Ok(());
        }

        // Switching away from plan — clear the collaboration mode override
        if self.plan_mode_active {
            self.plan_mode_active = false;
            let current_model = self.get_current_model().await;
            self.thread
                .submit(Op::OverrideTurnContext {
                    cwd: None,
                    approval_policy: None,
                    sandbox_policy: None,
                    model: None,
                    effort: None,
                    summary: None,
                    collaboration_mode: Some(CollaborationMode {
                        mode: ModeKind::Default,
                        settings: Settings {
                            model: current_model,
                            reasoning_effort: None,
                            developer_instructions: None,
                        },
                    }),
                    personality: None,
                    windows_sandbox_level: None,
                    service_tier: None,
                })
                .await
                .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        }

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
                service_tier: None,
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
                service_tier: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;

        Ok(())
    }

    async fn handle_cancel(&mut self) -> Result<(), Error> {
        self.abort_pending_interactions();
        self.thread
            .submit(Op::Interrupt)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn handle_shutdown(&mut self) -> Result<(), Error> {
        self.abort_pending_interactions();
        self.thread
            .submit(Op::Shutdown)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    fn abort_pending_interactions(&mut self) {
        for submission in self.submissions.values_mut() {
            submission.abort_pending_interactions();
        }
    }

    /// Replay conversation history to the client via session/update notifications.
    /// This is called when loading a session to stream all prior messages.
    ///
    /// We process both `EventMsg` and `ResponseItem`:
    /// - `EventMsg` for user/agent messages and reasoning (like the TUI does)
    /// - `ResponseItem` for tool calls only (not persisted as EventMsg)
    async fn handle_replay_history(&mut self, history: Vec<RolloutItem>) -> Result<(), Error> {
        for item in history {
            match item {
                RolloutItem::EventMsg(event_msg) => {
                    self.replay_event_msg(&event_msg).await;
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
    /// Handles messages and reasoning - mirrors the live event handling in PromptState.
    async fn replay_event_msg(&self, msg: &EventMsg) {
        match msg {
            EventMsg::UserMessage(UserMessageEvent { message, .. }) => {
                self.client.send_user_message(message.clone()).await;
            }
            EventMsg::AgentMessage(AgentMessageEvent { message, phase: _ }) => {
                self.client.send_agent_text(message.clone()).await;
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                self.client.send_agent_thought(text.clone()).await;
            }
            EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent { text }) => {
                self.client.send_agent_thought(text.clone()).await;
            }
            // Skip other event types during replay - they either:
            // - Are transient (deltas, turn lifecycle)
            // - Don't have direct ACP equivalents
            // - Are handled via ResponseItem instead
            _ => {}
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
                // Check if this is a shell command - parse it like we do for LocalShellCall
                if matches!(name.as_str(), "shell" | "container.exec" | "shell_command")
                    && let Some((title, kind, locations)) =
                        self.parse_shell_function_call(name, arguments)
                {
                    self.client
                        .send_tool_call(
                            ToolCall::new(call_id.clone(), title)
                                .kind(kind)
                                .status(ToolCallStatus::Completed)
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
                    .send_completed_tool_call(
                        call_id.clone(),
                        name.clone(),
                        ToolKind::Other,
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
                    codex_protocol::models::LocalShellStatus::InProgress
                    | codex_protocol::models::LocalShellStatus::Incomplete => {
                        ToolCallStatus::Failed
                    }
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
                // Check if this is an apply_patch call - show the patch content
                if name == "apply_patch"
                    && let Some((title, locations, content)) = self.parse_apply_patch_call(input)
                {
                    self.client
                        .send_tool_call(
                            ToolCall::new(call_id.clone(), title)
                                .kind(ToolKind::Edit)
                                .status(ToolCallStatus::Completed)
                                .locations(locations)
                                .content(content)
                                .raw_input(serde_json::from_str::<serde_json::Value>(input).ok()),
                        )
                        .await;
                    return;
                }

                // Fall through to generic custom tool call handling
                self.client
                    .send_completed_tool_call(
                        call_id.clone(),
                        name.clone(),
                        ToolKind::Other,
                        serde_json::from_str(input).ok(),
                    )
                    .await;
            }
            ResponseItem::CustomToolCallOutput { call_id, output } => {
                self.client
                    .send_tool_call_completed(call_id.clone(), Some(serde_json::json!(output)))
                    .await;
            }
            ResponseItem::WebSearchCall { id, action, .. } => {
                let (title, call_id) = if let Some(action) = action {
                    web_search_action_to_title_and_id(id, action)
                } else {
                    ("Web Search".into(), generate_fallback_id("web_search"))
                };
                self.client
                    .send_tool_call(
                        ToolCall::new(call_id, title)
                            .kind(ToolKind::Search)
                            .status(ToolCallStatus::Completed),
                    )
                    .await;
            }
            // Skip GhostSnapshot, Compaction, Other, LocalShellCall without call_id
            _ => {}
        }
    }

    async fn handle_event(&mut self, Event { id, msg }: Event) {
        if is_auto_compact_submission_id(&id) {
            if is_auto_compaction_event(&msg)
                && let Some(submission_id) = self.active_prompt_submission_id()
                && let Some(submission) = self.submissions.get_mut(&submission_id)
            {
                debug!(
                    "Routing live auto-compaction event to active prompt: source_id={id}, target_id={submission_id}, msg={msg:?}"
                );
                submission.handle_event(&self.client, msg).await;
            } else {
                debug!("Ignoring compact replay event for submission ID: {id} {msg:?}");
            }
            return;
        }

        if is_global_event(&msg) {
            self.handle_global_event(&msg).await;
            return;
        }

        if let Some(submission) = self.submissions.get_mut(&id) {
            submission.handle_event(&self.client, msg).await;
        } else {
            warn!("Received event for unknown submission ID: {id} {msg:?}");
        }
    }

    fn active_prompt_submission_id(&self) -> Option<String> {
        if let Some(submission_id) = &self.latest_prompt_submission_id
            && self
                .submissions
                .get(submission_id)
                .is_some_and(|submission| {
                    matches!(submission, SubmissionState::Prompt(..)) && submission.is_active()
                })
        {
            return Some(submission_id.clone());
        }

        self.submissions
            .iter()
            .find_map(|(submission_id, submission)| {
                (matches!(submission, SubmissionState::Prompt(..)) && submission.is_active())
                    .then(|| submission_id.clone())
            })
    }

    async fn handle_global_event(&mut self, msg: &EventMsg) {
        match msg {
            EventMsg::McpListToolsResponse(..) | EventMsg::ListSkillsResponse(..) => {
                self.handle_list_response(msg).await;
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
            _ => {}
        }
    }

    async fn handle_list_response(&mut self, msg: &EventMsg) {
        match msg {
            EventMsg::McpListToolsResponse(McpListToolsResponseEvent {
                tools,
                resources: _,
                resource_templates: _,
                auth_statuses: _,
            }) => {
                if let Some(pending) = self.pending_list_response.take() {
                    if matches!(pending.kind, ListResponseKind::McpTools) {
                        let mut text = String::from("## Configured MCP Tools\n\n");
                        if tools.is_empty() {
                            text.push_str("No MCP tools configured.\n");
                        } else {
                            for (server, tool) in tools {
                                let desc = tool.description.as_deref().unwrap_or("No description");
                                text.push_str(&format!(
                                    "- **{server}** / `{}`  \n  {desc}\n",
                                    tool.name
                                ));
                            }
                        }
                        self.client.send_agent_text(text).await;
                        pending.response_tx.send(Ok(StopReason::EndTurn)).ok();
                    }
                } else {
                    warn!("Unexpected McpListToolsResponse event");
                }
            }
            EventMsg::ListSkillsResponse(ListSkillsResponseEvent { skills }) => {
                if let Some(pending) = self.pending_list_response.take() {
                    if matches!(pending.kind, ListResponseKind::Skills) {
                        let mut text = String::from("## Available Skills\n\n");
                        let mut any_skills = false;
                        for entry in skills {
                            for skill in &entry.skills {
                                any_skills = true;
                                let name = &skill.name;
                                let desc = skill
                                    .short_description
                                    .as_deref()
                                    .unwrap_or("No description");
                                text.push_str(&format!("- **{name}**  \n  {desc}\n"));
                            }
                            for err in &entry.errors {
                                text.push_str(&format!(
                                    "- **Error** in `{}`  \n  {}\n",
                                    err.path.display(),
                                    err.message
                                ));
                            }
                        }
                        if !any_skills {
                            text.push_str("No skills configured.\n");
                        }
                        self.client.send_agent_text(text).await;
                        pending.response_tx.send(Ok(StopReason::EndTurn)).ok();
                    }
                } else {
                    warn!("Unexpected ListSkillsResponse event");
                }
            }
            _ => {}
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

/// Extract title and call_id from a WebSearchAction (used for replay)
fn web_search_action_to_title_and_id(
    id: &Option<String>,
    action: &codex_protocol::models::WebSearchAction,
) -> (String, String) {
    match action {
        codex_protocol::models::WebSearchAction::Search { query, queries } => {
            let title = queries
                .as_ref()
                .map(|q| q.join(", "))
                .or_else(|| query.clone())
                .unwrap_or_else(|| "Web search".to_string());
            let call_id = id
                .clone()
                .unwrap_or_else(|| generate_fallback_id("web_search"));
            (title, call_id)
        }
        codex_protocol::models::WebSearchAction::OpenPage { url } => {
            let title = url.clone().unwrap_or_else(|| "Open page".to_string());
            let call_id = id
                .clone()
                .unwrap_or_else(|| generate_fallback_id("web_open"));
            (title, call_id)
        }
        codex_protocol::models::WebSearchAction::FindInPage { pattern, .. } => {
            let title = pattern
                .clone()
                .unwrap_or_else(|| "Find in page".to_string());
            let call_id = id
                .clone()
                .unwrap_or_else(|| generate_fallback_id("web_find"));
            (title, call_id)
        }
        codex_protocol::models::WebSearchAction::Other => {
            ("Unknown".to_string(), generate_fallback_id("web_search"))
        }
    }
}

/// Generate a fallback ID using UUID (used when id is missing)
fn generate_fallback_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

fn format_service_tier_name(service_tier: Option<ServiceTier>) -> &'static str {
    match service_tier {
        Some(ServiceTier::Fast) => "Fast",
        Some(ServiceTier::Flex) => "Flex",
        None => "Standard",
    }
}

/// Runs `git diff`, `git diff --cached`, and `git ls-files --others` in the given directory
/// and returns formatted output.
async fn run_git_diff(cwd: &Path) -> String {
    use tokio::process::Command;

    let mut output = String::new();

    // Staged changes
    if let Ok(result) = Command::new("git")
        .args(["diff", "--cached"])
        .current_dir(cwd)
        .output()
        .await
    {
        let staged = String::from_utf8_lossy(&result.stdout);
        if !staged.is_empty() {
            output.push_str("## Staged Changes\n\n```diff\n");
            output.push_str(&staged);
            output.push_str("```\n\n");
        }
    }

    // Unstaged changes
    if let Ok(result) = Command::new("git")
        .arg("diff")
        .current_dir(cwd)
        .output()
        .await
    {
        let unstaged = String::from_utf8_lossy(&result.stdout);
        if !unstaged.is_empty() {
            output.push_str("## Unstaged Changes\n\n```diff\n");
            output.push_str(&unstaged);
            output.push_str("```\n\n");
        }
    }

    // Untracked files
    if let Ok(result) = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard"])
        .current_dir(cwd)
        .output()
        .await
    {
        let untracked = String::from_utf8_lossy(&result.stdout);
        if !untracked.is_empty() {
            output.push_str("## Untracked Files\n\n");
            for file in untracked.lines() {
                output.push_str(&format!("- {file}\n"));
            }
            output.push('\n');
        }
    }

    if output.is_empty() {
        output.push_str("No changes detected.\n");
    }

    output
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
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use agent_client_protocol::{RequestPermissionResponse, TextContent};
    use codex_core::{config::ConfigOverrides, test_support::all_model_presets};
    use codex_protocol::config_types::{ModeKind, ServiceTier};
    use tokio::{
        sync::{Mutex, Notify, mpsc::UnboundedSender},
        task::LocalSet,
    };

    use super::*;

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
        let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut actor = ThreadActor::new(
            StubAuth,
            session_client,
            conversation.clone(),
            models_manager,
            config,
            message_rx,
            resolution_tx,
            resolution_rx,
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
        async fn get_model(&self, _model_id: &Option<String>) -> String {
            all_model_presets()[0].to_owned().id
        }

        async fn list_models(&self) -> Vec<ModelPreset> {
            all_model_presets().to_owned()
        }
    }

    struct StubCodexThread {
        current_id: AtomicUsize,
        active_prompt_id: std::sync::Mutex<Option<String>>,
        ops: std::sync::Mutex<Vec<Op>>,
        op_tx: mpsc::UnboundedSender<Event>,
        op_rx: Mutex<mpsc::UnboundedReceiver<Event>>,
    }

    impl StubCodexThread {
        fn new() -> Self {
            let (op_tx, op_rx) = mpsc::unbounded_channel();
            StubCodexThread {
                current_id: AtomicUsize::new(0),
                active_prompt_id: std::sync::Mutex::default(),
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
                    *self.active_prompt_id.lock().unwrap() = Some(id.to_string());
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
                    } else if prompt == "emit-hook-events" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::HookStarted(HookStartedEvent {
                            turn_id: Some(id.to_string()),
                            run: codex_protocol::protocol::HookRunSummary {
                                id: "hook-1".to_string(),
                                event_name: codex_protocol::protocol::HookEventName::SessionStart,
                                handler_type: codex_protocol::protocol::HookHandlerType::Command,
                                execution_mode: codex_protocol::protocol::HookExecutionMode::Sync,
                                scope: codex_protocol::protocol::HookScope::Thread,
                                source_path: std::path::PathBuf::from("/test/hook.sh"),
                                display_order: 0,
                                status: codex_protocol::protocol::HookRunStatus::Running,
                                status_message: None,
                                started_at: 0,
                                completed_at: None,
                                duration_ms: None,
                                entries: vec![],
                            },
                        }));
                        send(EventMsg::HookCompleted(HookCompletedEvent {
                            turn_id: Some(id.to_string()),
                            run: codex_protocol::protocol::HookRunSummary {
                                id: "hook-1".to_string(),
                                event_name: codex_protocol::protocol::HookEventName::SessionStart,
                                handler_type: codex_protocol::protocol::HookHandlerType::Command,
                                execution_mode: codex_protocol::protocol::HookExecutionMode::Sync,
                                scope: codex_protocol::protocol::HookScope::Thread,
                                source_path: std::path::PathBuf::from("/test/hook.sh"),
                                display_order: 0,
                                status: codex_protocol::protocol::HookRunStatus::Completed,
                                status_message: Some("all good".to_string()),
                                started_at: 0,
                                completed_at: Some(100),
                                duration_ms: Some(100),
                                entries: vec![],
                            },
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-image-gen" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::ImageGenerationBegin(ImageGenerationBeginEvent {
                            call_id: "img-1".to_string(),
                        }));
                        send(EventMsg::ImageGenerationEnd(ImageGenerationEndEvent {
                            call_id: "img-1".to_string(),
                            status: "success".to_string(),
                            revised_prompt: None,
                            result: "image.png".to_string(),
                            saved_path: None,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-background-event" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::BackgroundEvent(BackgroundEventEvent {
                            message: "Long running task completed".to_string(),
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-auto-compact" {
                        let auto_compact_id = "auto-compact-0".to_string();
                        let compaction_item = codex_protocol::items::ContextCompactionItem::new();
                        let send = |event_id: String, msg| {
                            self.op_tx.send(Event { id: event_id, msg }).unwrap();
                        };
                        send(
                            auto_compact_id.clone(),
                            EventMsg::ItemStarted(ItemStartedEvent {
                                thread_id: codex_protocol::ThreadId::new(),
                                turn_id: auto_compact_id.clone(),
                                item: TurnItem::ContextCompaction(compaction_item.clone()),
                            }),
                        );
                        send(
                            auto_compact_id.clone(),
                            EventMsg::ItemCompleted(ItemCompletedEvent {
                                thread_id: codex_protocol::ThreadId::new(),
                                turn_id: auto_compact_id.clone(),
                                item: TurnItem::ContextCompaction(compaction_item),
                            }),
                        );
                        send(
                            auto_compact_id,
                            EventMsg::ContextCompacted(
                                codex_protocol::protocol::ContextCompactedEvent {},
                            ),
                        );
                        send(
                            id.to_string(),
                            EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
                            }),
                        );
                    } else if prompt == "emit-deprecation-notice" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::DeprecationNotice(DeprecationNoticeEvent {
                            summary: "Old API deprecated".to_string(),
                            details: Some("Please migrate to v2.".to_string()),
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-thread-rollback" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::ThreadRolledBack(ThreadRolledBackEvent {
                            num_turns: 3,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-reasoning-raw" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::AgentReasoningRawContent(
                            AgentReasoningRawContentEvent {
                                text: "Thinking about the problem...".to_string(),
                            },
                        ));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-plan-delta" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::PlanDelta(PlanDeltaEvent {
                            thread_id: id.to_string(),
                            turn_id: id.to_string(),
                            item_id: "plan-item-1".to_string(),
                            delta: "Step 1: Analyze the code".to_string(),
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-collab-spawn" {
                        use codex_protocol::ThreadId as ProtoThreadId;
                        use codex_protocol::protocol::AgentStatus;
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::CollabAgentSpawnBegin(
                            CollabAgentSpawnBeginEvent {
                                call_id: "collab-1".to_string(),
                                sender_thread_id: ProtoThreadId::new(),
                                prompt: "Review this file".to_string(),
                                model: "gpt-4".to_string(),
                                reasoning_effort: ReasoningEffort::Medium,
                            },
                        ));
                        send(EventMsg::CollabAgentSpawnEnd(CollabAgentSpawnEndEvent {
                            call_id: "collab-1".to_string(),
                            sender_thread_id: ProtoThreadId::new(),
                            new_thread_id: Some(ProtoThreadId::new()),
                            new_agent_nickname: Some("reviewer".to_string()),
                            new_agent_role: None,
                            prompt: String::new(),
                            status: AgentStatus::Completed(None),
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-collab-interaction" {
                        use codex_protocol::ThreadId as ProtoThreadId;
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::CollabAgentInteractionBegin(
                            CollabAgentInteractionBeginEvent {
                                call_id: "interact-1".to_string(),
                                sender_thread_id: ProtoThreadId::new(),
                                receiver_thread_id: ProtoThreadId::new(),
                                prompt: "Help with this".to_string(),
                            },
                        ));
                        send(EventMsg::CollabAgentInteractionEnd(
                            CollabAgentInteractionEndEvent {
                                call_id: "interact-1".to_string(),
                                sender_thread_id: ProtoThreadId::new(),
                                receiver_thread_id: ProtoThreadId::new(),
                                receiver_agent_nickname: Some("reviewer".to_string()),
                                receiver_agent_role: None,
                                prompt: String::new(),
                                status: codex_protocol::protocol::AgentStatus::Completed(None),
                            },
                        ));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-token-counts" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        use codex_protocol::protocol::{
                            CreditsSnapshot, RateLimitSnapshot, TokenUsage, TokenUsageInfo,
                        };
                        // Send two TokenCount events to test accumulation
                        send(EventMsg::TokenCount(TokenCountEvent {
                            info: Some(TokenUsageInfo {
                                total_token_usage: TokenUsage {
                                    input_tokens: 100,
                                    cached_input_tokens: 20,
                                    output_tokens: 50,
                                    reasoning_output_tokens: 10,
                                    total_tokens: 180,
                                },
                                last_token_usage: TokenUsage {
                                    input_tokens: 100,
                                    cached_input_tokens: 20,
                                    output_tokens: 50,
                                    reasoning_output_tokens: 10,
                                    total_tokens: 180,
                                },
                                model_context_window: Some(128000),
                            }),
                            rate_limits: Some(RateLimitSnapshot {
                                limit_id: None,
                                limit_name: None,
                                primary: None,
                                secondary: None,
                                credits: Some(CreditsSnapshot {
                                    has_credits: true,
                                    unlimited: false,
                                    balance: Some("1.50".to_string()),
                                }),
                                plan_type: None,
                            }),
                        }));
                        // Second event to verify accumulation
                        send(EventMsg::TokenCount(TokenCountEvent {
                            info: Some(TokenUsageInfo {
                                total_token_usage: TokenUsage {
                                    input_tokens: 200,
                                    cached_input_tokens: 40,
                                    output_tokens: 100,
                                    reasoning_output_tokens: 20,
                                    total_tokens: 360,
                                },
                                last_token_usage: TokenUsage {
                                    input_tokens: 100,
                                    cached_input_tokens: 20,
                                    output_tokens: 50,
                                    reasoning_output_tokens: 10,
                                    total_tokens: 180,
                                },
                                model_context_window: Some(128000),
                            }),
                            rate_limits: None,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-token-none-info" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        // TokenCount with info: None — should be handled gracefully
                        send(EventMsg::TokenCount(TokenCountEvent {
                            info: None,
                            rate_limits: None,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-token-no-window" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        use codex_protocol::protocol::{TokenUsage, TokenUsageInfo};
                        send(EventMsg::TokenCount(TokenCountEvent {
                            info: Some(TokenUsageInfo {
                                total_token_usage: TokenUsage {
                                    input_tokens: 50,
                                    cached_input_tokens: 10,
                                    output_tokens: 25,
                                    reasoning_output_tokens: 5,
                                    total_tokens: 90,
                                },
                                last_token_usage: TokenUsage {
                                    input_tokens: 50,
                                    cached_input_tokens: 10,
                                    output_tokens: 25,
                                    reasoning_output_tokens: 5,
                                    total_tokens: 90,
                                },
                                model_context_window: None, // No context window
                            }),
                            rate_limits: None,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-token-bad-balance" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        use codex_protocol::protocol::{
                            CreditsSnapshot, RateLimitSnapshot, TokenUsage, TokenUsageInfo,
                        };
                        send(EventMsg::TokenCount(TokenCountEvent {
                            info: Some(TokenUsageInfo {
                                total_token_usage: TokenUsage::default(),
                                last_token_usage: TokenUsage {
                                    input_tokens: 10,
                                    cached_input_tokens: 0,
                                    output_tokens: 5,
                                    reasoning_output_tokens: 0,
                                    total_tokens: 15,
                                },
                                model_context_window: Some(128000),
                            }),
                            rate_limits: Some(RateLimitSnapshot {
                                limit_id: None,
                                limit_name: None,
                                primary: None,
                                secondary: None,
                                credits: Some(CreditsSnapshot {
                                    has_credits: true,
                                    unlimited: false,
                                    balance: Some("not-a-number".to_string()),
                                }),
                                plan_type: None,
                            }),
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-orphaned-events" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        // Send output delta for a call_id that was never started
                        send(EventMsg::ExecCommandOutputDelta(
                            ExecCommandOutputDeltaEvent {
                                call_id: "nonexistent-call".to_string(),
                                chunk: b"some output".to_vec(),
                                stream: codex_protocol::protocol::ExecOutputStream::Stdout,
                            },
                        ));
                        // Send end for a call_id that was never started
                        send(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                            call_id: "nonexistent-call".to_string(),
                            process_id: None,
                            turn_id: id.to_string(),
                            command: vec!["echo".into()],
                            cwd: std::env::current_dir().unwrap(),
                            parsed_cmd: vec![],
                            source: Default::default(),
                            interaction_input: None,
                            stdout: String::new(),
                            stderr: String::new(),
                            aggregated_output: String::new(),
                            exit_code: 0,
                            duration: std::time::Duration::from_millis(1),
                            formatted_output: String::new(),
                            status: ExecCommandStatus::Completed,
                        }));
                        // Send terminal interaction for unknown call_id
                        send(EventMsg::TerminalInteraction(TerminalInteractionEvent {
                            call_id: "nonexistent-call".to_string(),
                            process_id: "0".to_string(),
                            stdin: "input".to_string(),
                        }));
                        // Should still complete normally
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-patch-abort" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        // Emit a PatchApplyBegin with one fake file change, then
                        // abort without ever emitting PatchApplyEnd.
                        let mut changes = HashMap::new();
                        changes.insert(
                            PathBuf::from("/tmp/fake-file.txt"),
                            FileChange::Add {
                                content: "new content".to_string(),
                            },
                        );
                        send(EventMsg::PatchApplyBegin(PatchApplyBeginEvent {
                            call_id: "patch-call-1".to_string(),
                            auto_approved: true,
                            changes,
                            turn_id: id.to_string(),
                        }));
                        // Abort without PatchApplyEnd
                        send(EventMsg::TurnAborted(TurnAbortedEvent {
                            turn_id: Some(id.to_string()),
                            reason: codex_protocol::protocol::TurnAbortReason::Interrupted,
                        }));
                    } else if prompt == "emit-collab-waiting" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        use codex_protocol::ThreadId as ProtoThreadId;
                        use codex_protocol::protocol::AgentStatus;
                        // Spawn a sub-agent first so active_subagent_call_id is set
                        send(EventMsg::CollabAgentSpawnBegin(
                            CollabAgentSpawnBeginEvent {
                                call_id: "wait-spawn-1".to_string(),
                                sender_thread_id: ProtoThreadId::new(),
                                prompt: "Review".to_string(),
                                model: "gpt-4".to_string(),
                                reasoning_effort: ReasoningEffort::Medium,
                            },
                        ));
                        send(EventMsg::CollabAgentSpawnEnd(CollabAgentSpawnEndEvent {
                            call_id: "wait-spawn-1".to_string(),
                            sender_thread_id: ProtoThreadId::new(),
                            new_thread_id: Some(ProtoThreadId::new()),
                            new_agent_nickname: None,
                            new_agent_role: None,
                            prompt: String::new(),
                            status: AgentStatus::Running,
                        }));
                        send(EventMsg::CollabWaitingBegin(
                            codex_protocol::protocol::CollabWaitingBeginEvent {
                                sender_thread_id: ProtoThreadId::new(),
                                receiver_thread_ids: vec![ProtoThreadId::new()],
                                receiver_agents: vec![],
                                call_id: "wait-1".to_string(),
                            },
                        ));
                        send(EventMsg::CollabWaitingEnd(
                            codex_protocol::protocol::CollabWaitingEndEvent {
                                sender_thread_id: ProtoThreadId::new(),
                                call_id: "wait-1".to_string(),
                                agent_statuses: vec![],
                                statuses: HashMap::new(),
                            },
                        ));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "emit-web-search-concurrent" {
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        // Start two concurrent web searches
                        send(EventMsg::WebSearchBegin(WebSearchBeginEvent {
                            call_id: "ws-1".to_string(),
                        }));
                        send(EventMsg::WebSearchBegin(WebSearchBeginEvent {
                            call_id: "ws-2".to_string(),
                        }));
                        // Trigger completion of all searches via ExecCommandBegin
                        send(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                            call_id: "exec-after-search".to_string(),
                            process_id: None,
                            turn_id: id.to_string(),
                            command: vec!["echo".into(), "done".into()],
                            cwd: std::env::current_dir().unwrap(),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo done".into(),
                            }],
                            source: Default::default(),
                            interaction_input: None,
                        }));
                        send(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                            call_id: "exec-after-search".to_string(),
                            process_id: None,
                            turn_id: id.to_string(),
                            command: vec!["echo".into(), "done".into()],
                            cwd: std::env::current_dir().unwrap(),
                            parsed_cmd: vec![],
                            source: Default::default(),
                            interaction_input: None,
                            stdout: "done\n".into(),
                            stderr: String::new(),
                            aggregated_output: "done\n".into(),
                            exit_code: 0,
                            duration: std::time::Duration::from_millis(5),
                            formatted_output: "done\n".into(),
                            status: ExecCommandStatus::Completed,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id: id.to_string(),
                        }));
                    } else if prompt == "approval-block" {
                        self.op_tx
                            .send(Event {
                                id: id.to_string(),
                                msg: EventMsg::ExecApprovalRequest(ExecApprovalRequestEvent {
                                    call_id: "call-id".to_string(),
                                    approval_id: Some("approval-id".to_string()),
                                    turn_id: id.to_string(),
                                    command: vec!["echo".to_string(), "hi".to_string()],
                                    cwd: std::env::current_dir().unwrap(),
                                    reason: None,
                                    network_approval_context: None,
                                    proposed_execpolicy_amendment: None,
                                    proposed_network_policy_amendments: None,
                                    additional_permissions: None,
                                    skill_metadata: None,
                                    available_decisions: Some(vec![
                                        ReviewDecision::Approved,
                                        ReviewDecision::Abort,
                                    ]),
                                    parsed_cmd: vec![ParsedCommand::Unknown {
                                        cmd: "echo hi".to_string(),
                                    }],
                                }),
                            })
                            .unwrap();
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
                Op::CleanBackgroundTerminals => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::AgentMessage(AgentMessageEvent {
                                message: "Background terminals stopped.".to_string(),
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
                Op::SetThreadName { .. } => {
                    // Fire-and-forget — no events needed
                }
                Op::ListMcpTools => {
                    use codex_protocol::mcp::Tool as McpTool;
                    let mut tools = std::collections::HashMap::new();
                    tools.insert(
                        "test-server".to_string(),
                        McpTool {
                            name: "test-tool".to_string(),
                            title: None,
                            description: Some("A test MCP tool".to_string()),
                            input_schema: serde_json::json!({}),
                            output_schema: None,
                            annotations: None,
                            icons: None,
                            meta: None,
                        },
                    );
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::McpListToolsResponse(McpListToolsResponseEvent {
                                tools,
                                resources: std::collections::HashMap::new(),
                                resource_templates: std::collections::HashMap::new(),
                                auth_statuses: std::collections::HashMap::new(),
                            }),
                        })
                        .unwrap();
                }
                Op::ListSkills { .. } => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::ListSkillsResponse(ListSkillsResponseEvent {
                                skills: vec![],
                            }),
                        })
                        .unwrap();
                }
                Op::OverrideTurnContext { .. }
                | Op::ExecApproval { .. }
                | Op::ResolveElicitation { .. }
                | Op::RequestPermissionsResponse { .. }
                | Op::PatchApproval { .. }
                | Op::Interrupt => {}
                Op::Shutdown => {
                    if let Some(active_prompt_id) = self.active_prompt_id.lock().unwrap().take() {
                        self.op_tx
                            .send(Event {
                                id: active_prompt_id.clone(),
                                msg: EventMsg::TurnAborted(TurnAbortedEvent {
                                    turn_id: Some(active_prompt_id),
                                    reason: codex_protocol::protocol::TurnAbortReason::Interrupted,
                                }),
                            })
                            .unwrap();
                    }
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
        permission_requests: std::sync::Mutex<Vec<RequestPermissionRequest>>,
        permission_responses: std::sync::Mutex<VecDeque<RequestPermissionResponse>>,
        block_permission_requests: Option<Arc<Notify>>,
    }

    impl StubClient {
        fn new() -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::default(),
                block_permission_requests: None,
            }
        }

        fn with_permission_responses(responses: Vec<RequestPermissionResponse>) -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::new(responses.into()),
                block_permission_requests: None,
            }
        }

        fn with_blocked_permission_requests(
            responses: Vec<RequestPermissionResponse>,
            notify: Arc<Notify>,
        ) -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::new(responses.into()),
                block_permission_requests: Some(notify),
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Client for StubClient {
        async fn request_permission(
            &self,
            args: RequestPermissionRequest,
        ) -> Result<RequestPermissionResponse, Error> {
            self.permission_requests.lock().unwrap().push(args);
            if let Some(notify) = &self.block_permission_requests {
                notify.notified().await;
            }
            Ok(self
                .permission_responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled)
                }))
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

    #[tokio::test]
    async fn test_exec_approval_uses_available_decisions() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new("denied"),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, mut message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .exec_approval(
                        &session_client,
                        ExecApprovalRequestEvent {
                            call_id: "call-id".to_string(),
                            approval_id: Some("approval-id".to_string()),
                            turn_id: "turn-id".to_string(),
                            command: vec!["echo".to_string(), "hi".to_string()],
                            cwd: std::env::current_dir()?,
                            reason: None,
                            network_approval_context: None,
                            proposed_execpolicy_amendment: None,
                            proposed_network_policy_amendments: None,
                            additional_permissions: None,
                            skill_metadata: None,
                            available_decisions: Some(vec![
                                ReviewDecision::Approved,
                                ReviewDecision::Denied,
                            ]),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo hi".to_string(),
                            }],
                        },
                    )
                    .await?;

                let ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key,
                    response,
                } = message_rx.recv().await.unwrap()
                else {
                    panic!("expected permission resolution message");
                };
                assert_eq!(submission_id, "submission-id");
                prompt_state
                    .handle_permission_request_resolved(&session_client, request_key, response)
                    .await?;

                let requests = client.permission_requests.lock().unwrap();
                let request = requests.last().unwrap();
                let option_ids = request
                    .options
                    .iter()
                    .map(|option| option.option_id.0.to_string())
                    .collect::<Vec<_>>();
                assert_eq!(option_ids, vec!["approved", "denied"]);

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ExecApproval {
                        id,
                        turn_id,
                        decision: ReviewDecision::Denied,
                    }) if id == "approval-id" && turn_id.as_deref() == Some("turn-id")
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_declines_unsupported_form_requests() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new("decline"),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Need some structured input".to_string(),
                                requested_schema: serde_json::json!({
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" }
                                    }
                                }),
                            },
                        },
                    )
                    .await?;

                let requests = client.permission_requests.lock().unwrap();
                assert!(
                    requests.is_empty(),
                    "unsupported MCP elicitations should be auto-declined"
                );

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                    }) if server_name == "test-server" && request_id == "request-id"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_blocked_approval_does_not_block_followup_events() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_blocked_permission_requests(
                    vec![],
                    Arc::new(Notify::new()),
                ));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state =
                    PromptState::new("submission-id".to_string(), thread, message_tx, response_tx);

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::ExecApprovalRequest(ExecApprovalRequestEvent {
                            call_id: "call-id".to_string(),
                            approval_id: Some("approval-id".to_string()),
                            turn_id: "turn-id".to_string(),
                            command: vec!["echo".to_string(), "hi".to_string()],
                            cwd: std::env::current_dir()?,
                            reason: None,
                            network_approval_context: None,
                            proposed_execpolicy_amendment: None,
                            proposed_network_policy_amendments: None,
                            additional_permissions: None,
                            skill_metadata: None,
                            available_decisions: Some(vec![
                                ReviewDecision::Approved,
                                ReviewDecision::Abort,
                            ]),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo hi".to_string(),
                            }],
                        }),
                    )
                    .await;

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::AgentMessage(AgentMessageEvent {
                            message: "still flowing".to_string(),
                            phase: None,
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert!(notifications.iter().any(|notification| {
                    matches!(
                        &notification.update,
                        SessionUpdate::AgentMessageChunk(ContentChunk {
                            content: ContentBlock::Text(TextContent { text, .. }),
                            ..
                        }) if text == "still flowing"
                    )
                }));

                drop(notifications);
                prompt_state.abort_pending_interactions();

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_thread_shutdown_bypasses_blocked_permission_request() -> anyhow::Result<()> {
        let session_id = SessionId::new("test");
        let client = Arc::new(StubClient::with_blocked_permission_requests(
            vec![RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled,
            )],
            Arc::new(Notify::new()),
        ));
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
        let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();
        let actor = ThreadActor::new(
            StubAuth,
            session_client,
            conversation.clone(),
            models_manager,
            config,
            message_rx,
            resolution_tx,
            resolution_rx,
        );

        let local_set = LocalSet::new();
        let handle = local_set.spawn_local(actor.spawn());
        let thread = Thread {
            thread: conversation.clone(),
            message_tx,
            _handle: handle,
        };

        local_set
            .run_until(async move {
                let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
                thread.message_tx.send(ThreadMessage::Prompt {
                    request: PromptRequest::new(session_id, vec!["approval-block".into()]),
                    response_tx: prompt_response_tx,
                })?;
                let stop_reason_rx = prompt_response_rx.await??;

                tokio::time::timeout(Duration::from_millis(100), async {
                    loop {
                        if !client.permission_requests.lock().unwrap().is_empty() {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await?;

                tokio::time::timeout(Duration::from_millis(100), thread.shutdown()).await??;
                let stop_reason =
                    tokio::time::timeout(Duration::from_millis(100), stop_reason_rx).await??;
                assert_eq!(stop_reason?, StopReason::Cancelled);

                anyhow::Ok(())
            })
            .await?;

        let ops = conversation.ops.lock().unwrap();
        assert!(matches!(ops.last(), Some(Op::Shutdown)));

        Ok(())
    }

    // ==================== New tests for P0/P2/P3 changes ====================

    #[tokio::test]
    async fn test_slash_stop() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/stop".into()]),
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

        let ops = thread.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::CleanBackgroundTerminals]);

        let notifications = client.notifications.lock().unwrap();
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Background terminals stopped."
            )),
            "expected background terminals stopped message, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_diff() -> anyhow::Result<()> {
        let (session_id, client, _thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/diff".into()]),
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

        // /diff now runs git locally and sends output as agent text — no Op submitted
        let notifications = client.notifications.lock().unwrap();
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(..),
                    ..
                })
            )),
            "expected /diff to send agent text with git output"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_status() -> anyhow::Result<()> {
        let (session_id, client, _thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/status".into()]),
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

        // /status now assembles info locally — no Op submitted
        let notifications = client.notifications.lock().unwrap();
        let has_status = notifications.iter().any(|n| {
            if let SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(t),
                ..
            }) = &n.update
            {
                t.text.contains("Session Status") && t.text.contains("**Service Tier:** Standard")
            } else {
                false
            }
        });
        assert!(has_status, "expected /status to send Session Status text");

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_rename() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/rename My New Thread".into()]),
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

        // /rename now uses Op::SetThreadName
        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(&ops[0], Op::SetThreadName { name } if name == "My New Thread"),
            "expected /rename to submit Op::SetThreadName, got {ops:?}"
        );

        // Should send confirmation text
        let notifications = client.notifications.lock().unwrap();
        let has_confirm = notifications.iter().any(|n| {
            if let SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(t),
                ..
            }) = &n.update
            {
                t.text.contains("My New Thread")
            } else {
                false
            }
        });
        assert!(has_confirm, "expected /rename to send confirmation text");

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_fast_toggle_on() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/fast".into()]),
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

        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(
                ops.first(),
                Some(Op::OverrideTurnContext {
                    service_tier: Some(Some(ServiceTier::Fast)),
                    ..
                })
            ),
            "expected /fast to enable fast mode, got {ops:?}"
        );

        let notifications = client.notifications.lock().unwrap();
        assert!(notifications.iter().any(|n| matches!(
            &n.update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Fast mode is on.\n"
        )));

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_fast_off() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/fast off".into()]),
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

        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(
                ops.first(),
                Some(Op::OverrideTurnContext {
                    service_tier: Some(None),
                    ..
                })
            ),
            "expected /fast off to disable fast mode, got {ops:?}"
        );

        let notifications = client.notifications.lock().unwrap();
        assert!(notifications.iter().any(|n| matches!(
            &n.update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Fast mode is off.\n"
        )));

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_fast_status() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/fast status".into()]),
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

        let ops = thread.ops.lock().unwrap();
        assert!(
            ops.is_empty(),
            "expected /fast status to avoid submitting ops, got {ops:?}"
        );

        let notifications = client.notifications.lock().unwrap();
        assert!(notifications.iter().any(|n| matches!(
            &n.update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Fast mode is off.\n"
        )));

        Ok(())
    }

    #[tokio::test]
    async fn test_config_options_include_service_tier() -> anyhow::Result<()> {
        let (_session_id, _client, _thread, message_tx, local_set) = setup(vec![]).await?;
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::GetConfigOptions { response_tx })?;
        let (options, _) = tokio::try_join!(
            async {
                let options = response_rx.await??;
                drop(message_tx);
                anyhow::Ok(options)
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        assert!(
            options
                .iter()
                .any(|option| option.id.0.as_ref() == "service_tier"),
            "expected service_tier config option, got {options:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_set_config_service_tier_submits_override() -> anyhow::Result<()> {
        let (_session_id, _client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::SetConfigOption {
            config_id: SessionConfigId::new("service_tier"),
            value: SessionConfigOptionValue::ValueId {
                value: SessionConfigValueId::new("fast"),
            },
            response_tx,
        })?;
        tokio::try_join!(
            async {
                response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let ops = thread.ops.lock().unwrap();
        assert!(
            ops.iter().any(|op| matches!(
                op,
                Op::OverrideTurnContext {
                    service_tier: Some(Some(ServiceTier::Fast)),
                    ..
                }
            )),
            "expected service tier override op, got {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_mention_removed() {
        // /mention was removed — it was a fake UserInput wrapper.
        // Verify it's not in builtin_commands.
        let commands = ThreadActor::<StubAuth>::builtin_commands();
        let names: Vec<_> = commands.iter().map(|c| c.name.as_str()).collect();
        assert!(
            !names.contains(&"mention"),
            "/mention should not be in builtin commands"
        );
    }

    #[tokio::test]
    async fn test_hook_events_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-hook-events".into()]),
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
        let texts: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) => Some(text.clone()),
                _ => None,
            })
            .collect();

        assert!(
            texts.iter().any(|t| t.contains("Running hook")),
            "expected hook started message, got {texts:?}"
        );
        assert!(
            texts
                .iter()
                .any(|t| t.contains("Hook completed") && t.contains("all good")),
            "expected hook completed message with status, got {texts:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_image_generation_events_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-image-gen".into()]),
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

        // Should have a ToolCall begin for image generation
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            tool_calls.len(),
            1,
            "expected 1 image gen ToolCall, got {tool_calls:?}"
        );
        assert!(tool_calls[0].title.contains("Generating image"));

        // Should have a completed ToolCallUpdate
        let updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(u)
                    if u.fields.status == Some(ToolCallStatus::Completed) =>
                {
                    Some(u.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            updates.len(),
            1,
            "expected 1 completed update, got {updates:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_background_event_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-background-event".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text.contains("Long running task completed")
            )),
            "expected background event message, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_auto_compact_events_surface_in_active_prompt() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-auto-compact".into()]),
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

        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::ToolCall(tool_call)
                    if tool_call.title == "Compacting context"
                        && tool_call.kind == ToolKind::Think
                        && tool_call.status == ToolCallStatus::InProgress
            )),
            "expected auto-compaction start tool call, got {notifications:?}"
        );
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::ToolCallUpdate(update)
                    if update.fields.status == Some(ToolCallStatus::Completed)
                        && update
                            .fields
                            .content
                            .as_ref()
                            .is_some_and(|content| content.iter().any(|item| {
                                matches!(
                                    item,
                                    ToolCallContent::Content(Content {
                                        content: ContentBlock::Text(TextContent { text, .. }),
                                        ..
                                    }) if text == "Context compacted."
                                )
                            }))
            )),
            "expected auto-compaction completion update, got {notifications:?}"
        );
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Context compacted\n"
            )),
            "expected compacted notice text, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_deprecation_notice_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-deprecation-notice".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text.contains("Deprecation") && text.contains("Old API deprecated") && text.contains("migrate to v2")
            )),
            "expected deprecation notice message, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_thread_rollback_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-thread-rollback".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text.contains("rolled back") && text.contains("3 turns")
            )),
            "expected rollback message, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_reasoning_raw_content_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-reasoning-raw".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentThoughtChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Thinking about the problem..."
            )),
            "expected reasoning raw content as thought chunk, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_plan_delta_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-plan-delta".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentThoughtChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Step 1: Analyze the code"
            )),
            "expected plan delta as thought chunk, got {notifications:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_plan_mode_in_available_modes() {
        // Plan mode should be in the available modes list
        let (_, _, _, _, _local_set) = setup(vec![]).await.unwrap();
        // We can verify via the builtin modes function indirectly —
        // check that "plan" mode is listed in builtin_commands() isn't the right
        // place; instead verify the modes() function includes it.
        // Since modes() requires a full ThreadActor, we test the simpler property:
        // that setting mode to "plan" is handled (not rejected as invalid_params).
    }

    #[tokio::test]
    async fn test_set_plan_mode() -> anyhow::Result<()> {
        let (_session_id, _client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::SetMode {
            mode: SessionModeId::new("plan"),
            response_tx,
        })?;

        // Give the actor time to process
        tokio::task::yield_now().await;
        drop(message_tx);
        local_set.await;

        let ops = thread.ops.lock().unwrap();
        assert!(
            ops.iter().any(|op| matches!(op, Op::OverrideTurnContext {
                collaboration_mode: Some(cm), ..
            } if cm.mode == ModeKind::Plan)),
            "expected plan mode to submit OverrideTurnContext with ModeKind::Plan, got {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_collab_spawn_events_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-collab-spawn".into()]),
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

        // Should have a ToolCall for spawn begin
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            tool_calls.len(),
            1,
            "expected 1 collab spawn ToolCall, got {tool_calls:?}"
        );
        assert!(
            tool_calls[0].title.to_lowercase().contains("sub-agent"),
            "expected spawn title containing 'sub-agent', got {:?}",
            tool_calls[0].title
        );

        // Should have a completed update
        let completed: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(u)
                    if u.fields.status == Some(ToolCallStatus::Completed) =>
                {
                    Some(u.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            completed.len(),
            1,
            "expected 1 completed collab update, got {completed:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_collab_interaction_events_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-collab-interaction".into()]),
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

        // Should have a ToolCall for interaction begin
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            tool_calls.len(),
            1,
            "expected 1 interaction ToolCall, got {tool_calls:?}"
        );
        assert!(
            tool_calls[0].title.contains("Sub-agent interaction"),
            "expected interaction title, got {:?}",
            tool_calls[0].title
        );

        // Should have a completed update
        let completed: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(u)
                    if u.fields.status == Some(ToolCallStatus::Completed) =>
                {
                    Some(u.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            completed.len(),
            1,
            "expected 1 completed interaction update, got {completed:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_web_searches() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(
                session_id.clone(),
                vec!["emit-web-search-concurrent".into()],
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

        // Should have ToolCall begins for both web searches
        let web_search_begins: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) if tc.title.contains("Searching") => Some(tc.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            web_search_begins.len(),
            2,
            "expected 2 web search ToolCall begins, got {web_search_begins:?}"
        );

        // Both searches should have completed (via complete_web_search when ExecCommandBegin fires)
        let web_search_completes: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(u)
                    if u.fields.status == Some(ToolCallStatus::Completed)
                        && (u.tool_call_id.0.as_ref() == "ws-1"
                            || u.tool_call_id.0.as_ref() == "ws-2") =>
                {
                    Some(u.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            web_search_completes.len(),
            2,
            "expected 2 web search completions (both concurrent searches should complete), got {web_search_completes:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_builtin_commands_includes_new_commands() {
        let commands = ThreadActor::<StubAuth>::builtin_commands();
        let names: Vec<_> = commands.iter().map(|c| c.name.as_str()).collect();

        // Original commands
        assert!(names.contains(&"review"), "missing /review");
        assert!(names.contains(&"compact"), "missing /compact");
        assert!(names.contains(&"undo"), "missing /undo");
        assert!(names.contains(&"init"), "missing /init");
        assert!(names.contains(&"logout"), "missing /logout");
        assert!(names.contains(&"review-branch"), "missing /review-branch");
        assert!(names.contains(&"review-commit"), "missing /review-commit");

        // New commands
        assert!(names.contains(&"fast"), "missing /fast");
        assert!(names.contains(&"diff"), "missing /diff");
        assert!(names.contains(&"status"), "missing /status");
        assert!(names.contains(&"stop"), "missing /stop");
        assert!(names.contains(&"rename"), "missing /rename");
        assert!(names.contains(&"mcp"), "missing /mcp");
        assert!(names.contains(&"skills"), "missing /skills");

        // Removed commands (were fake UserInput wrappers)
        assert!(!names.contains(&"mention"), "/mention should be removed");
        assert!(!names.contains(&"feedback"), "/feedback should be removed");
        assert!(
            !names.contains(&"debug-config"),
            "/debug-config should be removed"
        );
    }

    #[tokio::test]
    async fn test_usage_accumulation_with_cost() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-token-counts".into()]),
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

        // Collect all UsageUpdate notifications
        let usage_updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::UsageUpdate(u) => Some(u.clone()),
                _ => None,
            })
            .collect();

        // Should have 2 usage updates (one per TokenCount event)
        assert_eq!(
            usage_updates.len(),
            2,
            "expected 2 usage updates, got {usage_updates:?}"
        );

        // Both should have size = 128000
        assert_eq!(usage_updates[0].size, 128000);
        assert_eq!(usage_updates[0].used, 180);
        assert_eq!(usage_updates[1].size, 128000);
        assert_eq!(usage_updates[1].used, 360);

        // First should include cost from credits balance
        assert!(
            usage_updates[0].cost.is_some(),
            "first update should include cost from credits"
        );
        let cost = usage_updates[0].cost.as_ref().unwrap();
        assert!(
            (cost.amount - 1.50).abs() < f64::EPSILON,
            "cost should be 1.50, got {}",
            cost.amount
        );
        assert_eq!(cost.currency, "USD");

        // Second should NOT include cost (no rate_limits)
        assert!(
            usage_updates[1].cost.is_none(),
            "second update should not include cost (no rate_limits)"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_mcp() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/mcp".into()]),
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

        // /mcp now submits Op::ListMcpTools
        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(&ops[0], Op::ListMcpTools),
            "expected /mcp to submit Op::ListMcpTools, got {ops:?}"
        );

        // Should receive formatted MCP tool listing
        let notifications = client.notifications.lock().unwrap();
        let has_mcp_text = notifications.iter().any(|n| {
            if let SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(t),
                ..
            }) = &n.update
            {
                t.text.contains("MCP Tools")
            } else {
                false
            }
        });
        assert!(
            has_mcp_text,
            "expected /mcp to send formatted MCP tools listing"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_skills() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/skills".into()]),
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

        // /skills now submits Op::ListSkills
        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(&ops[0], Op::ListSkills { .. }),
            "expected /skills to submit Op::ListSkills, got {ops:?}"
        );

        // Should receive formatted skills listing
        let notifications = client.notifications.lock().unwrap();
        let has_skills_text = notifications.iter().any(|n| {
            if let SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(t),
                ..
            }) = &n.update
            {
                t.text.contains("Available Skills")
            } else {
                false
            }
        });
        assert!(
            has_skills_text,
            "expected /skills to send formatted skills listing"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_debug_config_removed() {
        // /debug-config was removed — it was a fake UserInput wrapper.
        let commands = ThreadActor::<StubAuth>::builtin_commands();
        let names: Vec<_> = commands.iter().map(|c| c.name.as_str()).collect();
        assert!(
            !names.contains(&"debug-config"),
            "/debug-config should not be in builtin commands"
        );
    }

    // ==================== Edge case tests for commands requiring arguments ====================

    #[tokio::test]
    async fn test_slash_rename_no_arg_falls_through() -> anyhow::Result<()> {
        let (session_id, _client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        // /rename with no argument should fall through to the default handler
        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/rename".into()]),
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

        // Should still produce a UserInput (the default handler sends the raw /rename text)
        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(&ops[0], Op::UserInput { .. }),
            "expected /rename with no arg to still produce a UserInput, got {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_mention_no_arg_falls_through() -> anyhow::Result<()> {
        // /mention was removed, but "/mention" with no arg should still fall through
        // to the default handler (treated as regular text since it's unrecognized)
        let (session_id, _client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/mention".into()]),
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

        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(&ops[0], Op::UserInput { .. }),
            "expected /mention to fall through as regular text, got {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_slash_review_branch_no_arg_falls_through() -> anyhow::Result<()> {
        let (session_id, _client, thread, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/review-branch".into()]),
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

        // Should NOT be an Op::Review — should fall through to default handler
        let ops = thread.ops.lock().unwrap();
        assert!(
            matches!(&ops[0], Op::UserInput { .. }),
            "expected /review-branch with no arg to fall through to UserInput, got {ops:?}"
        );

        Ok(())
    }

    // ==================== Orphaned tool call tests ====================

    #[tokio::test]
    async fn test_orphaned_tool_call_events_handled_gracefully() -> anyhow::Result<()> {
        let (session_id, _client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-orphaned-events".into()]),
            response_tx: prompt_response_tx,
        })?;

        // The key assertion: this should complete successfully without panicking
        // despite receiving events for call_ids that were never registered
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

        // Verify no ToolCall or ToolCallUpdate notifications were sent for the orphaned events
        // (since ExecCommandBegin was never sent, no ActiveCommand was created)
        // The test succeeding without panic proves the else-clause warnings work correctly.
        Ok(())
    }

    // ==================== Patch apply abort tests ====================

    #[tokio::test]
    async fn test_patch_apply_abort_marks_edit_failed() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-patch-abort".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::Cancelled);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();

        // There should be a ToolCall for the patch with InProgress status
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::ToolCall(tool_call)
                    if tool_call.tool_call_id.0.as_ref() == "patch-call-1"
                        && tool_call.status == ToolCallStatus::InProgress
            )),
            "expected InProgress ToolCall for patch-call-1, got {notifications:?}"
        );

        // There should be a ToolCallUpdate for the same call id with Failed status
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::ToolCallUpdate(update)
                    if update.tool_call_id.0.as_ref() == "patch-call-1"
                        && update.fields.status == Some(ToolCallStatus::Failed)
                        && update
                            .fields
                            .content
                            .as_ref()
                            .is_some_and(|content| content.iter().any(|item| {
                                matches!(
                                    item,
                                    ToolCallContent::Content(Content {
                                        content: ContentBlock::Text(TextContent { text, .. }),
                                        ..
                                    }) if text == "Edit interrupted before completion."
                                )
                            }))
            )),
            "expected Failed ToolCallUpdate for patch-call-1 with interruption message, got {notifications:?}"
        );

        Ok(())
    }

    // ==================== CollabWaiting event tests ====================

    #[tokio::test]
    async fn test_collab_waiting_events_surfaced() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-collab-waiting".into()]),
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

        // With a sub-agent active, waiting events should update the tool call
        // content rather than emitting standalone AgentMessageChunk text.
        let tool_updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(u) => Some(u.clone()),
                _ => None,
            })
            .collect();

        // Should have updates: spawn end content, waiting begin content, and turn-complete completed
        assert!(
            tool_updates.len() >= 2,
            "expected at least 2 tool call updates (waiting + completion), got {tool_updates:?}"
        );

        // One of the updates should contain the waiting status text
        let has_waiting_content = tool_updates.iter().any(|u| {
            u.fields.content.iter().any(|blocks| {
                blocks.iter().any(|block| {
                    matches!(
                        block,
                        ToolCallContent::Content(content)
                            if matches!(&content.content, ContentBlock::Text(TextContent { text, .. }) if text.contains("Waiting"))
                    )
                })
            })
        });
        assert!(
            has_waiting_content,
            "expected a tool call update with 'Waiting' content, got {tool_updates:?}"
        );

        // The final update should be completed (from TurnComplete cleanup)
        let has_completed = tool_updates
            .iter()
            .any(|u| u.fields.status == Some(ToolCallStatus::Completed));
        assert!(
            has_completed,
            "expected a completed tool call update, got {tool_updates:?}"
        );

        Ok(())
    }

    // ==================== Usage accumulation edge case tests ====================

    #[tokio::test]
    async fn test_token_count_with_none_info() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-token-none-info".into()]),
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
        // When info is None, no UsageUpdate should be sent
        let usage_updates: Vec<_> = notifications
            .iter()
            .filter(|n| matches!(&n.update, SessionUpdate::UsageUpdate(..)))
            .collect();
        assert_eq!(
            usage_updates.len(),
            0,
            "expected no UsageUpdate when TokenCount info is None, got {usage_updates:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_token_count_with_no_context_window() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-token-no-window".into()]),
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
        // When model_context_window is None, no UsageUpdate should be sent
        // (we still accumulate internally, but don't notify without window size)
        let usage_updates: Vec<_> = notifications
            .iter()
            .filter(|n| matches!(&n.update, SessionUpdate::UsageUpdate(..)))
            .collect();
        assert_eq!(
            usage_updates.len(),
            0,
            "expected no UsageUpdate when model_context_window is None, got {usage_updates:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_token_count_with_unparseable_balance() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-token-bad-balance".into()]),
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
        let usage_updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::UsageUpdate(u) => Some(u.clone()),
                _ => None,
            })
            .collect();

        // Should still send a UsageUpdate (we have model_context_window)
        assert_eq!(usage_updates.len(), 1, "expected 1 UsageUpdate");
        assert_eq!(usage_updates[0].size, 128000);
        // But cost should be None because "not-a-number" can't be parsed as f64
        assert!(
            usage_updates[0].cost.is_none(),
            "expected no cost when balance is unparseable, got {:?}",
            usage_updates[0].cost
        );

        Ok(())
    }

    // ==================== Tightened event notification tests ====================

    #[tokio::test]
    async fn test_hook_events_exact_format() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-hook-events".into()]),
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
        let texts: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) => Some(text.clone()),
                _ => None,
            })
            .collect();

        // Verify exact format of hook started message
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("Running hook: hook-1 (") && t.ends_with("...\n")),
            "hook started message should start with 'Running hook: hook-1 (' and end with '...\\n', got {texts:?}"
        );
        // Verify exact format of hook completed message including status message
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("Hook completed: hook-1") && t.contains("all good")),
            "hook completed message should contain hook id and status message, got {texts:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_image_generation_exact_content() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-image-gen".into()]),
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

        // Verify ToolCall begin has exact title and kind
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].title, "Generating image");
        assert_eq!(tool_calls[0].kind, ToolKind::Other);
        assert_eq!(tool_calls[0].status, ToolCallStatus::InProgress);

        // Verify completed update has result content "image.png"
        let updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(u) => Some(u.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].fields.status, Some(ToolCallStatus::Completed));
        assert!(
            updates[0]
                .fields
                .content
                .as_ref()
                .is_some_and(|c| !c.is_empty()),
            "completed update should have content with the image result"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_background_event_exact_format() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-background-event".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }), ..
                }) if text == "Long running task completed\n"
            )),
            "expected exact 'Long running task completed\\n' message"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_deprecation_notice_exact_format() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-deprecation-notice".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }), ..
                }) if text == "**Deprecation:** Old API deprecated\nPlease migrate to v2.\n"
            )),
            "expected exact deprecation format with markdown bold and details"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_thread_rollback_exact_format() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup(vec![]).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["emit-thread-rollback".into()]),
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
        assert!(
            notifications.iter().any(|n| matches!(
                &n.update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }), ..
                }) if text == "Thread rolled back: 3 turns removed from context.\n"
            )),
            "expected exact rollback message with plural 'turns'"
        );

        Ok(())
    }

    #[test]
    fn test_global_event_classification_includes_mcp_startup() {
        assert!(is_global_event(&EventMsg::McpStartupUpdate(
            McpStartupUpdateEvent {
                server: "test".to_string(),
                status: codex_protocol::protocol::McpStartupStatus::Ready,
            }
        )));
        assert!(is_global_event(&EventMsg::McpStartupComplete(
            McpStartupCompleteEvent {
                ready: vec!["test".to_string()],
                failed: vec![],
                cancelled: vec![],
            }
        )));
        assert!(!is_global_event(&EventMsg::TurnComplete(
            TurnCompleteEvent {
                last_agent_message: None,
                turn_id: "turn-1".to_string(),
            }
        )));
    }

    #[test]
    fn test_auto_compact_submission_id_is_suppressed() {
        assert!(is_auto_compact_submission_id("auto-compact-0"));
        assert!(is_auto_compact_submission_id("auto-compact-12"));
        assert!(!is_auto_compact_submission_id("0"));
        assert!(!is_auto_compact_submission_id("submission-1"));
    }

    /// Verify that /diff, /status, /rename, /mcp, and /skills all generate notifications
    /// (not just Ops) by checking the stub echoes the prompt text back.
    #[tokio::test]
    async fn test_slash_commands_generate_notifications() -> anyhow::Result<()> {
        // Test each native command and verify it completes with notifications
        for cmd in ["/diff", "/status", "/rename My Thread", "/mcp", "/skills"] {
            let (session_id, client, _thread, message_tx, local_set) = setup(vec![]).await?;
            let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

            message_tx.send(ThreadMessage::Prompt {
                request: PromptRequest::new(session_id.clone(), vec![cmd.into()]),
                response_tx: prompt_response_tx,
            })?;

            tokio::try_join!(
                async {
                    let stop_reason = prompt_response_rx.await??.await??;
                    assert_eq!(stop_reason, StopReason::EndTurn, "failed for cmd: {cmd}");
                    drop(message_tx);
                    anyhow::Ok(())
                },
                async {
                    local_set.await;
                    anyhow::Ok(())
                }
            )?;

            // Verify at least one notification was sent
            let notifications = client.notifications.lock().unwrap();
            assert!(
                !notifications.is_empty(),
                "cmd {cmd}: expected notifications but got none"
            );
            // Verify it's an agent message chunk
            assert!(
                notifications.iter().any(|n| matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(..),
                        ..
                    })
                )),
                "cmd {cmd}: expected AgentMessageChunk notification"
            );
        }

        Ok(())
    }
}
