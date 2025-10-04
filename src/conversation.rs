use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use agent_client_protocol::{
    AgentSideConnection, Client as _, ContentBlock, EmbeddedResource, EmbeddedResourceResource,
    Error, LoadSessionResponse, ModelId, ModelInfo, PromptRequest, ResourceLink, SessionId,
    SessionMode, SessionModeId, SessionModeState, SessionModelState, SessionNotification,
    SessionUpdate, StopReason, TextResourceContents, ToolCallId, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields,
};
use codex_common::{
    approval_presets::{ApprovalPreset, builtin_approval_presets},
    model_presets::ModelPreset,
};
use codex_core::{
    CodexConversation,
    config::Config,
    protocol::{Event, EventMsg, InputItem, Op, ReviewDecision},
};
use codex_protocol::config_types::ReasoningEffort;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use crate::codex_agent::ACP_CLIENT;

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);

enum ConversationMessage {
    Load {
        response_tx: oneshot::Sender<Result<LoadSessionResponse, Error>>,
    },
    Prompt {
        request: PromptRequest,
        response_tx: oneshot::Sender<Result<(String, mpsc::UnboundedReceiver<EventMsg>), Error>>,
    },
    ExecApproval {
        submission_id: String,
        decision: ReviewDecision,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    PatchApproval {
        submission_id: String,
        decision: ReviewDecision,
        response_tx: oneshot::Sender<Result<(), Error>>,
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

pub struct ConversationHandle {
    /// A sender for interacting with the conversation.
    message_tx: mpsc::UnboundedSender<ConversationMessage>,
    /// A handle to the spawned task.
    _handle: tokio::task::JoinHandle<()>,
}

impl ConversationHandle {
    pub fn new(
        session_id: SessionId,
        conversation: Arc<CodexConversation>,
        config: Config,
        model_presets: Arc<Vec<ModelPreset>>,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        let actor = ConversationActor::new(
            session_id,
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

    pub async fn prompt(
        &self,
        request: PromptRequest,
    ) -> Result<(String, mpsc::UnboundedReceiver<EventMsg>), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::Prompt {
            request,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }

    pub async fn exec_approval(
        &self,
        submission_id: String,
        decision: ReviewDecision,
    ) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::ExecApproval {
            submission_id,
            decision,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().with_data(e.to_string()))?
    }

    pub async fn patch_approval(
        &self,
        submission_id: String,
        decision: ReviewDecision,
    ) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ConversationMessage::PatchApproval {
            submission_id,
            decision,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
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
    Prompt(PromptState),
}

impl SubmissionState {
    fn is_active(&self) -> bool {
        match self {
            SubmissionState::Prompt(submission) => !submission.sender.is_closed(),
        }
    }

    async fn handle_event(&mut self, client: &Client, event: EventMsg) {
        match self {
            SubmissionState::Prompt(submission) => submission.handle_event(client, event).await,
        }
    }
}

struct PromptState {
    active_command: Option<(String, ToolCallId)>,
    active_web_search: Option<String>,
    event_count: usize,
    submission_id: String,
    stop_reason: Option<StopReason>,
    sender: mpsc::UnboundedSender<EventMsg>,
}

impl PromptState {
    fn new(submission_id: String, sender: mpsc::UnboundedSender<EventMsg>) -> Self {
        info!("Submitted prompt with submission_id: {submission_id}");
        info!("Starting to wait for conversation events for submission_id: {submission_id}");

        Self {
            active_command: None,
            active_web_search: None,
            event_count: 0,
            sender,
            submission_id,
            stop_reason: None,
        }
    }

    async fn handle_event(&mut self, client: &Client, event: EventMsg) {
        self.event_count += 1;

        // Complete any previous web search before starting a new one
        // match &event {
        //     EventMsg::Error(..)
        //     | EventMsg::StreamError(..)
        //     | EventMsg::WebSearchBegin(..)
        //     | EventMsg::UserMessage(..)
        //     | EventMsg::ExecApprovalRequest(..)
        //     | EventMsg::ExecCommandBegin(..)
        //     | EventMsg::ExecCommandOutputDelta(..)
        //     | EventMsg::ExecCommandEnd(..)
        //     | EventMsg::McpToolCallBegin(..)
        //     | EventMsg::McpToolCallEnd(..)
        //     | EventMsg::ApplyPatchApprovalRequest(..)
        //     | EventMsg::PatchApplyBegin(..)
        //     | EventMsg::PatchApplyEnd(..)
        //     | EventMsg::TaskComplete(..)
        //     | EventMsg::TokenCount(..)
        //     | EventMsg::TurnDiff(..)
        //     | EventMsg::TurnAborted(..)
        //     | EventMsg::ShutdownComplete => {
        //         self.complete_web_search(client).await;
        //     }
        //     _ => {}
        // }

        self.sender.send(event).ok();
    }

    async fn complete_web_search(&mut self, client: &Client) {
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

struct Client {
    session_id: SessionId,
}

impl Client {
    fn new(session_id: SessionId) -> Self {
        Self { session_id }
    }

    fn client() -> &'static AgentSideConnection {
        ACP_CLIENT.get().expect("Client should be set")
    }

    async fn send_notification(&self, update: SessionUpdate) {
        let notification = SessionNotification {
            session_id: self.session_id.clone(),
            update,
            meta: None,
        };

        if let Err(e) = Self::client().session_notification(notification).await {
            error!("Failed to send session notification: {:?}", e);
        }
    }
}

struct ConversationActor {
    /// Used for sending messages back to the client.
    client: Client,
    /// The conversation associated with this task.
    conversation: Arc<CodexConversation>,
    /// The configuration for the conversation.
    config: Config,
    /// The model presets for the conversation.
    model_presets: Arc<Vec<ModelPreset>>,
    /// A sender for each interested `Op` submission that needs events routed.
    submissions: HashMap<String, SubmissionState>,
    /// A receiver for incoming conversation messages.
    message_rx: mpsc::UnboundedReceiver<ConversationMessage>,
}

impl ConversationActor {
    fn new(
        session_id: SessionId,
        conversation: Arc<CodexConversation>,
        config: Config,
        model_presets: Arc<Vec<ModelPreset>>,
        message_rx: mpsc::UnboundedReceiver<ConversationMessage>,
    ) -> Self {
        Self {
            client: Client::new(session_id),
            conversation,
            config,
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
            }
            ConversationMessage::Prompt {
                request,
                response_tx,
            } => {
                let result = self.handle_prompt(request).await;
                drop(response_tx.send(result));
            }
            ConversationMessage::ExecApproval {
                submission_id,
                decision,
                response_tx,
            } => {
                let result = self.handle_exec_approval(submission_id, decision).await;
                drop(response_tx.send(result));
            }
            ConversationMessage::PatchApproval {
                submission_id,
                decision,
                response_tx,
            } => {
                let result = self.handle_patch_approval(submission_id, decision).await;
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

    fn find_model_preset(&self) -> Option<&ModelPreset> {
        if let Some(preset) = self.model_presets.iter().find(|preset| {
            preset.model == self.config.model && preset.effort == self.config.model_reasoning_effort
        }) {
            return Some(preset);
        }

        // If we didn't find it, and it is set to none, see if we can find one with the default value
        if self.config.model_reasoning_effort.is_none()
            && let Some(preset) = self.model_presets.iter().find(|preset| {
                preset.model == self.config.model
                    && preset.effort == Some(ReasoningEffort::default())
            })
        {
            return Some(preset);
        }

        None
    }

    fn models(&self) -> Result<SessionModelState, Error> {
        let current_model_id = self
            .find_model_preset()
            .map(|preset| ModelId(preset.id.into()))
            .ok_or_else(|| {
                anyhow::anyhow!("No valid model preset for model {}", self.config.model)
            })?;

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
    ) -> Result<(String, mpsc::UnboundedReceiver<EventMsg>), Error> {
        let (session_tx, session_rx) = mpsc::unbounded_channel();
        let items = build_prompt_items(request.prompt);

        let submission_id = match self.conversation.submit(Op::UserInput { items }).await {
            Ok(submission_id) => submission_id,
            Err(e) => {
                error!("Failed to submit prompt: {e:?}");
                return Err(Error::internal_error());
            }
        };

        self.submissions.insert(
            submission_id.clone(),
            SubmissionState::Prompt(PromptState::new(submission_id.clone(), session_tx)),
        );

        Ok((submission_id, session_rx))
    }

    async fn handle_exec_approval(
        &mut self,
        submission_id: String,
        decision: ReviewDecision,
    ) -> Result<(), Error> {
        self.conversation
            .submit(Op::ExecApproval {
                id: submission_id,
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn handle_patch_approval(
        &mut self,
        submission_id: String,
        decision: ReviewDecision,
    ) -> Result<(), Error> {
        self.conversation
            .submit(Op::PatchApproval {
                id: submission_id,
                decision,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
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
        let preset = self
            .model_presets
            .iter()
            .find(|p| p.id == model.0.as_ref())
            .ok_or_else(|| Error::invalid_params().with_data("Model not found"))?;

        self.conversation
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

        self.config.model = preset.model.into();
        self.config.model_reasoning_effort = preset.effort;

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
            ContentBlock::ResourceLink(ResourceLink {
                annotations: _,
                description: _,
                mime_type: _,
                name,
                size: _,
                title: _,
                uri,
                meta: _,
            }) => Some(InputItem::Text {
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
            }) => Some(InputItem::Text {
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
