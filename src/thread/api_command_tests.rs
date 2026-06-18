use super::*;
use agent_client_protocol::schema::SessionNotification;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

#[test]
fn build_prompt_items_maps_supported_prompt_content_blocks() {
    let items = build_prompt_items(vec![
        ContentBlock::Text(TextContent::new("hello")),
        ContentBlock::Image(ImageContent::new("Zm9v", "image/png")),
        ContentBlock::ResourceLink(ResourceLink::new("file.rs", "file:///tmp/file.rs")),
        ContentBlock::Resource(EmbeddedResource::new(
            EmbeddedResourceResource::TextResourceContents(TextResourceContents::new(
                "let x = 1;",
                "file:///tmp/file.rs",
            )),
        )),
    ]);

    assert_eq!(items.len(), 4);
    assert!(matches!(
        &items[0],
        UserInput::Text { text, .. } if text == "hello"
    ));
    assert!(matches!(
        &items[1],
        UserInput::Image { image_url, .. } if image_url == "data:image/png;base64,Zm9v"
    ));
    assert!(matches!(
        &items[2],
        UserInput::Text { text, .. } if text == "[@file.rs](file:///tmp/file.rs)"
    ));
    assert!(matches!(
        &items[3],
        UserInput::Text { text, .. }
            if text == "[@file.rs](file:///tmp/file.rs)\n<context ref=\"file:///tmp/file.rs\">\nlet x = 1;\n</context>"
    ));
}

#[test]
fn builtin_commands_cover_advertised_slash_commands() {
    let commands = ThreadActor::<StubAuth>::builtin_commands();
    let command_names = commands
        .iter()
        .map(|command| command.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        command_names,
        vec![
            "review",
            "review-branch",
            "review-commit",
            "init",
            "compact",
            "logout",
        ]
    );
    assert!(commands[0].input.is_some());
    assert!(commands[1].input.is_some());
    assert!(commands[2].input.is_some());
    assert!(commands[3].input.is_none());
    assert!(commands[4].input.is_none());
    assert!(commands[5].input.is_none());
}

#[tokio::test]
async fn load_returns_config_options_and_announces_builtin_commands() -> anyhow::Result<()> {
    let (_session_id, client, _, message_tx, _handle) = setup().await?;
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::Load { response_tx })?;

    let response = response_rx.await??;
    assert_config_option_ids_include(
        response
            .config_options
            .as_deref()
            .expect("load response should include config options"),
        &["mode", "model"],
    );

    wait_for_notification(client.as_ref(), |notification| {
        matches!(
            &notification.update,
            SessionUpdate::AvailableCommandsUpdate(update)
                if update
                    .available_commands
                    .iter()
                    .map(|command| command.name.as_str())
                    .collect::<Vec<_>>()
                    == vec![
                        "review",
                        "review-branch",
                        "review-commit",
                        "init",
                        "compact",
                        "logout",
                    ]
        )
    })
    .await?;

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn get_config_options_returns_mode_and_model_options() -> anyhow::Result<()> {
    let (_session_id, _client, _, message_tx, _handle) = setup().await?;
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::GetConfigOptions { response_tx })?;

    let options = response_rx.await??;
    assert_config_option_ids_include(&options, &["mode", "model"]);

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn set_mode_submits_thread_settings_and_emits_config_options() -> anyhow::Result<()> {
    let (_session_id, client, thread, message_tx, _handle) = setup().await?;
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::SetMode {
        mode: SessionModeId::new("read-only"),
        response_tx,
    })?;

    response_rx.await??;

    let ops = thread.ops.lock().unwrap();
    assert!(
        matches!(ops.last(), Some(Op::ThreadSettings { .. })),
        "expected set mode to submit ThreadSettings, got {ops:?}"
    );
    drop(ops);

    wait_for_notification(client.as_ref(), |notification| {
        matches!(
            &notification.update,
            SessionUpdate::ConfigOptionUpdate(update)
                if update
                    .config_options
                    .iter()
                    .any(|option| option.id.0.as_ref() == "mode")
        )
    })
    .await?;

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn set_config_option_model_submits_thread_settings() -> anyhow::Result<()> {
    let (_session_id, _client, thread, message_tx, _handle) = setup().await?;
    let preset = all_model_presets()
        .first()
        .expect("test model presets should not be empty")
        .clone();
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::SetConfigOption {
        config_id: SessionConfigId::new("model"),
        value: SessionConfigOptionValue::ValueId {
            value: SessionConfigValueId::new(preset.id.clone()),
        },
        response_tx,
    })?;

    response_rx.await??;

    let ops = thread.ops.lock().unwrap();
    assert!(
        matches!(ops.last(), Some(Op::ThreadSettings { .. })),
        "expected model config update to submit ThreadSettings, got {ops:?}"
    );

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn set_config_option_reasoning_effort_submits_thread_settings() -> anyhow::Result<()> {
    let (_session_id, _client, thread, message_tx, _handle) = setup().await?;
    let preset = all_model_presets()
        .iter()
        .find(|preset| preset.supported_reasoning_efforts.len() > 1)
        .expect("at least one test model preset should expose reasoning effort")
        .clone();
    let effort = preset.supported_reasoning_efforts[0].effort;

    let (model_response_tx, model_response_rx) = tokio::sync::oneshot::channel();
    message_tx.send(ThreadMessage::SetConfigOption {
        config_id: SessionConfigId::new("model"),
        value: SessionConfigOptionValue::ValueId {
            value: SessionConfigValueId::new(preset.id.clone()),
        },
        response_tx: model_response_tx,
    })?;
    model_response_rx.await??;

    let (effort_response_tx, effort_response_rx) = tokio::sync::oneshot::channel();
    message_tx.send(ThreadMessage::SetConfigOption {
        config_id: SessionConfigId::new("reasoning_effort"),
        value: SessionConfigOptionValue::ValueId {
            value: SessionConfigValueId::new(effort.to_string()),
        },
        response_tx: effort_response_tx,
    })?;

    effort_response_rx.await??;

    let ops = thread.ops.lock().unwrap();
    assert!(
        matches!(ops.last(), Some(Op::ThreadSettings { .. })),
        "expected reasoning config update to submit ThreadSettings, got {ops:?}"
    );

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn set_config_option_rejects_unknown_option() -> anyhow::Result<()> {
    let (_session_id, _client, _, message_tx, _handle) = setup().await?;
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::SetConfigOption {
        config_id: SessionConfigId::new("unknown"),
        value: SessionConfigOptionValue::ValueId {
            value: SessionConfigValueId::new("value"),
        },
        response_tx,
    })?;

    assert!(response_rx.await?.is_err());

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn cancel_submits_interrupt() -> anyhow::Result<()> {
    let (_session_id, _client, thread, message_tx, _handle) = setup().await?;
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::Cancel { response_tx })?;

    response_rx.await??;
    let ops = thread.ops.lock().unwrap();
    assert!(
        matches!(ops.last(), Some(Op::Interrupt)),
        "expected cancel to submit Interrupt, got {ops:?}"
    );

    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn replay_history_sends_persisted_agent_updates() -> anyhow::Result<()> {
    let (_session_id, client, _, message_tx, _handle) = setup().await?;
    let thread_id = ThreadId::default();
    let history = vec![
        RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
            message: "replayed answer".to_string(),
            phase: None,
            memory_citation: None,
        })),
        RolloutItem::EventMsg(EventMsg::AgentReasoning(AgentReasoningEvent {
            text: "replayed reasoning".to_string(),
        })),
        RolloutItem::EventMsg(EventMsg::ThreadGoalUpdated(ThreadGoalUpdatedEvent {
            thread_id,
            turn_id: Some("turn-1".to_string()),
            goal: ThreadGoal {
                thread_id,
                objective: "Replay the goal".to_string(),
                status: ThreadGoalStatus::Complete,
                token_budget: None,
                tokens_used: 0,
                time_used_seconds: 0,
                created_at: 1,
                updated_at: 2,
            },
        })),
    ];
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::ReplayHistory {
        history,
        response_tx,
    })?;

    response_rx.await??;
    let notifications = client.notifications.lock().unwrap();
    assert!(notifications.iter().any(|notification| {
        matches!(
            &notification.update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "replayed answer"
        )
    }));
    assert!(notifications.iter().any(|notification| {
        matches!(
            &notification.update,
            SessionUpdate::AgentThoughtChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "replayed reasoning"
        )
    }));
    assert!(notifications.iter().any(|notification| {
        matches!(
            &notification.update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Goal updated (complete): Replay the goal"
        )
    }));

    drop(notifications);
    drop(message_tx);
    Ok(())
}

#[tokio::test]
async fn logout_slash_command_logs_out_and_returns_auth_required() -> anyhow::Result<()> {
    let session_id = SessionId::new("test");
    let client = Arc::new(StubClient::new());
    let session_client = SessionClient::with_client(session_id.clone(), client, Arc::default());
    let conversation = Arc::new(StubCodexThread::new());
    let models_manager = Arc::new(StubModelsManager);
    let config = Config::load_with_cli_overrides_and_harness_overrides(
        vec![],
        ConfigOverrides::default(),
    )
    .await?;
    let logout_calls = Arc::new(AtomicUsize::new(0));
    let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();
    let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();
    let actor = ThreadActor::new(
        RecordingAuth {
            logout_calls: logout_calls.clone(),
        },
        session_client,
        conversation.clone(),
        models_manager,
        config,
        message_rx,
        resolution_tx,
        resolution_rx,
    );
    let _handle = tokio::spawn(actor.spawn());
    let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

    message_tx.send(ThreadMessage::Prompt {
        request: PromptRequest::new(session_id, vec!["/logout".into()]),
        response_tx: prompt_response_tx,
    })?;

    assert!(prompt_response_rx.await?.is_err());
    assert_eq!(logout_calls.load(Ordering::SeqCst), 1);
    assert!(
        conversation.ops.lock().unwrap().is_empty(),
        "logout should not submit a Codex op"
    );

    drop(message_tx);
    Ok(())
}

fn assert_config_option_ids_include(options: &[SessionConfigOption], expected: &[&str]) {
    for expected_id in expected {
        assert!(
            options
                .iter()
                .any(|option| option.id.0.as_ref() == *expected_id),
            "expected config option `{expected_id}` in {options:?}"
        );
    }
}

async fn wait_for_notification(
    client: &StubClient,
    mut matches_notification: impl FnMut(&SessionNotification) -> bool,
) -> anyhow::Result<()> {
    tokio::time::timeout(Duration::from_millis(500), async {
        loop {
            if client
                .notifications
                .lock()
                .unwrap()
                .iter()
                .any(&mut matches_notification)
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await?;

    Ok(())
}

struct RecordingAuth {
    logout_calls: Arc<AtomicUsize>,
}

impl Auth for RecordingAuth {
    async fn logout(&self) -> Result<bool, Error> {
        self.logout_calls.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }
}
