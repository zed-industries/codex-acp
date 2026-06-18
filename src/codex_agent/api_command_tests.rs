use super::super::*;
use acp::schema::{
    ContentBlock, SessionConfigOptionValue, SessionConfigValueId, StopReason, TextContent,
};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Default)]
struct RecordingAgentApi {
    calls: Mutex<Vec<&'static str>>,
}

impl RecordingAgentApi {
    fn record(&self, name: &'static str) {
        self.calls.lock().unwrap().push(name);
    }

    fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().clone()
    }
}

impl AgentApi for RecordingAgentApi {
    fn initialize(&self, _request: InitializeRequest) -> AgentApiFuture<'_, InitializeResponse> {
        Box::pin(async {
            self.record("initialize");
            Ok(InitializeResponse::new(ProtocolVersion::V1))
        })
    }

    fn authenticate(
        &self,
        _request: AuthenticateRequest,
    ) -> AgentApiFuture<'_, AuthenticateResponse> {
        Box::pin(async {
            self.record("authenticate");
            Ok(AuthenticateResponse::new())
        })
    }

    fn logout(&self, _request: LogoutRequest) -> AgentApiFuture<'_, LogoutResponse> {
        Box::pin(async {
            self.record("logout");
            Ok(LogoutResponse::new())
        })
    }

    fn new_session(
        &self,
        _request: NewSessionRequest,
        _cx: ConnectionTo<Client>,
    ) -> AgentApiFuture<'_, NewSessionResponse> {
        Box::pin(async {
            self.record("new_session");
            Ok(NewSessionResponse::new(SessionId::new("new-session")))
        })
    }

    fn load_session(
        &self,
        _request: LoadSessionRequest,
        _cx: ConnectionTo<Client>,
    ) -> AgentApiFuture<'_, LoadSessionResponse> {
        Box::pin(async {
            self.record("load_session");
            Ok(LoadSessionResponse::new())
        })
    }

    fn resume_session(
        &self,
        _request: ResumeSessionRequest,
        _cx: ConnectionTo<Client>,
    ) -> AgentApiFuture<'_, ResumeSessionResponse> {
        Box::pin(async {
            self.record("resume_session");
            Ok(ResumeSessionResponse::new())
        })
    }

    fn list_sessions(
        &self,
        _request: ListSessionsRequest,
    ) -> AgentApiFuture<'_, ListSessionsResponse> {
        Box::pin(async {
            self.record("list_sessions");
            Ok(ListSessionsResponse::new(vec![]))
        })
    }

    fn close_session(
        &self,
        _request: CloseSessionRequest,
    ) -> AgentApiFuture<'_, CloseSessionResponse> {
        Box::pin(async {
            self.record("close_session");
            Ok(CloseSessionResponse::new())
        })
    }

    fn prompt(&self, _request: PromptRequest) -> AgentApiFuture<'_, PromptResponse> {
        Box::pin(async {
            self.record("prompt");
            Ok(PromptResponse::new(StopReason::EndTurn))
        })
    }

    fn cancel(&self, _notification: CancelNotification) -> AgentApiFuture<'_, ()> {
        Box::pin(async {
            self.record("cancel");
            Ok(())
        })
    }

    fn set_session_mode(
        &self,
        _request: SetSessionModeRequest,
    ) -> AgentApiFuture<'_, SetSessionModeResponse> {
        Box::pin(async {
            self.record("set_session_mode");
            Ok(SetSessionModeResponse::new())
        })
    }

    fn set_session_config_option(
        &self,
        _request: SetSessionConfigOptionRequest,
    ) -> AgentApiFuture<'_, SetSessionConfigOptionResponse> {
        Box::pin(async {
            self.record("set_session_config_option");
            Ok(SetSessionConfigOptionResponse::new(vec![]))
        })
    }
}

struct TestAgentComponent(Arc<RecordingAgentApi>);

impl ConnectTo<Client> for TestAgentComponent {
    fn connect_to(
        self,
        client: impl ConnectTo<Agent>,
    ) -> impl std::future::Future<Output = acp::Result<()>> + Send {
        serve_agent_api(self.0, client)
    }
}

async fn wait_for_call(api: &RecordingAgentApi, name: &'static str) -> Result<(), Error> {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if api.calls().contains(&name) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .map_err(|_| Error::internal_error().data(format!("timed out waiting for {name}")))?;

    Ok(())
}

#[tokio::test]
async fn serve_dispatches_all_acp_requests_and_notifications() -> anyhow::Result<()> {
    let api = Arc::new(RecordingAgentApi::default());
    let cwd = std::env::current_dir()?;
    let api_for_client = api.clone();

    Client
        .connect_with(TestAgentComponent(api.clone()), async |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            cx.send_request(AuthenticateRequest::new(AuthMethodId::new("chatgpt")))
                .block_task()
                .await?;
            cx.send_request(LogoutRequest::new()).block_task().await?;
            cx.send_request(NewSessionRequest::new(cwd.clone()))
                .block_task()
                .await?;
            cx.send_request(LoadSessionRequest::new(
                SessionId::new("load-session"),
                cwd.clone(),
            ))
            .block_task()
            .await?;
            cx.send_request(ResumeSessionRequest::new(
                SessionId::new("resume-session"),
                cwd.clone(),
            ))
            .block_task()
            .await?;
            cx.send_request(ListSessionsRequest::new())
                .block_task()
                .await?;
            cx.send_request(CloseSessionRequest::new(SessionId::new("close-session")))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                SessionId::new("prompt-session"),
                vec![ContentBlock::Text(TextContent::new("hello"))],
            ))
            .block_task()
            .await?;
            cx.send_notification(CancelNotification::new(SessionId::new("cancel-session")))?;
            cx.send_request(SetSessionModeRequest::new(
                SessionId::new("mode-session"),
                "read-only",
            ))
            .block_task()
            .await?;
            cx.send_request(SetSessionConfigOptionRequest::new(
                SessionId::new("config-session"),
                "model",
                SessionConfigOptionValue::value_id(SessionConfigValueId::new("gpt-5")),
            ))
            .block_task()
            .await?;

            wait_for_call(&api_for_client, "cancel").await?;

            Ok(())
        })
        .await?;

    let mut calls = api.calls();
    calls.sort_unstable();
    assert_eq!(
        calls,
        vec![
            "authenticate",
            "cancel",
            "close_session",
            "initialize",
            "list_sessions",
            "load_session",
            "logout",
            "new_session",
            "prompt",
            "resume_session",
            "set_session_config_option",
            "set_session_mode",
        ]
    );

    Ok(())
}

#[test]
fn auth_method_ids_round_trip() {
    for method in [
        CodexAuthMethod::ChatGpt,
        CodexAuthMethod::CodexApiKey,
        CodexAuthMethod::OpenAiApiKey,
    ] {
        let id = AuthMethodId::from(method);
        assert_eq!(CodexAuthMethod::try_from(id).unwrap(), method);
    }
}

#[test]
fn unsupported_auth_method_is_rejected() {
    assert!(CodexAuthMethod::try_from(AuthMethodId::new("unsupported")).is_err());
}
