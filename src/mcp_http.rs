use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::rc::Rc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

use agent_client_protocol::{
    Client as _, CreateTerminalRequest, KillTerminalCommandRequest, ReleaseTerminalRequest,
    SessionNotification, SessionUpdate, TerminalOutputRequest, ToolCallContent, ToolCallId,
    ToolCallUpdate, ToolCallUpdateFields, WaitForTerminalExitRequest,
};

pub struct AcpMcpHttpServer {
    local_addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,

    // Shared state for MCP tool state (e.g. bash_output last offsets).
    state: Rc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    last_offsets: std::collections::HashMap<String, usize>,
    client: Option<Rc<agent_client_protocol::AgentSideConnection>>,
    invocations: std::collections::HashMap<
        (String, String),
        Vec<(String, agent_client_protocol::SessionId)>,
    >,
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: serde_json::Value,
    method: String,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum JsonRpcResponse {
    Result {
        jsonrpc: &'static str,
        id: serde_json::Value,
        result: serde_json::Value,
    },
    Error {
        jsonrpc: &'static str,
        id: serde_json::Value,
        error: JsonRpcError,
    },
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

impl AcpMcpHttpServer {
    pub async fn bind_and_spawn() -> Result<Self> {
        let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
            .await
            .context("failed to bind MCP HTTP server")?;

        let local_addr = listener
            .local_addr()
            .context("failed to resolve MCP HTTP server local_addr")?;

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let state = Rc::new(Mutex::new(State::default()));
        let state_clone = Rc::clone(&state);

        let join = tokio::task::spawn_local(async move {
            if let Err(err) = accept_loop(listener, shutdown_rx, state_clone).await {
                // In a skeleton, we just log to stderr; production should use tracing.
                tracing::error!("MCP HTTP server accept loop error: {err:#}");
            }
        });

        Ok(Self {
            local_addr,
            shutdown_tx: Some(shutdown_tx),
            join: Some(join),
            state,
        })
    }

    /// Return the stream URL (for GET).
    ///
    /// Example: http://127.0.0.1:<port>/stream
    pub fn stream_url(&self) -> String {
        format!(
            "http://{}:{}/stream",
            self.local_addr.ip(),
            self.local_addr.port()
        )
    }

    /// Return the messages URL (for POST).
    ///
    /// Example: http://127.0.0.1:<port>/messages
    pub fn messages_url(&self) -> String {
        format!(
            "http://{}:{}/messages",
            self.local_addr.ip(),
            self.local_addr.port()
        )
    }

    /// Record an in‑flight MCP tool invocation so the HTTP handler can correlate to the correct ToolCallId/SessionId.
    pub async fn record_invocation(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
        tool_call_id: &str,
        session_id: &agent_client_protocol::SessionId,
    ) {
        let key = (tool.to_string(), canonicalize_json(arguments));
        let mut s = self.state.lock().await;
        s.invocations
            .entry(key)
            .or_default()
            .push((tool_call_id.to_string(), session_id.clone()));
    }

    pub async fn set_client(&self, client: Rc<agent_client_protocol::AgentSideConnection>) {
        let mut s = self.state.lock().await;
        s.client = Some(client);
    }
}

impl Drop for AcpMcpHttpServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

async fn accept_loop(
    listener: TcpListener,
    mut shutdown_rx: oneshot::Receiver<()>,
    state: Rc<Mutex<State>>,
) -> Result<()> {
    loop {
        tokio::select! {
            accept_res = listener.accept() => {
                let (stream, peer_addr) = match accept_res {
                    Ok((s, a)) => (s, a),
                    Err(err) => {
                        tracing::error!("accept error: {err:#}");
                        continue;
                    }
                };
                let state = Rc::clone(&state);
                tokio::task::spawn_local(async move {
                    if let Err(err) = handle_connection(stream, state).await {
                        tracing::error!("connection {peer_addr} error: {err:#}");
                    }
                });
            }
            _ = &mut shutdown_rx => {
                break;
            }
        }
    }
    Ok(())
}

async fn handle_connection(mut stream: TcpStream, state: Rc<Mutex<State>>) -> Result<()> {
    let mut buf = Vec::with_capacity(8 * 1024);
    let (method, path, headers, body) = read_http_request(&mut stream, &mut buf).await?;
    match (method.as_str(), path.as_str()) {
        ("GET", "/stream") => {
            let headers = concat_strings(&[
                "HTTP/1.1 200 OK\r\n",
                "Content-Type: application/x-ndjson\r\n",
                "Cache-Control: no-cache\r\n",
                "Connection: close\r\n",
                "\r\n",
            ]);
            stream.write_all(headers.as_bytes()).await?;
            Ok(())
        }
        ("POST", "/messages") => {
            let content_length = headers
                .iter()
                .find_map(|(k, v)| {
                    if k.eq_ignore_ascii_case("content-length") {
                        v.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            let mut body_buf = body;
            while body_buf.len() < content_length {
                let mut tmp = vec![0u8; content_length - body_buf.len()];
                let n = stream.read(&mut tmp).await?;
                if n == 0 {
                    break;
                }
                body_buf.extend_from_slice(&tmp[..n]);
            }

            let response = match serde_json::from_slice::<JsonRpcRequest>(&body_buf) {
                Ok(req) => handle_jsonrpc_request(req, &state).await,
                Err(err) => JsonRpcResponse::Error {
                    jsonrpc: "2.0",
                    id: json!(null),
                    error: JsonRpcError {
                        code: -32700,
                        message: format!("invalid JSON-RPC: {err}"),
                        data: None,
                    },
                },
            };

            let payload = serde_json::to_string(&response)?;
            let headers = concat_strings(&[
                "HTTP/1.1 200 OK\r\n",
                "Content-Type: application/x-ndjson\r\n",
                "Cache-Control: no-cache\r\n",
                "Connection: close\r\n",
                "\r\n",
            ]);
            stream.write_all(headers.as_bytes()).await?;
            stream.write_all(payload.as_bytes()).await?;
            stream.write_all(b"\n").await?;
            Ok(())
        }
        _ => {
            let headers = concat_strings(&[
                "HTTP/1.1 404 Not Found\r\n",
                "Content-Type: text/plain; charset=utf-8\r\n",
                "Cache-Control: no-cache\r\n",
                "Connection: close\r\n",
                "\r\n",
            ]);
            stream.write_all(headers.as_bytes()).await?;
            stream.write_all(b"not found").await?;
            Ok(())
        }
    }
}

async fn handle_jsonrpc_request(req: JsonRpcRequest, state: &Rc<Mutex<State>>) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => {
            let result = json!({
                "capabilities": {},
                "serverInfo": {
                    "name": "codex-acp-mcp",
                    "version": "0.1.0"
                },
                "protocolVersion": "2025-06-18"
            });
            JsonRpcResponse::Result {
                jsonrpc: "2.0",
                id: req.id,
                result,
            }
        }

        // List our terminal tools (bash, bash_output, kill_bash)
        "tools/list" => {
            // JSON Schema helper constructors
            let string_schema = |desc: Option<&str>| {
                json!({
                    "type": "string",
                    "description": desc
                })
            };
            let array_of_strings = |desc: Option<&str>| {
                json!({
                    "type": "array",
                    "items": { "type": "string" },
                    "description": desc
                })
            };

            // bash tool schema
            let bash_tool = json!({
                "name": "bash",
                "description": "Run a command using the editor-owned terminal. The client manages terminal lifecycle and UI.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": string_schema(Some("The command to execute")),
                        "args": array_of_strings(Some("Optional arguments for the command")),
                        "cwd": string_schema(Some("Working directory for the command")),
                        // Context injected by the caller (codex-rs) so we can embed the terminal UI on the correct ToolCall.
                        "tool_call_id": string_schema(Some("Tool call id for embedding updates")),
                        "conversation_id": string_schema(Some("Conversation id for routing updates")),
                        "background": { "type": "boolean", "description": "If true, return immediately and keep the process running" }
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }
            });

            // bash_output tool schema
            let bash_output_tool = json!({
                "name": "bash_output",
                "description": "Return only new output since last call for a given terminal.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "terminal_id": string_schema(Some("Terminal id returned by bash")),
                    },
                    "required": ["terminal_id"],
                    "additionalProperties": false
                }


            });

            // kill_bash tool schema
            let kill_bash_tool = json!({
                "name": "kill_bash",
                "description": "Kill a running terminal session (and release it).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "terminal_id": string_schema(Some("Terminal id returned by bash")),
                    },
                    "required": ["terminal_id"],
                    "additionalProperties": false
                }
            });

            let result = json!({
                "tools": [bash_tool, bash_output_tool, kill_bash_tool]
            });
            JsonRpcResponse::Result {
                jsonrpc: "2.0",
                id: req.id,
                result,
            }
        }

        // Call one of our tools
        "tools/call" => {
            // Expect params with shape { name: string, arguments?: any }
            let (name, args) = match req.params.clone() {
                Some(p) => {
                    let name = p
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let args = p.get("arguments").cloned().unwrap_or(json!({}));
                    (name, args)
                }
                None => ("".to_string(), json!({})),
            };

            // Resolve ACP client
            let client = {
                let guard = state.lock().await;
                guard.client.as_ref().map(Rc::clone)
            };
            let client = match client {
                Some(c) => c,
                None => {
                    return JsonRpcResponse::Error {
                        jsonrpc: "2.0",
                        id: req.id,
                        error: JsonRpcError {
                            code: -32000,
                            message: "acp client not initialized".to_string(),
                            data: None,
                        },
                    };
                }
            };

            let canonical_args = canonicalize_json(&args);
            let (tool_call_id, session_id) = {
                let mut s = state.lock().await;
                match s
                    .invocations
                    .get_mut(&(name.clone(), canonical_args.clone()))
                    .and_then(|v| v.pop())
                {
                    Some((call_id_str, sid)) => (ToolCallId(call_id_str.into()), sid),
                    None => {
                        return JsonRpcResponse::Error {
                            jsonrpc: "2.0",
                            id: req.id,
                            error: JsonRpcError {
                                code: -32001,
                                message: "no matching invocation for tools/call".to_string(),
                                data: Some(json!({
                                    "tool": name,
                                    "canonical_args": canonical_args
                                })),
                            },
                        };
                    }
                }
            };

            let result_payload = match name.as_str() {
                "bash" => {
                    let command = args
                        .get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let cmd_args = args
                        .get("args")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let cwd = args.get("cwd").and_then(|v| v.as_str()).map(PathBuf::from);
                    let background = args
                        .get("background")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    let create_res = client
                        .create_terminal(CreateTerminalRequest {
                            session_id: session_id.clone(),
                            command,
                            args: cmd_args,
                            env: vec![],
                            cwd,
                            output_byte_limit: Some(2_000_000),
                            meta: None,
                        })
                        .await;

                    let create_ok = match create_res {
                        Ok(r) => r,
                        Err(e) => {
                            return JsonRpcResponse::Result {
                                jsonrpc: "2.0",
                                id: req.id,
                                result: json!({
                                    "content": [],
                                    "structuredContent": { "error": format!("terminal/create failed: {e}") },
                                    "isError": true
                                }),
                            };
                        }
                    };

                    let terminal_id = create_ok.terminal_id;

                    let _ = client
                        .session_notification(SessionNotification {
                            session_id: session_id.clone(),
                            update: SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                id: tool_call_id.clone(),
                                fields: ToolCallUpdateFields {
                                    content: Some(vec![ToolCallContent::Terminal {
                                        terminal_id: terminal_id.clone(),
                                    }]),
                                    ..Default::default()
                                },
                                meta: None,
                            }),
                            meta: None,
                        })
                        .await;

                    if background {
                        json!({
                            "content": [],
                            "structuredContent": {
                                "terminal_id": terminal_id.to_string(),
                                "background": true
                            },
                            "isError": false
                        })
                    } else {
                        // Wait for exit
                        let _ = client
                            .wait_for_terminal_exit(WaitForTerminalExitRequest {
                                session_id: session_id.clone(),
                                terminal_id: terminal_id.clone(),
                                meta: None,
                            })
                            .await;

                        // Get final output
                        let out_res = client
                            .terminal_output(TerminalOutputRequest {
                                session_id: session_id.clone(),
                                terminal_id: terminal_id.clone(),
                                meta: None,
                            })
                            .await;

                        // Release terminal (best effort)
                        let _ = client
                            .release_terminal(ReleaseTerminalRequest {
                                session_id: session_id.clone(),
                                terminal_id: terminal_id.clone(),
                                meta: None,
                            })
                            .await;

                        match out_res {
                            Ok(o) => json!({
                                "content": [],
                                "structuredContent": {
                                    "output": o.output,
                                    "exit_status": o.exit_status,
                                    "truncated": o.truncated
                                },
                                "isError": false
                            }),
                            Err(e) => json!({
                                "content": [],
                                "structuredContent": { "error": format!("terminal/output failed: {e}") },
                                "isError": true
                            }),
                        }
                    }
                }
                "bash_output" => {
                    let terminal_id = args
                        .get("terminal_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();

                    let out_res = client
                        .terminal_output(TerminalOutputRequest {
                            session_id: session_id.clone(),
                            terminal_id: agent_client_protocol::TerminalId(
                                terminal_id.clone().into(),
                            ),
                            meta: None,
                        })
                        .await;

                    match out_res {
                        Ok(o) => {
                            // Compute delta based on last_offsets
                            let mut guard = state.lock().await;
                            let last = guard.last_offsets.get(&terminal_id).copied().unwrap_or(0);
                            let full = o.output;
                            let new_bytes = full.as_bytes();
                            let start = last.min(new_bytes.len());
                            let delta = String::from_utf8_lossy(&new_bytes[start..]).to_string();
                            guard
                                .last_offsets
                                .insert(terminal_id.clone(), new_bytes.len());

                            json!({
                                "content": [],
                                "structuredContent": {
                                    "output": delta,
                                    "truncated": o.truncated,
                                    "exit_status": o.exit_status
                                },
                                "isError": false
                            })
                        }
                        Err(e) => json!({
                            "content": [],
                            "structuredContent": { "error": format!("terminal/output failed: {e}") },
                            "isError": true
                        }),
                    }
                }
                "kill_bash" => {
                    let terminal_id = args
                        .get("terminal_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();

                    // Kill then release
                    let _ = client
                        .kill_terminal_command(KillTerminalCommandRequest {
                            session_id: session_id.clone(),
                            terminal_id: agent_client_protocol::TerminalId(
                                terminal_id.clone().into(),
                            ),
                            meta: None,
                        })
                        .await;
                    let _ = client
                        .release_terminal(ReleaseTerminalRequest {
                            session_id: session_id.clone(),
                            terminal_id: agent_client_protocol::TerminalId(
                                terminal_id.clone().into(),
                            ),
                            meta: None,
                        })
                        .await;

                    json!({
                        "content": [],
                        "structuredContent": { "ok": true },
                        "isError": false
                    })
                }
                // Unknown tool name
                other => {
                    json!({
                        "content": [],
                        "structuredContent": {
                            "error": format!("unknown tool: {}", other)
                        },
                        "isError": true
                    })
                }
            };

            JsonRpcResponse::Result {
                jsonrpc: "2.0",
                id: req.id,
                result: result_payload,
            }
        }

        // Default: method not found
        _ => JsonRpcResponse::Error {
            jsonrpc: "2.0",
            id: req.id,
            error: JsonRpcError {
                code: -32601,
                message: "method not implemented".to_string(),
                data: None,
            },
        },
    }
}

/// Read an HTTP request line and headers (very minimal), then return:
/// (method, path, headers, partial_body)
///
/// This simple parser:
/// - Expects HTTP/1.1 style requests
/// - Assumes ASCII header keys
/// - Does not validate all request correctness
async fn read_http_request(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
) -> Result<(String, String, Vec<(String, String)>, Vec<u8>)> {
    let header_end;
    loop {
        let mut temp = [0u8; 4096];
        let n = stream.read(&mut temp).await?;
        if n == 0 {
            return Err(anyhow!("connection closed before headers"));
        }
        buf.extend_from_slice(&temp[..n]);
        if let Some(pos) = find_headers_end(buf) {
            header_end = pos;
            break;
        }
        if buf.len() > 1024 * 1024 {
            return Err(anyhow!("request headers too large"));
        }
    }

    // Split into header section and initial body tail
    let (headers_part, body_part) = buf.split_at(header_end);
    let (request_line, header_lines) = split_request_lines(headers_part)?;

    let mut iter = request_line.split_whitespace();
    let method = iter.next().unwrap_or_default().to_string();
    let path = iter.next().unwrap_or_default().to_string();

    let mut headers = Vec::new();
    for line in header_lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    Ok((method, path, headers, body_part.to_vec()))
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    // Look for CRLF CRLF or LF LF
    for i in 0..buf.len().saturating_sub(3) {
        if &buf[i..i + 4] == b"\r\n\r\n" {
            return Some(i + 4);
        }
    }
    for i in 0..buf.len().saturating_sub(1) {
        if &buf[i..i + 2] == b"\n\n" {
            return Some(i + 2);
        }
    }
    None
}

fn split_request_lines(headers_part: &[u8]) -> Result<(&str, Vec<&str>)> {
    let s = std::str::from_utf8(headers_part).context("invalid HTTP request UTF-8")?;
    let mut lines = s.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow!("missing request line"))?;
    let header_lines = lines.collect::<Vec<_>>();
    Ok((request_line, header_lines))
}

fn concat_strings(parts: &[&str]) -> String {
    let total = parts.iter().map(|s| s.len()).sum();
    let mut s = String::with_capacity(total);
    for p in parts {
        s.push_str(p);
    }
    s
}

/// Produce a deterministic string form of a JSON value by sorting all object keys recursively.
fn canonicalize_json(value: &serde_json::Value) -> String {
    let normalized = normalize_json(value);
    serde_json::to_string(&normalized).unwrap_or_default()
}

fn normalize_json(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value as JsonValue;

    match value {
        JsonValue::Object(map) => {
            // Sort keys using a BTreeMap, normalizing children recursively.
            let mut sorted = std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), normalize_json(v));
            }
            let mut new_map = serde_json::Map::new();
            for (k, v) in sorted {
                new_map.insert(k, v);
            }
            JsonValue::Object(new_map)
        }
        JsonValue::Array(arr) => JsonValue::Array(arr.iter().map(normalize_json).collect()),
        // Primitives: pass through unchanged.
        _ => value.clone(),
    }
}
