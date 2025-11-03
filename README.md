# ACP adapter for Codex

Use [Codex](https://github.com/openai/codex) from [ACP-compatible](https://agentclientprotocol.com) clients such as [Zed](https://zed.dev)!

This tool implements an ACP adapter around the Codex CLI, supporting:

- Context @-mentions
- Images
- Tool calls (with permission requests)
- Following
- Edit review
- TODO lists
- Slash commands:
  - /review (with optional instructions)
  - /review-branch
  - /review-commit
  - /init
  - /compact
  - /logout
  - Custom Prompts
- Client MCP servers
- Auth Methods:
  - ChatGPT subscription (requires paid subscription and doesn't work in remote projects)
  - CODEX_API_KEY
  - OPENAI_API_KEY

Learn more about the [Agent Client Protocol](https://agentclientprotocol.com/).

## How to use

### Zed

The latest version of Zed can already use this adapter out of the box.

To use Codex, open the Agent Panel and click "New Codex Thread" from the `+` button menu in the top-right.

Read the docs on [External Agent](https://zed.dev/docs/ai/external-agents) support.

### Other clients

[Submit a PR](https://github.com/zed-industries/codex-acp/pulls) to add yours!

#### Installation

Install the adapter from the latest release for your architecture and OS: https://github.com/zed-industries/codex-acp/releases

You can then use `codex-acp` as a regular ACP agent:

```
OPENAI_API_KEY=sk-... codex-acp
```

Or via npm:

```
npx @zed-industries/codex-acp
```

### Session persistence

Enable persistent sessions with either:

- `codex-acp --session-persist` (or `--session-persist=/custom/path`)
- Environment variables: `CODEX_SESSION_PERSIST=1` and optionally `CODEX_SESSION_DIR=/custom/path`

By default, metadata lives alongside rollout JSONL files under `${CODEX_HOME}/sessions`. The
manifest keeps track of the rollout path, model/mode overrides, and MCP servers so `/session/load`
can resume a conversation after the agent restarts. Disable persistence with `--no-session-persist`
or `CODEX_SESSION_PERSIST=0` if you want the in-memory behavior back.

## License

Apache-2.0
