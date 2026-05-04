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

### JetBrains IDEs

JetBrains IDEs expose ACP agents through the JetBrains AI Assistant agent registry. That registry is separate from Zed's registry, so Codex may not appear in the built-in JetBrains agent list even though this adapter is available in Zed.

Add Codex manually by creating or updating `~/.jetbrains/acp.json`:

```json
{
  "default_mcp_settings": {
    "use_custom_mcp": true,
    "use_idea_mcp": true
  },
  "agent_servers": {
    "Codex": {
      "command": "npx",
      "args": ["-y", "@zed-industries/codex-acp"],
      "env": {}
    }
  }
}
```

If you installed a release binary instead of using npm, set `command` to the absolute path of the `codex-acp` binary and remove `args`.

Some JetBrains AI Assistant builds still expect the older ACP `env_var.varName` auth field and may fail to initialize when the agent advertises API-key auth methods. If you already authenticate Codex through the regular Codex CLI or have a `CODEX_HOME` with valid credentials, you can hide those env-var auth methods for JetBrains:

```json
{
  "default_mcp_settings": {
    "use_custom_mcp": true,
    "use_idea_mcp": true
  },
  "agent_servers": {
    "Codex": {
      "command": "npx",
      "args": ["-y", "@zed-industries/codex-acp"],
      "env": {
        "CODEX_ACP_DISABLE_ENV_AUTH_METHODS": "1"
      }
    }
  }
}
```

Restart the IDE or start a new AI Assistant chat after changing `~/.jetbrains/acp.json`.

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

## License

Apache-2.0
