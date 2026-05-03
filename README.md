**English** | [中文](README_CN.md)

# Anima — AI Agent Personality Framework

> **Agents are not tools, they are "people".**

> **Alpha** — Core features working. Feedback welcome.

Anima is a personality layer for AI agents. It can be used as a **plugin for Claude Code** (or other MCP-compatible agents) or as a **standalone AI assistant**. After installation, your agent gains memory and grows with you — the more you use it, the better it understands you.

**What does it do?** It inserts a "personality layer" between you and the LLM:
- **Experience Graph**: Graph-structured memory with spreading activation — determines what the agent "recalls" for each new task
- **Strategy Network**: Learns from your feedback how to approach tasks — which methods work, which skills to prefer
- **Competence Embedding**: Automatically forms the agent's "identity" — strengths, work style, domain expertise

Two agents with identical skills but different owners will gradually develop completely different behavior patterns.

**LLM-agnostic** — works with Qwen, DeepSeek, GPT, Ollama, and any OpenAI-compatible API.

---

## Two Ways to Use

| Mode | For whom | Description |
|------|----------|-------------|
| **Standalone** | Want an AI assistant with memory | Chat directly, or call from your code |
| **Plugin** | Already using Claude Code, etc. | Add memory to your existing agent |

---

## Mode 1: Standalone

### Install

```bash
git clone https://github.com/guoy0701/AnimaAgent.git
cd AnimaAgent
pip install -e .
pip install openai          # For connecting to Qwen/DeepSeek/GPT APIs
```

### Configure

Create a `.env` file in the project directory (see `.env.example`):

```bash
# .env
ANIMA_API_KEY=sk-your-api-key
ANIMA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1    # Qwen
ANIMA_CHAT_MODEL=qwen-plus
ANIMA_EMBED_MODEL=text-embedding-v4
ANIMA_AGENT_NAME=MyAssistant
```

<details>
<summary>Other API configurations</summary>

**DeepSeek**
```
ANIMA_API_KEY=sk-xxx
ANIMA_BASE_URL=https://api.deepseek.com
ANIMA_CHAT_MODEL=deepseek-chat
ANIMA_EMBED_MODEL=deepseek-chat
```

**OpenAI**
```
ANIMA_API_KEY=sk-xxx
ANIMA_CHAT_MODEL=gpt-4o-mini
ANIMA_EMBED_MODEL=text-embedding-3-small
```
(No ANIMA_BASE_URL needed for OpenAI)
</details>

### CLI Chat

```bash
python -m anima
```

```
═══════════════════════════════════════════════════
  Anima — AI Assistant with Soul
  Agent: MyAssistant
  Model: qwen-plus
═══════════════════════════════════════════════════

You: Analyze user retention data
MyAssistant: [Personalized response based on past experience]

You: /feedback 9       # Rate the response (0-10), agent learns from it
You: /status           # View agent growth status
You: /sleep            # Consolidate memories
You: /help             # Show all commands
You: /quit             # Exit
```

Try without an API key using mock mode: `python -m anima --mock`

### Use in Code

```python
from anima import AnimaAgent
from anima.provider import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    chat_model="qwen-plus",
    embed_model="text-embedding-v4",
)

agent = AnimaAgent("MyAssistant")
agent.configure(provider)

response = agent.chat("Analyze user retention data")
agent.feedback(0.9)    # Agent learns from feedback
```

---

## Mode 2: As a Claude Code Plugin

Give Claude Code "memory" — it remembers your work habits and gets better over time.

### Install

```bash
git clone https://github.com/guoy0701/AnimaAgent.git
cd AnimaAgent
pip install -e .
pip install openai
```

### Configure

**Step 1:** Create a `.env` file in the project directory (same as above).

**Step 2:** Add MCP Server to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "anima": {
      "command": "python",
      "args": ["-m", "anima.integrations.claude_code.mcp_server"]
    }
  }
}
```

**Step 3:** Restart Claude Code.

After installation, Claude Code gains 4 new tools:

| Tool | Purpose |
|------|---------|
| `anima_think` | Get personalized context before handling a task (past experiences + strategy) |
| `anima_feedback` | Record feedback so the agent learns |
| `anima_status` | View agent growth status |
| `anima_register_skill` | Register a new skill |

Claude Code will automatically use these tools at the right time. No change to your workflow needed.

---

## Uninstall

### Standalone

```bash
pip uninstall anima-agent
```

### Claude Code Plugin

```bash
# Step 1: Remove "anima" section from .claude/settings.json
# Step 2: Restart Claude Code (fully restored to original state)
# Step 3: Uninstall the package
pip uninstall anima-agent
```

### About Agent Data

After uninstalling, agent data in `anima_data/` (experiences, strategies, competence profile) remains on disk.

- **Keep data**: Next time you reinstall, the agent resumes from where it left off
- **Delete data**: `rm -rf anima_data/` — the agent is permanently gone

**Uninstalling does not affect Claude Code or any other agent.** AnimaAgent runs as a separate process — no hooks, no background services left behind.

---

## Supported LLMs

AnimaAgent is not tied to any specific LLM. Via the OpenAI-compatible interface, it works with all major APIs:

| Platform | Status |
|----------|--------|
| Qwen (通义千问) | ✅ |
| DeepSeek | ✅ |
| OpenAI (GPT) | ✅ |
| Ollama (local models) | ✅ |
| vLLM | ✅ |
| Any OpenAI-format API | ✅ |

---

## Development

```bash
git clone https://github.com/guoy0701/AnimaAgent
cd AnimaAgent
pip install -e ".[dev]"
python -m pytest tests/ -v          # 83 tests
python -X utf8 dry_comparison.py    # Dry comparison experiment (no API key needed)
```

## Design Philosophy

> LLMs are infrastructure. Skills are textbooks. The Agent itself is the "person".

An agent's true value is not how many skills it has installed, but how deep its foundation is. That foundation is shaped by time and experience — it cannot be compressed or easily replicated.

## License

MIT
