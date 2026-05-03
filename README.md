# Anima — 有灵魂的AI Agent框架

> **Agent不是工具，是"人"。**

Anima 是一个 AI Agent 个性层框架。它让 Agent 拥有记忆和成长性——越用越懂你。

核心能力：
- **经验图谱**：用图结构记忆（不是向量数据库），通过激活扩散决定面对新任务时"想起"什么
- **策略网络**：从你的反馈中学习"怎么做事"——哪些方法有效、哪些技能好用
- **能力画像**：自动形成 Agent 的"身份"——擅长什么、什么做事风格

两个 Agent 学了同样的技能，但因为跟不同的主人工作，会逐渐表现出完全不同的行为模式。

---

## 两种使用方式

| 方式 | 适合谁 | 说明 |
|------|--------|------|
| **独立使用** | 想要一个有记忆的 AI 助手 | 直接对话，或在你的代码里调用 |
| **作为插件** | 已经在用 Claude Code 等 Agent | 给现有 Agent 加上记忆能力 |

---

## 方式一：独立使用

### 安装

```bash
pip install anima-agent
pip install openai          # 用于连接 Qwen/DeepSeek/GPT 等 API
```

### 配置

在项目目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# .env
ANIMA_API_KEY=sk-你的API密钥
ANIMA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1    # 通义千问
ANIMA_CHAT_MODEL=qwen-plus
ANIMA_EMBED_MODEL=text-embedding-v3
ANIMA_AGENT_NAME=我的助手
```

<details>
<summary>其他 API 的配置</summary>

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
（OpenAI 不需要设置 ANIMA_BASE_URL）
</details>

### 命令行对话

```bash
python -m anima
```

```
═══════════════════════════════════════════════════
  Anima — 有灵魂的AI助手
  Agent: 我的助手
  模型: qwen-plus
═══════════════════════════════════════════════════

你: 帮我分析用户留存数据
我的助手: [基于历史经验的个性化回答]

你: /feedback 9       # 给回答打分（0-10），Agent 从中学习
你: /status           # 查看 Agent 成长状态
你: /sleep            # 让 Agent 整理记忆
你: /help             # 查看所有命令
你: /quit             # 退出
```

无 API key 时可用 Mock 模式体验框架效果：`python -m anima --mock`

### 在代码中使用

```python
from anima import AnimaAgent
from anima.provider import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    chat_model="qwen-plus",
    embed_model="text-embedding-v3",
)

agent = AnimaAgent("我的助手")
agent.configure(provider)

response = agent.chat("帮我分析用户留存数据")
agent.feedback(0.9)    # Agent 从反馈中学习
```

---

## 方式二：作为 Claude Code 插件

让 Claude Code 拥有"记忆"——它会记住你的工作习惯，越用越懂你。

### 安装

```bash
pip install anima-agent
pip install openai          # 用于语义理解
```

### 配置

**第一步：** 在项目目录创建 `.env` 文件（同上）。

**第二步：** 在 `.claude/settings.json` 中添加 MCP Server：

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

**第三步：** 重启 Claude Code。

安装后 Claude Code 会多出 4 个工具：

| 工具 | 作用 |
|------|------|
| `anima_think` | 处理任务前获取个性化上下文（历史经验 + 策略建议） |
| `anima_feedback` | 记录反馈，让 Agent 学习 |
| `anima_status` | 查看 Agent 成长状态 |
| `anima_register_skill` | 注册新技能 |

Claude Code 会自动在合适的时候调用这些工具。你不需要改变使用习惯。

---

## 卸载

### 如果是独立使用

```bash
pip uninstall anima-agent
```

### 如果是 Claude Code 插件

```bash
# 第一步：从 .claude/settings.json 中删除 "anima" 配置段
# 第二步：重启 Claude Code（此时 Claude Code 完全恢复原样）
# 第三步：卸载包
pip uninstall anima-agent
```

### 关于 Agent 数据

卸载后，`anima_data/` 目录中的 Agent 数据（经验、策略、能力画像）会保留在磁盘上。

- **保留数据**：下次重新安装后，Agent 从上次的状态继续成长
- **删除数据**：`rm -rf anima_data/`，Agent 彻底消失，重装也是全新的

**卸载不会影响 Claude Code 或任何其他 Agent。** AnimaAgent 是一个独立进程，卸载后不留任何钩子或后台程序。

---

## 支持的 LLM

AnimaAgent 不绑定任何特定大模型。通过 OpenAI 兼容接口，支持所有主流 API：

| 平台 | 支持情况 |
|------|---------|
| 通义千问（Qwen） | ✅ |
| DeepSeek | ✅ |
| OpenAI (GPT) | ✅ |
| Ollama (本地模型) | ✅ |
| vLLM | ✅ |
| 任何 OpenAI 格式 API | ✅ |

---

## 开发

```bash
git clone https://github.com/guoy0701/AnimaAgent
cd AnimaAgent
pip install -e ".[dev]"
python -m pytest tests/ -v          # 83 个测试
python -X utf8 dry_comparison.py    # 干对比实验（无需 API key）
```

## 设计哲学

> 大模型是基础设施，Skill是教材，Agent本身才是"人"。

Agent 的真正价值不在于装了多少 Skill，而在于"底子"有多厚。这个底子是被时间和经历塑造的，不可压缩、不可轻易复制。

## License

MIT
