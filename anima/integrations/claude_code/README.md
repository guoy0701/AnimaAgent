# AnimaAgent for Claude Code

让 Claude Code 获得"记忆和个性化"能力。安装后，Claude Code 会记住你的工作习惯，越用越懂你。

## 安装

```bash
pip install anima-agent[claude-code,qwen]   # 或 [claude-code,openai]
```

## 配置

在项目的 `.claude/settings.json` 中添加 MCP Server：

```json
{
  "mcpServers": {
    "anima": {
      "command": "python",
      "args": ["-m", "anima.integrations.claude_code.mcp_server"],
      "env": {
        "ANIMA_AGENT_NAME": "我的助手",
        "ANIMA_API_KEY": "你的API密钥",
        "ANIMA_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "ANIMA_CHAT_MODEL": "qwen-plus",
        "ANIMA_EMBED_MODEL": "text-embedding-v3"
      }
    }
  }
}
```

## 暴露的工具

| 工具 | 用途 |
|------|------|
| `anima_think` | 获取个性化上下文（历史经验 + 策略建议 + 能力画像） |
| `anima_feedback` | 记录反馈，让 Agent 学习 |
| `anima_status` | 查看 Agent 状态 |
| `anima_register_skill` | 注册新技能 |

## 工作原理

```
用户输入任务
    │
    ▼
Claude Code 调用 anima_think(任务描述)
    │
    ▼
AnimaAgent 查经验图谱 → 选策略 → 生成个性化上下文
    │
    ▼
Claude Code 参考上下文回答（更个性化、更懂你）
    │
    ▼
用户满意 → Claude Code 调用 anima_feedback(0.9)
    │
    ▼
AnimaAgent 学习更新 → 下次更好
```

## 无 API Key 模式

不设置 `ANIMA_API_KEY` 时，自动使用 Mock 模式（基于字符重叠的 embedding + 基于关键词的概念提取）。功能正常但语义理解能力有限。
