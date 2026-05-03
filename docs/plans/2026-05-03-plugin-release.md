# AnimaAgent Plugin Release Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 AnimaAgent 的 MCP 插件达到可发布状态：真实可用、容易安装、容易卸载、有 .env 配置、有错误处理、发到 PyPI。

**Architecture:** MCP Server 是一个独立 Python 进程，通过 stdio 与宿主 Agent 通信。所有状态存在本地 `anima_data/` 目录。支持 .env 文件配置。安装和卸载都不修改宿主 Agent 的任何文件（用户手动加/删 MCP 配置）。

**Tech Stack:** Python 3.10+, mcp SDK, python-dotenv, 现有 anima 模块

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `anima/dotenv_loader.py` | 加载 .env 文件中的环境变量 |
| Modify | `anima/__main__.py` | 启动时加载 .env |
| Modify | `anima/integrations/claude_code/mcp_server.py` | 错误处理、.env 加载、健壮性 |
| Create | `.env.example` | 配置模板，用户复制为 .env |
| Create | `tests/test_mcp_server.py` | MCP Server 单元测试 |
| Modify | `pyproject.toml` | 加入 mcp + python-dotenv 依赖，配置 PyPI 元数据 |
| Modify | `anima/__init__.py` | 确保 __version__ 正确 |
| Modify | `README.md` | 加入完整的安装/卸载指南 |

---

### Task 1: .env 文件支持

**Files:**
- Create: `.env.example`
- Create: `anima/dotenv_loader.py`
- Modify: `anima/__main__.py`
- Modify: `anima/integrations/claude_code/mcp_server.py`

- [ ] **Step 1: Create .env.example**

```
# AnimaAgent 配置
# 复制此文件为 .env 并填入你的值

# LLM API 配置（必需）
ANIMA_API_KEY=

# API 地址（可选，不填则使用 OpenAI 默认地址）
# 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1
# DeepSeek: https://api.deepseek.com
ANIMA_BASE_URL=

# 模型名称（可选）
# 通义千问: qwen-plus / text-embedding-v3
# DeepSeek: deepseek-chat / deepseek-chat
# OpenAI: gpt-4o-mini / text-embedding-3-small
ANIMA_CHAT_MODEL=qwen-plus
ANIMA_EMBED_MODEL=text-embedding-v3

# Agent 名称（可选）
ANIMA_AGENT_NAME=Anima
```

- [ ] **Step 2: Create anima/dotenv_loader.py**

```python
# anima/dotenv_loader.py
"""
加载 .env 文件中的环境变量。
不依赖 python-dotenv——用纯 Python 实现，保持零外部依赖的核心原则。
"""

import os
from pathlib import Path


def load_dotenv(start_dir: str = None):
    """从当前目录或指定目录向上查找 .env 文件并加载。"""
    search_dir = Path(start_dir) if start_dir else Path.cwd()

    for directory in [search_dir] + list(search_dir.parents):
        env_file = directory / ".env"
        if env_file.exists():
            _parse_env_file(env_file)
            return str(env_file)
    return None


def _parse_env_file(filepath: Path):
    """解析 .env 文件，只设置尚未存在的环境变量（不覆盖已有值）。"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
```

- [ ] **Step 3: Modify anima/__main__.py — add .env loading at startup**

At the very beginning of `main()`, before creating provider:

```python
from .dotenv_loader import load_dotenv
env_path = load_dotenv()
if env_path:
    print(f"[Anima] 已加载配置: {env_path}")
```

- [ ] **Step 4: Modify MCP Server — add .env loading**

At the top of `anima/integrations/claude_code/mcp_server.py`, in `_get_agent()`, before reading env vars:

```python
from anima.dotenv_loader import load_dotenv
load_dotenv()
```

- [ ] **Step 5: Add .env to .gitignore**

Append to `.gitignore`:
```
.env
```

- [ ] **Step 6: Run tests, commit**

```bash
python -m pytest tests/ -v --tb=short
git add .env.example anima/dotenv_loader.py anima/__main__.py anima/integrations/claude_code/mcp_server.py .gitignore
git commit -m "feat: add .env file support for configuration"
```

---

### Task 2: MCP Server 错误处理和健壮性

**Files:**
- Modify: `anima/integrations/claude_code/mcp_server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_mcp_server.py
"""测试 MCP Server 的工具函数（不启动真实 MCP 连接）。"""
import os
import json
from unittest.mock import patch, MagicMock


class TestAnimaThink:
    def test_returns_valid_json(self):
        # 需要 mock 环境让 _get_agent 不需要真实 API
        with patch.dict(os.environ, {}, clear=False):
            from anima.integrations.claude_code.mcp_server import anima_think, _get_agent
            # 先 reset agent
            import anima.integrations.claude_code.mcp_server as srv
            srv._agent = None

            result = anima_think(task="测试任务")
            data = json.loads(result)
            assert "system_prompt_addition" in data
            assert "task_category" in data

    def test_handles_empty_task(self):
        import anima.integrations.claude_code.mcp_server as srv
        srv._agent = None
        result = anima_think(task="")
        data = json.loads(result)
        assert isinstance(data, dict)


class TestAnimaFeedback:
    def test_accepts_valid_reward(self):
        import anima.integrations.claude_code.mcp_server as srv
        srv._agent = None
        # Need to call think first to set current_task
        anima_think(task="测试任务")
        from anima.integrations.claude_code.mcp_server import anima_feedback
        result = anima_feedback(reward=0.8)
        assert "已记录" in result

    def test_handles_no_current_task(self):
        import anima.integrations.claude_code.mcp_server as srv
        srv._agent = None
        from anima.integrations.claude_code.mcp_server import anima_feedback
        # feedback without prior think — should not crash
        result = anima_feedback(reward=0.5)
        assert isinstance(result, str)


class TestAnimaStatus:
    def test_returns_valid_json(self):
        import anima.integrations.claude_code.mcp_server as srv
        srv._agent = None
        from anima.integrations.claude_code.mcp_server import anima_status
        result = anima_status()
        data = json.loads(result)
        assert "agent_name" in data
        assert "interactions" in data


class TestAnimaRegisterSkill:
    def test_registers_skill(self):
        import anima.integrations.claude_code.mcp_server as srv
        srv._agent = None
        from anima.integrations.claude_code.mcp_server import anima_register_skill
        result = anima_register_skill(name="test_skill", description="测试技能")
        assert "test_skill" in result
```

- [ ] **Step 2: Add error handling to MCP Server tools**

Wrap each tool function with try/except:

```python
@mcp.tool()
def anima_think(task: str) -> str:
    """获取 AnimaAgent 的个性化上下文..."""
    try:
        agent = _get_agent()
        context = agent.think(task)
        result = { ... }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
```

Do this for ALL 4 tools.

- [ ] **Step 3: Handle feedback without prior think gracefully**

In `anima_feedback`, check if agent has current task:

```python
@mcp.tool()
def anima_feedback(reward: float, ...) -> str:
    try:
        agent = _get_agent()
        if agent._current_task is None:
            return "没有待反馈的任务。请先调用 anima_think 处理一个任务。"
        # ... existing logic
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_mcp_server.py -v
python -m pytest tests/ --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add anima/integrations/claude_code/mcp_server.py tests/test_mcp_server.py
git commit -m "fix: add error handling to MCP Server, add MCP tool tests"
```

---

### Task 3: PyPI 发布准备

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Update pyproject.toml with full metadata**

```toml
[project]
name = "anima-agent"
version = "0.2.0"
description = "有灵魂的AI Agent框架 — 让Agent不再是工具，是人"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "AnimaAgent Team", email = "guoy0702@gmail.com"},
]
keywords = ["ai", "agent", "personality", "memory", "llm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "jieba>=0.42",
    "mcp>=1.0",
]

[project.optional-dependencies]
anthropic = [
    "anthropic>=0.49",
    "voyageai>=0.3",
]
dev = [
    "pytest>=8.0",
]

[project.urls]
Homepage = "https://github.com/你的用户名/AnimaAgent"
Repository = "https://github.com/你的用户名/AnimaAgent"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
include = ["anima*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

注意：`openai` 不放在 dependencies 里（用户自己装），`mcp` 放在 dependencies 里（插件核心依赖）。

- [ ] **Step 2: Update .gitignore for build artifacts**

Append:
```
dist/
build/
*.egg-info/
```

- [ ] **Step 3: Test build**

```bash
pip install build
python -m build
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: prepare pyproject.toml for PyPI release"
```

---

### Task 4: 安装/卸载指南 + README 更新

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add install/uninstall guide to README**

Add after the quick start section:

```markdown
## 安装

```bash
pip install anima-agent
```

如果要使用 OpenAI 兼容 API（Qwen/DeepSeek/GPT 等），还需要：
```bash
pip install openai
```

## 作为 Claude Code 插件使用

1. 在项目目录创建 `.env` 文件（参考 `.env.example`）
2. 在 `.claude/settings.json` 中添加 MCP Server 配置
3. 重启 Claude Code

详见 [Claude Code 插件文档](anima/integrations/claude_code/README.md)

## 卸载

```bash
# 1. 删除 MCP 配置（如果用了插件模式）
#    从 .claude/settings.json 中删除 "anima" 配置段

# 2. 卸载包
pip uninstall anima-agent

# 3. 删除 Agent 数据（可选，删了就丢失所有经验）
rm -rf anima_data/
```

卸载后宿主 Agent 不受任何影响。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add install/uninstall guide"
```

---

### Task 5: End-to-End 验证

- [ ] **Step 1: 在 Mock 模式下验证完整插件流程**

```bash
# 启动 MCP Server（另一个终端）
python -m anima.integrations.claude_code.mcp_server

# 或者直接测试工具函数
python -c "
from anima.integrations.claude_code.mcp_server import anima_think, anima_feedback, anima_status, anima_register_skill
print('=== register ===')
print(anima_register_skill('coding', '编程'))
print('=== think ===')
print(anima_think('帮我写一个排序算法'))
print('=== feedback ===')
print(anima_feedback(0.9, skills_used='coding'))
print('=== status ===')
print(anima_status())
"
```

- [ ] **Step 2: Run ALL tests**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all pass

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "release: AnimaAgent v0.2.0 - plugin ready for release"
```

---

## Post-Release

1. **发到 PyPI**: `python -m build && twine upload dist/*`
2. **发到 GitHub**: push + create release tag
3. **收集反馈**: 看用户安装后的实际体验
4. **独立 Agent 方向**: 根据反馈决定是否投入
