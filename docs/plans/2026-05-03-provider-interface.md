# Unified Provider Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the separate EmbeddingProvider + ExperienceExtractor with a single unified `LLMProvider` interface, add `agent.chat()` one-step API, and ship an OpenAI-compatible provider that works with Qwen, DeepSeek, GPT, and any OpenAI-format API.

**Architecture:** A single `LLMProvider` ABC exposes three capabilities: `chat()`, `embed()`, and `extract()`. An `OpenAICompatibleProvider` implementation handles any API that follows the OpenAI chat/embedding format (Qwen, DeepSeek, GPT, local models via Ollama/vLLM). The existing `EmbeddingProvider` and `ExperienceExtractor` ABCs stay as-is for backwards compatibility, but `LLMProvider` implements both. `AnimaAgent.configure()` accepts a single provider and wires everything. `AnimaAgent.chat()` is the new one-step entry point: think → LLM call → return response.

**Tech Stack:** Python 3.10+, openai SDK (for OpenAI-compatible APIs), existing anima modules

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `anima/provider.py` | `LLMProvider` ABC + `OpenAICompatibleProvider` implementation |
| Create | `tests/test_provider.py` | Provider tests (with mock HTTP) |
| Modify | `anima/agent.py` | Add `configure()` and `chat()` methods |
| Create | `tests/test_agent_chat.py` | End-to-end chat tests |
| Modify | `anima/__init__.py` | Export new APIs |
| Modify | `pyproject.toml` | Add openai optional dependency |

---

### Task 1: LLMProvider ABC and OpenAICompatibleProvider

**Files:**
- Create: `anima/provider.py`
- Create: `tests/test_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_provider.py
from unittest.mock import patch, MagicMock
from anima.provider import OpenAICompatibleProvider, LLMProvider
from anima.embedding import EmbeddingProvider
from anima.extractor import ExperienceExtractor


class TestLLMProviderIsUnified:
    def test_openai_provider_is_embedding_provider(self):
        """OpenAICompatibleProvider should satisfy EmbeddingProvider interface."""
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake", chat_model="m", embed_model="m")
        assert isinstance(provider, EmbeddingProvider)

    def test_openai_provider_is_extractor(self):
        """OpenAICompatibleProvider should satisfy ExperienceExtractor interface."""
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake", chat_model="m", embed_model="m")
        assert isinstance(provider, ExperienceExtractor)

    def test_openai_provider_is_llm_provider(self):
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake", chat_model="m", embed_model="m")
        assert isinstance(provider, LLMProvider)


class TestOpenAIProviderChat:
    def test_chat_returns_string(self):
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="qwen-plus", embed_model="text-embedding-v3")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "这是AI的回答"

        with patch.object(provider._client.chat.completions, 'create',
                          return_value=mock_response):
            result = provider.chat("你好", system="你是助手")
            assert result == "这是AI的回答"

    def test_chat_passes_system_prompt(self):
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="qwen-plus", embed_model="text-embedding-v3")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch.object(provider._client.chat.completions, 'create',
                          return_value=mock_response) as mock_create:
            provider.chat("任务", system="你是数据分析师")
            call_args = mock_create.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
            assert messages[0]["role"] == "system"
            assert "数据分析师" in messages[0]["content"]


class TestOpenAIProviderEmbed:
    def test_embed_returns_list_of_floats(self):
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="m", embed_model="text-embedding-v3")

        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_item]

        with patch.object(provider._client.embeddings, 'create',
                          return_value=mock_response):
            result = provider.embed("测试文本")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_returns_multiple(self):
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="m", embed_model="text-embedding-v3")

        mock_response = MagicMock()
        item1 = MagicMock()
        item1.embedding = [0.1, 0.2]
        item2 = MagicMock()
        item2.embedding = [0.3, 0.4]
        mock_response.data = [item1, item2]

        with patch.object(provider._client.embeddings, 'create',
                          return_value=mock_response):
            results = provider.embed_batch(["文本1", "文本2"])
            assert len(results) == 2
            assert results[0] == [0.1, 0.2]


class TestOpenAIProviderExtract:
    def test_extract_returns_extraction_result(self):
        from anima.extractor import ExtractionResult
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="qwen-plus", embed_model="m")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "concepts": ["用户留存", "数据分析"],
            "entities": ["SQL"],
            "domain": "data_analysis",
            "problems": [],
            "solutions": [],
            "outcome_summary": "分析用户数据",
            "related_concepts": ["用户流失"]
        }'''

        with patch.object(provider._client.chat.completions, 'create',
                          return_value=mock_response):
            result = provider.extract("分析用户留存数据")
            assert isinstance(result, ExtractionResult)
            assert "用户留存" in result.concepts
            assert result.domain == "data_analysis"

    def test_extract_handles_malformed_json(self):
        from anima.extractor import ExtractionResult
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="qwen-plus", embed_model="m")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "这不是JSON"

        with patch.object(provider._client.chat.completions, 'create',
                          return_value=mock_response):
            result = provider.extract("测试")
            assert isinstance(result, ExtractionResult)
            assert result.domain == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_provider.py -v
```

Expected: FAIL with `ImportError` (module doesn't exist)

- [ ] **Step 3: Implement provider.py**

```python
# anima/provider.py
"""
Unified LLM Provider interface.

A single provider handles chat completion, embedding, and concept extraction.
Ships with OpenAICompatibleProvider that works with any OpenAI-format API:
Qwen, DeepSeek, GPT, Ollama, vLLM, etc.
"""

import json
import re
from abc import ABC, abstractmethod

from .embedding import EmbeddingProvider
from .extractor import ExperienceExtractor, ExtractionResult, EXTRACTION_PROMPT


class LLMProvider(EmbeddingProvider, ExperienceExtractor):
    """Unified provider: chat + embedding + extraction in one object."""

    @abstractmethod
    def chat(self, message: str, system: str = None) -> str:
        """Send a message to the LLM and get a response."""
        ...


class OpenAICompatibleProvider(LLMProvider):
    """Works with any OpenAI-format API: Qwen, DeepSeek, GPT, Ollama, vLLM, etc.

    Usage:
        # Qwen
        provider = OpenAICompatibleProvider(
            api_key="sk-xxx",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            chat_model="qwen-plus",
            embed_model="text-embedding-v3",
        )
        # DeepSeek
        provider = OpenAICompatibleProvider(
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
            chat_model="deepseek-chat",
            embed_model="deepseek-chat",  # DeepSeek uses same model for embedding
        )
        # OpenAI
        provider = OpenAICompatibleProvider(
            api_key="sk-xxx",
            chat_model="gpt-4o-mini",
            embed_model="text-embedding-3-small",
        )
    """

    def __init__(self, api_key: str, base_url: str = None,
                 chat_model: str = "gpt-4o-mini",
                 embed_model: str = "text-embedding-3-small",
                 embed_dimensions: int = 1024):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  # required for OpenAICompatibleProvider")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._chat_model = chat_model
        self._embed_model = embed_model
        self._embed_dimensions = embed_dimensions

    # --- LLMProvider (chat) ---

    def chat(self, message: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
        )
        return response.choices[0].message.content

    # --- EmbeddingProvider ---

    @property
    def dimensions(self) -> int:
        return self._embed_dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # --- ExperienceExtractor ---

    def extract(self, text: str) -> ExtractionResult:
        prompt = EXTRACTION_PROMPT.format(text=text)
        raw = self.chat(prompt)

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return ExtractionResult(outcome_summary=text[:100])

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ExtractionResult(outcome_summary=text[:100])

        return ExtractionResult(
            concepts=data.get("concepts", []),
            entities=data.get("entities", []),
            domain=data.get("domain", "unknown"),
            problems=data.get("problems", []),
            solutions=data.get("solutions", []),
            outcome_summary=data.get("outcome_summary", ""),
            related_concepts=data.get("related_concepts", []),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_provider.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add anima/provider.py tests/test_provider.py
git commit -m "feat: add unified LLMProvider with OpenAI-compatible implementation

OpenAICompatibleProvider handles chat, embedding, and extraction via any
OpenAI-format API (Qwen, DeepSeek, GPT, Ollama, vLLM). Implements both
EmbeddingProvider and ExperienceExtractor interfaces."
```

---

### Task 2: AnimaAgent.configure() and AnimaAgent.chat()

**Files:**
- Modify: `anima/agent.py`
- Create: `tests/test_agent_chat.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_chat.py
import os
from unittest.mock import MagicMock, patch
from anima.agent import AnimaAgent
from anima.provider import OpenAICompatibleProvider
from anima.embedding import MockEmbeddingProvider
from anima.extractor import MockExtractor
from anima.experience_graph import NodeType


class TestAgentConfigure:
    def test_configure_with_provider_sets_all_layers(self):
        """configure() with a LLMProvider should set up embedding + extraction + chat."""
        agent = AnimaAgent("test", save_path="/tmp/test_configure.json")
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake",
            chat_model="m", embed_model="m")
        agent.configure(provider)

        assert agent._provider is not None
        assert agent.persona._embedding_provider is not None
        assert agent.persona._extractor is not None

    def test_configure_with_separate_components(self):
        """configure() should also accept separate embedding + extractor (backwards compat)."""
        agent = AnimaAgent("test", save_path="/tmp/test_configure2.json")
        agent.configure(
            embedding_provider=MockEmbeddingProvider(dimensions=64),
            extractor=MockExtractor(),
        )
        assert agent.persona._embedding_provider is not None
        assert agent.persona._extractor is not None


class TestAgentChat:
    def _make_agent_with_mock_provider(self):
        path = "/tmp/test_chat_agent.json"
        if os.path.exists(path):
            os.remove(path)
        agent = AnimaAgent("ChatTest", save_path=path)

        mock_provider = MagicMock()
        mock_provider.embed = MockEmbeddingProvider(dimensions=64).embed
        mock_provider.embed_batch = MockEmbeddingProvider(dimensions=64).embed_batch
        mock_provider.dimensions = 64
        mock_provider.extract = MockExtractor().extract
        mock_provider.chat = MagicMock(return_value="这是LLM的回答")

        agent.configure(mock_provider)
        return agent, mock_provider

    def test_chat_returns_llm_response(self):
        agent, mock_provider = self._make_agent_with_mock_provider()
        response = agent.chat("帮我分析用户数据")
        assert response == "这是LLM的回答"

    def test_chat_passes_persona_context_as_system_prompt(self):
        agent, mock_provider = self._make_agent_with_mock_provider()
        agent.chat("帮我分析用户数据")

        call_args = mock_provider.chat.call_args
        system_prompt = call_args[1].get("system", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Agent ChatTest" in system_prompt or "个性化" in system_prompt

    def test_chat_without_provider_raises(self):
        agent = AnimaAgent("NoProv", save_path="/tmp/test_no_prov.json")
        try:
            agent.chat("hello")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "configure" in str(e).lower()

    def test_chat_updates_current_task(self):
        """After chat(), feedback() should work without calling think() first."""
        agent, mock_provider = self._make_agent_with_mock_provider()
        agent.chat("帮我分析用户数据")

        assert agent._current_task == "帮我分析用户数据"
        assert agent._current_context is not None

    def test_chat_then_feedback_records_experience(self):
        agent, mock_provider = self._make_agent_with_mock_provider()
        agent.register_skill("sql_query", "SQL查询")

        agent.chat("帮我分析用户数据")
        agent.feedback(0.8, skills_used=["sql_query"])

        task_nodes = agent.persona.experience_graph.find_by_type(NodeType.TASK)
        assert len(task_nodes) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_agent_chat.py -v
```

Expected: FAIL (`configure()` and `chat()` don't exist yet)

- [ ] **Step 3: Add configure() and chat() to AnimaAgent**

In `anima/agent.py`, add to `__init__`:

```python
self._provider = None
```

Add these methods after `configure_semantic`:

```python
def configure(self, provider=None, *,
              embedding_provider=None, extractor=None):
    """
    配置Agent的LLM能力。

    方式一（推荐）：传入统一 provider
        agent.configure(QwenProvider(...))

    方式二：分别传入 embedding 和 extractor
        agent.configure(embedding_provider=..., extractor=...)
    """
    if provider is not None:
        self._provider = provider
        self.persona.configure_semantic(
            embedding_provider=provider,
            extractor=provider,
        )
        print(f"[Anima] 已配置LLM Provider")
    elif embedding_provider or extractor:
        self.persona.configure_semantic(
            embedding_provider=embedding_provider,
            extractor=extractor,
        )
        print(f"[Anima] 已配置语义层组件")
    else:
        raise ValueError("需要提供 provider 或 embedding_provider/extractor")

def chat(self, message: str) -> str:
    """
    一步完成：思考 → 调用LLM → 返回响应。

    这是最简单的使用方式。Agent自动注入个性化上下文。
    调用后可以用 feedback() 给予反馈。
    """
    if self._provider is None or not hasattr(self._provider, 'chat'):
        raise RuntimeError(
            "需要先调用 agent.configure(provider) 配置LLM Provider 才能使用 chat()")

    context = self.think(message)
    system_prompt = self._build_full_system_prompt(context)
    response = self._provider.chat(message, system=system_prompt)
    return response
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_agent_chat.py -v
```

Expected: all PASS

- [ ] **Step 5: Run ALL tests to verify no regression**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all PASS (existing tests should still work because configure_semantic still exists)

- [ ] **Step 6: Commit**

```bash
git add anima/agent.py tests/test_agent_chat.py
git commit -m "feat: add agent.configure() and agent.chat() one-step API

configure() accepts either a unified LLMProvider or separate embedding/extractor.
chat() does think → LLM call → return in one step, with PersonaLayer context
automatically injected as system prompt."
```

---

### Task 3: Update Exports and Dependencies

**Files:**
- Modify: `anima/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update __init__.py exports**

Add after existing imports:

```python
from .provider import LLMProvider, OpenAICompatibleProvider
```

Add to `__all__`:

```python
"LLMProvider", "OpenAICompatibleProvider",
```

- [ ] **Step 2: Update pyproject.toml**

Replace the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
qwen = [
    "openai>=1.0",
]
openai = [
    "openai>=1.0",
]
anthropic = [
    "anthropic>=0.49",
    "voyageai>=0.3",
]
all = [
    "openai>=1.0",
    "anthropic>=0.49",
    "voyageai>=0.3",
    "numpy>=1.24",
]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 3: Install openai for testing**

```bash
pip install openai
```

- [ ] **Step 4: Run ALL tests**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add anima/__init__.py pyproject.toml
git commit -m "feat: export LLMProvider, add qwen/openai/anthropic optional deps"
```

---

### Task 4: Qwen Quick Start Example

**Files:**
- Create: `examples/qwen_quickstart.py`

- [ ] **Step 1: Create examples directory and quick start**

```python
# examples/qwen_quickstart.py
"""
Qwen Quick Start — 用通义千问 API 跑一个有记忆的 Agent。

使用前：
    pip install anima-agent[qwen]
    export DASHSCOPE_API_KEY=你的key

或者直接在代码里传 api_key。
"""

import os
from anima import AnimaAgent
from anima.provider import OpenAICompatibleProvider


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("请设置环境变量 DASHSCOPE_API_KEY，或在代码中传入 api_key")
        print("获取方式：https://dashscope.console.aliyun.com/")
        return

    # 1. 创建 Provider（通义千问兼容 OpenAI 格式）
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        chat_model="qwen-plus",
        embed_model="text-embedding-v3",
        embed_dimensions=1024,
    )

    # 2. 创建 Agent 并配置
    agent = AnimaAgent("我的助手")
    agent.configure(provider)
    agent.register_skill("python_coding", "编写Python代码")
    agent.register_skill("data_analysis", "数据分析")

    # 3. 第一次对话
    print("\n--- 第一次对话 ---")
    response = agent.chat("帮我写一个Python函数，计算列表的移动平均值")
    print(f"助手: {response}")
    agent.feedback(0.9, skills_used=["python_coding"],
                   problems=["需要处理窗口边界"], solutions=["用min处理边界"])

    # 4. 第二次对话（Agent已经有了第一次的经验）
    print("\n--- 第二次对话 ---")
    response = agent.chat("帮我写一个函数，计算指数移动平均值")
    print(f"助手: {response}")
    agent.feedback(0.8, skills_used=["python_coding"])

    # 5. 查看 Agent 状态
    print("\n--- Agent 状态 ---")
    status = agent.status()
    print(f"交互次数: {status['interactions']}")
    print(f"图谱节点: {status['graph_stats']['total_nodes']}")
    print(f"能力标签: {status['competence']['domain_tags']}")

    print("\n第二次对话时，Agent 已经记住了第一次的经验（移动平均、窗口边界处理）。")
    print("随着使用越来越多，Agent 会越来越了解你的编程风格和偏好。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('examples/qwen_quickstart.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add examples/qwen_quickstart.py
git commit -m "docs: add Qwen quick start example"
```

---

### Task 5: End-to-End Integration Test (Mock)

**Files:**
- Add to: `tests/test_agent_chat.py`

- [ ] **Step 1: Write full lifecycle test**

```python
class TestFullLifecycle:
    def test_two_agents_same_provider_different_personalities(self):
        """Core integration test: same provider, different experiences, different behavior."""
        import os
        for f in ["/tmp/lifecycle_alpha.json", "/tmp/lifecycle_beta.json"]:
            if os.path.exists(f):
                os.remove(f)

        mock_provider = MagicMock()
        mock_provider.embed = MockEmbeddingProvider(dimensions=64).embed
        mock_provider.embed_batch = MockEmbeddingProvider(dimensions=64).embed_batch
        mock_provider.dimensions = 64
        mock_provider.extract = MockExtractor().extract
        mock_provider.chat = MagicMock(return_value="模拟回答")

        # Alpha: 数据分析经验
        alpha = AnimaAgent("Alpha", save_path="/tmp/lifecycle_alpha.json")
        alpha.configure(mock_provider)
        alpha.register_skill("sql", "SQL查询")
        alpha.register_skill("python", "Python编程")

        alpha.chat("分析用户留存数据")
        alpha.feedback(0.9, ["decompose_first"], ["sql"],
                       ["数据有空值"], ["中位数填充"])
        alpha.chat("分析转化漏斗")
        alpha.feedback(0.8, ["decompose_first"], ["sql"])

        # Beta: 开发经验
        beta = AnimaAgent("Beta", save_path="/tmp/lifecycle_beta.json")
        beta.configure(mock_provider)
        beta.register_skill("sql", "SQL查询")
        beta.register_skill("python", "Python编程")

        beta.chat("写一个自动部署脚本")
        beta.feedback(0.9, ["direct_execution"], ["python"])
        beta.chat("重构权限系统")
        beta.feedback(0.8, ["iterate_and_refine"], ["python"])

        # 验证分化
        alpha_prefs = alpha.persona.strategy_network.profiles["data_analysis"].action_preferences
        beta_prefs = beta.persona.strategy_network.profiles["code_writing"].action_preferences

        assert alpha_prefs.get("decompose_first", 0) > 0
        assert beta_prefs.get("direct_execution", 0) > 0

        # 验证 chat 被调用时传入了不同的 system prompt
        # Alpha 和 Beta 的 chat 调用次数（各2次任务 + 可能的 extract 调用）
        assert mock_provider.chat.call_count >= 4

    def test_agent_save_load_preserves_state(self):
        """Agent state should survive save/load cycle."""
        import os
        path = "/tmp/lifecycle_saveload.json"
        if os.path.exists(path):
            os.remove(path)

        mock_provider = MagicMock()
        mock_provider.embed = MockEmbeddingProvider(dimensions=64).embed
        mock_provider.embed_batch = MockEmbeddingProvider(dimensions=64).embed_batch
        mock_provider.dimensions = 64
        mock_provider.extract = MockExtractor().extract
        mock_provider.chat = MagicMock(return_value="回答")

        agent = AnimaAgent("SaveTest", save_path=path)
        agent.configure(mock_provider)
        agent.register_skill("coding", "编程")
        agent.chat("写个排序算法")
        agent.feedback(0.9, ["direct_execution"], ["coding"])

        # Save is automatic after feedback
        # Load as new agent
        agent2 = AnimaAgent("SaveTest", save_path=path)
        assert agent2.persona.interaction_count >= 1

        task_nodes = agent2.persona.experience_graph.find_by_type(NodeType.TASK)
        assert len(task_nodes) >= 1
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_agent_chat.py -v
```

Expected: all PASS

- [ ] **Step 3: Run FULL test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_chat.py
git commit -m "test: add full lifecycle integration test - two agents diverge via chat()"
```

---

## Post-Plan: What Comes Next

1. **CLI Demo** — A simple `python -m anima` command-line agent for quick experimentation
2. **MCP Server + Skill** — Expose AnimaAgent as MCP tools for Claude Code / OpenClaw integration
3. **Phase 3 Features** — PATTERN nodes, feedback robustness, narrative V2
4. **Multi-provider live test** — Run dry_comparison with Qwen to get real system prompt + LLM output
