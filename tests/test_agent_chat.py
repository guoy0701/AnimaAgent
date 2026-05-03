import os
from unittest.mock import MagicMock
from anima.agent import AnimaAgent
from anima.embedding import MockEmbeddingProvider
from anima.extractor import MockExtractor
from anima.experience_graph import NodeType


class TestAgentConfigure:
    def test_configure_with_unified_provider(self):
        agent = AnimaAgent("test_cfg", save_path="/tmp/test_cfg.json")
        mock_provider = MagicMock()
        agent.configure(mock_provider)

        assert agent._provider is mock_provider
        assert agent.persona._embedding_provider is mock_provider
        assert agent.persona._extractor is mock_provider

    def test_configure_with_separate_components(self):
        agent = AnimaAgent("test_cfg2", save_path="/tmp/test_cfg2.json")
        emb = MockEmbeddingProvider(dimensions=64)
        ext = MockExtractor()
        agent.configure(embedding_provider=emb, extractor=ext)

        assert agent.persona._embedding_provider is emb
        assert agent.persona._extractor is ext

    def test_configure_with_nothing_raises(self):
        agent = AnimaAgent("test_cfg3", save_path="/tmp/test_cfg3.json")
        try:
            agent.configure()
            assert False, "Should have raised"
        except ValueError:
            pass


class TestAgentChat:
    def _make_agent(self):
        path = "/tmp/test_chat_agent.json"
        if os.path.exists(path):
            os.remove(path)
        agent = AnimaAgent("ChatTest", save_path=path)

        mock_provider = MagicMock()
        mock_emb = MockEmbeddingProvider(dimensions=64)
        mock_provider.embed = mock_emb.embed
        mock_provider.embed_batch = mock_emb.embed_batch
        mock_provider.dimensions = 64
        mock_provider.extract = MockExtractor().extract
        mock_provider.chat = MagicMock(return_value="这是LLM的回答")

        agent.configure(mock_provider)
        return agent, mock_provider

    def test_chat_returns_llm_response(self):
        agent, provider = self._make_agent()
        response = agent.chat("帮我分析用户数据")
        assert response == "这是LLM的回答"

    def test_chat_calls_provider_with_system_prompt(self):
        agent, provider = self._make_agent()
        agent.chat("帮我分析用户数据")

        provider.chat.assert_called_once()
        call_kwargs = provider.chat.call_args[1]
        assert "system" in call_kwargs
        assert "ChatTest" in call_kwargs["system"]

    def test_chat_without_configure_raises(self):
        agent = AnimaAgent("NoConfig", save_path="/tmp/test_noconfig.json")
        try:
            agent.chat("hello")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "configure" in str(e).lower()

    def test_chat_sets_current_task_for_feedback(self):
        agent, provider = self._make_agent()
        agent.chat("帮我分析用户数据")

        assert agent._current_task == "帮我分析用户数据"
        assert agent._current_context is not None

    def test_chat_then_feedback_works(self):
        agent, provider = self._make_agent()
        agent.register_skill("sql_query", "SQL查询")

        agent.chat("帮我分析用户数据")
        agent.feedback(0.8, skills_used=["sql_query"])

        task_nodes = agent.persona.experience_graph.find_by_type(NodeType.TASK)
        assert len(task_nodes) >= 1
