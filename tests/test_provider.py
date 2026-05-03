from unittest.mock import patch, MagicMock
from anima.provider import OpenAICompatibleProvider, LLMProvider
from anima.embedding import EmbeddingProvider
from anima.extractor import ExperienceExtractor


class TestLLMProviderIsUnified:
    def test_openai_provider_is_embedding_provider(self):
        provider = OpenAICompatibleProvider(
            api_key="fake", base_url="http://fake", chat_model="m", embed_model="m")
        assert isinstance(provider, EmbeddingProvider)

    def test_openai_provider_is_extractor(self):
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
            messages = mock_create.call_args[1]["messages"]
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
        mock_response.choices[0].message.content = '{"concepts": ["用户留存", "数据分析"], "entities": ["SQL"], "domain": "data_analysis", "problems": [], "solutions": [], "outcome_summary": "分析用户数据", "related_concepts": ["用户流失"]}'

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
