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
        ...


class OpenAICompatibleProvider(LLMProvider):
    """Works with any OpenAI-format API: Qwen, DeepSeek, GPT, Ollama, vLLM, etc.

    Usage:
        # Qwen (通义千问)
        provider = OpenAICompatibleProvider(
            api_key="sk-xxx",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            chat_model="qwen-plus",
            embed_model="text-embedding-v3",
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
