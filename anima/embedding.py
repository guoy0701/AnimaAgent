"""
Embedding provider interface.

Provides vector embeddings for text, enabling semantic similarity search.
Ships with:
- MockEmbeddingProvider: deterministic, character-overlap-based (for testing)
- AnthropicEmbeddingProvider: uses Anthropic's voyage embeddings (for production)
"""

import hashlib
import math
from abc import ABC, abstractmethod


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock: produces embeddings based on character n-gram hashing.
    Texts sharing more characters produce more similar vectors.
    For testing only."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        chars = list(text.lower())
        lower_text = text.lower()
        ngrams = [lower_text[i:i+2] for i in range(len(lower_text) - 1)] + chars
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
            idx = h % self._dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """Uses Anthropic's Voyage embeddings. Requires: pip install 'anima-agent[semantic]'"""

    def __init__(self, model: str = "voyage-3", api_key: str = None):
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "pip install voyageai for Anthropic/Voyage embeddings")
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimensions = 1024

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model)
        return result.embeddings
