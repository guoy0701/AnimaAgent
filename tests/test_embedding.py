import math
from anima.embedding import MockEmbeddingProvider, cosine_similarity


class TestCosineSimiarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.01

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 0.01

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) < -0.9

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert cosine_similarity(a, b) == 0.0


class TestMockEmbeddingProvider:
    def test_returns_consistent_embeddings(self):
        provider = MockEmbeddingProvider(dimensions=64)
        emb1 = provider.embed("用户留存")
        emb2 = provider.embed("用户留存")
        assert emb1 == emb2, "Same text should produce same embedding"

    def test_returns_correct_dimensions(self):
        provider = MockEmbeddingProvider(dimensions=128)
        emb = provider.embed("test")
        assert len(emb) == 128
        assert provider.dimensions == 128

    def test_embeddings_are_normalized(self):
        provider = MockEmbeddingProvider(dimensions=64)
        emb = provider.embed("用户留存分析")
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.01, f"Embedding should be normalized, got norm={norm}"

    def test_similar_texts_have_higher_similarity(self):
        provider = MockEmbeddingProvider(dimensions=64)
        emb_retain = provider.embed("用户留存")
        emb_churn = provider.embed("用户流失")
        emb_server = provider.embed("服务器部署")

        sim_related = cosine_similarity(emb_retain, emb_churn)
        sim_unrelated = cosine_similarity(emb_retain, emb_server)
        assert sim_related > sim_unrelated, \
            f"Related texts should be more similar: {sim_related} vs {sim_unrelated}"

    def test_batch_embed(self):
        provider = MockEmbeddingProvider(dimensions=64)
        texts = ["用户留存", "数据分析", "服务器"]
        results = provider.embed_batch(texts)
        assert len(results) == 3
        assert results[0] == provider.embed("用户留存")
