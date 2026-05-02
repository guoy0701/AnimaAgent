from anima.experience_graph import ExperienceGraph, NodeType, EdgeType


class TestEdgeDirectionality:
    def test_causal_edge_only_forward_in_activation(self):
        """CAUSAL edges should only propagate activation forward (source→target)."""
        g = ExperienceGraph()
        problem = g.add_node(NodeType.PROBLEM, "数据缺失")
        solution = g.add_node(NodeType.SOLUTION, "用中位数填充")
        g.add_edge(problem.id, solution.id, EdgeType.SOLVED_BY)

        # Activate from problem → should reach solution
        activated_from_problem = g.spreading_activation([problem.id])
        activated_ids = {n.id for n, _ in activated_from_problem}
        assert solution.id in activated_ids

        # Activate from solution → should NOT reach problem via directed edge
        activated_from_solution = g.spreading_activation([solution.id])
        activated_ids = {n.id for n, _ in activated_from_solution}
        assert problem.id not in activated_ids

    def test_similar_edge_propagates_both_ways(self):
        """SIMILAR edges are undirected, should propagate both ways."""
        g = ExperienceGraph()
        a = g.add_node(NodeType.CONCEPT, "用户留存")
        b = g.add_node(NodeType.CONCEPT, "用户流失")
        g.add_edge(a.id, b.id, EdgeType.SIMILAR)

        activated_from_a = g.spreading_activation([a.id])
        assert any(n.id == b.id for n, _ in activated_from_a)

        activated_from_b = g.spreading_activation([b.id])
        assert any(n.id == a.id for n, _ in activated_from_b)


class TestSerialization:
    def test_roundtrip_preserves_edge_directionality(self):
        g = ExperienceGraph()
        a = g.add_node(NodeType.PROBLEM, "问题A")
        b = g.add_node(NodeType.SOLUTION, "方案B")
        g.add_edge(a.id, b.id, EdgeType.SOLVED_BY)

        data = g.to_dict()
        g2 = ExperienceGraph.from_dict(data)

        # Forward: problem → solution should work
        activated = g2.spreading_activation([a.id])
        assert any(n.id == b.id for n, _ in activated)

        # Backward: solution → problem should NOT work
        activated = g2.spreading_activation([b.id])
        assert not any(n.id == a.id for n, _ in activated)

    def test_roundtrip_preserves_all_nodes_and_edges(self):
        g = ExperienceGraph()
        a = g.add_node(NodeType.TASK, "任务1")
        b = g.add_node(NodeType.CONCEPT, "概念1")
        c = g.add_node(NodeType.CONCEPT, "概念2")
        g.add_edge(a.id, b.id, EdgeType.COMPOSED_OF)
        g.add_edge(b.id, c.id, EdgeType.SIMILAR)

        data = g.to_dict()
        g2 = ExperienceGraph.from_dict(data)

        assert len(g2.nodes) == 3
        assert len(g2.edges) == 2


from anima.embedding import MockEmbeddingProvider, cosine_similarity


class TestSemanticSearch:
    def test_find_by_embedding_returns_semantically_similar(self):
        provider = MockEmbeddingProvider(dimensions=64)
        g = ExperienceGraph()

        n1 = g.add_node(NodeType.CONCEPT, "用户留存分析",
                         embedding=provider.embed("用户留存分析"))
        n2 = g.add_node(NodeType.CONCEPT, "服务器部署配置",
                         embedding=provider.embed("服务器部署配置"))
        n3 = g.add_node(NodeType.CONCEPT, "用户流失原因",
                         embedding=provider.embed("用户流失原因"))

        query_emb = provider.embed("用户留存")
        results = g.find_by_embedding(query_emb, top_k=2)

        result_ids = [n.id for n, _ in results]
        assert n1.id in result_ids, "Should find '用户留存分析'"
        assert n2.id not in result_ids, "Should NOT find '服务器部署配置'"

    def test_find_by_embedding_skips_nodes_without_embeddings(self):
        g = ExperienceGraph()
        g.add_node(NodeType.CONCEPT, "用户留存分析")  # no embedding

        results = g.find_by_embedding([0.1] * 64, top_k=5)
        assert len(results) == 0

    def test_find_by_embedding_respects_min_similarity(self):
        provider = MockEmbeddingProvider(dimensions=64)
        g = ExperienceGraph()

        g.add_node(NodeType.CONCEPT, "用户留存",
                   embedding=provider.embed("用户留存"))
        g.add_node(NodeType.CONCEPT, "完全无关的话题",
                   embedding=provider.embed("完全无关的话题"))

        query_emb = provider.embed("用户留存分析")
        results = g.find_by_embedding(query_emb, top_k=10, min_similarity=0.5)

        # Only highly similar results should be returned
        for node, sim in results:
            assert sim >= 0.5, f"Node '{node.content}' has similarity {sim} < 0.5"


class TestGraphTopologyStats:
    def test_returns_domain_stats(self):
        g = ExperienceGraph()
        t1 = g.add_node(NodeType.TASK, "任务1",
                         metadata={"category": "data_analysis"})
        c1 = g.add_node(NodeType.CONCEPT, "用户留存")
        g.add_edge(t1.id, c1.id, EdgeType.COMPOSED_OF)

        stats = g.get_topology_stats()
        assert "data_analysis" in stats
        assert stats["data_analysis"]["node_count"] >= 1

    def test_empty_graph_returns_empty_stats(self):
        g = ExperienceGraph()
        stats = g.get_topology_stats()
        assert stats == {}

    def test_counts_concepts_correctly(self):
        g = ExperienceGraph()
        t1 = g.add_node(NodeType.TASK, "任务1",
                         metadata={"category": "data_analysis"})
        c1 = g.add_node(NodeType.CONCEPT, "概念1")
        c2 = g.add_node(NodeType.CONCEPT, "概念2")
        g.add_edge(t1.id, c1.id, EdgeType.COMPOSED_OF)
        g.add_edge(t1.id, c2.id, EdgeType.COMPOSED_OF)

        stats = g.get_topology_stats()
        assert stats["data_analysis"]["concept_count"] == 2
