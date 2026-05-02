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
