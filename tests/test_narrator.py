from anima.experience_graph import ExperienceGraph, NodeType, EdgeType
from anima.narrator import narrate_subgraph


class TestNarrator:
    def _build_sample_graph(self):
        g = ExperienceGraph()
        task = g.add_node(NodeType.TASK, "分析用户留存数据")
        problem = g.add_node(NodeType.PROBLEM, "数据有缺失值")
        solution = g.add_node(NodeType.SOLUTION, "用中位数填充")
        outcome = g.add_node(NodeType.FEEDBACK, "结果: 成功")
        skill = g.add_node(NodeType.SKILL, "Skill: sql_query - SQL查询")

        g.add_edge(task.id, problem.id, EdgeType.CAUSAL)
        g.add_edge(problem.id, solution.id, EdgeType.SOLVED_BY)
        g.add_edge(task.id, outcome.id, EdgeType.CAUSAL)
        g.add_edge(task.id, skill.id, EdgeType.REQUIRES)

        activated = [(task, 1.0), (problem, 0.6), (solution, 0.5),
                     (outcome, 0.4), (skill, 0.3)]
        return g, activated

    def test_narrate_produces_readable_text(self):
        g, activated = self._build_sample_graph()
        text = narrate_subgraph(g, activated)
        assert "用户留存" in text
        assert "缺失" in text
        assert len(text) > 20

    def test_narrate_includes_problem_and_solution(self):
        g, activated = self._build_sample_graph()
        text = narrate_subgraph(g, activated)
        assert "缺失" in text
        assert "中位数" in text

    def test_narrate_empty_activation(self):
        g = ExperienceGraph()
        text = narrate_subgraph(g, [])
        assert "没有" in text or len(text) < 50

    def test_narrate_includes_skill_info(self):
        g, activated = self._build_sample_graph()
        text = narrate_subgraph(g, activated)
        assert "sql_query" in text or "SQL" in text

    def test_narrate_multiple_tasks(self):
        g = ExperienceGraph()
        t1 = g.add_node(NodeType.TASK, "分析用户留存")
        t2 = g.add_node(NodeType.TASK, "分析销售趋势")
        o1 = g.add_node(NodeType.FEEDBACK, "结果: 成功")
        o2 = g.add_node(NodeType.FEEDBACK, "结果: 一般")
        g.add_edge(t1.id, o1.id, EdgeType.CAUSAL)
        g.add_edge(t2.id, o2.id, EdgeType.CAUSAL)

        activated = [(t1, 1.0), (t2, 0.8), (o1, 0.5), (o2, 0.4)]
        text = narrate_subgraph(g, activated, max_stories=2)
        assert "留存" in text
        assert "销售" in text
