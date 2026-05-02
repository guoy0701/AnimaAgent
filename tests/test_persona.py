from anima.persona import PersonaLayer
from anima.strategy import TaskCategory
from anima.experience_graph import NodeType, EdgeType


class TestExperienceRecording:
    def test_solutions_pair_with_corresponding_problems(self):
        """Each solution should connect to its corresponding problem, not all problems."""
        persona = PersonaLayer("test")
        persona.record_experience(
            "测试任务", TaskCategory.DATA_ANALYSIS,
            ["direct_execution"], ["python_coding"], "成功",
            problems_encountered=["问题A", "问题B"],
            solutions_found=["方案A", "方案B"],
        )
        graph = persona.experience_graph

        solution_nodes = graph.find_by_type(NodeType.SOLUTION)
        assert len(solution_nodes) == 2

        # Each solution should connect to exactly 1 problem
        for sol in solution_nodes:
            connected_problems = []
            for _, edge in graph._backward.get(sol.id, []):
                if edge.edge_type == EdgeType.SOLVED_BY:
                    src_node = graph.nodes.get(edge.source_id)
                    if src_node and src_node.node_type == NodeType.PROBLEM:
                        connected_problems.append(src_node)
            assert len(connected_problems) == 1, f"Solution '{sol.content}' connected to {len(connected_problems)} problems"


class TestSkillRegistration:
    def test_duplicate_skill_does_not_create_new_node(self):
        persona = PersonaLayer("test")
        persona.register_skill("python_coding", "编写Python代码", ["code_writing"])
        persona.register_skill("python_coding", "编写Python代码v2", ["code_writing"])

        skill_nodes = persona.experience_graph.find_by_type(NodeType.SKILL)
        assert len(skill_nodes) == 1

    def test_duplicate_skill_updates_metadata(self):
        persona = PersonaLayer("test")
        persona.register_skill("python_coding", "编写Python代码", ["code_writing"])
        persona.register_skill("python_coding", "编写Python代码v2", ["code_writing", "data_analysis"])

        assert persona.skills["python_coding"]["description"] == "编写Python代码v2"
        assert "data_analysis" in persona.skills["python_coding"]["categories"]


class TestFeedbackRecording:
    def test_feedback_does_not_double_record(self):
        import os
        from anima.agent import AnimaAgent
        save_path = "/tmp/test_no_double.json"
        if os.path.exists(save_path):
            os.remove(save_path)
        agent = AnimaAgent("test_no_double", save_path=save_path)
        agent.register_skill("coding", "写代码")

        agent.think("写一个排序算法")
        agent.feedback(0.8, ["direct_execution"], ["coding"])

        task_nodes = agent.persona.experience_graph.find_by_type(NodeType.TASK)
        assert len(task_nodes) == 1, f"Expected 1 task node, got {len(task_nodes)}"
