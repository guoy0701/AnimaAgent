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


class TestChineseKeywords:
    def test_extracts_meaningful_chinese_keywords(self):
        persona = PersonaLayer("test")
        keywords = persona._extract_keywords("分析上个月的用户留存数据")

        assert len(keywords) >= 2, f"Got only {keywords}"
        found_meaningful = any(
            kw in ["分析", "用户", "留存", "数据", "用户留存", "留存数据", "上个月"]
            for kw in keywords
        )
        assert found_meaningful, f"No meaningful keywords in {keywords}"

    def test_chinese_keywords_can_find_related_nodes(self):
        persona = PersonaLayer("test")
        from anima.experience_graph import NodeType
        persona.experience_graph.add_node(NodeType.CONCEPT, "用户留存分析方法")

        keywords = persona._extract_keywords("帮我看看用户留存的情况")
        results = persona.experience_graph.find_by_content(keywords)
        assert len(results) > 0, "Should find related node via Chinese keywords"

    def test_english_keywords_still_work(self):
        persona = PersonaLayer("test")
        keywords = persona._extract_keywords("analyze user retention data")
        assert len(keywords) >= 2


class TestChineseConnections:
    def test_discovers_similar_chinese_tasks(self):
        from anima.experience_graph import NodeType, EdgeType
        persona = PersonaLayer("test")
        persona.record_experience(
            "分析用户留存数据", TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], "成功")
        persona.record_experience(
            "分析用户流失原因", TaskCategory.DATA_ANALYSIS,
            ["search_first"], ["sql_query"], "成功")

        task_nodes = persona.experience_graph.find_by_type(NodeType.TASK)
        assert len(task_nodes) == 2

        edges = persona.experience_graph.edges
        similar_edges = [e for e in edges if e.edge_type == EdgeType.SIMILAR]
        assert len(similar_edges) > 0, "Should discover similarity between related Chinese tasks"


class TestPhase1Integration:
    def test_two_agents_diverge_with_different_experiences(self):
        """Core claim: two agents with same skills but different experiences behave differently."""
        from anima.agent import AnimaAgent
        import os

        # Clean up any previous test files
        for f in ["/tmp/test_alpha_int.json", "/tmp/test_beta_int.json"]:
            if os.path.exists(f):
                os.remove(f)

        alpha = AnimaAgent("AlphaInt", save_path="/tmp/test_alpha_int.json")
        beta = AnimaAgent("BetaInt", save_path="/tmp/test_beta_int.json")

        for name, desc in [("sql_query", "SQL查询"), ("python_coding", "Python编程")]:
            alpha.register_skill(name, desc)
            beta.register_skill(name, desc)

        # Alpha: data analysis experiences (decompose_first works well)
        alpha.think("分析用户留存数据")
        alpha.feedback(0.9, ["decompose_first", "use_skill"], ["sql_query"],
                       ["数据有缺失值"], ["中位数填充"])
        alpha.think("分析销售趋势")
        alpha.feedback(0.8, ["decompose_first", "search_first"], ["sql_query"])

        # Beta: coding experiences (direct_execution works well)
        beta.think("编写自动部署的Python代码")
        beta.feedback(0.9, ["direct_execution", "iterate_and_refine"], ["python_coding"])
        beta.think("编程实现权限系统重构")
        beta.feedback(0.8, ["direct_execution", "use_skill"], ["python_coding"])

        # Verify strategy preferences diverged
        alpha_da = alpha.persona.strategy_network.profiles["data_analysis"].action_preferences
        beta_cw = beta.persona.strategy_network.profiles["code_writing"].action_preferences

        # Alpha should prefer decompose_first for data_analysis
        assert alpha_da.get("decompose_first", 0) > alpha_da.get("direct_execution", 0), \
            f"Alpha should prefer decompose_first but got {alpha_da}"
        # Beta should prefer direct_execution for code_writing
        assert beta_cw.get("direct_execution", 0) > beta_cw.get("decompose_first", 0), \
            f"Beta should prefer direct_execution but got {beta_cw}"

    def test_experience_graph_grows_with_interactions(self):
        from anima.agent import AnimaAgent
        import os

        path = "/tmp/test_growth.json"
        if os.path.exists(path):
            os.remove(path)

        agent = AnimaAgent("Growth", save_path=path)
        agent.register_skill("coding", "编程")

        initial_nodes = len(agent.persona.experience_graph.nodes)

        agent.think("写一个排序算法")
        agent.feedback(0.8, ["direct_execution"], ["coding"], ["效率低"], ["改用快排"])

        final_nodes = len(agent.persona.experience_graph.nodes)
        assert final_nodes > initial_nodes, "Graph should grow after recording experience"

        # Verify problem-solution pair exists
        from anima.experience_graph import NodeType
        problems = agent.persona.experience_graph.find_by_type(NodeType.PROBLEM)
        solutions = agent.persona.experience_graph.find_by_type(NodeType.SOLUTION)
        assert len(problems) == 1
        assert len(solutions) == 1

    def test_experience_graph_has_no_duplicate_skills(self):
        from anima.agent import AnimaAgent
        from anima.experience_graph import NodeType
        import os

        path = "/tmp/test_dedup_int.json"
        if os.path.exists(path):
            os.remove(path)

        agent = AnimaAgent("Dedup", save_path=path)
        agent.register_skill("sql", "SQL")
        agent.register_skill("sql", "SQL updated")

        skill_nodes = agent.persona.experience_graph.find_by_type(NodeType.SKILL)
        assert len(skill_nodes) == 1
