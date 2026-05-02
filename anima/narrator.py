"""
经验叙事化模块 (Experience Narrativization)

将激活的子图转化为叙事文本，供 LLM 上下文注入使用。
核心思路：沿 CAUSAL / SOLVED_BY 等有向边追踪，构建故事，而非简单列举节点。
"""

from anima.experience_graph import ExperienceGraph, Node, NodeType, EdgeType


def narrate_subgraph(graph: ExperienceGraph,
                     activated: list[tuple[Node, float]],
                     max_stories: int = 3) -> str:
    """
    将激活的子图转化为自然语言叙事。

    Args:
        graph: 经验图谱实例
        activated: 已激活节点列表，每项为 (Node, activation_score)
        max_stories: 最多生成几条故事

    Returns:
        供 LLM 阅读的叙事文本
    """
    if not activated:
        return "没有找到相关历史经验。"

    activated_ids = {n.id for n, _ in activated}
    task_groups = _group_by_task(graph, activated, activated_ids)

    if not task_groups:
        return _fallback_list(activated)

    stories = []
    for task_node, related_nodes in task_groups[:max_stories]:
        story = _build_story(graph, task_node, related_nodes, activated_ids)
        if story:
            stories.append(story)

    if not stories:
        return _fallback_list(activated)

    header = "以下是与当前任务相关的历史经验：\n"
    return header + "\n\n".join(stories)


def _group_by_task(graph: ExperienceGraph,
                   activated: list[tuple[Node, float]],
                   activated_ids: set[str]) -> list[tuple[Node, list]]:
    """
    将激活节点按任务节点分组，每个任务节点携带其相关节点。
    返回按激活强度降序排列的 [(task_node, [(related_node, edge), ...]), ...] 列表。
    """
    task_nodes = [(n, a) for n, a in activated if n.node_type == NodeType.TASK]
    task_nodes.sort(key=lambda x: x[1], reverse=True)

    groups = []
    for task_node, _ in task_nodes:
        related = []
        for neighbor_id, edge in graph._forward.get(task_node.id, []):
            if neighbor_id in activated_ids:
                neighbor = graph.nodes.get(neighbor_id)
                if neighbor:
                    related.append((neighbor, edge))
        groups.append((task_node, related))
    return groups


def _build_story(graph: ExperienceGraph,
                 task_node: Node,
                 related_nodes: list[tuple[Node, object]],
                 activated_ids: set[str]) -> str:
    """
    根据单个任务节点及其相关节点，构建一段叙事文本。
    """
    parts = [f"你之前做过一个任务：{task_node.content}。"]

    problems = []
    solutions = []   # list of (problem_node, solution_node)
    outcomes = []
    skills = []

    for node, edge in related_nodes:
        if node.node_type == NodeType.PROBLEM:
            problems.append(node)
            # 沿 SOLVED_BY 边查找该问题的解决方案
            for sol_id, sol_edge in graph._forward.get(node.id, []):
                if sol_edge.edge_type == EdgeType.SOLVED_BY and sol_id in activated_ids:
                    sol_node = graph.nodes.get(sol_id)
                    if sol_node:
                        solutions.append((node, sol_node))
        elif node.node_type == NodeType.FEEDBACK:
            outcomes.append(node)
        elif node.node_type == NodeType.SKILL:
            skills.append(node)

    # 使用的技能
    if skills:
        skill_names = []
        for s in skills:
            # 格式: "Skill: skill_name - description"
            name = s.content.replace("Skill: ", "").split(" - ")[0]
            skill_names.append(name)
        parts.append(f"使用了{', '.join(skill_names)}。")

    # 遇到的问题及解决方案
    if solutions:
        for prob, sol in solutions:
            parts.append(f"遇到了「{prob.content}」的问题，通过「{sol.content}」解决。")
    elif problems:
        for p in problems:
            parts.append(f"遇到了「{p.content}」的问题。")

    # 最终结果
    if outcomes:
        outcome_text = outcomes[0].content.replace("结果: ", "")
        parts.append(f"最终结果：{outcome_text}。")

    return "".join(parts)


def _fallback_list(activated: list[tuple[Node, float]]) -> str:
    """
    当没有任务节点时的降级处理：以列表形式输出相关节点。
    """
    lines = ["以下是相关的历史信息："]
    for node, act in activated[:5]:
        relevance = "高度相关" if act > 0.5 else "可能相关"
        lines.append(f"- {node.content}（{relevance}）")
    return "\n".join(lines)
