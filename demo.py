#!/usr/bin/env python3
"""
Anima Demo: 两个Agent的分化实验

这个演示创建两个从相同初始状态出发的Agent，
让它们经历不同的"人生"，然后观察它们的分化。

Agent Alpha: 跟随一个数据分析师主人
Agent Beta:  跟随一个全栈开发者主人

两个Agent学了完全相同的Skill，但因为经历不同，
面对同一个新任务时会给出不同的策略建议。
"""

import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anima import AnimaAgent, TaskCategory


def create_agents():
    """创建两个初始状态相同的Agent"""
    print("=" * 60)
    print("  Anima Demo: Agent的分化实验")
    print("=" * 60)
    print()

    alpha = AnimaAgent("Alpha", save_path="/tmp/anima_alpha.json")
    beta = AnimaAgent("Beta", save_path="/tmp/anima_beta.json")

    # 注册完全相同的Skill
    skills = [
        ("python_coding", "编写Python代码", ["code_writing", "data_analysis"]),
        ("sql_query", "编写SQL查询", ["data_analysis", "code_writing"]),
        ("data_viz", "数据可视化", ["data_analysis", "content_creation"]),
        ("web_dev", "Web开发", ["code_writing", "creative"]),
        ("writing", "文案写作", ["content_creation", "communication"]),
        ("research", "信息调研", ["research"]),
    ]

    for name, desc, cats in skills:
        alpha.register_skill(name, desc, cats)
        beta.register_skill(name, desc, cats)

    return alpha, beta


def simulate_alpha_life(alpha: AnimaAgent):
    """
    模拟Alpha的"人生"——跟随一个数据分析师主人

    Alpha会经历大量的数据分析任务，偶尔写代码，
    主人偏好先看数据再下结论的风格。
    """
    print("\n" + "=" * 60)
    print("  Phase 1: Alpha的成长历程（数据分析师的助手）")
    print("=" * 60)

    experiences = [
        # (任务, 类别, 动作, 使用的Skill, 反馈, 遇到的问题, 解决方案)
        ("分析上个月的用户留存数据",
         TaskCategory.DATA_ANALYSIS,
         ["decompose_first", "use_skill"],
         ["sql_query", "data_viz"],
         0.9, ["数据中有大量空值"], ["用中位数填充空值"]),

        ("找出销售额下降的原因",
         TaskCategory.DATA_ANALYSIS,
         ["search_first", "decompose_first"],
         ["sql_query", "python_coding"],
         0.8, ["需要关联多个数据表"], ["用LEFT JOIN关联用户表和订单表"]),

        ("做一份季度数据报告",
         TaskCategory.CONTENT_CREATION,
         ["decompose_first", "use_skill"],
         ["data_viz", "writing"],
         0.7, [], []),

        ("预测下季度的用户增长",
         TaskCategory.DATA_ANALYSIS,
         ["search_first", "use_skill"],
         ["python_coding", "data_viz"],
         0.85, ["线性模型拟合效果差"], ["改用ARIMA时间序列模型"]),

        ("帮我写个自动化取数脚本",
         TaskCategory.CODE_WRITING,
         ["direct_execution", "use_skill"],
         ["python_coding", "sql_query"],
         0.95, [], []),

        ("对比A/B测试的结果",
         TaskCategory.DATA_ANALYSIS,
         ["decompose_first", "use_skill"],
         ["python_coding", "data_viz"],
         0.9, ["样本量不够大"], ["用Bootstrap方法估计置信区间"]),

        ("分析用户行为漏斗",
         TaskCategory.DATA_ANALYSIS,
         ["decompose_first", "consult_experience"],
         ["sql_query", "data_viz"],
         0.85, ["漏斗定义不明确"], ["先跟主人确认关键事件定义"]),

        ("写一个数据质量监控看板",
         TaskCategory.CODE_WRITING,
         ["decompose_first", "combine_skills"],
         ["python_coding", "data_viz", "sql_query"],
         0.8, ["实时数据延迟问题"], ["改用增量更新而非全量刷新"]),
    ]

    for task, cat, actions, skills, reward, problems, solutions in experiences:
        print(f"\n  任务: {task}")
        context = alpha.think(task)
        alpha.persona.record_experience(
            task, cat, actions, skills,
            "成功" if reward > 0.5 else "一般",
            problems, solutions)
        alpha.feedback(reward, actions, skills, problems, solutions)
        print(f"  反馈: {'⭐' * int(reward * 5)}")


def simulate_beta_life(beta: AnimaAgent):
    """
    模拟Beta的"人生"——跟随一个全栈开发者主人

    Beta会经历大量的编码任务，偶尔做数据分析，
    主人偏好快速迭代、先出MVP再优化的风格。
    """
    print("\n" + "=" * 60)
    print("  Phase 2: Beta的成长历程（全栈开发者的助手）")
    print("=" * 60)

    experiences = [
        ("搭建一个用户注册登录系统",
         TaskCategory.CODE_WRITING,
         ["direct_execution", "iterate_and_refine"],
         ["web_dev", "python_coding"],
         0.9, ["密码加密方案选择"], ["使用bcrypt"]),

        ("做一个商品展示页面",
         TaskCategory.CREATIVE,
         ["direct_execution", "use_skill"],
         ["web_dev"],
         0.85, ["移动端适配问题"], ["用CSS Grid + 媒体查询"]),

        ("优化API响应速度",
         TaskCategory.PROBLEM_SOLVING,
         ["direct_execution", "iterate_and_refine"],
         ["python_coding", "sql_query"],
         0.8, ["N+1查询问题"], ["用JOIN替代循环查询"]),

        ("写一个自动部署脚本",
         TaskCategory.CODE_WRITING,
         ["direct_execution", "use_skill"],
         ["python_coding"],
         0.95, [], []),

        ("做一个数据看板给运营看",
         TaskCategory.DATA_ANALYSIS,
         ["iterate_and_refine", "combine_skills"],
         ["web_dev", "data_viz", "sql_query"],
         0.7, ["图表加载太慢"], ["前端做数据缓存"]),

        ("重构后端的权限系统",
         TaskCategory.CODE_WRITING,
         ["decompose_first", "iterate_and_refine"],
         ["python_coding"],
         0.9, ["需要兼容旧接口"], ["用装饰器模式做权限检查"]),

        ("写个爬虫抓竞品价格",
         TaskCategory.CODE_WRITING,
         ["direct_execution", "iterate_and_refine"],
         ["python_coding"],
         0.85, ["反爬机制"], ["加随机延迟+代理IP"]),

        ("做一个实时聊天功能",
         TaskCategory.CODE_WRITING,
         ["decompose_first", "iterate_and_refine"],
         ["web_dev", "python_coding"],
         0.8, ["WebSocket连接不稳定"], ["加心跳检测和自动重连"]),
    ]

    for task, cat, actions, skills, reward, problems, solutions in experiences:
        print(f"\n  任务: {task}")
        context = beta.think(task)
        beta.persona.record_experience(
            task, cat, actions, skills,
            "成功" if reward > 0.5 else "一般",
            problems, solutions)
        beta.feedback(reward, actions, skills, problems, solutions)
        print(f"  反馈: {'⭐' * int(reward * 5)}")


def the_test(alpha: AnimaAgent, beta: AnimaAgent):
    """
    关键实验：同一个任务，两个Agent的反应

    任务："帮我分析一下我们网站最近一个月的用户行为数据，
          然后做一个可视化报告页面"

    这个任务同时涉及数据分析和前端开发，
    让我们看看两个有不同经历的Agent会如何处理。
    """
    print("\n" + "=" * 60)
    print("  Phase 3: 关键实验——同一个任务，不同的Agent")
    print("=" * 60)

    test_task = ("帮我分析一下我们网站最近一个月的用户行为数据，"
                 "然后做一个可视化报告页面")

    print(f"\n  任务: {test_task}")
    print("\n" + "-" * 50)

    # Alpha的反应
    print(f"\n  >>> Alpha（数据分析师的助手）的思考：")
    alpha_context = alpha.think(test_task)
    alpha_response = alpha.process_task(test_task)
    print(f"\n{alpha_response}")

    print("\n" + "-" * 50)

    # Beta的反应
    print(f"\n  >>> Beta（全栈开发者的助手）的思考：")
    beta_context = beta.think(test_task)
    beta_response = beta.process_task(test_task)
    print(f"\n{beta_response}")

    return alpha_context, beta_context


def compare_agents(alpha: AnimaAgent, beta: AnimaAgent):
    """对比两个Agent的差异"""
    print("\n" + "=" * 60)
    print("  Phase 4: 两个Agent的全面对比")
    print("=" * 60)

    comparison = alpha.compare_with(beta)

    print(f"\n  整体相似度: {comparison['overall_similarity']:.0%}")
    print(f"  差异评估: {comparison['interpretation']}")

    for name, info in comparison["comparison"].items():
        print(f"\n  {name}:")
        print(f"    交互次数: {info['interactions']}")
        print(f"    擅长领域: {', '.join(info['domain_tags']) if info['domain_tags'] else '尚未形成'}")
        print(f"    画像置信度: {info['confidence']:.0%}")

    # 详细的策略对比
    print("\n  策略偏好对比：")
    alpha_strategy = alpha.persona.strategy_network.get_profile_summary()
    beta_strategy = beta.persona.strategy_network.get_profile_summary()

    all_cats = set(list(alpha_strategy.keys()) + list(beta_strategy.keys()))
    for cat in sorted(all_cats):
        a_info = alpha_strategy.get(cat, {})
        b_info = beta_strategy.get(cat, {})
        if a_info or b_info:
            print(f"\n    [{cat}]")
            if a_info:
                print(f"      Alpha: 成功率{a_info.get('success_rate', 0):.0%}, "
                      f"偏好动作: {[a for a, _ in a_info.get('top_actions', [])[:2]]}")
            if b_info:
                print(f"      Beta:  成功率{b_info.get('success_rate', 0):.0%}, "
                      f"偏好动作: {[a for a, _ in b_info.get('top_actions', [])[:2]]}")

    # 能力画像对比
    print("\n  能力画像对比：")
    alpha_comp = alpha.persona.competence.competence_scores
    beta_comp = beta.persona.competence.competence_scores
    for dim in alpha_comp:
        a_score = alpha_comp.get(dim, 0)
        b_score = beta_comp.get(dim, 0)
        if a_score > 0.01 or b_score > 0.01:
            a_bar = "█" * int(a_score * 10)
            b_bar = "█" * int(b_score * 10)
            print(f"    {dim:20s}  Alpha [{a_bar:10s}] {a_score:.0%}"
                  f"  Beta [{b_bar:10s}] {b_score:.0%}")


def show_experience_graph_diff(alpha: AnimaAgent, beta: AnimaAgent):
    """展示经验图谱的差异"""
    print("\n" + "=" * 60)
    print("  Phase 5: 经验图谱的差异")
    print("=" * 60)

    a_stats = alpha.persona.experience_graph.get_stats()
    b_stats = beta.persona.experience_graph.get_stats()

    print(f"\n  Alpha的经验图谱:")
    print(f"    节点数: {a_stats['total_nodes']}")
    print(f"    连接数: {a_stats['total_edges']}")
    print(f"    节点类型分布: {json.dumps(a_stats['node_types'], indent=6)}")

    print(f"\n  Beta的经验图谱:")
    print(f"    节点数: {b_stats['total_nodes']}")
    print(f"    连接数: {b_stats['total_edges']}")
    print(f"    节点类型分布: {json.dumps(b_stats['node_types'], indent=6)}")

    # 激活测试
    print("\n  激活扩散测试 - 关键词 '数据分析':")
    for name, agent in [("Alpha", alpha), ("Beta", beta)]:
        seeds = agent.persona.experience_graph.find_by_content(["数据", "分析"])
        if seeds:
            activated = agent.persona.experience_graph.spreading_activation(
                [n.id for n in seeds[:3]])
            print(f"\n    {name} 激活了 {len(activated)} 个节点:")
            for node, act in activated[:5]:
                print(f"      {node.content[:50]:50s} (激活值: {act:.2f})")
        else:
            print(f"\n    {name} 没有找到相关节点")


def main():
    # 创建两个Agent
    alpha, beta = create_agents()

    # 模拟不同的人生经历
    simulate_alpha_life(alpha)
    simulate_beta_life(beta)

    # 关键测试：同一个任务，不同的反应
    the_test(alpha, beta)

    # 全面对比
    compare_agents(alpha, beta)

    # 经验图谱差异
    show_experience_graph_diff(alpha, beta)

    # 总结
    print("\n" + "=" * 60)
    print("  实验结论")
    print("=" * 60)
    print("""
  两个Agent从完全相同的初始状态出发，学了完全相同的Skill，
  但因为跟随不同的主人、经历了不同的任务，最终演化出了
  截然不同的能力结构和行为模式。

  这验证了我们的核心命题：
  1. Agent的能力 ≠ Skill的累加
  2. 同一个Skill在不同Agent上效果不同
  3. Agent的"人格"是由经历塑造的，不可轻易复制
  4. 真正的差异化发生在个性层，不在大模型层

  这就是Anima的设计理念：Agent不是工具，是"人"。
    """)

    print(f"  Alpha的灵魂已保存到: /tmp/anima_alpha.json")
    print(f"  Beta的灵魂已保存到: /tmp/anima_beta.json")


if __name__ == "__main__":
    main()
