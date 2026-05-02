#!/usr/bin/env python3
"""
干对比实验：不调用任何 LLM API，只看 PersonaLayer 为两个不同 Agent 生成的 system prompt 差异。

验证核心命题：同一任务，不同经历的 Agent，送给 LLM 的上下文是否有意义地不同。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anima import AnimaAgent, TaskCategory
from anima.embedding import MockEmbeddingProvider
from anima.extractor import MockExtractor


def create_agent(name, save_path):
    agent = AnimaAgent(name, save_path=save_path)
    agent.configure_semantic(
        embedding_provider=MockEmbeddingProvider(dimensions=64),
        extractor=MockExtractor(),
    )
    for skill_name, desc in [
        ("sql_query", "SQL查询"),
        ("python_coding", "Python编程"),
        ("data_viz", "数据可视化"),
        ("web_dev", "Web开发"),
        ("writing", "文案写作"),
    ]:
        agent.register_skill(skill_name, desc)
    return agent


def train_alpha(alpha):
    """Alpha: 数据分析师的助手，15 轮数据分析相关经验"""
    experiences = [
        ("分析上个月的用户留存数据", 0.9,
         ["decompose_first", "use_skill"], ["sql_query", "data_viz"],
         ["数据中有大量空值"], ["用中位数填充空值"]),
        ("找出销售额下降的原因", 0.8,
         ["search_first", "decompose_first"], ["sql_query", "python_coding"],
         ["需要关联多个数据表"], ["用LEFT JOIN关联用户表和订单表"]),
        ("做一份季度数据报告", 0.7,
         ["decompose_first", "use_skill"], ["data_viz", "writing"],
         [], []),
        ("预测下季度的用户增长", 0.85,
         ["search_first", "use_skill"], ["python_coding", "data_viz"],
         ["线性模型拟合效果差"], ["改用ARIMA时间序列模型"]),
        ("帮我写个自动化取数脚本", 0.95,
         ["direct_execution", "use_skill"], ["python_coding", "sql_query"],
         [], []),
        ("对比AB测试的结果", 0.9,
         ["decompose_first", "use_skill"], ["python_coding", "data_viz"],
         ["样本量不够大"], ["用Bootstrap方法估计置信区间"]),
        ("分析用户行为漏斗", 0.85,
         ["decompose_first", "consult_experience"], ["sql_query", "data_viz"],
         ["漏斗定义不明确"], ["先跟主人确认关键事件定义"]),
        ("写一个数据质量监控看板", 0.8,
         ["decompose_first", "combine_skills"], ["python_coding", "data_viz", "sql_query"],
         ["实时数据延迟问题"], ["改用增量更新而非全量刷新"]),
        ("分析新渠道的获客质量", 0.9,
         ["decompose_first", "use_skill"], ["sql_query", "data_viz"],
         ["新渠道数据不完整"], ["用已有渠道的均值做对照组"]),
        ("计算各产品线的ROI", 0.75,
         ["decompose_first", "search_first"], ["sql_query", "python_coding"],
         ["成本数据分散在多个系统"], ["写ETL脚本统一归集"]),
        ("做用户分群画像", 0.85,
         ["decompose_first", "use_skill"], ["python_coding", "data_viz"],
         ["分群标准难以确定"], ["用RFM模型做初始分群"]),
        ("分析页面转化率变化", 0.9,
         ["search_first", "use_skill"], ["sql_query", "data_viz"],
         [], []),
        ("建立数据指标体系", 0.7,
         ["decompose_first", "consult_experience"], ["writing", "data_viz"],
         ["指标定义不统一"], ["参照AARRR模型重新定义"]),
        ("做一个实时数据大屏", 0.8,
         ["decompose_first", "combine_skills"], ["python_coding", "data_viz", "sql_query"],
         ["性能瓶颈"], ["加Redis缓存层"]),
        ("分析竞品的定价策略", 0.6,
         ["search_first", "decompose_first"], ["python_coding", "writing"],
         ["公开数据不够"], ["结合爬虫数据和行业报告"]),
    ]

    for task, reward, actions, skills, problems, solutions in experiences:
        alpha.think(task)
        alpha.feedback(reward, actions, skills, problems, solutions)


def train_beta(beta):
    """Beta: 全栈开发者的助手，15 轮编码相关经验"""
    experiences = [
        ("搭建一个用户注册登录系统", 0.9,
         ["direct_execution", "iterate_and_refine"], ["web_dev", "python_coding"],
         ["密码加密方案选择"], ["使用bcrypt"]),
        ("做一个商品展示页面", 0.85,
         ["direct_execution", "use_skill"], ["web_dev"],
         ["移动端适配问题"], ["用CSS Grid加媒体查询"]),
        ("优化API响应速度", 0.8,
         ["direct_execution", "iterate_and_refine"], ["python_coding", "sql_query"],
         ["N+1查询问题"], ["用JOIN替代循环查询"]),
        ("写一个自动部署脚本", 0.95,
         ["direct_execution", "use_skill"], ["python_coding"],
         [], []),
        ("做一个数据看板给运营看", 0.7,
         ["iterate_and_refine", "combine_skills"], ["web_dev", "data_viz", "sql_query"],
         ["图表加载太慢"], ["前端做数据缓存"]),
        ("重构后端的权限系统", 0.9,
         ["decompose_first", "iterate_and_refine"], ["python_coding"],
         ["需要兼容旧接口"], ["用装饰器模式做权限检查"]),
        ("写个爬虫抓竞品价格", 0.85,
         ["direct_execution", "iterate_and_refine"], ["python_coding"],
         ["反爬机制"], ["加随机延迟加代理IP"]),
        ("做一个实时聊天功能", 0.8,
         ["decompose_first", "iterate_and_refine"], ["web_dev", "python_coding"],
         ["WebSocket连接不稳定"], ["加心跳检测和自动重连"]),
        ("实现第三方支付对接", 0.85,
         ["direct_execution", "use_skill"], ["python_coding", "web_dev"],
         ["回调验签失败"], ["仔细对照文档的签名算法"]),
        ("写一个任务队列系统", 0.9,
         ["decompose_first", "iterate_and_refine"], ["python_coding"],
         ["任务重试逻辑复杂"], ["用指数退避加死信队列"]),
        ("做一个文件上传组件", 0.85,
         ["direct_execution", "iterate_and_refine"], ["web_dev", "python_coding"],
         ["大文件上传超时"], ["改用分片上传"]),
        ("优化数据库索引", 0.8,
         ["direct_execution", "use_skill"], ["sql_query"],
         ["慢查询日志分析困难"], ["用EXPLAIN分析执行计划"]),
        ("实现消息推送功能", 0.75,
         ["decompose_first", "iterate_and_refine"], ["python_coding", "web_dev"],
         ["推送延迟高"], ["改用消息队列异步发送"]),
        ("写一个CI/CD流水线", 0.9,
         ["direct_execution", "use_skill"], ["python_coding"],
         [], []),
        ("做一个管理后台", 0.8,
         ["iterate_and_refine", "combine_skills"], ["web_dev", "python_coding", "sql_query"],
         ["权限粒度不够细"], ["复用之前的装饰器权限系统"]),
    ]

    for task, reward, actions, skills, problems, solutions in experiences:
        beta.think(task)
        beta.feedback(reward, actions, skills, problems, solutions)


def compare_on_task(alpha, beta, task_description):
    """对比两个 Agent 面对同一任务时的完整上下文"""
    print("=" * 70)
    print(f"  测试任务: {task_description}")
    print("=" * 70)

    for name, agent in [("Alpha（数据分析师助手）", alpha), ("Beta（全栈开发者助手）", beta)]:
        print(f"\n{'─' * 70}")
        print(f"  {name} 的 PersonaLayer 输出")
        print(f"{'─' * 70}")

        context = agent.think(task_description)

        print(f"\n【任务分类】{context.get('task_category', '?')}")

        strategy = context.get("strategy", {})
        print(f"\n【决策模式】{strategy.get('mode', '?')}")
        print(f"【置信度】{strategy.get('confidence', 0):.0%}")
        print(f"【建议动作】{', '.join(strategy.get('actions', []))}")
        print(f"【建议技能】{', '.join(strategy.get('skills', ['无']))}")
        print(f"【决策依据】{strategy.get('reasoning', '?')}")

        activated = context.get("activated_experiences", [])
        if activated:
            print(f"\n【激活的经验】（共 {len(activated)} 条）")
            for content, score in activated[:5]:
                print(f"  {score:.2f} | {content[:60]}")

        print(f"\n【完整 System Prompt Addition】")
        prompt = context.get("system_prompt_addition", "")
        print(prompt)


def show_graph_diff(alpha, beta):
    """展示两个 Agent 的图谱差异"""
    print("\n" + "=" * 70)
    print("  图谱结构对比")
    print("=" * 70)

    for name, agent in [("Alpha", alpha), ("Beta", beta)]:
        stats = agent.persona.experience_graph.get_stats()
        topo = agent.persona.experience_graph.get_topology_stats()
        print(f"\n  {name}:")
        print(f"    总节点: {stats['total_nodes']}, 总边: {stats['total_edges']}")
        print(f"    节点类型: {stats['node_types']}")
        for domain, t in sorted(topo.items()):
            print(f"    [{domain}] 节点{t['node_count']} 边{t['edge_count']} "
                  f"密度{t['edge_density']:.1f} 概念{t['concept_count']}")

    print(f"\n  能力画像对比:")
    a_comp = alpha.persona.competence.competence_scores
    b_comp = beta.persona.competence.competence_scores
    for dim in sorted(set(list(a_comp.keys()) + list(b_comp.keys()))):
        a = a_comp.get(dim, 0)
        b = b_comp.get(dim, 0)
        if a > 0.01 or b > 0.01:
            a_bar = "█" * int(a * 20)
            b_bar = "█" * int(b * 20)
            print(f"    {dim:20s}  A[{a_bar:20s}]{a:.0%}  B[{b_bar:20s}]{b:.0%}")

    similarity = alpha.persona.competence.similarity(beta.persona.competence)
    print(f"\n  整体能力相似度: {similarity:.0%}")


def main():
    # 清理旧数据
    for f in ["/tmp/dry_alpha.json", "/tmp/dry_beta.json"]:
        if os.path.exists(f):
            os.remove(f)

    print("创建两个 Agent（相同 Skill，相同语义层配置）...")
    alpha = create_agent("Alpha", "/tmp/dry_alpha.json")
    beta = create_agent("Beta", "/tmp/dry_beta.json")

    print("\n训练 Alpha（数据分析师助手，15 轮经验）...")
    train_alpha(alpha)
    print("训练 Beta（全栈开发者助手，15 轮经验）...")
    train_beta(beta)

    # 测试任务 1：混合型任务
    compare_on_task(alpha, beta,
        "帮我分析一下我们网站最近一个月的用户行为数据，然后做一个可视化报告页面")

    # 测试任务 2：数据分析偏向
    compare_on_task(alpha, beta,
        "这个月用户留存率下降了，帮我看看原因")

    # 测试任务 3：开发偏向
    compare_on_task(alpha, beta,
        "帮我写一个后台管理系统的用户权限模块")

    # 图谱对比
    show_graph_diff(alpha, beta)

    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    print("""
  以上输出展示了两个 Agent 面对同一任务时，PersonaLayer 生成的
  system prompt 有多大差异。差异越大、越具体，说明框架的"经历
  塑造个性"机制越有效。

  重点关注：
  1. 两个 Agent 激活的经验是否不同？
  2. 策略建议是否不同？
  3. 身份描述是否反映了各自的经历？
  4. 差异是否"合理"（Alpha 偏数据，Beta 偏开发）？
    """)


if __name__ == "__main__":
    main()
