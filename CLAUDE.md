# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima 是一个纯 Python AI Agent 框架，核心理念是 "Agent 不是工具，是人"。Agent 的行为由其独特的交互历史塑造，而非简单的 Skill 累加。零外部依赖，仅使用 Python 标准库。

## Commands

```bash
# 运行 demo（创建两个 Agent，模拟不同经历后观察行为差异）
python demo.py
```

无构建步骤、无测试框架、无 lint 配置。项目为研究/演示性质。

## Architecture

核心数据流：`AnimaAgent` → `PersonaLayer` → 三大组件 → 生成 system prompt 注入 LLM。

```
anima/
├── agent.py              # AnimaAgent: 用户 API 入口（think/feedback/export_soul）
├── persona.py            # PersonaLayer: 整合三大组件，生成 LLM system prompt 片段
├── experience_graph.py   # ExperienceGraph: 图结构记忆（激活扩散 + 赫布学习 + 遗忘衰减）
├── strategy.py           # StrategyNetwork: 行为决策（探索/利用平衡 + 反馈强化学习）
├── competence.py         # CompetenceEmbedding: 能力画像（12维能力 + 5维风格）
└── __init__.py           # 公共 API 导出
```

### 三大组件的职责边界

- **ExperienceGraph** 决定"想起什么"：节点类型（TASK/CONCEPT/SKILL/PROBLEM/SOLUTION/FEEDBACK/PATTERN）通过边类型（CAUSAL/SIMILAR/TEMPORAL 等）关联。面对新任务时，从关键词匹配的种子节点出发做 BFS 激活扩散，返回 top-K 激活节点作为相关记忆。
- **StrategyNetwork** 决定"怎么做"：按 TaskCategory（9类）维护 action 偏好分布。exploration_rate 随经验指数衰减（`exp(-0.01 × attempts)`），年轻 Agent 探索多，成熟 Agent 利用多。
- **CompetenceEmbedding** 决定"是什么样的"：从前两个组件的统计数据动态计算，生成自然语言身份描述注入 system prompt。不直接学习，而是被动反映。

### 学习闭环

所有学习由 `agent.feedback(reward, ...)` 触发（reward 范围 -1.0 到 1.0）：
1. ExperienceGraph 添加新节点/边，赫布强化共激活节点间的连接
2. StrategyNetwork 更新对应 category 的 action 偏好分数
3. CompetenceEmbedding 从更新后的图谱和策略网络重新计算
4. 自动持久化到 `anima_data/{agent_name}.json`

## Key Patterns

- **节点 ID 生成**：MD5 hash of `{type}:{content}:{timestamp}`
- **图边双向**：所有边在邻接表中双向存储
- **激活扩散上限**：`initial_activation × 3.0`，防止爆炸
- **历史记录上限**：sequence patterns 限 20 条，history 限 100 条
- **持久化格式**：单个 JSON 文件包含全部 Agent 状态（图谱 + 策略 + 能力向量）
- **关键词提取**：简单停用词过滤（中英文），无 ML/embedding 依赖
- **代码注释**：中英文混合，用户侧文档和示例全中文

## License

MIT
