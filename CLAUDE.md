# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima 是一个 Python AI Agent 个性层框架，核心理念是 "Agent 不是工具，是人"。Agent 的行为由其独特的交互历史塑造，而非简单的 Skill 累加。通过经验图谱（激活扩散 + 赫布学习）让不同 Agent 面对同一任务产生不同的行为。

## Commands

```bash
# 运行测试
python -m pytest tests/ -v

# 运行干对比实验（不需要 API key）
python -X utf8 dry_comparison.py

# 运行 Qwen 示例（需要 DASHSCOPE_API_KEY）
python examples/qwen_quickstart.py

# 安装开发依赖
pip install -e ".[dev]"

# 安装 Qwen/OpenAI 支持
pip install -e ".[qwen]"
```

## Architecture

核心数据流：`AnimaAgent.chat()` → `PersonaLayer` → 三大组件 → 生成 system prompt → `LLMProvider.chat()` → 返回响应。

```
anima/
├── agent.py              # AnimaAgent: 用户 API 入口（configure/chat/feedback/think）
├── provider.py           # LLMProvider ABC + OpenAICompatibleProvider（Qwen/DeepSeek/GPT）
├── persona.py            # PersonaLayer: 整合三大组件，生成 LLM system prompt
├── experience_graph.py   # ExperienceGraph: 图结构记忆（激活扩散 + 赫布学习 + 遗忘衰减）
├── strategy.py           # StrategyNetwork: 行为决策（EMA 学习 + 探索/利用平衡）
├── competence.py         # CompetenceEmbedding: 能力画像（12维能力 + 5维风格 + 图拓扑）
├── embedding.py          # EmbeddingProvider ABC + MockEmbeddingProvider
├── extractor.py          # ExperienceExtractor ABC + MockExtractor（LLM 概念提取）
├── narrator.py           # 子图叙事化（激活的经验 → 因果链故事）
└── __init__.py           # 公共 API 导出
```

### 三大组件的职责边界

- **ExperienceGraph** 决定"想起什么"：节点类型（TASK/CONCEPT/SKILL/PROBLEM/SOLUTION/FEEDBACK/PATTERN）通过有向边（CAUSAL/TEMPORAL/SOLVED_BY/REQUIRES/COMPOSED_OF）和无向边（SIMILAR/CONFLICTS/REINFORCES）关联。面对新任务时，从 embedding 语义相似度找种子节点，做 BFS 激活扩散，返回 top-K 激活节点。SKILL 节点被排除在种子选择之外。
- **StrategyNetwork** 决定"怎么做"：按 TaskCategory 维护 action 偏好分布（EMA 学习，分数收敛于 [-1, 1]）。有 embedding 时通过余弦相似度查找历史相似任务来做决策，无 embedding 时回退到偏好分数。exploration_rate 随经验衰减，track attempt_counts 驱动探索。
- **CompetenceEmbedding** 决定"是什么样的"：融合策略网络成功率（70%）和图谱拓扑特征（30%，边密度 + 概念数），生成自然语言身份描述 + 具体经验亮点注入 system prompt。

### Provider 接口（解耦多 LLM）

`LLMProvider` 是统一接口，同时继承 `EmbeddingProvider` 和 `ExperienceExtractor`，提供 `chat()` + `embed()` + `extract()` 三合一能力。`OpenAICompatibleProvider` 兼容所有 OpenAI 格式 API（Qwen/DeepSeek/GPT/Ollama/vLLM）。

### 学习闭环

`agent.chat(task)` → PersonaLayer 注入个性化上下文 → LLM 回答 → `agent.feedback(reward)` 触发学习：
1. ExperienceGraph 添加 TASK/CONCEPT/PROBLEM/SOLUTION 节点，赫布强化共激活连接
2. StrategyNetwork 用 EMA 更新 action/skill 偏好，记录 task_embedding 供相似检索
3. CompetenceEmbedding 从图谱拓扑 + 策略统计重新计算
4. narrator 将激活子图转化为因果叙事注入 system prompt
5. 自动持久化到 `anima_data/{agent_name}.json`

## Key Patterns

- **边方向性**：CAUSAL/TEMPORAL/SOLVED_BY/REQUIRES/COMPOSED_OF 是有向边（只正向传播激活），SIMILAR/CONFLICTS/REINFORCES 是无向边
- **自环保护**：`add_edge(A, A)` 被拒绝，防止激活爆炸
- **EMA 学习**：策略偏好用 EMA（alpha=0.2）而非累加，分数收敛不发散
- **语义层可选**：未 configure 时回退到 jieba 关键词匹配，功能不降级
- **中文分词**：使用 jieba，关键词提取和连接发现都基于分词结果
- **概念去重**：embedding 余弦相似度 > 0.7 视为同一概念，复用已有节点
- **action 验证**：learn_from_feedback 过滤无效 action 字符串

## License

MIT
