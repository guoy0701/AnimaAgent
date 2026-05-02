"""
个性层 (Persona Layer)

核心理念：在大模型和外部世界之间，插入一个可演化的"人格"层。
这个层整合了经验图谱、策略网络和能力向量三个组件，
让Agent不再是一个无状态的Prompt中转站，而是一个有积淀、有个性的"人"。

架构示意：
                ┌─────────────────────┐
                │      大模型          │  ← 通用能力，所有Agent共享
                │   (不被修改)         │
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────┐
                │     个性层           │  ← 每个Agent独有，持续演化
                │  (Persona Layer)     │
                │                     │
                │  · 经验图谱          │
                │  · 策略网络          │
                │  · 能力向量          │
                └─────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     ┌────────┐     ┌────────┐     ┌────────┐
     │ Skill A│     │ Skill B│     │ Skill C│
     └────────┘     └────────┘     └────────┘
"""

import json
import time
from pathlib import Path

import jieba

from .experience_graph import ExperienceGraph, NodeType, EdgeType
from .strategy import StrategyNetwork, TaskCategory, ActionType
from .competence import CompetenceEmbedding


class PersonaLayer:
    """
    个性层：三大组件的整合器。

    负责：
    1. 协调三个组件的工作
    2. 在任务到来时，构建完整的"个性化上下文"注入LLM
    3. 在任务结束后，从反馈中更新三个组件
    4. 持久化和加载Agent的"灵魂"
    """

    def __init__(self, agent_name: str = "Anima"):
        self.agent_name = agent_name
        self.created_at = time.time()

        # 三大核心组件
        self.experience_graph = ExperienceGraph()
        self.strategy_network = StrategyNetwork()
        self.competence = CompetenceEmbedding()

        # 已注册的Skill
        self.skills: dict[str, dict] = {}

        # 交互计数
        self.interaction_count = 0

        # 语义层（可选）
        self._embedding_provider = None
        self._extractor = None

    def configure_semantic(self, embedding_provider=None, extractor=None):
        """配置语义层：embedding provider 和概念提取器（均可选）。"""
        self._embedding_provider = embedding_provider
        self._extractor = extractor

    def register_skill(self, skill_name: str, description: str,
                       categories: list[str] = None):
        """注册一个Skill"""
        if skill_name in self.skills:
            # 更新元数据，但不创建重复节点
            self.skills[skill_name]["description"] = description
            self.skills[skill_name]["categories"] = categories or []
            return

        self.skills[skill_name] = {
            "name": skill_name,
            "description": description,
            "categories": categories or [],
            "usage_count": 0,
            "success_count": 0,
        }
        # 在经验图谱中创建Skill节点
        self.experience_graph.add_node(
            NodeType.SKILL, f"Skill: {skill_name} - {description}")

    def prepare_context(self, task_description: str,
                        task_category: TaskCategory = None) -> dict:
        """
        为一个新任务准备完整的"个性化上下文"。

        这是个性层的核心输出——将三个组件的信息整合成
        一个可以注入到LLM的system prompt中的上下文包。
        """
        self.interaction_count += 1

        # 1. 在经验图谱中搜索相关经验（排除 SKILL 节点，它们是结构节点不是经验）
        if self._embedding_provider:
            query_emb = self._embedding_provider.embed(task_description)
            seed_results = self.experience_graph.find_by_embedding(query_emb, top_k=10)
            seed_ids = [n.id for n, _ in seed_results
                        if n.node_type != NodeType.SKILL][:5]
        else:
            keywords = self._extract_keywords(task_description)
            seed_nodes = self.experience_graph.find_by_content(keywords)
            seed_ids = [n.id for n in seed_nodes
                        if n.node_type != NodeType.SKILL][:5]

        activated = []
        experience_context = "没有找到相关历史经验。"
        if seed_ids:
            activated = self.experience_graph.spreading_activation(seed_ids)
            from anima.narrator import narrate_subgraph
            experience_context = narrate_subgraph(
                self.experience_graph, activated, max_stories=3)

        # 2. 从策略网络获取策略建议
        if not task_category:
            task_category = self._infer_category(task_description)
        available_skills = list(self.skills.keys())
        strategy_context = {"task": task_description}
        if self._embedding_provider:
            # reuse query_emb already computed above for seed selection
            strategy_context["task_embedding"] = query_emb
        strategy = self.strategy_network.decide_strategy(
            task_category, strategy_context, available_skills)
        strategy_context = self.strategy_network.generate_strategy_prompt(
            task_category)

        # 3. 获取能力画像（含经验亮点）
        highlights = self._get_experience_highlights()
        identity_context = self.competence.generate_identity_prompt(highlights)

        # 4. 整合成完整的个性化上下文
        full_context = self._build_system_prompt(
            experience_context, strategy_context,
            identity_context, strategy)

        return {
            "system_prompt_addition": full_context,
            "strategy": strategy,
            "activated_experiences": [(n.content, round(a, 2))
                                     for n, a in activated
                                     if n.node_type != NodeType.SKILL][:5],
            "task_category": task_category.value,
        }

    def _build_system_prompt(self, experience: str, strategy: str,
                              identity: str, strategy_plan: dict) -> str:
        """构建注入到LLM的个性化system prompt"""
        sections = [
            f"# Agent {self.agent_name} 的个性化上下文",
            f"（第 {self.interaction_count} 次交互）\n",
            identity,
            "\n## 相关历史经验\n",
            experience,
            "\n## 策略建议\n",
            strategy,
        ]

        if strategy_plan.get("reasoning"):
            sections.append(f"\n当前决策模式：{strategy_plan['reasoning']}")
            sections.append(f"置信度：{strategy_plan.get('confidence', 0):.0%}")

        if strategy_plan.get("skills"):
            sections.append(f"\n建议使用的Skill：{', '.join(strategy_plan['skills'])}")

        return "\n".join(sections)

    def record_experience(self, task_description: str,
                          task_category: TaskCategory,
                          actions_taken: list[str],
                          skills_used: list[str],
                          outcome: str,
                          problems_encountered: list[str] = None,
                          solutions_found: list[str] = None):
        """
        记录一次完整的任务经历到经验图谱中。
        这是Agent"积累经验"的核心机制。
        """
        # 计算任务节点的 embedding（如果配置了 embedding provider）
        task_embedding = []
        if self._embedding_provider:
            task_embedding = self._embedding_provider.embed(task_description)

        # 创建任务节点
        task_node = self.experience_graph.add_node(
            NodeType.TASK, task_description,
            embedding=task_embedding,
            metadata={"category": task_category.value,
                      "actions": actions_taken,
                      "skills": skills_used})

        # 如果配置了提取器，提取概念节点
        if self._extractor:
            extraction = self._extractor.extract(task_description)
            for concept_text in extraction.concepts:
                concept_emb = []
                if self._embedding_provider:
                    concept_emb = self._embedding_provider.embed(concept_text)

                existing = self._find_existing_concept(concept_text, concept_emb)
                if existing:
                    self.experience_graph.add_edge(
                        task_node.id, existing.id, EdgeType.COMPOSED_OF)
                    existing.activation_count += 1
                else:
                    concept_node = self.experience_graph.add_node(
                        NodeType.CONCEPT, concept_text, embedding=concept_emb)
                    self.experience_graph.add_edge(
                        task_node.id, concept_node.id, EdgeType.COMPOSED_OF)

        # 创建结果节点并连接
        outcome_node = self.experience_graph.add_node(
            NodeType.FEEDBACK, f"结果: {outcome}")
        self.experience_graph.add_edge(
            task_node.id, outcome_node.id, EdgeType.CAUSAL)

        # 连接到使用的Skill
        for skill_name in skills_used:
            skill_nodes = [n for n in self.experience_graph.find_by_type(
                NodeType.SKILL) if skill_name in n.content]
            for sn in skill_nodes:
                self.experience_graph.add_edge(
                    task_node.id, sn.id, EdgeType.REQUIRES)

        # 记录问题和解决方案（按索引 1:1 配对，避免笛卡尔积）
        if problems_encountered:
            for i, prob in enumerate(problems_encountered):
                prob_node = self.experience_graph.add_node(
                    NodeType.PROBLEM, prob)
                self.experience_graph.add_edge(
                    task_node.id, prob_node.id, EdgeType.CAUSAL)

                # 按索引配对（1:1），而非每个方案连接所有问题
                if solutions_found and i < len(solutions_found):
                    sol_node = self.experience_graph.add_node(
                        NodeType.SOLUTION, solutions_found[i])
                    self.experience_graph.add_edge(
                        prob_node.id, sol_node.id, EdgeType.SOLVED_BY)

        # 处理多出的解决方案（没有对应问题的部分）
        if solutions_found and len(solutions_found) > len(problems_encountered or []):
            for sol in solutions_found[len(problems_encountered or []):]:
                sol_node = self.experience_graph.add_node(
                    NodeType.SOLUTION, sol)
                self.experience_graph.add_edge(
                    task_node.id, sol_node.id, EdgeType.CAUSAL)

        # 尝试发现与之前任务的关联
        self._discover_connections(task_node)

    def learn_from_feedback(self, task_category: TaskCategory,
                            actions_taken: list[str],
                            skills_used: list[str],
                            reward: float,
                            task_embedding: list = None):
        """
        从主人的反馈中学习，更新所有三个组件。
        """
        # 1. 更新策略网络（附带 task_embedding 让策略记录可被相似检索）
        context = {}
        if task_embedding:
            context["task_embedding"] = task_embedding
        self.strategy_network.learn_from_feedback(
            task_category, actions_taken, skills_used, reward, context)

        # 2. 对经验图谱做赫布学习（强化共同激活的节点间的连接）
        keywords = actions_taken + skills_used
        seed_nodes = self.experience_graph.find_by_content(keywords)
        seed_ids = [n.id for n in seed_nodes[:5]]
        if seed_ids:
            activated = self.experience_graph.spreading_activation(seed_ids)
            self.experience_graph.hebbian_update(activated)

        # 3. 更新能力向量
        graph_stats = self.experience_graph.get_stats()
        strategy_summary = self.strategy_network.get_profile_summary()
        topology_stats = self.experience_graph.get_topology_stats()
        self.competence.update_from_graph_and_strategy(
            graph_stats, strategy_summary, topology_stats)

        # 4. 更新Skill统计
        for skill in skills_used:
            if skill in self.skills:
                self.skills[skill]["usage_count"] += 1
                if reward > 0.5:
                    self.skills[skill]["success_count"] += 1

    def _find_existing_concept(self, concept_text: str, concept_embedding: list):
        """查找是否已存在相似的 CONCEPT 节点（用于去重）。"""
        if concept_embedding and self._embedding_provider:
            from anima.embedding import cosine_similarity
            best_match = None
            best_sim = 0.7  # 相似度阈值，超过则视为同一概念
            for node in self.experience_graph.find_by_type(NodeType.CONCEPT):
                if node.embedding:
                    sim = cosine_similarity(concept_embedding, node.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = node
            return best_match

        # 无 embedding 时退回精确文本匹配
        for node in self.experience_graph.find_by_type(NodeType.CONCEPT):
            if node.content == concept_text:
                return node
        return None

    def _discover_connections(self, new_node):
        """发现新节点与已有节点之间的潜在关联"""
        new_words = set(jieba.lcut(new_node.content.lower()))
        new_words = {w for w in new_words if len(w) > 1}
        if not new_words:
            return

        for existing_node in self.experience_graph.nodes.values():
            if existing_node.id == new_node.id:
                continue
            if existing_node.node_type != new_node.node_type:
                continue

            existing_words = set(jieba.lcut(existing_node.content.lower()))
            existing_words = {w for w in existing_words if len(w) > 1}
            if not existing_words:
                continue

            overlap = len(new_words & existing_words)
            total = len(new_words | existing_words)
            if total > 0 and overlap / total > 0.3:
                self.experience_graph.add_edge(
                    new_node.id, existing_node.id,
                    EdgeType.SIMILAR, weight=overlap / total)

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词（使用 jieba 分词，支持中文）"""
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "它", "们", "可以", "什么", "怎么", "那", "吗", "吧", "啊",
            "帮", "请", "想", "能", "把", "让", "给", "用", "做", "写",
            "看看", "一下", "下", "个", "来", "过", "被", "比", "从",
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "with", "and", "or", "but", "not",
            "this", "that", "i", "me", "my", "you", "your", "he", "she",
            "it", "we", "they", "can", "will", "do", "does",
            "help", "please", "want", "need", "make",
        }
        words = jieba.lcut(text)
        return [w.strip() for w in words
                if w.strip() and w.strip() not in stop_words and len(w.strip()) > 1]

    def _get_experience_highlights(self) -> list:
        """从成功的任务节点中提取经验亮点，用于身份提示词。"""
        highlights = []
        task_nodes = self.experience_graph.find_by_type(NodeType.TASK)
        for task in task_nodes[-10:]:
            for neighbor_id, edge in self.experience_graph._forward.get(task.id, []):
                neighbor = self.experience_graph.nodes.get(neighbor_id)
                if neighbor and neighbor.node_type == NodeType.FEEDBACK:
                    if "成功" in neighbor.content:
                        highlights.append(task.content)
                        break
        return highlights[-5:]

    def _infer_category(self, task: str) -> TaskCategory:
        """从任务描述推断任务类别"""
        category_keywords = {
            TaskCategory.DATA_ANALYSIS: [
                "数据", "分析", "统计", "图表", "报表", "指标", "留存",
                "转化", "漏斗", "趋势", "预测", "ROI", "AB测试",
                "取数", "看板", "大屏", "画像", "分群", "可视化",
                "data", "analysis", "chart", "metric", "dashboard",
                "retention", "funnel", "cohort",
            ],
            TaskCategory.CODE_WRITING: [
                "代码", "编程", "函数", "程序", "脚本", "模块", "接口",
                "系统", "开发", "搭建", "实现", "重构", "优化", "部署",
                "后端", "前端", "API", "SDK", "爬虫", "权限", "登录",
                "注册", "数据库", "索引", "缓存", "队列", "CI/CD",
                "code", "script", "function", "module", "api", "bug",
                "deploy", "refactor", "backend", "frontend",
            ],
            TaskCategory.CONTENT_CREATION: [
                "文章", "文案", "内容", "报告", "文档", "说明",
                "write", "article", "content", "document", "report",
            ],
            TaskCategory.PROBLEM_SOLVING: [
                "问题", "解决", "修复", "错误", "排查", "定位", "故障",
                "problem", "solve", "fix", "error", "debug", "issue",
            ],
            TaskCategory.COMMUNICATION: [
                "邮件", "回复", "沟通", "消息", "通知", "反馈",
                "email", "reply", "message", "communicate",
            ],
            TaskCategory.RESEARCH: [
                "调研", "研究", "查找", "搜索", "对比", "竞品",
                "research", "search", "find", "compare", "survey",
            ],
            TaskCategory.PLANNING: [
                "计划", "规划", "安排", "策略", "方案", "架构",
                "plan", "schedule", "strategy", "roadmap",
            ],
            TaskCategory.CREATIVE: [
                "创意", "设计", "创作", "灵感", "原型", "UI", "UX",
                "creative", "design", "idea", "prototype",
            ],
        }

        task_words = set(jieba.lcut(task.lower()))
        best_cat = TaskCategory.UNKNOWN
        best_score = 0

        for cat, kw_list in category_keywords.items():
            score = sum(1 for kw in kw_list if kw in task_words or kw in task.lower())
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    def maintenance(self):
        """
        定期维护——相当于Agent的"睡眠"。
        整理经验、衰减不重要的记忆、更新能力画像。
        """
        # 衰减长期不活跃的节点
        self.experience_graph.decay_all()

        # 更新能力画像
        graph_stats = self.experience_graph.get_stats()
        strategy_summary = self.strategy_network.get_profile_summary()
        topology_stats = self.experience_graph.get_topology_stats()
        self.competence.update_from_graph_and_strategy(
            graph_stats, strategy_summary, topology_stats)

    def get_full_status(self) -> dict:
        """获取Agent的完整状态概览"""
        return {
            "agent_name": self.agent_name,
            "interactions": self.interaction_count,
            "created_at": self.created_at,
            "graph_stats": self.experience_graph.get_stats(),
            "strategy_summary": self.strategy_network.get_profile_summary(),
            "competence": {
                "scores": self.competence.competence_scores,
                "style": self.competence.style_tendencies,
                "domain_tags": self.competence.domain_tags,
                "confidence": self.competence.get_confidence(),
            },
            "skills": {name: {
                "usage": info["usage_count"],
                "success": info["success_count"],
            } for name, info in self.skills.items()},
        }

    def save(self, filepath: str):
        """将Agent的"灵魂"持久化到文件"""
        data = {
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "interaction_count": self.interaction_count,
            "experience_graph": self.experience_graph.to_dict(),
            "strategy_network": self.strategy_network.to_dict(),
            "competence": self.competence.to_dict(),
            "skills": self.skills,
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PersonaLayer":
        """从文件加载Agent的"灵魂" """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        persona = cls(agent_name=data.get("agent_name", "Anima"))
        persona.created_at = data.get("created_at", time.time())
        persona.interaction_count = data.get("interaction_count", 0)
        persona.experience_graph = ExperienceGraph.from_dict(
            data["experience_graph"])
        persona.strategy_network = StrategyNetwork.from_dict(
            data["strategy_network"])
        persona.competence = CompetenceEmbedding.from_dict(
            data["competence"])
        persona.skills = data.get("skills", {})
        return persona
