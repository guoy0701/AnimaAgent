"""
Anima Agent - 主Agent类

这是用户直接交互的入口。它整合了个性层(Persona Layer)与大模型(LLM)，
实现了"有人格的Agent"的完整闭环：

1. 接收任务 → 个性层准备上下文 → 大模型生成响应
2. 主人反馈 → 个性层学习更新 → Agent演化

每一次交互都在改变Agent的行为模式，让它逐渐成为主人的数字分身。
"""

import json
import time
from pathlib import Path

from .persona import PersonaLayer
from .strategy import TaskCategory
from .experience_graph import NodeType


class AnimaAgent:
    """
    Anima Agent：有灵魂的AI Agent。

    区别于传统Agent框架的核心特点：
    1. 能力不等于Skill的简单累加，而是经验结构化后的涌现
    2. 同一个Skill在不同Agent上效果不同
    3. 每个Agent是独一无二的，不可轻易复制
    4. Agent的"人格"会随交互持续演化
    """

    def __init__(self, name: str = "Anima",
                 save_path: str = None):
        self.name = name
        self.save_path = save_path or f"./anima_data/{name}.json"

        # 尝试加载已有的灵魂
        if Path(self.save_path).exists():
            self.persona = PersonaLayer.load(self.save_path)
            print(f"[Anima] 已加载Agent '{name}'的灵魂"
                  f"（{self.persona.interaction_count}次交互经验）")
        else:
            self.persona = PersonaLayer(agent_name=name)
            print(f"[Anima] 创建了新的Agent '{name}'")

        # 当前任务的上下文（用于反馈时关联）
        self._current_task = None
        self._current_context = None

    def register_skill(self, name: str, description: str,
                       categories: list[str] = None):
        """注册Skill"""
        self.persona.register_skill(name, description, categories)
        print(f"[Anima] 注册Skill: {name}")

    def think(self, task: str) -> dict:
        """
        Agent的"思考"过程——面对一个新任务时的内部处理。

        返回：
        - system_prompt_addition: 注入到LLM的个性化上下文
        - strategy: 策略建议
        - activated_experiences: 被激活的相关经验
        """
        context = self.persona.prepare_context(task)

        self._current_task = task
        self._current_context = context

        return context

    def process_task(self, task: str, llm_caller=None) -> str:
        """
        完整的任务处理流程。

        如果提供了llm_caller，会调用大模型生成响应。
        llm_caller应该是一个函数：(system_prompt, user_message) -> response_text
        """
        # Step 1: 思考——准备个性化上下文
        context = self.think(task)

        # Step 2: 如果有LLM，调用它
        if llm_caller:
            system_prompt = self._build_full_system_prompt(context)
            response = llm_caller(system_prompt, task)
        else:
            response = self._simulate_response(task, context)

        return response

    def _build_full_system_prompt(self, context: dict) -> str:
        """构建完整的system prompt"""
        base = f"你是{self.name}，一个有独特经验和个性的AI助手。\n\n"
        persona_context = context.get("system_prompt_addition", "")
        return base + persona_context

    def _simulate_response(self, task: str, context: dict) -> str:
        """无LLM时的模拟响应（用于演示）"""
        strategy = context.get("strategy", {})
        experiences = context.get("activated_experiences", [])

        lines = [f"[{self.name}的思考过程]",
                 f"任务类别：{context.get('task_category', 'unknown')}",
                 f"决策模式：{strategy.get('mode', 'unknown')}",
                 f"置信度：{strategy.get('confidence', 0):.0%}",
                 f"建议动作：{', '.join(strategy.get('actions', []))}",
                 f"建议Skill：{', '.join(strategy.get('skills', ['无']))}"]

        if experiences:
            lines.append(f"\n激活的相关经验：")
            for content, activation in experiences:
                lines.append(f"  - {content} (相关度: {activation})")

        lines.append(f"\n[这里是LLM基于以上上下文生成的实际响应]")
        return "\n".join(lines)

    def feedback(self, reward: float,
                 actions_taken: list[str] = None,
                 skills_used: list[str] = None,
                 problems: list[str] = None,
                 solutions: list[str] = None):
        """
        主人给予反馈——这是Agent成长的核心驱动力。

        reward: -1.0 到 1.0
        """
        if not self._current_task or not self._current_context:
            print("[Anima] 没有当前任务上下文，无法学习")
            return

        task_category = TaskCategory(
            self._current_context.get("task_category", "unknown"))
        strategy = self._current_context.get("strategy", {})

        actions = actions_taken or strategy.get("actions", [])
        skills = skills_used or strategy.get("skills", [])

        # 记录经验
        outcome = "成功" if reward > 0.5 else "一般" if reward > 0 else "失败"
        self.persona.record_experience(
            self._current_task, task_category,
            actions, skills, outcome, problems, solutions)

        # 从反馈中学习
        self.persona.learn_from_feedback(
            task_category, actions, skills, reward)

        # 自动保存
        self._auto_save()

        reward_desc = "很满意" if reward > 0.7 else "满意" if reward > 0.3 \
            else "一般" if reward > -0.3 else "不满意"
        print(f"[Anima] 收到反馈: {reward_desc} ({reward:.1f})，已更新学习")

    def status(self) -> dict:
        """查看Agent的当前状态"""
        return self.persona.get_full_status()

    def sleep(self):
        """
        让Agent"睡眠"——执行记忆整合和遗忘。
        建议定期调用（比如每天结束时）。
        """
        self.persona.maintenance()
        self._auto_save()
        print(f"[Anima] {self.name}已完成记忆整合")

    def _auto_save(self):
        """自动保存"""
        self.persona.save(self.save_path)

    def compare_with(self, other: "AnimaAgent") -> dict:
        """
        与另一个Agent比较——直观展示两个Agent的分化程度。
        """
        similarity = self.persona.competence.similarity(
            other.persona.competence)

        my_status = self.status()
        other_status = other.status()

        return {
            "overall_similarity": round(similarity, 2),
            "comparison": {
                self.name: {
                    "interactions": my_status["interactions"],
                    "domain_tags": my_status["competence"]["domain_tags"],
                    "confidence": round(
                        my_status["competence"]["confidence"], 2),
                },
                other.name: {
                    "interactions": other_status["interactions"],
                    "domain_tags": other_status["competence"]["domain_tags"],
                    "confidence": round(
                        other_status["competence"]["confidence"], 2),
                },
            },
            "interpretation": (
                "几乎相同" if similarity > 0.9 else
                "非常相似" if similarity > 0.7 else
                "有一定差异" if similarity > 0.5 else
                "差异明显" if similarity > 0.3 else
                "完全不同的Agent"
            ),
        }

    def export_soul(self, filepath: str = None):
        """导出Agent的灵魂（完整状态）"""
        path = filepath or f"./{self.name}_soul_export.json"
        self.persona.save(path)
        print(f"[Anima] 灵魂已导出到: {path}")

    @classmethod
    def import_soul(cls, filepath: str,
                    new_name: str = None) -> "AnimaAgent":
        """
        导入Agent的灵魂。

        注意：导入的灵魂在新主人手下可能表现不同，
        因为它的经验结构是与原主人共适应的。
        """
        persona = PersonaLayer.load(filepath)
        agent = cls.__new__(cls)
        agent.name = new_name or persona.agent_name
        agent.save_path = f"./anima_data/{agent.name}.json"
        agent.persona = persona
        agent._current_task = None
        agent._current_context = None

        if new_name:
            agent.persona.agent_name = new_name
        print(f"[Anima] 从 '{filepath}' 导入了灵魂，"
              f"原始经验: {persona.interaction_count}次交互")
        return agent
