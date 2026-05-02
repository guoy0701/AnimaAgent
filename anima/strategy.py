"""
策略网络 (Strategy Network)

核心理念：Agent面对任务时"怎么做"的决策，不应该完全依赖大模型的临场判断，
而应该有一个专门的、可学习的决策层。

这个网络通过主人的反馈来训练：
- 满意 → 强化当前策略
- 不满意 → 削弱当前策略
- 主人亲自修改 → 学习主人的偏好

时间长了，策略网络就学会了"在我主人的场景下，面对这类问题，最好的策略是什么"。
这就是为什么"强Agent用同一个Skill效果更好"——策略网络决定了Skill被如何调用。
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ActionType(Enum):
    """Agent可以采取的策略动作类型"""
    ASK_CLARIFICATION = "ask_clarification"     # 先问清楚再做
    DIRECT_EXECUTION = "direct_execution"       # 直接动手做
    DECOMPOSE_FIRST = "decompose_first"         # 先分解任务
    SEARCH_FIRST = "search_first"               # 先搜索相关信息
    USE_SKILL = "use_skill"                     # 调用特定Skill
    COMBINE_SKILLS = "combine_skills"           # 组合多个Skill
    ITERATE_AND_REFINE = "iterate_and_refine"   # 先出草稿再迭代
    CONSULT_EXPERIENCE = "consult_experience"   # 先查经验图谱


class TaskCategory(Enum):
    """任务的大类，用于策略索引"""
    DATA_ANALYSIS = "data_analysis"
    CODE_WRITING = "code_writing"
    CONTENT_CREATION = "content_creation"
    PROBLEM_SOLVING = "problem_solving"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    PLANNING = "planning"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


@dataclass
class StrategyRecord:
    """一条策略记录：在什么情境下，选了什么策略，效果如何"""
    task_category: TaskCategory
    context_features: dict           # 任务的特征向量
    actions_taken: list[ActionType]  # 采取的策略序列
    skills_used: list[str]           # 使用的Skill列表
    reward: float = 0.0              # 主人的反馈 (-1到1)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StrategyProfile:
    """
    某个任务类别下的策略偏好画像。

    这不是一个神经网络，而是一个更直观的"偏好分布"——
    记录了在该类别下，各种策略和Skill组合的历史成功率。
    这个设计让策略的形成过程可解释、可调试。
    """
    category: TaskCategory
    # 每种动作类型的偏好分数（通过反馈累积）
    action_preferences: dict[str, float] = field(default_factory=dict)
    # 每种动作类型的尝试次数（用于探索时优先选最少尝试的）
    action_attempt_counts: dict[str, int] = field(default_factory=dict)
    # Skill的偏好分数
    skill_preferences: dict[str, float] = field(default_factory=dict)
    # 策略序列模式的偏好（记录成功的动作组合）
    sequence_patterns: list[dict] = field(default_factory=list)
    # 总尝试次数
    total_attempts: int = 0
    # 成功次数（reward > 0.5）
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_attempts, 1)


class StrategyNetwork:
    """
    策略网络的核心实现。

    不是传统意义上的神经网络，而是一个基于经验统计的策略决策系统。
    它维护每个任务类别下的策略偏好画像，通过主人的反馈持续更新。

    关键设计：
    1. 探索与利用(Exploration vs Exploitation) - 新Agent更多探索，
       成熟Agent更多利用已知的好策略
    2. 上下文敏感 - 同一类任务，不同的上下文特征可能导向不同策略
    3. 组合学习 - 学习哪些Skill组合在一起效果好
    """

    def __init__(self, exploration_rate: float = 0.3):
        self.profiles: dict[str, StrategyProfile] = {}
        self.history: list[StrategyRecord] = []
        self.exploration_rate = exploration_rate  # 会随经验积累而降低
        self._total_decisions = 0

        # 初始化所有类别的画像
        for cat in TaskCategory:
            self.profiles[cat.value] = StrategyProfile(category=cat)

    def decide_strategy(self, task_category: TaskCategory,
                        context: dict,
                        available_skills: list[str]) -> dict:
        """
        为一个新任务决定策略。

        返回一个策略计划，包括：
        - 建议的动作序列
        - 建议使用的Skill
        - 决策的置信度
        - 决策依据
        """
        profile = self.profiles[task_category.value]
        self._total_decisions += 1

        # 动态调整探索率：经验越多，探索越少
        effective_exploration = self.exploration_rate * math.exp(
            -0.01 * profile.total_attempts)
        effective_exploration = max(0.05, effective_exploration)  # 最低5%探索

        # 决定是探索还是利用
        if random.random() < effective_exploration:
            strategy = self._explore(task_category, available_skills, context)
            strategy["mode"] = "exploration"
        else:
            strategy = self._exploit(task_category, available_skills, context)
            strategy["mode"] = "exploitation"

        strategy["confidence"] = self._calculate_confidence(profile)
        strategy["category"] = task_category.value

        return strategy

    def _explore(self, category: TaskCategory,
                 available_skills: list[str], context: dict) -> dict:
        """探索模式：尝试不太常用的策略组合"""
        profile = self.profiles[category.value]

        # 找出尝试次数较少的动作类型（按 attempt_counts 排序，而非偏好分数）
        all_actions = list(ActionType)

        # 倾向于选择尝试次数最少的动作
        actions = sorted(all_actions,
                         key=lambda a: profile.action_attempt_counts.get(a.value, 0))
        selected_actions = actions[:2]  # 选两个最少尝试的

        # 随机选择Skill组合
        selected_skills = []
        if available_skills:
            n_skills = min(random.randint(1, 3), len(available_skills))
            selected_skills = random.sample(available_skills, n_skills)

        return {
            "actions": [a.value for a in selected_actions],
            "skills": selected_skills,
            "reasoning": "探索模式：尝试不太常用的策略以发现更好的方法"
        }

    def _exploit(self, category: TaskCategory,
                 available_skills: list[str], context: dict) -> dict:
        """利用模式：使用历史上效果最好的策略"""
        profile = self.profiles[category.value]

        # 按偏好分数排序选择动作
        sorted_actions = sorted(
            profile.action_preferences.items(),
            key=lambda x: x[1], reverse=True
        )

        if sorted_actions:
            selected_actions = [a for a, _ in sorted_actions[:3] if _ > 0]
        else:
            selected_actions = [ActionType.DIRECT_EXECUTION.value]

        # 选择历史效果好的Skill
        sorted_skills = sorted(
            profile.skill_preferences.items(),
            key=lambda x: x[1], reverse=True
        )
        selected_skills = [s for s, score in sorted_skills
                           if s in available_skills and score > 0][:3]

        if not selected_skills and available_skills:
            selected_skills = available_skills[:1]

        # 查找成功的序列模式
        reasoning = "利用模式：基于历史经验选择最佳策略"
        if profile.sequence_patterns:
            best_pattern = max(profile.sequence_patterns,
                               key=lambda p: p.get("score", 0))
            if best_pattern.get("score", 0) > 0.5:
                reasoning += f"\n参考成功模式：{best_pattern.get('description', '')}"

        return {
            "actions": selected_actions if selected_actions
                       else [ActionType.DIRECT_EXECUTION.value],
            "skills": selected_skills,
            "reasoning": reasoning,
        }

    def _calculate_confidence(self, profile: StrategyProfile) -> float:
        """计算策略决策的置信度"""
        if profile.total_attempts == 0:
            return 0.1  # 完全没有经验
        experience_factor = min(1.0, profile.total_attempts / 20)
        success_factor = profile.success_rate
        return 0.1 + 0.9 * experience_factor * (0.3 + 0.7 * success_factor)

    def learn_from_feedback(self, task_category: TaskCategory,
                            actions_taken: list[str],
                            skills_used: list[str],
                            reward: float,
                            context: dict = None):
        """
        从主人的反馈中学习。

        reward: -1.0（非常不满意）到 1.0（非常满意）
        正向反馈强化当前策略，负向反馈削弱。
        """
        profile = self.profiles[task_category.value]
        profile.total_attempts += 1
        if reward > 0.5:
            profile.success_count += 1

        # EMA alpha：防止分数无界增长（代替原来的累加方式）
        ema_alpha = 0.2

        # 更新动作偏好（EMA）并记录尝试次数
        for action in actions_taken:
            current = profile.action_preferences.get(action, 0.0)
            profile.action_preferences[action] = ema_alpha * reward + (1 - ema_alpha) * current
            profile.action_attempt_counts[action] = profile.action_attempt_counts.get(action, 0) + 1

        # 更新Skill偏好（EMA）
        for skill in skills_used:
            current = profile.skill_preferences.get(skill, 0.0)
            profile.skill_preferences[skill] = ema_alpha * reward + (1 - ema_alpha) * current

        # 记录策略序列模式
        if reward > 0.5:
            pattern = {
                "actions": actions_taken,
                "skills": skills_used,
                "score": reward,
                "description": f"使用 {', '.join(actions_taken)} + "
                               f"{', '.join(skills_used) if skills_used else '无特定Skill'}",
                "timestamp": time.time(),
            }
            profile.sequence_patterns.append(pattern)
            # 只保留最好的20个模式
            profile.sequence_patterns.sort(
                key=lambda p: p["score"], reverse=True)
            profile.sequence_patterns = profile.sequence_patterns[:20]

        # 记录历史
        self.history.append(StrategyRecord(
            task_category=task_category,
            context_features=context or {},
            actions_taken=[ActionType(a) for a in actions_taken
                           if a in [at.value for at in ActionType]],
            skills_used=skills_used,
            reward=reward,
        ))

    def get_profile_summary(self, category: TaskCategory = None) -> dict:
        """获取策略画像摘要"""
        if category:
            profiles = {category.value: self.profiles[category.value]}
        else:
            profiles = self.profiles

        summary = {}
        for cat_name, profile in profiles.items():
            if profile.total_attempts == 0:
                continue
            top_actions = sorted(profile.action_preferences.items(),
                                 key=lambda x: x[1], reverse=True)[:3]
            top_skills = sorted(profile.skill_preferences.items(),
                                key=lambda x: x[1], reverse=True)[:3]
            summary[cat_name] = {
                "attempts": profile.total_attempts,
                "success_rate": round(profile.success_rate, 2),
                "top_actions": [(a, round(s, 2)) for a, s in top_actions],
                "top_skills": [(s, round(sc, 2)) for s, sc in top_skills],
                "known_patterns": len(profile.sequence_patterns),
            }
        return summary

    def generate_strategy_prompt(self, task_category: TaskCategory) -> str:
        """
        将策略偏好转化为文本提示，注入到LLM的上下文中。
        这是策略网络与大模型之间的接口。
        """
        profile = self.profiles[task_category.value]

        if profile.total_attempts == 0:
            return "这是一个新的任务类型，没有历史策略经验。请根据任务本身灵活决策。"

        lines = [f"基于历史经验（{profile.total_attempts}次，"
                 f"成功率{profile.success_rate:.0%}），以下是策略建议：\n"]

        top_actions = sorted(profile.action_preferences.items(),
                             key=lambda x: x[1], reverse=True)[:3]
        if top_actions:
            lines.append("推荐的做事方式：")
            for action, score in top_actions:
                if score > 0:
                    lines.append(f"  - {action}（置信度：{min(score, 5):.1f}/5）")

        top_skills = sorted(profile.skill_preferences.items(),
                            key=lambda x: x[1], reverse=True)[:3]
        if top_skills:
            lines.append("推荐优先使用的Skill：")
            for skill, score in top_skills:
                if score > 0:
                    lines.append(f"  - {skill}（效果：{min(score, 5):.1f}/5）")

        if profile.sequence_patterns:
            best = profile.sequence_patterns[0]
            lines.append(f"\n最成功的策略模式：{best.get('description', '')}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "exploration_rate": self.exploration_rate,
            "total_decisions": self._total_decisions,
            "profiles": {
                cat: {
                    "category": p.category.value,
                    "action_preferences": p.action_preferences,
                    "action_attempt_counts": p.action_attempt_counts,
                    "skill_preferences": p.skill_preferences,
                    "sequence_patterns": p.sequence_patterns,
                    "total_attempts": p.total_attempts,
                    "success_count": p.success_count,
                } for cat, p in self.profiles.items()
            },
            "history": [{
                "task_category": r.task_category.value,
                "context_features": r.context_features,
                "actions_taken": [a.value for a in r.actions_taken],
                "skills_used": r.skills_used,
                "reward": r.reward,
                "timestamp": r.timestamp,
            } for r in self.history[-100:]],  # 只保留最近100条
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyNetwork":
        """反序列化"""
        net = cls(exploration_rate=data.get("exploration_rate", 0.3))
        net._total_decisions = data.get("total_decisions", 0)

        for cat, pd in data.get("profiles", {}).items():
            if cat in net.profiles:
                profile = net.profiles[cat]
                profile.action_preferences = pd.get("action_preferences", {})
                profile.action_attempt_counts = pd.get("action_attempt_counts", {})
                profile.skill_preferences = pd.get("skill_preferences", {})
                profile.sequence_patterns = pd.get("sequence_patterns", [])
                profile.total_attempts = pd.get("total_attempts", 0)
                profile.success_count = pd.get("success_count", 0)

        # 恢复历史记录
        for record_data in data.get("history", []):
            try:
                record = StrategyRecord(
                    task_category=TaskCategory(record_data["task_category"]),
                    context_features=record_data.get("context_features", {}),
                    actions_taken=[ActionType(a) for a in record_data.get("actions_taken", [])],
                    skills_used=record_data.get("skills_used", []),
                    reward=record_data.get("reward", 0),
                    timestamp=record_data.get("timestamp", 0),
                )
                net.history.append(record)
            except (ValueError, KeyError):
                continue

        return net
