"""
能力向量 (Competence Embedding)

核心理念：把Agent当前的整体"能力状态"压缩成一个可解释的画像。
它不做决策（那是策略网络的事），它描述"这个Agent是什么样的"。

能力向量从经验图谱和策略网络的状态中提取，
被注入到大模型的推理过程中，让大模型知道自己服务的是一个什么样的Agent。

打个比方：策略网络是行为习惯，能力向量是身份画像。
"""

import math
import time
from dataclasses import dataclass, field


# 能力维度的定义
COMPETENCE_DIMENSIONS = {
    "data_analysis": "数据分析能力",
    "code_writing": "代码编写能力",
    "content_creation": "内容创作能力",
    "problem_solving": "问题解决能力",
    "communication": "沟通表达能力",
    "research": "调研分析能力",
    "planning": "规划能力",
    "creative": "创意思维能力",
    "domain_depth": "领域深度",
    "adaptability": "适应性",
    "reliability": "可靠性",
    "initiative": "主动性",
}

# 风格维度的定义
STYLE_DIMENSIONS = {
    "verbosity": ("简洁", "详细"),          # -1=极简, 1=详尽
    "formality": ("随意", "正式"),           # -1=口语, 1=书面
    "proactivity": ("被动", "主动"),         # -1=只做被要求的, 1=主动扩展
    "risk_tolerance": ("保守", "大胆"),      # -1=保守安全, 1=敢于创新
    "structure": ("灵活", "结构化"),         # -1=自由发挥, 1=严格结构
}


@dataclass
class CompetenceEmbedding:
    """
    Agent的能力画像。

    由两部分组成：
    1. 能力分数(competence_scores) - 各能力维度的量化评估
    2. 风格倾向(style_tendencies) - 做事风格的偏好

    这些值不是静态配置的，而是从经验图谱和策略网络的状态中动态计算的。
    """
    # 各能力维度的分数 (0-1)
    competence_scores: dict[str, float] = field(default_factory=dict)
    # 各风格维度的倾向 (-1到1)
    style_tendencies: dict[str, float] = field(default_factory=dict)
    # 主要领域标签
    domain_tags: list[str] = field(default_factory=list)
    # 上次更新时间
    last_updated: float = field(default_factory=time.time)
    # 总经验量（影响画像的置信度）
    total_experience: int = 0

    def __post_init__(self):
        # 初始化所有维度为默认值
        for dim in COMPETENCE_DIMENSIONS:
            self.competence_scores.setdefault(dim, 0.0)
        for dim in STYLE_DIMENSIONS:
            self.style_tendencies.setdefault(dim, 0.0)

    def update_from_graph_and_strategy(self, graph_stats: dict,
                                        strategy_summary: dict):
        """
        从经验图谱和策略网络的状态中更新能力画像。
        这是能力向量"自动演化"的核心机制。
        """
        self.last_updated = time.time()

        # 从策略网络的成功率更新能力分数
        category_to_competence = {
            "data_analysis": "data_analysis",
            "code_writing": "code_writing",
            "content_creation": "content_creation",
            "problem_solving": "problem_solving",
            "communication": "communication",
            "research": "research",
            "planning": "planning",
            "creative": "creative",
        }

        total_attempts = 0
        for cat, info in strategy_summary.items():
            if cat in category_to_competence:
                comp_dim = category_to_competence[cat]
                attempts = info.get("attempts", 0)
                success_rate = info.get("success_rate", 0)
                total_attempts += attempts

                # 能力分数 = 经验量加权的成功率
                experience_weight = min(1.0, attempts / 10)
                self.competence_scores[comp_dim] = (
                    experience_weight * success_rate
                )

        self.total_experience = total_attempts

        # 从策略偏好推断风格倾向
        all_action_prefs = {}
        for cat, info in strategy_summary.items():
            for action, score in info.get("top_actions", []):
                all_action_prefs[action] = all_action_prefs.get(
                    action, 0) + score

        # 推断主动性
        if "ask_clarification" in all_action_prefs:
            ask_score = all_action_prefs["ask_clarification"]
            direct_score = all_action_prefs.get("direct_execution", 0)
            if ask_score + direct_score > 0:
                # 更倾向直接执行 → 更主动
                self.style_tendencies["proactivity"] = (
                    (direct_score - ask_score) /
                    (ask_score + direct_score + 0.01)
                )

        # 推断结构化倾向
        decompose_score = all_action_prefs.get("decompose_first", 0)
        iterate_score = all_action_prefs.get("iterate_and_refine", 0)
        if decompose_score + iterate_score > 0:
            self.style_tendencies["structure"] = (
                (decompose_score - iterate_score) /
                (decompose_score + iterate_score + 0.01)
            )

        # 从经验图谱的统计推断领域深度
        total_nodes = graph_stats.get("total_nodes", 0)
        if total_nodes > 50:
            self.competence_scores["domain_depth"] = min(
                1.0, total_nodes / 200)
        avg_strength = graph_stats.get("avg_strength", 0)
        self.competence_scores["reliability"] = min(1.0, avg_strength)

        # 更新领域标签
        self._update_domain_tags()

    def _update_domain_tags(self):
        """根据能力分数更新领域标签"""
        sorted_competences = sorted(
            self.competence_scores.items(),
            key=lambda x: x[1], reverse=True
        )
        self.domain_tags = [
            COMPETENCE_DIMENSIONS.get(dim, dim)
            for dim, score in sorted_competences
            if score > 0.3
        ][:5]

    def manual_adjust(self, dimension: str, value: float):
        """主人手动调整某个维度（覆盖自动计算）"""
        if dimension in self.competence_scores:
            self.competence_scores[dimension] = max(0, min(1, value))
        elif dimension in self.style_tendencies:
            self.style_tendencies[dimension] = max(-1, min(1, value))

    def get_confidence(self) -> float:
        """画像的整体置信度——经验越多越可信"""
        return min(1.0, self.total_experience / 50)

    def generate_identity_prompt(self) -> str:
        """
        将能力画像转化为文本提示，注入到LLM的system prompt中。
        这是能力向量与大模型之间的接口。

        大模型读到这段描述后，会自然地调整自己的输出风格和深度，
        使其符合这个特定Agent的特点。
        """
        confidence = self.get_confidence()
        if confidence < 0.1:
            return ("这是一个新的Agent，还没有积累足够的经验来形成明确的能力画像。"
                    "请根据任务需求灵活应对。")

        lines = ["## 当前Agent能力画像\n"]

        # 能力描述
        strong = [(dim, score) for dim, score in self.competence_scores.items()
                  if score > 0.5]
        developing = [(dim, score) for dim, score
                      in self.competence_scores.items()
                      if 0.2 < score <= 0.5]

        if strong:
            lines.append("**擅长领域：**")
            for dim, score in sorted(strong, key=lambda x: x[1],
                                     reverse=True):
                desc = COMPETENCE_DIMENSIONS.get(dim, dim)
                level = "精通" if score > 0.8 else "熟练"
                lines.append(f"  - {desc}（{level}，{score:.0%}）")

        if developing:
            lines.append("**成长中的领域：**")
            for dim, score in sorted(developing, key=lambda x: x[1],
                                     reverse=True):
                desc = COMPETENCE_DIMENSIONS.get(dim, dim)
                lines.append(f"  - {desc}（{score:.0%}）")

        # 风格描述
        style_desc = []
        for dim, value in self.style_tendencies.items():
            if abs(value) > 0.3:
                labels = STYLE_DIMENSIONS.get(dim, (dim, dim))
                if value < 0:
                    style_desc.append(f"偏{labels[0]}")
                else:
                    style_desc.append(f"偏{labels[1]}")

        if style_desc:
            lines.append(f"\n**做事风格：** {', '.join(style_desc)}")

        lines.append(f"\n画像置信度：{confidence:.0%}"
                     f"（基于{self.total_experience}次交互）")

        return "\n".join(lines)

    def similarity(self, other: "CompetenceEmbedding") -> float:
        """计算两个Agent的能力相似度"""
        comp_sim = 0
        for dim in COMPETENCE_DIMENSIONS:
            diff = (self.competence_scores.get(dim, 0) -
                    other.competence_scores.get(dim, 0))
            comp_sim += diff ** 2
        comp_sim = 1 - math.sqrt(comp_sim / len(COMPETENCE_DIMENSIONS))

        style_sim = 0
        for dim in STYLE_DIMENSIONS:
            diff = (self.style_tendencies.get(dim, 0) -
                    other.style_tendencies.get(dim, 0))
            style_sim += diff ** 2
        style_sim = 1 - math.sqrt(style_sim / len(STYLE_DIMENSIONS))

        return 0.6 * comp_sim + 0.4 * style_sim

    def to_dict(self) -> dict:
        return {
            "competence_scores": self.competence_scores,
            "style_tendencies": self.style_tendencies,
            "domain_tags": self.domain_tags,
            "last_updated": self.last_updated,
            "total_experience": self.total_experience,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompetenceEmbedding":
        return cls(
            competence_scores=data.get("competence_scores", {}),
            style_tendencies=data.get("style_tendencies", {}),
            domain_tags=data.get("domain_tags", []),
            last_updated=data.get("last_updated", 0),
            total_experience=data.get("total_experience", 0),
        )
