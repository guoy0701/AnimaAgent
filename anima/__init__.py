"""
Anima - 有灵魂的AI Agent框架

让Agent不再是工具，而是"人"。

核心组件：
- ExperienceGraph: 经验图谱，Agent的结构化记忆
- StrategyNetwork: 策略网络，Agent的行为模式
- CompetenceEmbedding: 能力向量，Agent的身份画像
- PersonaLayer: 个性层，整合以上三者
- AnimaAgent: 主Agent类，用户交互入口
"""

from .experience_graph import ExperienceGraph, NodeType, EdgeType
from .strategy import StrategyNetwork, TaskCategory, ActionType
from .competence import CompetenceEmbedding
from .persona import PersonaLayer
from .agent import AnimaAgent

__version__ = "0.1.0"
__all__ = [
    "ExperienceGraph", "NodeType", "EdgeType",
    "StrategyNetwork", "TaskCategory", "ActionType",
    "CompetenceEmbedding",
    "PersonaLayer",
    "AnimaAgent",
]
