"""
Anima - 有灵魂的AI Agent框架

让Agent不再是工具，而是"人"。

核心组件：
- ExperienceGraph: 经验图谱，Agent的结构化记忆
- StrategyNetwork: 策略网络，Agent的行为模式
- CompetenceEmbedding: 能力向量，Agent的身份画像
- PersonaLayer: 个性层，整合以上三者
- AnimaAgent: 主Agent类，用户交互入口
- EmbeddingProvider: 语义向量接口
- ExperienceExtractor: 经验结构化提取接口
- LLMProvider: 统一LLM接口（chat + embedding + extraction）
- OpenAICompatibleProvider: 兼容Qwen/DeepSeek/GPT/Ollama等所有OpenAI格式API
"""

from .experience_graph import ExperienceGraph, NodeType, EdgeType
from .strategy import StrategyNetwork, TaskCategory, ActionType
from .competence import CompetenceEmbedding
from .persona import PersonaLayer
from .agent import AnimaAgent
from .embedding import EmbeddingProvider, MockEmbeddingProvider, cosine_similarity
from .extractor import ExperienceExtractor, MockExtractor, ExtractionResult
from .narrator import narrate_subgraph
from .provider import LLMProvider

try:
    from .provider import OpenAICompatibleProvider
except ImportError:
    pass

try:
    from .embedding import AnthropicEmbeddingProvider
except ImportError:
    pass

try:
    from .extractor import AnthropicExtractor
except ImportError:
    pass

__version__ = "0.2.0"
__all__ = [
    "ExperienceGraph", "NodeType", "EdgeType",
    "StrategyNetwork", "TaskCategory", "ActionType",
    "CompetenceEmbedding",
    "PersonaLayer",
    "AnimaAgent",
    "EmbeddingProvider", "MockEmbeddingProvider", "cosine_similarity",
    "ExperienceExtractor", "MockExtractor", "ExtractionResult",
    "narrate_subgraph",
    "LLMProvider", "OpenAICompatibleProvider",
]
