"""
LLM-based concept extraction for experience integration.

Extracts structured information from natural language experience descriptions,
following a fixed schema (concepts, entities, domain, problems, solutions).
"""

import json
import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ExtractionResult:
    concepts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    domain: str = "unknown"
    problems: list[str] = field(default_factory=list)
    solutions: list[str] = field(default_factory=list)
    outcome_summary: str = ""
    related_concepts: list[str] = field(default_factory=list)


class ExperienceExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        ...


EXTRACTION_PROMPT = """从以下经验描述中提取结构化信息。严格按JSON格式输出，不要输出其他内容。

Schema:
{{
  "concepts": ["核心概念列表，2-5个"],
  "entities": ["涉及的工具/技术/产品名"],
  "domain": "领域: data_analysis|code_writing|content_creation|problem_solving|communication|research|planning|creative|unknown",
  "problems": ["遇到的问题"],
  "solutions": ["解决方案"],
  "outcome_summary": "一句话总结结果",
  "related_concepts": ["可能相关但未直接提及的概念"]
}}

经验描述：
{text}

JSON输出："""


class MockExtractor(ExperienceExtractor):
    """Deterministic mock for testing. Extracts based on jieba + keyword rules."""

    def extract(self, text: str) -> ExtractionResult:
        import jieba
        words = list(jieba.cut(text))

        domain_keywords = {
            "data_analysis": ["数据", "分析", "统计", "留存", "指标", "趋势"],
            "code_writing": ["代码", "编程", "函数", "脚本", "开发", "编写", "Python"],
            "content_creation": ["写", "文章", "文案", "内容", "报告"],
            "problem_solving": ["问题", "解决", "修复", "错误", "bug"],
            "research": ["调研", "研究", "搜索", "查找"],
        }

        domain = "unknown"
        best_score = 0
        for d, kws in domain_keywords.items():
            score = sum(1 for kw in kws if kw in text)
            if score > best_score:
                best_score = score
                domain = d

        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不",
                      "帮", "请", "想", "能", "把", "让", "给", "用", "做",
                      "一", "一个", "上", "也", "很", "到", "要", "去",
                      "看", "好", "这", "那", "都", "来", "过", "下"}
        concepts = [w for w in words
                    if len(w) >= 2 and w not in stop_words][:5]

        return ExtractionResult(
            concepts=concepts,
            entities=[],
            domain=domain,
            problems=[],
            solutions=[],
            outcome_summary=text[:50],
            related_concepts=[],
        )


class AnthropicExtractor(ExperienceExtractor):
    """Uses Claude for structured extraction."""

    def __init__(self, api_key: str = None,
                 model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "pip install 'anima-agent[semantic]' for LLM extraction")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def extract(self, text: str) -> ExtractionResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(text=text),
            }],
        )
        raw = response.content[0].text.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return ExtractionResult(outcome_summary=text[:100])

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ExtractionResult(outcome_summary=text[:100])

        return ExtractionResult(
            concepts=data.get("concepts", []),
            entities=data.get("entities", []),
            domain=data.get("domain", "unknown"),
            problems=data.get("problems", []),
            solutions=data.get("solutions", []),
            outcome_summary=data.get("outcome_summary", ""),
            related_concepts=data.get("related_concepts", []),
        )
