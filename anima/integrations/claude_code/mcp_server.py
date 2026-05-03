"""
AnimaAgent MCP Server — 让 Claude Code 获得"记忆和个性化"能力。

启动方式：
    python -m anima.integrations.claude_code.mcp_server

Claude Code 配置（.claude/settings.json）：
    {
      "mcpServers": {
        "anima": {
          "command": "python",
          "args": ["-m", "anima.integrations.claude_code.mcp_server"],
          "env": {
            "ANIMA_AGENT_NAME": "我的助手",
            "ANIMA_API_KEY": "sk-xxx",
            "ANIMA_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "ANIMA_CHAT_MODEL": "qwen-plus",
            "ANIMA_EMBED_MODEL": "text-embedding-v3"
          }
        }
      }
    }
"""

import os
import sys
import json
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from anima.dotenv_loader import load_dotenv
load_dotenv()

mcp = FastMCP("AnimaAgent")

_agent = None


def _get_agent():
    """延迟初始化 Agent（首次调用时创建）。"""
    global _agent
    if _agent is not None:
        return _agent

    from anima import AnimaAgent

    name = os.environ.get("ANIMA_AGENT_NAME", "Anima")
    _agent = AnimaAgent(name)

    api_key = os.environ.get("ANIMA_API_KEY", "")
    if api_key:
        from anima.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=os.environ.get("ANIMA_BASE_URL"),
            chat_model=os.environ.get("ANIMA_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=os.environ.get("ANIMA_EMBED_MODEL", "text-embedding-3-small"),
        )
        _agent.configure(provider)
    else:
        from anima.embedding import MockEmbeddingProvider
        from anima.extractor import MockExtractor
        _agent.configure(
            embedding_provider=MockEmbeddingProvider(dimensions=64),
            extractor=MockExtractor(),
        )

    return _agent


@mcp.tool()
def anima_think(task: str) -> str:
    """获取 AnimaAgent 的个性化上下文。在开始处理用户任务前调用此工具，
    获取基于 Agent 历史经验的个性化建议（相关经验、策略建议、能力画像）。
    将返回的内容作为你回答时的参考背景。"""
    try:
        agent = _get_agent()
        context = agent.think(task)
        result = {
            "system_prompt_addition": context.get("system_prompt_addition", ""),
            "task_category": context.get("task_category", "unknown"),
            "strategy": {
                "mode": context.get("strategy", {}).get("mode", ""),
                "actions": context.get("strategy", {}).get("actions", []),
                "skills": context.get("strategy", {}).get("skills", []),
                "confidence": context.get("strategy", {}).get("confidence", 0),
            },
            "activated_experiences": context.get("activated_experiences", []),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": "anima_think failed. Check server logs."}, ensure_ascii=False)


@mcp.tool()
def anima_feedback(reward: float, skills_used: str = "",
                   problems: str = "", solutions: str = "") -> str:
    """记录用户对上一次回答的反馈，让 Agent 从中学习。
    reward: 满意度，0.0（非常不满意）到 1.0（非常满意）。
    skills_used: 使用的技能，逗号分隔（如 "python_coding,sql_query"）。
    problems: 遇到的问题，逗号分隔。
    solutions: 解决方案，逗号分隔。"""
    try:
        agent = _get_agent()
        if agent._current_task is None:
            return "没有待反馈的任务。请先使用 anima_think 处理一个任务。"

        skills = [s.strip() for s in skills_used.split(",") if s.strip()] if skills_used else None
        probs = [p.strip() for p in problems.split(",") if p.strip()] if problems else None
        sols = [s.strip() for s in solutions.split(",") if s.strip()] if solutions else None

        mapped_reward = reward * 2 - 1  # 0-1 映射到 -1 到 1

        agent.feedback(mapped_reward, skills_used=skills,
                       problems=probs, solutions=sols)

        return f"已记录反馈（满意度 {reward:.0%}），Agent 已学习更新。"
    except Exception as e:
        return json.dumps({"error": "anima_feedback failed. Check server logs."}, ensure_ascii=False)


@mcp.tool()
def anima_status() -> str:
    """查看 AnimaAgent 的当前状态，包括交互次数、能力画像、图谱统计等。"""
    try:
        agent = _get_agent()
        status = agent.status()
        summary = {
            "agent_name": status["agent_name"],
            "interactions": status["interactions"],
            "graph": {
                "nodes": status["graph_stats"]["total_nodes"],
                "edges": status["graph_stats"]["total_edges"],
            },
            "competence": {
                "domain_tags": status["competence"]["domain_tags"],
                "confidence": f"{status['competence']['confidence']:.0%}",
            },
            "skills": list(status.get("skills", {}).keys()),
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": "anima_status failed. Check server logs."}, ensure_ascii=False)


@mcp.tool()
def anima_register_skill(name: str, description: str) -> str:
    """为 Agent 注册一个新技能。注册后，Agent 在处理相关任务时会考虑使用此技能。"""
    try:
        agent = _get_agent()
        agent.register_skill(name, description)
        return f"已注册技能: {name}"
    except Exception as e:
        return json.dumps({"error": "anima_register_skill failed. Check server logs."}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
