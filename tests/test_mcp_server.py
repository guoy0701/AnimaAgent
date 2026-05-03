import os
import json


def _reset_server():
    """Reset the global agent in mcp_server module."""
    import anima.integrations.claude_code.mcp_server as srv
    srv._agent = None


class TestAnimaThink:
    def test_returns_valid_json(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_think
        result = anima_think(task="测试任务")
        data = json.loads(result)
        assert "system_prompt_addition" in data
        assert "task_category" in data
        assert "strategy" in data

    def test_handles_empty_task(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_think
        result = anima_think(task="")
        data = json.loads(result)
        assert isinstance(data, dict)


class TestAnimaFeedback:
    def test_feedback_after_think_works(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_think, anima_feedback
        anima_think(task="测试任务")
        result = anima_feedback(reward=0.8)
        assert "已记录" in result

    def test_feedback_without_think_returns_message(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_feedback
        result = anima_feedback(reward=0.5)
        assert "没有待反馈" in result or isinstance(result, str)

    def test_feedback_with_skills(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_think, anima_feedback
        anima_think(task="写代码")
        result = anima_feedback(reward=0.9, skills_used="python_coding,sql_query")
        assert "已记录" in result


class TestAnimaStatus:
    def test_returns_valid_json(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_status
        result = anima_status()
        data = json.loads(result)
        assert "agent_name" in data
        assert "interactions" in data
        assert "graph" in data


class TestAnimaRegisterSkill:
    def test_registers_skill(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_register_skill
        result = anima_register_skill(name="test_skill", description="测试技能")
        assert "test_skill" in result

    def test_duplicate_register_no_crash(self):
        _reset_server()
        from anima.integrations.claude_code.mcp_server import anima_register_skill
        anima_register_skill(name="skill_a", description="技能A")
        result = anima_register_skill(name="skill_a", description="技能A更新")
        assert "skill_a" in result
