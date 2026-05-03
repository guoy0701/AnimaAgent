#!/usr/bin/env python3
"""
Qwen Quick Start — 用通义千问 API 跑一个有记忆的 Agent。

使用前：
    pip install anima-agent[qwen]
    export DASHSCOPE_API_KEY=你的key

或者直接在代码里传 api_key。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anima import AnimaAgent
from anima.provider import OpenAICompatibleProvider


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("请设置环境变量 DASHSCOPE_API_KEY")
        print("获取方式：https://dashscope.console.aliyun.com/")
        return

    # 1. 创建 Provider（通义千问兼容 OpenAI 格式）
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        chat_model="qwen-plus",
        embed_model="text-embedding-v3",
        embed_dimensions=1024,
    )

    # 2. 创建 Agent 并配置
    agent = AnimaAgent("我的助手")
    agent.configure(provider)
    agent.register_skill("python_coding", "编写Python代码")
    agent.register_skill("data_analysis", "数据分析")

    # 3. 第一次对话
    print("\n--- 第一次对话 ---")
    response = agent.chat("帮我写一个Python函数，计算列表的移动平均值")
    print(f"助手: {response}")
    agent.feedback(0.9, skills_used=["python_coding"],
                   problems=["需要处理窗口边界"], solutions=["用min处理边界"])

    # 4. 第二次对话（Agent 已经有了第一次的经验）
    print("\n--- 第二次对话 ---")
    response = agent.chat("帮我写一个函数，计算指数移动平均值")
    print(f"助手: {response}")
    agent.feedback(0.8, skills_used=["python_coding"])

    # 5. 查看 Agent 状态
    print("\n--- Agent 状态 ---")
    status = agent.status()
    print(f"交互次数: {status['interactions']}")
    print(f"图谱节点: {status['graph_stats']['total_nodes']}")
    print(f"能力标签: {status['competence']['domain_tags']}")

    print("\n第二次对话时，Agent 已经记住了第一次的经验。")
    print("随着使用越来越多，Agent 会越来越了解你的编程风格和偏好。")


if __name__ == "__main__":
    main()
