"""
AnimaAgent CLI — 有灵魂的AI助手

Usage:
    python -m anima                          # 使用默认 Agent 名称
    python -m anima --name "我的助手"          # 指定 Agent 名称
    python -m anima --name "助手" --mock      # Mock 模式（无需 API key，用于测试）

环境变量配置 Provider：
    ANIMA_API_KEY       API 密钥（必需，除非 --mock）
    ANIMA_BASE_URL      API 地址（可选，默认 OpenAI）
    ANIMA_CHAT_MODEL    聊天模型（默认 gpt-4o-mini）
    ANIMA_EMBED_MODEL   Embedding 模型（默认 text-embedding-3-small）
    ANIMA_AGENT_NAME    Agent 名称（默认 Anima）

常用配置示例：
    # 通义千问
    ANIMA_API_KEY=sk-xxx
    ANIMA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    ANIMA_CHAT_MODEL=qwen-plus
    ANIMA_EMBED_MODEL=text-embedding-v3

    # DeepSeek
    ANIMA_API_KEY=sk-xxx
    ANIMA_BASE_URL=https://api.deepseek.com
    ANIMA_CHAT_MODEL=deepseek-chat
    ANIMA_EMBED_MODEL=deepseek-chat
"""

import os
import sys
import argparse


def create_provider(args):
    """根据环境变量或参数创建 Provider。"""
    if args.mock:
        from .embedding import MockEmbeddingProvider
        from .extractor import MockExtractor
        return None, MockEmbeddingProvider(dimensions=64), MockExtractor()

    api_key = os.environ.get("ANIMA_API_KEY", "")
    if not api_key:
        print("错误：未设置 ANIMA_API_KEY 环境变量")
        print("请设置后重试，或使用 --mock 模式测试")
        print("\n示例：")
        print("  export ANIMA_API_KEY=sk-xxx          # Linux/Mac")
        print("  set ANIMA_API_KEY=sk-xxx             # Windows")
        print("  python -m anima --mock               # Mock 模式")
        sys.exit(1)

    from .provider import OpenAICompatibleProvider

    base_url = os.environ.get("ANIMA_BASE_URL")
    chat_model = os.environ.get("ANIMA_CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.environ.get("ANIMA_EMBED_MODEL", "text-embedding-3-small")

    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embed_model=embed_model,
    )
    return provider, None, None


def print_help():
    print("""
可用命令：
  /feedback <分数>   给上一次回答打分（0-10，如 /feedback 8）
  /status           查看 Agent 当前状态
  /sleep            让 Agent 整理记忆（遗忘 + 整合）
  /help             显示此帮助
  /quit             退出

直接输入文字即可对话。Agent 会随着你的使用越来越了解你。
""")


def handle_feedback(agent, args_str):
    """处理 /feedback 命令。"""
    try:
        score = float(args_str.strip())
        if score < 0 or score > 10:
            print("分数范围 0-10")
            return
        reward = score / 10.0 * 2 - 1  # 0-10 映射到 -1 到 1
        agent.feedback(reward)
    except ValueError:
        print("用法：/feedback <0-10>   例如 /feedback 8")


def handle_status(agent):
    """处理 /status 命令。"""
    status = agent.status()
    print(f"\n{'─' * 40}")
    print(f"  Agent: {status['agent_name']}")
    print(f"  交互次数: {status['interactions']}")
    print(f"  图谱节点: {status['graph_stats']['total_nodes']}")
    print(f"  图谱连接: {status['graph_stats']['total_edges']}")
    comp = status['competence']
    if comp['domain_tags']:
        print(f"  擅长领域: {', '.join(comp['domain_tags'])}")
    print(f"  画像置信度: {comp['confidence']:.0%}")
    skills = status.get('skills', {})
    if skills:
        print(f"  已注册 Skill: {', '.join(skills.keys())}")
    print(f"{'─' * 40}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AnimaAgent — 有灵魂的AI助手")
    parser.add_argument("--name", default=os.environ.get("ANIMA_AGENT_NAME", "Anima"),
                        help="Agent 名称（默认 Anima）")
    parser.add_argument("--mock", action="store_true",
                        help="Mock 模式，无需 API key")
    args = parser.parse_args()

    provider, mock_emb, mock_ext = create_provider(args)

    from . import AnimaAgent
    agent = AnimaAgent(args.name)

    if provider:
        agent.configure(provider)
    elif mock_emb and mock_ext:
        agent.configure(embedding_provider=mock_emb, extractor=mock_ext)

    print(f"\n{'═' * 50}")
    print(f"  Anima — 有灵魂的AI助手")
    print(f"  Agent: {args.name}")
    if args.mock:
        print(f"  模式: Mock（无 LLM 调用）")
    else:
        chat_model = os.environ.get("ANIMA_CHAT_MODEL", "gpt-4o-mini")
        print(f"  模型: {chat_model}")
    print(f"  输入 /help 查看命令")
    print(f"{'═' * 50}\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("再见！")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/feedback":
                handle_feedback(agent, cmd_args)
            elif cmd == "/status":
                handle_status(agent)
            elif cmd == "/sleep":
                agent.sleep()
            else:
                print(f"未知命令: {cmd}（输入 /help 查看帮助）")
            continue

        # 正常对话
        if provider:
            try:
                response = agent.chat(user_input)
                print(f"\n{args.name}: {response}\n")
            except Exception as e:
                print(f"\n调用失败: {e}\n")
        else:
            # Mock 模式：用 think() + 模拟输出
            context = agent.think(user_input)
            strategy = context.get("strategy", {})
            print(f"\n{args.name} [Mock 模式]:")
            print(f"  任务分类: {context.get('task_category', '?')}")
            print(f"  决策模式: {strategy.get('mode', '?')}")
            print(f"  建议动作: {', '.join(strategy.get('actions', []))}")
            print(f"  建议技能: {', '.join(strategy.get('skills', ['无']))}")
            activated = context.get("activated_experiences", [])
            if activated:
                print(f"  激活经验:")
                for content, score in activated[:3]:
                    print(f"    {score:.2f} | {content[:50]}")
            print()


if __name__ == "__main__":
    main()
