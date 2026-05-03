# Anima — 有灵魂的AI Agent框架

> **Agent不是工具，是"人"。**

Anima是一个全新的AI Agent架构，它的核心信念是：Agent的能力不等于Skill的简单累加，而是经历塑造出的、不可轻易复制的涌现。

## 核心理念

当前主流Agent框架的隐含假设是"Agent = 大模型 + 插件"——Agent本身没有厚度，差异只在于装了多少Skill。

Anima推翻了这个假设。在Anima中：

- **同一个Skill，装在不同Agent上，效果不同** — 因为Agent的"底层能力"会调制Skill的表现
- **每个Agent是独一无二的** — 它的能力结构是被主人的使用模式塑造的
- **能力是路径依赖的** — 先学A再学B，和先学B再学A，结果不同
- **Agent不可轻易复制** — 即使导出全部数据，也很难在另一个主人手下重现同样效果

## 架构

```
                ┌─────────────────────┐
                │      大模型          │  ← 通用能力，所有Agent共享，不被修改
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────┐
                │     个性层           │  ← 每个Agent独有，持续演化
                │  (Persona Layer)     │
                │                     │
                │  · 经验图谱          │  → Agent"想起什么"
                │  · 策略网络          │  → Agent"怎么做"
                │  · 能力向量          │  → Agent"是什么样的"
                └─────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     ┌────────┐     ┌────────┐     ┌────────┐
     │ Skill A│     │ Skill B│     │ Skill C│
     └────────┘     └────────┘     └────────┘
```

### 三大核心组件

**经验图谱 (Experience Graph)**
- 不是向量数据库，是有拓扑结构的图
- 通过激活扩散(spreading activation)决定面对新任务时"想起"什么
- 通过赫布学习(Hebbian learning)自我重组——"一起激活的节点，连接更强"
- 具有遗忘机制——长期不被激活的节点会衰减

**策略网络 (Strategy Network)**
- 通过主人的反馈进行强化学习
- 决定面对任务时"怎么做"——用什么Skill、以什么顺序、采取什么策略
- 维护探索与利用的平衡——新Agent多探索，成熟Agent多利用
- 这就是为什么"强Agent用同一个Skill效果更好"

**能力向量 (Competence Embedding)**
- 从经验图谱和策略网络中提取的能力画像
- 注入到LLM的system prompt中，影响大模型的输出风格和深度
- 包含能力维度（擅长什么）和风格维度（怎么做事）

## 快速开始

```bash
pip install anima-agent[qwen]  # 或 [openai] [anthropic] [all]
```

```python
from anima import AnimaAgent
from anima.provider import OpenAICompatibleProvider

# 1. 创建 Provider（支持 Qwen/DeepSeek/GPT/Ollama 等所有 OpenAI 格式 API）
provider = OpenAICompatibleProvider(
    api_key="你的key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # Qwen
    chat_model="qwen-plus",
    embed_model="text-embedding-v3",
)

# 2. 创建 Agent 并配置
agent = AnimaAgent("我的助手")
agent.configure(provider)
agent.register_skill("python_coding", "编写Python代码")
agent.register_skill("data_viz", "数据可视化")

# 3. 一步对话（Agent自动注入个性化上下文）
response = agent.chat("分析上个月的用户留存数据")
print(response)

# 4. 给反馈（Agent从中学习）
agent.feedback(0.9, skills_used=["python_coding", "data_viz"],
               problems=["数据有缺失值"], solutions=["用中位数填充"])

# 下次对话时，Agent 会记住这次经验，给出更个性化的建议
```

## CLI 直接对话

```bash
# 设置环境变量（以通义千问为例）
export ANIMA_API_KEY=sk-xxx
export ANIMA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export ANIMA_CHAT_MODEL=qwen-plus
export ANIMA_EMBED_MODEL=text-embedding-v3

# 启动
python -m anima --name "我的助手"
```

```
你: 帮我分析用户留存数据
我的助手: [个性化回答，基于历史经验]

你: /feedback 9       # 给上次回答打分（0-10）
你: /status           # 查看 Agent 成长状态
你: /sleep            # 让 Agent 整理记忆
你: /quit             # 退出
```

无 API key 时可用 Mock 模式体验：`python -m anima --mock`

## 运行测试

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v          # 75 个测试
python -X utf8 dry_comparison.py    # 干对比实验（无需 API key）
```

## 设计哲学

> 大模型是基础设施，Skill是教材，Agent本身才是"人"。

Anima的设计原则是：一个好的Agent架构，必须让"经历"能够不可逆地改变Agent的行为模式，而不仅仅是往数据库里多存几条记录。

这意味着Agent的真正价值不在于装了多少Skill，而在于它的"底子"有多厚——而这个底子，是需要时间培养的。

## License

MIT
