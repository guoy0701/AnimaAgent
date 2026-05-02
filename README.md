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

```python
from anima import AnimaAgent, TaskCategory

# 创建Agent
agent = AnimaAgent("MyAgent")

# 注册Skill
agent.register_skill("python_coding", "编写Python代码")
agent.register_skill("data_viz", "数据可视化")

# 处理任务（获取个性化上下文）
context = agent.think("分析上个月的用户数据")
# context["system_prompt_addition"] 可以注入到任何LLM的system prompt中

# 给予反馈（Agent从中学习）
agent.feedback(
    reward=0.9,  # -1到1
    actions_taken=["decompose_first", "use_skill"],
    skills_used=["python_coding", "data_viz"],
    problems=["数据有缺失值"],
    solutions=["用插值法填充"]
)

# 查看Agent状态
print(agent.status())

# 保存Agent的"灵魂"
agent.export_soul("my_agent_soul.json")
```

## 运行Demo

```bash
python demo.py
```

Demo会创建两个初始状态完全相同的Agent，让它们经历不同的"人生"，
然后观察面对同一个任务时的行为差异。

## 设计哲学

> 大模型是基础设施，Skill是教材，Agent本身才是"人"。

Anima的设计原则是：一个好的Agent架构，必须让"经历"能够不可逆地改变Agent的行为模式，而不仅仅是往数据库里多存几条记录。

这意味着Agent的真正价值不在于装了多少Skill，而在于它的"底子"有多厚——而这个底子，是需要时间培养的。

## License

MIT
