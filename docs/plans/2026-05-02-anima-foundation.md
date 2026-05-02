# AnimaAgent Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the data model bugs that make the current codebase unreliable, then add the semantic layer (embeddings + LLM extraction) that makes the core vision ("experience shapes personality") actually work.

**Architecture:** Two-phase approach. Phase 1 fixes bugs in ExperienceGraph, StrategyNetwork, and PersonaLayer without adding new dependencies (except jieba for Chinese segmentation and pytest for testing). Phase 2 adds an embedding provider interface and LLM-based concept extraction, connecting the graph to real semantic understanding. The LLM is used in two distinct roles: as a "semantic engine" inside PersonaLayer (for experience processing), and as a "task executor" outside PersonaLayer (unchanged, user-provided).

**Tech Stack:** Python 3.10+, pytest, jieba (Phase 1), anthropic SDK + numpy (Phase 2)

---

## File Structure

### Phase 1: Data Model Correctness

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `pyproject.toml` | Project config, pytest settings, dependencies |
| Create | `tests/__init__.py` | Test package |
| Create | `tests/test_experience_graph.py` | ExperienceGraph unit tests |
| Create | `tests/test_strategy.py` | StrategyNetwork unit tests |
| Create | `tests/test_persona.py` | PersonaLayer integration tests |
| Modify | `anima/experience_graph.py` | Fix edge directionality, dedup, decay |
| Modify | `anima/strategy.py` | Fix EMA learning, exploration logic, history deserialization |
| Modify | `anima/persona.py` | Fix Chinese keywords, connection discovery, experience recording |
| Modify | `anima/agent.py` | Fix double-recording in feedback() |

### Phase 2: Semantic Layer

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `anima/embedding.py` | Embedding provider interface + Anthropic implementation |
| Create | `anima/extractor.py` | LLM-based concept/relationship extraction |
| Create | `anima/narrator.py` | Subgraph → narrative text conversion |
| Create | `tests/test_embedding.py` | Embedding tests (with mock provider) |
| Create | `tests/test_extractor.py` | Extraction tests |
| Create | `tests/test_narrator.py` | Narrative tests |
| Modify | `anima/experience_graph.py` | Add embedding storage, cosine similarity search |
| Modify | `anima/strategy.py` | Read similar past tasks from graph for decision-making |
| Modify | `anima/competence.py` | Derive from graph topology instead of strategy stats |
| Modify | `anima/persona.py` | Wire extraction + embedding + narrator into main flow |
| Modify | `anima/__init__.py` | Export new public APIs |

---

## Phase 1: Data Model Correctness

> **Phase 1 has no LLM dependency.** All fixes use pure Python + jieba.
> After Phase 1, the graph structure is correct, learning is bounded, Chinese text works, and all behavior is tested.

### Task 1: Project Infrastructure

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repository**

```bash
cd /d/claude-dev/AnimaAgent
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.pytest_cache/
anima_data/
*.json
!pyproject.toml
.venv/
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "anima-agent"
version = "0.2.0"
description = "有灵魂的AI Agent框架"
requires-python = ">=3.10"
dependencies = [
    "jieba>=0.42",
]

[project.optional-dependencies]
semantic = [
    "anthropic>=0.49",
    "numpy>=1.24",
]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 4: Install dependencies**

```bash
pip install jieba pytest
```

- [ ] **Step 5: Create test package**

Create empty `tests/__init__.py`.

- [ ] **Step 6: Verify pytest runs**

```bash
pytest --co -q
```

Expected: `no tests ran` (no errors)

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml tests/__init__.py
git commit -m "chore: add project config, test infrastructure, gitignore"
```

---

### Task 2: Fix Edge Directionality

**Problem:** All edges are stored bidirectionally in the adjacency list, but `CAUSAL`, `TEMPORAL`, `SOLVED_BY`, `REQUIRES` are inherently directed. Spreading activation and subgraph extraction traverse them backwards, producing nonsensical results like "solution solved_by problem."

**Fix:** Add a `directed` property to EdgeType. Only add undirected edges to both sides of the adjacency. Directed edges only go source→target in forward adjacency, but store a separate reverse adjacency for backward lookup (needed for serialization, not for activation).

**Files:**
- Modify: `anima/experience_graph.py`
- Create: `tests/test_experience_graph.py`

- [ ] **Step 1: Write failing test for directed edge traversal**

```python
# tests/test_experience_graph.py
from anima.experience_graph import ExperienceGraph, NodeType, EdgeType


class TestEdgeDirectionality:
    def test_causal_edge_only_forward_in_activation(self):
        """CAUSAL edges should only propagate activation forward (source→target)."""
        g = ExperienceGraph()
        problem = g.add_node(NodeType.PROBLEM, "数据缺失")
        solution = g.add_node(NodeType.SOLUTION, "用中位数填充")
        g.add_edge(problem.id, solution.id, EdgeType.SOLVED_BY)

        # Activate from problem → should reach solution
        activated_from_problem = g.spreading_activation([problem.id])
        activated_ids = {n.id for n, _ in activated_from_problem}
        assert solution.id in activated_ids

        # Activate from solution → should NOT reach problem via directed edge
        activated_from_solution = g.spreading_activation([solution.id])
        activated_ids = {n.id for n, _ in activated_from_solution}
        assert problem.id not in activated_ids

    def test_similar_edge_propagates_both_ways(self):
        """SIMILAR edges are undirected, should propagate both ways."""
        g = ExperienceGraph()
        a = g.add_node(NodeType.CONCEPT, "用户留存")
        b = g.add_node(NodeType.CONCEPT, "用户流失")
        g.add_edge(a.id, b.id, EdgeType.SIMILAR)

        activated_from_a = g.spreading_activation([a.id])
        assert any(n.id == b.id for n, _ in activated_from_a)

        activated_from_b = g.spreading_activation([b.id])
        assert any(n.id == a.id for n, _ in activated_from_b)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_experience_graph.py::TestEdgeDirectionality -v
```

Expected: `test_causal_edge_only_forward_in_activation` FAILS (currently propagates both ways)

- [ ] **Step 3: Add DIRECTED_TYPES constant and modify adjacency**

In `anima/experience_graph.py`, after the `EdgeType` enum:

```python
DIRECTED_EDGE_TYPES = {
    EdgeType.CAUSAL,
    EdgeType.TEMPORAL,
    EdgeType.SOLVED_BY,
    EdgeType.REQUIRES,
    EdgeType.COMPOSED_OF,
}

UNDIRECTED_EDGE_TYPES = {
    EdgeType.SIMILAR,
    EdgeType.CONFLICTS,
    EdgeType.REINFORCES,
}
```

Modify `ExperienceGraph.__init__`:

```python
def __init__(self):
    self.nodes: dict[str, Node] = {}
    self.edges: list[Edge] = []
    self._forward: dict[str, list[tuple[str, Edge]]] = {}   # source → [(target, edge)]
    self._backward: dict[str, list[tuple[str, Edge]]] = {}  # target → [(source, edge)]
```

Modify `add_edge`:

```python
def add_edge(self, source_id: str, target_id: str,
             edge_type: EdgeType, weight: float = 1.0) -> Optional[Edge]:
    if source_id not in self.nodes or target_id not in self.nodes:
        return None

    for _, edge in self._forward.get(source_id, []):
        if edge.target_id == target_id and edge.edge_type == edge_type:
            edge.weight = min(edge.weight + 0.1, 5.0)
            return edge
    for _, edge in self._forward.get(target_id, []):
        if edge.target_id == source_id and edge.edge_type == edge_type:
            edge.weight = min(edge.weight + 0.1, 5.0)
            return edge

    edge = Edge(source_id=source_id, target_id=target_id,
                edge_type=edge_type, weight=weight)
    self.edges.append(edge)

    self._forward.setdefault(source_id, []).append((target_id, edge))
    self._backward.setdefault(target_id, []).append((source_id, edge))

    if edge_type in UNDIRECTED_EDGE_TYPES:
        self._forward.setdefault(target_id, []).append((source_id, edge))
        self._backward.setdefault(source_id, []).append((target_id, edge))

    return edge
```

Modify `spreading_activation` to use `self._forward` instead of `self._adjacency`:

```python
for neighbor_id, edge in self._forward.get(node_id, []):
```

Update all other references from `self._adjacency` to `self._forward` (in `hebbian_update`, `extract_subgraph`) or `self._backward` where reverse lookup is needed.

Update `from_dict` similarly — when reconstructing adjacency, use the same directed/undirected logic.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_experience_graph.py::TestEdgeDirectionality -v
```

Expected: both tests PASS

- [ ] **Step 5: Write test for serialization roundtrip with directed edges**

```python
class TestSerialization:
    def test_roundtrip_preserves_edge_directionality(self):
        g = ExperienceGraph()
        a = g.add_node(NodeType.PROBLEM, "问题A")
        b = g.add_node(NodeType.SOLUTION, "方案B")
        g.add_edge(a.id, b.id, EdgeType.SOLVED_BY)

        data = g.to_dict()
        g2 = ExperienceGraph.from_dict(data)

        # Forward: problem → solution should work
        activated = g2.spreading_activation([a.id])
        assert any(n.id == b.id for n, _ in activated)

        # Backward: solution → problem should NOT work
        activated = g2.spreading_activation([b.id])
        assert not any(n.id == a.id for n, _ in activated)
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/test_experience_graph.py -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add anima/experience_graph.py tests/test_experience_graph.py
git commit -m "fix: separate directed and undirected edges in ExperienceGraph

Directed edge types (CAUSAL, TEMPORAL, SOLVED_BY, REQUIRES, COMPOSED_OF)
now only propagate activation in the forward direction. Undirected types
(SIMILAR, CONFLICTS, REINFORCES) propagate both ways. This prevents
nonsensical traversals like 'solution solved_by problem'."
```

---

### Task 3: Fix Experience Recording

**Problems:**
1. Multiple solutions get connected to ALL problems (should be paired or sequential)
2. Skill nodes are never deduplicated
3. `feedback()` can double-record experiences (demo calls `record_experience` then `feedback` which also records)

**Files:**
- Modify: `anima/persona.py`
- Modify: `anima/agent.py`
- Create: `tests/test_persona.py`

- [ ] **Step 1: Write failing test for solution-problem pairing**

```python
# tests/test_persona.py
from anima.persona import PersonaLayer
from anima.strategy import TaskCategory
from anima.experience_graph import NodeType, EdgeType


class TestExperienceRecording:
    def test_solutions_pair_with_corresponding_problems(self):
        """Each solution should connect to its corresponding problem, not all problems."""
        persona = PersonaLayer("test")
        persona.record_experience(
            "测试任务", TaskCategory.DATA_ANALYSIS,
            ["direct_execution"], ["python_coding"], "成功",
            problems_encountered=["问题A", "问题B"],
            solutions_found=["方案A", "方案B"],
        )
        graph = persona.experience_graph

        # Find solution nodes
        solution_nodes = graph.find_by_type(NodeType.SOLUTION)
        assert len(solution_nodes) == 2

        # 方案A should connect to 问题A only, 方案B to 问题B only
        for sol in solution_nodes:
            connected_problems = []
            for _, edge in graph._forward.get(sol.id, []) + graph._backward.get(sol.id, []):
                if edge.edge_type == EdgeType.SOLVED_BY:
                    other_id = edge.source_id if edge.target_id == sol.id else edge.target_id
                    node = graph.nodes.get(other_id)
                    if node and node.node_type == NodeType.PROBLEM:
                        connected_problems.append(node)
            assert len(connected_problems) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_persona.py::TestExperienceRecording::test_solutions_pair_with_corresponding_problems -v
```

Expected: FAILS (currently each solution connects to all problems)

- [ ] **Step 3: Fix solution-problem pairing in persona.py**

Replace the nested loop in `record_experience`:

```python
# Old code (in record_experience):
#   if problems_encountered:
#       for prob in problems_encountered:
#           ...
#           if solutions_found:
#               for sol in solutions_found:
#                   ...

# New code:
if problems_encountered:
    for i, prob in enumerate(problems_encountered):
        prob_node = self.experience_graph.add_node(
            NodeType.PROBLEM, prob)
        self.experience_graph.add_edge(
            task_node.id, prob_node.id, EdgeType.CAUSAL)

        if solutions_found and i < len(solutions_found):
            sol_node = self.experience_graph.add_node(
                NodeType.SOLUTION, solutions_found[i])
            self.experience_graph.add_edge(
                prob_node.id, sol_node.id, EdgeType.SOLVED_BY)

# Handle extra solutions not paired with problems
if solutions_found and len(solutions_found) > len(problems_encountered or []):
    for sol in solutions_found[len(problems_encountered or []):]:
        sol_node = self.experience_graph.add_node(
            NodeType.SOLUTION, sol)
        self.experience_graph.add_edge(
            task_node.id, sol_node.id, EdgeType.CAUSAL)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_persona.py::TestExperienceRecording -v
```

Expected: PASS

- [ ] **Step 5: Write test for skill node deduplication**

```python
class TestSkillRegistration:
    def test_duplicate_skill_does_not_create_new_node(self):
        persona = PersonaLayer("test")
        persona.register_skill("python_coding", "编写Python代码", ["code_writing"])
        persona.register_skill("python_coding", "编写Python代码", ["code_writing"])

        skill_nodes = persona.experience_graph.find_by_type(NodeType.SKILL)
        assert len(skill_nodes) == 1
```

- [ ] **Step 6: Fix register_skill to deduplicate**

In `persona.py`, modify `register_skill`:

```python
def register_skill(self, skill_name: str, description: str,
                   categories: list[str] = None):
    if skill_name in self.skills:
        self.skills[skill_name]["description"] = description
        self.skills[skill_name]["categories"] = categories or []
        return

    self.skills[skill_name] = {
        "name": skill_name,
        "description": description,
        "categories": categories or [],
        "usage_count": 0,
        "success_count": 0,
    }
    self.experience_graph.add_node(
        NodeType.SKILL, f"Skill: {skill_name} - {description}")
```

- [ ] **Step 7: Write test for no double-recording in feedback**

```python
class TestFeedbackRecording:
    def test_feedback_does_not_double_record(self):
        """Calling feedback() should not create duplicate task nodes."""
        from anima.agent import AnimaAgent
        agent = AnimaAgent("test", save_path="/tmp/test_agent.json")
        agent.register_skill("coding", "写代码")

        agent.think("写一个排序算法")
        agent.feedback(0.8, ["direct_execution"], ["coding"])

        task_nodes = agent.persona.experience_graph.find_by_type(NodeType.TASK)
        # Should have exactly 1 task node, not 2
        assert len(task_nodes) == 1
```

- [ ] **Step 8: Fix agent.py feedback() to avoid double-recording**

In `agent.py`, the `feedback` method currently calls `persona.record_experience()` then `persona.learn_from_feedback()`. But `learn_from_feedback` in `persona.py` does NOT call `record_experience` — so the double-recording only happens if user code (like demo.py) manually calls `record_experience` AND then `feedback`.

The fix: make `feedback()` the single entry point for recording + learning. Remove the manual `record_experience` calls from demo.py later.

Check the current `feedback()` in `agent.py`:

```python
def feedback(self, reward: float,
             actions_taken: list[str] = None,
             skills_used: list[str] = None,
             problems: list[str] = None,
             solutions: list[str] = None):
    if not self._current_task or not self._current_context:
        print("[Anima] 没有当前���务上下文，无法学习")
        return

    task_category = TaskCategory(
        self._current_context.get("task_category", "unknown"))
    strategy = self._current_context.get("strategy", {})

    actions = actions_taken or strategy.get("actions", [])
    skills = skills_used or strategy.get("skills", [])

    outcome = "成功" if reward > 0.5 else "一般" if reward > 0 else "失败"
    self.persona.record_experience(
        self._current_task, task_category,
        actions, skills, outcome, problems, solutions)

    self.persona.learn_from_feedback(
        task_category, actions, skills, reward)

    self._auto_save()
```

This is correct — `feedback()` calls `record_experience` exactly once. The bug is in `demo.py` which calls `persona.record_experience` manually before calling `feedback()`. The fix is to remove the manual calls from demo.py (or note in docstring that `feedback()` handles recording).

Add docstring to `feedback()`:

```python
def feedback(self, reward: float, ...):
    """
    主人给予反馈——这是Agent成长的核心驱动力。
    此方法会自动记录经验并触发学习，不需要手动调用record_experience。
    """
```

- [ ] **Step 9: Run all persona tests**

```bash
pytest tests/test_persona.py -v
```

Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add anima/persona.py anima/agent.py tests/test_persona.py
git commit -m "fix: correct experience recording - pair solutions with problems, deduplicate skills

- Solutions now pair 1:1 with problems by index, not cartesian product
- register_skill checks for existing skill before creating node
- Clarified that feedback() is the single entry point for recording + learning"
```

---

### Task 4: Fix StrategyNetwork Learning

**Problems:**
1. Preference scores grow unbounded (current += learning_rate * reward). Need EMA.
2. Exploration logic selects lowest-scored actions (which may be tried-and-failed), not least-tried.
3. `from_dict()` doesn't restore history.
4. `context` parameter is accepted but never used in decision-making.

**Files:**
- Modify: `anima/strategy.py`
- Create: `tests/test_strategy.py`

- [ ] **Step 1: Write failing test for bounded learning**

```python
# tests/test_strategy.py
from anima.strategy import StrategyNetwork, TaskCategory


class TestBoundedLearning:
    def test_preference_scores_stay_bounded(self):
        """After many positive feedbacks, scores should not grow past a reasonable bound."""
        net = StrategyNetwork()
        cat = TaskCategory.DATA_ANALYSIS

        for _ in range(200):
            net.learn_from_feedback(
                cat, ["decompose_first"], ["sql_query"], reward=1.0)

        profile = net.profiles[cat.value]
        score = profile.action_preferences.get("decompose_first", 0)
        # With EMA, score should converge near 1.0, not grow to 40+
        assert score <= 1.5, f"Score grew unbounded to {score}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_strategy.py::TestBoundedLearning -v
```

Expected: FAILS (`score` is around 40.0 with current accumulation)

- [ ] **Step 3: Implement EMA-based learning**

In `strategy.py`, modify `learn_from_feedback`:

```python
def learn_from_feedback(self, task_category: TaskCategory,
                        actions_taken: list[str],
                        skills_used: list[str],
                        reward: float,
                        context: dict = None):
    profile = self.profiles[task_category.value]
    profile.total_attempts += 1
    if reward > 0.5:
        profile.success_count += 1

    ema_alpha = 0.2  # EMA smoothing factor: higher = more responsive to recent

    for action in actions_taken:
        current = profile.action_preferences.get(action, 0.0)
        # EMA: new_value = alpha * observation + (1 - alpha) * old_value
        # observation = reward (positive reinforces, negative weakens)
        profile.action_preferences[action] = (
            ema_alpha * reward + (1 - ema_alpha) * current
        )

    for skill in skills_used:
        current = profile.skill_preferences.get(skill, 0.0)
        profile.skill_preferences[skill] = (
            ema_alpha * reward + (1 - ema_alpha) * current
        )

    if reward > 0.5:
        pattern = {
            "actions": actions_taken,
            "skills": skills_used,
            "score": reward,
            "description": f"使用 {', '.join(actions_taken)} + "
                           f"{', '.join(skills_used) if skills_used else '无特定Skill'}",
            "timestamp": time.time(),
        }
        profile.sequence_patterns.append(pattern)
        profile.sequence_patterns.sort(
            key=lambda p: p["score"], reverse=True)
        profile.sequence_patterns = profile.sequence_patterns[:20]

    self.history.append(StrategyRecord(
        task_category=task_category,
        context_features=context or {},
        actions_taken=[ActionType(a) for a in actions_taken
                       if a in [at.value for at in ActionType]],
        skills_used=skills_used,
        reward=reward,
    ))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_strategy.py::TestBoundedLearning -v
```

Expected: PASS (score converges near 1.0)

- [ ] **Step 5: Write failing test for exploration logic**

```python
class TestExplorationLogic:
    def test_exploration_picks_least_tried_not_lowest_scored(self):
        """Exploration should favor actions with fewest attempts, not lowest score."""
        net = StrategyNetwork(exploration_rate=1.0)  # always explore
        cat = TaskCategory.CODE_WRITING

        # Give negative feedback to action A (tried but bad)
        for _ in range(5):
            net.learn_from_feedback(cat, ["direct_execution"], [], reward=-0.8)
        # Give positive feedback to action B (tried and good)
        for _ in range(5):
            net.learn_from_feedback(cat, ["decompose_first"], [], reward=0.9)

        # Exploration should NOT pick direct_execution just because its score is lowest
        # It should pick actions that have been tried fewer times
        strategies = [net.decide_strategy(cat, {}, []) for _ in range(20)]
        action_lists = [s["actions"] for s in strategies]

        # direct_execution (10 attempts, bad score) should NOT dominate exploration
        de_count = sum(1 for acts in action_lists if "direct_execution" in acts)
        assert de_count < 15, "Exploration is re-trying failed actions too often"
```

- [ ] **Step 6: Fix exploration to track attempt counts separately**

Add `attempt_counts` to `StrategyProfile`:

```python
@dataclass
class StrategyProfile:
    category: TaskCategory
    action_preferences: dict[str, float] = field(default_factory=dict)
    action_attempt_counts: dict[str, int] = field(default_factory=dict)  # NEW
    skill_preferences: dict[str, float] = field(default_factory=dict)
    sequence_patterns: list[dict] = field(default_factory=list)
    total_attempts: int = 0
    success_count: int = 0
```

Update `learn_from_feedback` to track counts:

```python
for action in actions_taken:
    # ... existing EMA code ...
    profile.action_attempt_counts[action] = (
        profile.action_attempt_counts.get(action, 0) + 1
    )
```

Fix `_explore` to use attempt counts:

```python
def _explore(self, category, available_skills, context):
    profile = self.profiles[category.value]

    all_actions = list(ActionType)
    # Sort by attempt count (ascending), not by preference score
    actions = sorted(all_actions,
                     key=lambda a: profile.action_attempt_counts.get(a.value, 0))
    selected_actions = actions[:2]

    selected_skills = []
    if available_skills:
        n_skills = min(random.randint(1, 3), len(available_skills))
        selected_skills = random.sample(available_skills, n_skills)

    return {
        "actions": [a.value for a in selected_actions],
        "skills": selected_skills,
        "reasoning": "探索模式：尝试使用次数较少的策略",
    }
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_strategy.py::TestExplorationLogic -v
```

Expected: PASS

- [ ] **Step 8: Write failing test for history deserialization**

```python
class TestHistorySerialization:
    def test_history_survives_roundtrip(self):
        net = StrategyNetwork()
        net.learn_from_feedback(
            TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], reward=0.9)
        net.learn_from_feedback(
            TaskCategory.CODE_WRITING,
            ["direct_execution"], ["python_coding"], reward=0.7)

        data = net.to_dict()
        net2 = StrategyNetwork.from_dict(data)

        assert len(net2.history) == 2
        assert net2.history[0].reward == 0.9
        assert net2.history[1].task_category == TaskCategory.CODE_WRITING
```

- [ ] **Step 9: Fix from_dict to restore history**

In `strategy.py`, at the end of `from_dict`:

```python
@classmethod
def from_dict(cls, data: dict) -> "StrategyNetwork":
    net = cls(exploration_rate=data.get("exploration_rate", 0.3))
    net._total_decisions = data.get("total_decisions", 0)

    for cat, pd in data.get("profiles", {}).items():
        if cat in net.profiles:
            profile = net.profiles[cat]
            profile.action_preferences = pd.get("action_preferences", {})
            profile.action_attempt_counts = pd.get("action_attempt_counts", {})
            profile.skill_preferences = pd.get("skill_preferences", {})
            profile.sequence_patterns = pd.get("sequence_patterns", [])
            profile.total_attempts = pd.get("total_attempts", 0)
            profile.success_count = pd.get("success_count", 0)

    # Restore history
    for record_data in data.get("history", []):
        try:
            record = StrategyRecord(
                task_category=TaskCategory(record_data["task_category"]),
                context_features=record_data.get("context_features", {}),
                actions_taken=[ActionType(a) for a in record_data.get("actions_taken", [])],
                skills_used=record_data.get("skills_used", []),
                reward=record_data.get("reward", 0),
                timestamp=record_data.get("timestamp", 0),
            )
            net.history.append(record)
        except (ValueError, KeyError):
            continue

    return net
```

Also update `to_dict` to include `action_attempt_counts`:

```python
# Inside profiles serialization:
"action_attempt_counts": p.action_attempt_counts,
```

- [ ] **Step 10: Run all strategy tests**

```bash
pytest tests/test_strategy.py -v
```

Expected: all PASS

- [ ] **Step 11: Commit**

```bash
git add anima/strategy.py tests/test_strategy.py
git commit -m "fix: bounded EMA learning, correct exploration logic, restore history

- Preference scores now use exponential moving average (alpha=0.2), converge near [-1, 1]
- Exploration selects least-attempted actions, not lowest-scored
- Track action_attempt_counts separately from preference scores
- from_dict now restores history records"
```

---

### Task 5: Fix Chinese Text Processing

**Problem:** `_extract_keywords` uses `.split()` which doesn't work for Chinese (no spaces between words). `_discover_connections` uses the same broken approach. This makes the entire system unable to process Chinese text.

**Fix:** Use jieba for Chinese word segmentation. Apply it to both keyword extraction and connection discovery.

**Files:**
- Modify: `anima/persona.py`
- Add tests to: `tests/test_persona.py`

- [ ] **Step 1: Write failing test for Chinese keyword extraction**

```python
# Add to tests/test_persona.py
class TestChineseKeywords:
    def test_extracts_meaningful_chinese_keywords(self):
        persona = PersonaLayer("test")
        keywords = persona._extract_keywords("分析上个月的用户留存数据")

        # Should extract meaningful segments, not a single long string
        assert len(keywords) >= 2, f"Got only {keywords}"
        # Should contain meaningful words
        found_meaningful = any(
            kw in ["分析", "用户", "留存", "数据", "用户留存", "留存数据"]
            for kw in keywords
        )
        assert found_meaningful, f"No meaningful keywords in {keywords}"

    def test_chinese_keywords_can_find_related_nodes(self):
        persona = PersonaLayer("test")
        persona.experience_graph.add_node(NodeType.CONCEPT, "用户留存分析方法")

        keywords = persona._extract_keywords("帮我看看用户留存的情况")
        results = persona.experience_graph.find_by_content(keywords)
        assert len(results) > 0, "Should find related node via Chinese keywords"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_persona.py::TestChineseKeywords -v
```

Expected: FAILS (current `.split()` produces single-element list for Chinese)

- [ ] **Step 3: Rewrite _extract_keywords with jieba**

In `persona.py`, add import and rewrite:

```python
import jieba

# ... inside PersonaLayer class ...

def _extract_keywords(self, text: str) -> list[str]:
    stop_words = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "��个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "它", "们", "可以", "什么", "怎么", "那", "吗", "吧", "啊",
        "帮", "请", "想", "能", "把", "让", "给", "用", "做", "写",
        "看看", "一下", "下", "个", "来", "过", "被", "比", "从",
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "with", "and", "or", "but", "not",
        "this", "that", "i", "me", "my", "you", "your", "he", "she",
        "it", "we", "they", "can", "will", "do", "does",
        "help", "please", "want", "need", "make",
    }
    words = jieba.lcut(text)
    return [w.strip() for w in words
            if w.strip() and w.strip() not in stop_words and len(w.strip()) > 1]
```

- [ ] **Step 4: Rewrite _discover_connections with jieba**

```python
def _discover_connections(self, new_node):
    new_words = set(jieba.lcut(new_node.content.lower()))
    new_words = {w for w in new_words if len(w) > 1}
    if not new_words:
        return

    for existing_node in self.experience_graph.nodes.values():
        if existing_node.id == new_node.id:
            continue
        if existing_node.node_type != new_node.node_type:
            continue

        existing_words = set(jieba.lcut(existing_node.content.lower()))
        existing_words = {w for w in existing_words if len(w) > 1}
        if not existing_words:
            continue

        overlap = len(new_words & existing_words)
        total = len(new_words | existing_words)
        if total > 0 and overlap / total > 0.3:
            self.experience_graph.add_edge(
                new_node.id, existing_node.id,
                EdgeType.SIMILAR, weight=overlap / total)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_persona.py::TestChineseKeywords -v
```

Expected: PASS

- [ ] **Step 6: Write test for connection discovery between Chinese tasks**

```python
class TestChineseConnections:
    def test_discovers_similar_chinese_tasks(self):
        persona = PersonaLayer("test")
        persona.record_experience(
            "分析用户留存数据", TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], "成功")
        persona.record_experience(
            "分析用户���失原因", TaskCategory.DATA_ANALYSIS,
            ["search_first"], ["sql_query"], "成功")

        task_nodes = persona.experience_graph.find_by_type(NodeType.TASK)
        assert len(task_nodes) == 2

        # The two tasks share "分析" and "用户", should have a SIMILAR edge
        edges = persona.experience_graph.edges
        similar_edges = [e for e in edges if e.edge_type == EdgeType.SIMILAR]
        assert len(similar_edges) > 0, "Should discover similarity between related Chinese tasks"
```

- [ ] **Step 7: Run all tests**

```bash
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add anima/persona.py tests/test_persona.py
git commit -m "fix: use jieba for Chinese word segmentation

- _extract_keywords now uses jieba.lcut() instead of .split()
- _discover_connections uses jieba-based word sets for overlap
- Both functions now work correctly with Chinese text"
```

---

### Task 6: Phase 1 Integration Test

**Files:**
- Add to: `tests/test_persona.py`

- [ ] **Step 1: Write integration test for the full Phase 1 flow**

```python
class TestPhase1Integration:
    def test_two_agents_diverge_with_different_experiences(self):
        """Core claim: two agents with same skills but different experiences behave differently."""
        from anima.agent import AnimaAgent

        alpha = AnimaAgent("Alpha", save_path="/tmp/test_alpha.json")
        beta = AnimaAgent("Beta", save_path="/tmp/test_beta.json")

        for name, desc in [("sql_query", "SQL查询"), ("python_coding", "Python编程")]:
            alpha.register_skill(name, desc)
            beta.register_skill(name, desc)

        # Alpha: data analysis experiences
        alpha.think("分析用户留存数据")
        alpha.feedback(0.9, ["decompose_first", "use_skill"], ["sql_query"],
                       ["数据有缺失值"], ["中位数填充"])
        alpha.think("分析销售趋势")
        alpha.feedback(0.8, ["decompose_first", "search_first"], ["sql_query"])

        # Beta: coding experiences
        beta.think("编写自动部署脚本")
        beta.feedback(0.9, ["direct_execution", "iterate_and_refine"], ["python_coding"])
        beta.think("重构权限系统")
        beta.feedback(0.8, ["direct_execution", "use_skill"], ["python_coding"])

        # Verify they diverged
        alpha_strategy = alpha.persona.strategy_network.profiles
        beta_strategy = beta.persona.strategy_network.profiles

        alpha_da = alpha_strategy["data_analysis"].action_preferences
        beta_cw = beta_strategy["code_writing"].action_preferences

        # Alpha should prefer decompose_first for data_analysis
        assert alpha_da.get("decompose_first", 0) > alpha_da.get("direct_execution", 0)
        # Beta should prefer direct_execution for code_writing
        assert beta_cw.get("direct_execution", 0) > beta_cw.get("decompose_first", 0)

    def test_experience_graph_has_no_duplicate_skills(self):
        from anima.agent import AnimaAgent
        agent = AnimaAgent("Test", save_path="/tmp/test_dedup.json")
        agent.register_skill("sql", "SQL")
        agent.register_skill("sql", "SQL updated")
        skill_nodes = agent.persona.experience_graph.find_by_type(NodeType.SKILL)
        assert len(skill_nodes) == 1
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_persona.py::TestPhase1Integration -v
```

Expected: all PASS

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_persona.py
git commit -m "test: add Phase 1 integration tests - verify agent divergence and deduplication"
```

---

## Phase 2: Semantic Layer

> **Phase 2 adds LLM and embedding dependencies.**
> After Phase 2, the graph has real semantic structure, activation finds truly related experiences,
> strategy decisions consume graph state, and the LLM receives narrative context instead of node lists.

### Task 7: Embedding Provider Interface

**Purpose:** Abstract the embedding generation so it works with different providers (Anthropic, OpenAI, local models). Ship with an Anthropic implementation + a mock for testing.

**Files:**
- Create: `anima/embedding.py`
- Create: `tests/test_embedding.py`

- [ ] **Step 1: Write test with mock provider**

```python
# tests/test_embedding.py
import numpy as np
from anima.embedding import MockEmbeddingProvider, cosine_similarity


class TestEmbedding:
    def test_mock_provider_returns_consistent_embeddings(self):
        provider = MockEmbeddingProvider(dimensions=64)
        emb1 = provider.embed("用户留存")
        emb2 = provider.embed("用户留存")
        assert np.allclose(emb1, emb2), "Same text should produce same embedding"

    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.01

    def test_cosine_similarity_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 0.01

    def test_mock_provider_similar_texts_have_higher_similarity(self):
        provider = MockEmbeddingProvider(dimensions=64)
        emb_retain = provider.embed("用户留存")
        emb_churn = provider.embed("用户流失")
        emb_server = provider.embed("服务器部署")

        sim_related = cosine_similarity(emb_retain, emb_churn)
        sim_unrelated = cosine_similarity(emb_retain, emb_server)
        # Mock uses character overlap heuristic — related texts score higher
        assert sim_related > sim_unrelated
```

- [ ] **Step 2: Run test to verify it fails (module doesn't exist yet)**

```bash
pytest tests/test_embedding.py -v
```

Expected: FAILS with ImportError

- [ ] **Step 3: Implement embedding.py**

```python
# anima/embedding.py
"""
Embedding provider interface.

Provides vector embeddings for text, enabling semantic similarity search.
Ships with:
- MockEmbeddingProvider: deterministic, character-overlap-based (for testing)
- AnthropicEmbeddingProvider: uses Anthropic's voyage embeddings (for production)
"""

import hashlib
import math
from abc import ABC, abstractmethod


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock: produces embeddings based on character n-gram hashing.
    Texts sharing more characters produce more similar vectors.
    For testing only — not semantically meaningful beyond surface similarity."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        # Generate character bigrams
        chars = list(text.lower())
        ngrams = [text[i:i+2] for i in range(len(text) - 1)] + chars
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
            idx = h % self._dimensions
            vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """Uses Anthropic's Voyage embeddings via the Anthropic SDK."""

    def __init__(self, model: str = "voyage-3", api_key: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install 'anima-agent[semantic]' for Anthropic embeddings")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._dimensions = 1024  # voyage-3 default

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import anthropic
        voyageai = anthropic.Anthropic().embeddings
        response = voyageai.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_embedding.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add anima/embedding.py tests/test_embedding.py
git commit -m "feat: add embedding provider interface with mock and Anthropic implementations"
```

---

### Task 8: Semantic Seed Selection for ExperienceGraph

**Purpose:** Replace keyword substring matching with embedding cosine similarity for finding seed nodes. This is the single most impactful change — it makes "what the agent recalls" semantically meaningful.

**Files:**
- Modify: `anima/experience_graph.py`
- Add tests to: `tests/test_experience_graph.py`

- [ ] **Step 1: Write failing test for semantic search**

```python
# Add to tests/test_experience_graph.py
from anima.embedding import MockEmbeddingProvider


class TestSemanticSearch:
    def test_find_by_embedding_returns_semantically_similar(self):
        provider = MockEmbeddingProvider(dimensions=64)
        g = ExperienceGraph()

        n1 = g.add_node(NodeType.CONCEPT, "用户留���分析",
                         embedding=provider.embed("用户留存分析"))
        n2 = g.add_node(NodeType.CONCEPT, "服务器部署配置",
                         embedding=provider.embed("服务器部署配置"))
        n3 = g.add_node(NodeType.CONCEPT, "用户流失原因",
                         embedding=provider.embed("用户流失原因"))

        query_emb = provider.embed("用户留存")
        results = g.find_by_embedding(query_emb, top_k=2)

        result_ids = [n.id for n, _ in results]
        # "用户留存分析" and "用户流失原因" should rank above "服务器部署配置"
        assert n1.id in result_ids
        assert n2.id not in result_ids

    def test_find_by_embedding_falls_back_to_content_when_no_embeddings(self):
        g = ExperienceGraph()
        g.add_node(NodeType.CONCEPT, "用户留存分析")  # no embedding

        results = g.find_by_embedding([0.1] * 64, top_k=5)
        assert len(results) == 0  # nodes without embeddings are skipped
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_experience_graph.py::TestSemanticSearch -v
```

Expected: FAILS (method doesn't exist)

- [ ] **Step 3: Add find_by_embedding to ExperienceGraph**

```python
# In anima/experience_graph.py, add to ExperienceGraph class:

def find_by_embedding(self, query_embedding: list[float],
                      top_k: int = 5,
                      min_similarity: float = 0.1) -> list[tuple[Node, float]]:
    from anima.embedding import cosine_similarity

    results = []
    for node in self.nodes.values():
        if not node.embedding:
            continue
        sim = cosine_similarity(query_embedding, node.embedding)
        if sim >= min_similarity:
            results.append((node, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_experience_graph.py::TestSemanticSearch -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add anima/experience_graph.py tests/test_experience_graph.py
git commit -m "feat: add embedding-based semantic search to ExperienceGraph"
```

---

### Task 9: LLM-Based Concept Extraction

**Purpose:** When recording an experience, use LLM to extract structured concepts and relationships instead of just storing raw text. This is Foundation 1 — the quality of the graph depends on this.

**Schema** (constrained, per Codex's recommendation):

```json
{
  "concepts": ["用户留存", "渠道分析"],
  "entities": ["SQL", "Python"],
  "domain": "data_analysis",
  "problems": ["数据有缺失值"],
  "solutions": ["中位数填充"],
  "outcome_summary": "按渠道分群找到了留存下降的原因",
  "related_concepts": ["用户流失", "分群分析"]
}
```

**Files:**
- Create: `anima/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write test with mock extractor**

```python
# tests/test_extractor.py
from anima.extractor import ExperienceExtractor, MockExtractor, ExtractionResult


class TestExtraction:
    def test_mock_extractor_returns_valid_result(self):
        extractor = MockExtractor()
        result = extractor.extract("分析用户留存数据，用SQL取数后按渠道分群")

        assert isinstance(result, ExtractionResult)
        assert len(result.concepts) > 0
        assert result.domain is not None

    def test_extraction_result_has_required_fields(self):
        result = ExtractionResult(
            concepts=["用户留存", "渠道分析"],
            entities=["SQL"],
            domain="data_analysis",
            problems=["数据缺失"],
            solutions=["中位数填充"],
            outcome_summary="成功找到原因",
            related_concepts=["用户流失"],
        )
        assert "用户留存" in result.concepts
        assert result.domain == "data_analysis"
```

- [ ] **Step 2: Implement extractor.py**

```python
# anima/extractor.py
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
{
  "concepts": ["核心概念列表，2-5个"],
  "entities": ["涉及的工具/技术/产品名"],
  "domain": "领域分类: data_analysis|code_writing|content_creation|problem_solving|communication|research|planning|creative|unknown",
  "problems": ["遇到的问题"],
  "solutions": ["解决方案"],
  "outcome_summary": "一句话总结结果",
  "related_concepts": ["可能相关但未直接提及的概念"]
}

经验描述：
{text}

JSON输出："""


class MockExtractor(ExperienceExtractor):
    """Deterministic mock for testing. Extracts based on simple keyword rules."""

    def extract(self, text: str) -> ExtractionResult:
        import jieba
        words = list(jieba.cut(text))

        domain_keywords = {
            "data_analysis": ["数据", "分析", "统计", "留存", "指标"],
            "code_writing": ["代码", "编程", "函数", "脚本", "开发"],
        }

        domain = "unknown"
        for d, kws in domain_keywords.items():
            if any(kw in text for kw in kws):
                domain = d
                break

        concepts = [w for w in words if len(w) >= 2 and w not in
                    {"分析", "数据", "帮我", "一下", "看看", "上个"}][:5]

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

    def __init__(self, api_key: str = None, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install 'anima-agent[semantic]' for LLM extraction")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def extract(self, text: str) -> ExtractionResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        )
        raw = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_extractor.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add anima/extractor.py tests/test_extractor.py
git commit -m "feat: add LLM-based concept extractor with mock and Anthropic implementations"
```

---

### Task 10: Wire Semantic Layer into PersonaLayer

**Purpose:** Connect embedding + extraction into the main experience recording and retrieval flow. This is where Foundation 1 and Foundation 2 come together.

**Files:**
- Modify: `anima/persona.py`
- Modify: `anima/agent.py`
- Add tests to: `tests/test_persona.py`

- [ ] **Step 1: Write test for semantic experience recording**

```python
# Add to tests/test_persona.py
from anima.embedding import MockEmbeddingProvider
from anima.extractor import MockExtractor


class TestSemanticRecording:
    def _make_semantic_persona(self):
        persona = PersonaLayer("test")
        persona.configure_semantic(
            embedding_provider=MockEmbeddingProvider(dimensions=64),
            extractor=MockExtractor(),
        )
        return persona

    def test_recording_creates_concept_nodes(self):
        persona = self._make_semantic_persona()
        persona.record_experience(
            "分析用户留存数据", TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], "成功")

        concept_nodes = persona.experience_graph.find_by_type(NodeType.CONCEPT)
        assert len(concept_nodes) > 0, "Should create CONCEPT nodes from extraction"

    def test_concept_nodes_have_embeddings(self):
        persona = self._make_semantic_persona()
        persona.record_experience(
            "分析用户留存数据", TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], "成功")

        concept_nodes = persona.experience_graph.find_by_type(NodeType.CONCEPT)
        for node in concept_nodes:
            assert len(node.embedding) > 0, f"CONCEPT node '{node.content}' has no embedding"

    def test_semantic_activation_finds_related_experience(self):
        persona = self._make_semantic_persona()
        persona.record_experience(
            "分析用户留存数据", TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], "成功")

        # A new task about user churn (related to retention)
        context = persona.prepare_context("帮我看看用户流失的原因")
        activated = context.get("activated_experiences", [])
        # Should find related experiences via semantic similarity
        assert len(activated) > 0, "Semantic activation should find related experiences"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_persona.py::TestSemanticRecording -v
```

Expected: FAILS (`configure_semantic` doesn't exist)

- [ ] **Step 3: Add semantic configuration to PersonaLayer**

In `persona.py`, add to `PersonaLayer.__init__`:

```python
def __init__(self, agent_name: str = "Anima"):
    self.agent_name = agent_name
    self.created_at = time.time()

    self.experience_graph = ExperienceGraph()
    self.strategy_network = StrategyNetwork()
    self.competence = CompetenceEmbedding()

    self.skills: dict[str, dict] = {}
    self.interaction_count = 0

    # Semantic layer (optional, None = keyword fallback)
    self._embedding_provider = None
    self._extractor = None

def configure_semantic(self, embedding_provider=None, extractor=None):
    self._embedding_provider = embedding_provider
    self._extractor = extractor
```

- [ ] **Step 4: Modify record_experience to use extractor when available**

```python
def record_experience(self, task_description, task_category,
                      actions_taken, skills_used, outcome,
                      problems_encountered=None, solutions_found=None):
    task_embedding = []
    if self._embedding_provider:
        task_embedding = self._embedding_provider.embed(task_description)

    task_node = self.experience_graph.add_node(
        NodeType.TASK, task_description,
        embedding=task_embedding,
        metadata={"category": task_category.value,
                  "actions": actions_taken,
                  "skills": skills_used})

    # Semantic extraction: create CONCEPT nodes
    if self._extractor:
        extraction = self._extractor.extract(task_description)
        for concept in extraction.concepts:
            concept_emb = []
            if self._embedding_provider:
                concept_emb = self._embedding_provider.embed(concept)

            # Check if similar concept already exists
            existing = self._find_existing_concept(concept, concept_emb)
            if existing:
                self.experience_graph.add_edge(
                    task_node.id, existing.id, EdgeType.COMPOSED_OF)
                existing.activation_count += 1
            else:
                concept_node = self.experience_graph.add_node(
                    NodeType.CONCEPT, concept, embedding=concept_emb)
                self.experience_graph.add_edge(
                    task_node.id, concept_node.id, EdgeType.COMPOSED_OF)

    # ... rest of existing code (outcome, skills, problems, solutions) ...
    outcome_node = self.experience_graph.add_node(
        NodeType.FEEDBACK, f"结果: {outcome}")
    self.experience_graph.add_edge(
        task_node.id, outcome_node.id, EdgeType.CAUSAL)

    for skill_name in skills_used:
        skill_nodes = [n for n in self.experience_graph.find_by_type(
            NodeType.SKILL) if skill_name in n.content]
        for sn in skill_nodes:
            self.experience_graph.add_edge(
                task_node.id, sn.id, EdgeType.REQUIRES)

    if problems_encountered:
        for i, prob in enumerate(problems_encountered):
            prob_node = self.experience_graph.add_node(NodeType.PROBLEM, prob)
            self.experience_graph.add_edge(
                task_node.id, prob_node.id, EdgeType.CAUSAL)
            if solutions_found and i < len(solutions_found):
                sol_node = self.experience_graph.add_node(
                    NodeType.SOLUTION, solutions_found[i])
                self.experience_graph.add_edge(
                    prob_node.id, sol_node.id, EdgeType.SOLVED_BY)

    if solutions_found and len(solutions_found) > len(problems_encountered or []):
        for sol in solutions_found[len(problems_encountered or []):]:
            sol_node = self.experience_graph.add_node(NodeType.SOLUTION, sol)
            self.experience_graph.add_edge(
                task_node.id, sol_node.id, EdgeType.CAUSAL)

    self._discover_connections(task_node)


def _find_existing_concept(self, concept_text, concept_embedding):
    """Find an existing CONCEPT node that matches semantically."""
    if concept_embedding and self._embedding_provider:
        from anima.embedding import cosine_similarity
        best_match = None
        best_sim = 0.7  # threshold
        for node in self.experience_graph.find_by_type(NodeType.CONCEPT):
            if node.embedding:
                sim = cosine_similarity(concept_embedding, node.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_match = node
        return best_match

    # Fallback: exact text match
    for node in self.experience_graph.find_by_type(NodeType.CONCEPT):
        if node.content == concept_text:
            return node
    return None
```

- [ ] **Step 5: Modify prepare_context to use semantic seed selection**

```python
def prepare_context(self, task_description, task_category=None):
    self.interaction_count += 1

    # Semantic seed selection (if embedding available)
    if self._embedding_provider:
        query_emb = self._embedding_provider.embed(task_description)
        seed_results = self.experience_graph.find_by_embedding(query_emb, top_k=5)
        seed_ids = [n.id for n, _ in seed_results]
    else:
        # Fallback: keyword matching
        keywords = self._extract_keywords(task_description)
        seed_nodes = self.experience_graph.find_by_content(keywords)
        seed_ids = [n.id for n in seed_nodes[:5]]

    activated = []
    experience_context = "没有找到相关历史经验。"
    if seed_ids:
        activated = self.experience_graph.spreading_activation(seed_ids)
        experience_context = self.experience_graph.extract_subgraph(
            activated, max_nodes=8)

    # ... rest unchanged ...
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_persona.py::TestSemanticRecording -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add anima/persona.py anima/agent.py tests/test_persona.py
git commit -m "feat: wire semantic extraction and embedding into PersonaLayer

- record_experience uses LLM extractor to create CONCEPT nodes
- CONCEPT nodes get embeddings and link to existing similar concepts
- prepare_context uses embedding-based seed selection when available
- Falls back to keyword matching when no embedding provider configured"
```

---

### Task 11: Strategy-Graph Connection

**Purpose:** Make StrategyNetwork actually read the ExperienceGraph when deciding strategy. Currently they're parallel prompt contributions — this closes the loop.

**Files:**
- Modify: `anima/strategy.py`
- Modify: `anima/persona.py`
- Add tests to: `tests/test_strategy.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_strategy.py
class TestGraphInformedStrategy:
    def test_exploit_uses_similar_task_history(self):
        """When exploiting, strategy should consider what worked for similar past tasks."""
        net = StrategyNetwork(exploration_rate=0.0)  # always exploit

        # Record history with context: a data task where decompose worked well
        net.learn_from_feedback(
            TaskCategory.DATA_ANALYSIS,
            ["decompose_first", "use_skill"], ["sql_query"],
            reward=0.9,
            context={"task_embedding": [0.8, 0.3, 0.1]})

        # Record history: a data task where direct_execution worked poorly
        net.learn_from_feedback(
            TaskCategory.DATA_ANALYSIS,
            ["direct_execution"], ["python_coding"],
            reward=0.2,
            context={"task_embedding": [0.7, 0.3, 0.2]})

        # New task similar to the first one
        strategy = net.decide_strategy(
            TaskCategory.DATA_ANALYSIS,
            {"task_embedding": [0.8, 0.3, 0.1]},
            ["sql_query", "python_coding"])

        # Should prefer decompose_first (worked for similar task) over direct_execution
        assert "decompose_first" in strategy["actions"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_strategy.py::TestGraphInformedStrategy -v
```

Expected: FAILS (context is currently ignored)

- [ ] **Step 3: Make _exploit consider similar task history**

In `strategy.py`, modify `_exploit`:

```python
def _exploit(self, category, available_skills, context):
    profile = self.profiles[category.value]

    # NEW: Check for similar past tasks in history
    similar_strategies = self._find_similar_strategies(context)

    if similar_strategies:
        # Weight by reward: higher reward = more influence
        action_scores = {}
        skill_scores = {}
        for record, similarity in similar_strategies:
            weight = max(0, record.reward) * similarity
            for action in record.actions_taken:
                action_scores[action.value] = (
                    action_scores.get(action.value, 0) + weight)
            for skill in record.skills_used:
                skill_scores[skill] = skill_scores.get(skill, 0) + weight

        # Blend history-based scores with profile preferences
        for action, score in profile.action_preferences.items():
            action_scores[action] = action_scores.get(action, 0) + score

        sorted_actions = sorted(action_scores.items(),
                                key=lambda x: x[1], reverse=True)
        selected_actions = [a for a, s in sorted_actions[:3] if s > 0]

        sorted_skills = sorted(skill_scores.items(),
                               key=lambda x: x[1], reverse=True)
        selected_skills = [s for s, sc in sorted_skills
                           if s in available_skills and sc > 0][:3]
    else:
        # Fallback to existing preference-only logic
        sorted_actions = sorted(
            profile.action_preferences.items(),
            key=lambda x: x[1], reverse=True)
        selected_actions = [a for a, _ in sorted_actions[:3] if _ > 0]

        sorted_skills = sorted(
            profile.skill_preferences.items(),
            key=lambda x: x[1], reverse=True)
        selected_skills = [s for s, score in sorted_skills
                           if s in available_skills and score > 0][:3]

    if not selected_skills and available_skills:
        selected_skills = available_skills[:1]

    reasoning = "利用模式：基于历史经验选择最佳策略"
    if similar_strategies:
        best = similar_strategies[0][0]
        reasoning += f"\n参考相似任务（reward={best.reward:.1f}）的成功策略"

    return {
        "actions": selected_actions if selected_actions
                   else [ActionType.DIRECT_EXECUTION.value],
        "skills": selected_skills,
        "reasoning": reasoning,
    }


def _find_similar_strategies(self, context: dict,
                             top_k: int = 3) -> list[tuple]:
    """Find past strategy records with similar task context."""
    task_emb = context.get("task_embedding")
    if not task_emb or not self.history:
        return []

    from anima.embedding import cosine_similarity

    scored = []
    for record in self.history:
        record_emb = record.context_features.get("task_embedding")
        if record_emb:
            sim = cosine_similarity(task_emb, record_emb)
            if sim > 0.5:
                scored.append((record, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

- [ ] **Step 4: Update PersonaLayer to pass task embedding to strategy**

In `persona.py`, in `prepare_context`, pass embedding in context:

```python
# After computing seed_ids and before calling decide_strategy:
strategy_context = {}
if self._embedding_provider:
    strategy_context["task_embedding"] = query_emb

strategy = self.strategy_network.decide_strategy(
    task_category, strategy_context, available_skills)
```

And in `learn_from_feedback`, pass embedding:

```python
def learn_from_feedback(self, task_category, actions_taken, skills_used,
                        reward, task_embedding=None):
    context = {}
    if task_embedding:
        context["task_embedding"] = task_embedding

    self.strategy_network.learn_from_feedback(
        task_category, actions_taken, skills_used, reward, context)
    # ... rest unchanged ...
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_strategy.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add anima/strategy.py anima/persona.py tests/test_strategy.py
git commit -m "feat: StrategyNetwork reads similar task history for decision-making

Strategy decisions now consider embedding-similarity-matched past tasks,
not just category-level preferences. This closes the loop between
ExperienceGraph and StrategyNetwork."
```

---

### Task 12: Experience Narrativization

**Purpose:** Replace the flat node list output with narrative text that follows causal/temporal edges. This makes the LLM context actually useful.

**Files:**
- Create: `anima/narrator.py`
- Create: `tests/test_narrator.py`
- Modify: `anima/persona.py` (use narrator in prepare_context)

- [ ] **Step 1: Write test**

```python
# tests/test_narrator.py
from anima.experience_graph import ExperienceGraph, NodeType, EdgeType
from anima.narrator import narrate_subgraph


class TestNarrator:
    def _build_sample_graph(self):
        g = ExperienceGraph()
        task = g.add_node(NodeType.TASK, "分析用户留存数据")
        problem = g.add_node(NodeType.PROBLEM, "数据有缺失值")
        solution = g.add_node(NodeType.SOLUTION, "用中位数填充")
        outcome = g.add_node(NodeType.FEEDBACK, "结果: 成功")

        g.add_edge(task.id, problem.id, EdgeType.CAUSAL)
        g.add_edge(problem.id, solution.id, EdgeType.SOLVED_BY)
        g.add_edge(task.id, outcome.id, EdgeType.CAUSAL)

        activated = [(task, 1.0), (problem, 0.6), (solution, 0.5), (outcome, 0.4)]
        return g, activated

    def test_narrate_produces_readable_text(self):
        g, activated = self._build_sample_graph()
        text = narrate_subgraph(g, activated)

        assert "用户留存" in text
        assert "缺失值" in text
        assert "中位数" in text
        assert len(text) > 20

    def test_narrate_follows_causal_chain(self):
        g, activated = self._build_sample_graph()
        text = narrate_subgraph(g, activated)

        # Problem should appear after task, solution after problem (causal order)
        task_pos = text.find("留存")
        problem_pos = text.find("缺失")
        solution_pos = text.find("中位数")
        assert task_pos < problem_pos < solution_pos

    def test_narrate_empty_activation(self):
        g = ExperienceGraph()
        text = narrate_subgraph(g, [])
        assert "没有" in text or len(text) < 30
```

- [ ] **Step 2: Implement narrator.py**

```python
# anima/narrator.py
"""
Converts activated subgraphs into narrative text for LLM context injection.

Template-based: follows CAUSAL and TEMPORAL edges to build stories,
not just list nodes. No LLM call needed.
"""

from anima.experience_graph import ExperienceGraph, Node, NodeType, EdgeType


def narrate_subgraph(graph: ExperienceGraph,
                     activated: list[tuple[Node, float]],
                     max_stories: int = 3) -> str:
    if not activated:
        return "没有找到相关历史经验。"

    activated_ids = {n.id for n, _ in activated}

    # Group activated nodes by their root TASK
    task_groups = _group_by_task(graph, activated, activated_ids)

    if not task_groups:
        return _fallback_list(activated)

    stories = []
    for task_node, related_nodes in task_groups[:max_stories]:
        story = _build_story(graph, task_node, related_nodes, activated_ids)
        if story:
            stories.append(story)

    if not stories:
        return _fallback_list(activated)

    header = "以下是与当前任务相关的历史经验：\n"
    return header + "\n\n".join(stories)


def _group_by_task(graph, activated, activated_ids):
    """Group activated nodes by their nearest TASK ancestor."""
    task_nodes = [(n, a) for n, a in activated if n.node_type == NodeType.TASK]
    task_nodes.sort(key=lambda x: x[1], reverse=True)

    groups = []
    for task_node, _ in task_nodes:
        related = []
        for neighbor_id, edge in graph._forward.get(task_node.id, []):
            if neighbor_id in activated_ids:
                neighbor = graph.nodes.get(neighbor_id)
                if neighbor:
                    related.append((neighbor, edge))
        groups.append((task_node, related))

    return groups


def _build_story(graph, task_node, related_nodes, activated_ids):
    """Build a narrative from a task node and its related nodes."""
    parts = [f"你之前做过一个任务：{task_node.content}。"]

    # Collect by type
    problems = []
    solutions = []
    outcomes = []
    skills = []
    concepts = []

    for node, edge in related_nodes:
        if node.node_type == NodeType.PROBLEM:
            problems.append(node)
            # Follow SOLVED_BY edges from this problem
            for sol_id, sol_edge in graph._forward.get(node.id, []):
                if sol_edge.edge_type == EdgeType.SOLVED_BY and sol_id in activated_ids:
                    sol_node = graph.nodes.get(sol_id)
                    if sol_node:
                        solutions.append((node, sol_node))
        elif node.node_type == NodeType.FEEDBACK:
            outcomes.append(node)
        elif node.node_type == NodeType.SKILL:
            skills.append(node)
        elif node.node_type == NodeType.CONCEPT:
            concepts.append(node)

    if skills:
        skill_names = [s.content.replace("Skill: ", "").split(" - ")[0] for s in skills]
        parts.append(f"使用了{', '.join(skill_names)}。")

    if problems and solutions:
        for prob, sol in solutions:
            parts.append(f"遇到了「{prob.content}」的问题，通过「{sol.content}」解决。")
    elif problems:
        for p in problems:
            parts.append(f"遇到了「{p.content}」的问题。")

    if outcomes:
        outcome_text = outcomes[0].content.replace("结果: ", "")
        parts.append(f"最终结果：{outcome_text}。")

    return "".join(parts)


def _fallback_list(activated):
    """Fallback when no TASK nodes found."""
    lines = ["以下是相关的历史信息："]
    for node, act in activated[:5]:
        relevance = "高度相关" if act > 0.5 else "可能相关"
        lines.append(f"- {node.content}（{relevance}）")
    return "\n".join(lines)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_narrator.py -v
```

Expected: all PASS

- [ ] **Step 4: Wire narrator into PersonaLayer.prepare_context**

In `persona.py`, replace `extract_subgraph` call:

```python
# In prepare_context, replace:
#   experience_context = self.experience_graph.extract_subgraph(activated, max_nodes=8)
# With:
from anima.narrator import narrate_subgraph
experience_context = narrate_subgraph(self.experience_graph, activated, max_stories=3)
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add anima/narrator.py tests/test_narrator.py anima/persona.py
git commit -m "feat: replace flat node lists with narrative experience context

narrate_subgraph follows CAUSAL/SOLVED_BY edges to build stories like
'你之前做过X任务，遇到Y问题，通过Z解决' instead of listing nodes."
```

---

### Task 13: Graph-Derived CompetenceEmbedding

**Purpose:** Derive competence scores from graph topology (not just strategy stats), and generate specific identity narratives (not abstract percentages).

**Files:**
- Modify: `anima/competence.py`
- Add tests to: `tests/test_experience_graph.py`

- [ ] **Step 1: Write test for topology-based competence**

```python
# Add to tests/test_experience_graph.py
class TestGraphTopologyStats:
    def test_domain_density(self):
        g = ExperienceGraph()
        t1 = g.add_node(NodeType.TASK, "任务1", metadata={"category": "data_analysis"})
        c1 = g.add_node(NodeType.CONCEPT, "用户留存")
        c2 = g.add_node(NodeType.CONCEPT, "渠道分析")
        g.add_edge(t1.id, c1.id, EdgeType.COMPOSED_OF)
        g.add_edge(t1.id, c2.id, EdgeType.COMPOSED_OF)
        g.add_edge(c1.id, c2.id, EdgeType.SIMILAR)

        stats = g.get_topology_stats()
        da_stats = stats.get("data_analysis", {})
        assert da_stats.get("node_count", 0) >= 1
        assert da_stats.get("edge_density", 0) > 0
```

- [ ] **Step 2: Add get_topology_stats to ExperienceGraph**

```python
# In experience_graph.py, add to ExperienceGraph:
def get_topology_stats(self) -> dict:
    """Return per-domain topology statistics for competence derivation."""
    domain_nodes = {}  # domain -> set of node ids
    for node in self.nodes.values():
        domain = node.metadata.get("category", "unknown")
        if node.node_type == NodeType.TASK:
            domain_nodes.setdefault(domain, set()).add(node.id)
            for neighbor_id, edge in self._forward.get(node.id, []):
                domain_nodes[domain].add(neighbor_id)

    results = {}
    for domain, node_ids in domain_nodes.items():
        edge_count = sum(
            1 for e in self.edges
            if e.source_id in node_ids and e.target_id in node_ids
        )
        node_count = len(node_ids)
        results[domain] = {
            "node_count": node_count,
            "edge_count": edge_count,
            "edge_density": edge_count / max(node_count, 1),
            "concept_count": sum(
                1 for nid in node_ids
                if self.nodes.get(nid, None) and
                   self.nodes[nid].node_type == NodeType.CONCEPT
            ),
        }
    return results
```

- [ ] **Step 3: Update CompetenceEmbedding to use topology stats**

In `competence.py`, modify `update_from_graph_and_strategy`:

```python
def update_from_graph_and_strategy(self, graph_stats: dict,
                                    strategy_summary: dict,
                                    topology_stats: dict = None):
    self.last_updated = time.time()

    # From strategy network: success rates (existing logic, keep as-is)
    category_to_competence = { ... }  # unchanged
    total_attempts = 0
    for cat, info in strategy_summary.items():
        if cat in category_to_competence:
            comp_dim = category_to_competence[cat]
            attempts = info.get("attempts", 0)
            success_rate = info.get("success_rate", 0)
            total_attempts += attempts
            experience_weight = min(1.0, attempts / 10)
            self.competence_scores[comp_dim] = experience_weight * success_rate

    self.total_experience = total_attempts

    # NEW: From graph topology
    if topology_stats:
        for domain, topo in topology_stats.items():
            if domain in category_to_competence:
                dim = category_to_competence[domain]
                # Blend topology signal with strategy signal
                density_score = min(1.0, topo.get("edge_density", 0) / 3.0)
                concept_score = min(1.0, topo.get("concept_count", 0) / 20.0)
                current = self.competence_scores.get(dim, 0)
                # Topology contributes up to 30% of the score
                self.competence_scores[dim] = 0.7 * current + 0.3 * (
                    0.5 * density_score + 0.5 * concept_score)

        total_concepts = sum(t.get("concept_count", 0)
                             for t in topology_stats.values())
        self.competence_scores["domain_depth"] = min(1.0, total_concepts / 50)

    # Style inference (existing logic)
    # ... unchanged ...

    self._update_domain_tags()
```

Update `PersonaLayer.learn_from_feedback` to pass topology:

```python
def learn_from_feedback(self, ...):
    # ... existing code ...
    graph_stats = self.experience_graph.get_stats()
    topology_stats = self.experience_graph.get_topology_stats()
    strategy_summary = self.strategy_network.get_profile_summary()
    self.competence.update_from_graph_and_strategy(
        graph_stats, strategy_summary, topology_stats)
```

- [ ] **Step 4: Improve identity prompt to use specifics from graph**

In `competence.py`, modify `generate_identity_prompt` to accept optional experience details:

```python
def generate_identity_prompt(self, experience_highlights: list[str] = None) -> str:
    confidence = self.get_confidence()
    if confidence < 0.1:
        return ("这是一个新的Agent，还没有积累���够的经验来形成明确的能力画像。"
                "请根据任务需求灵活应对。")

    lines = ["## 当前Agent能力画像\n"]

    # ... existing strong/developing logic unchanged ...

    # NEW: Add specific experience highlights
    if experience_highlights:
        lines.append("\n**关键经验：**")
        for highlight in experience_highlights[:5]:
            lines.append(f"  - {highlight}")

    # ... existing style description unchanged ...

    return "\n".join(lines)
```

In `PersonaLayer._build_system_prompt`, pass highlights:

```python
# Derive highlights from recent successful tasks in graph
highlights = self._get_experience_highlights()
identity_context = self.competence.generate_identity_prompt(highlights)

def _get_experience_highlights(self) -> list[str]:
    """Extract specific experience highlights for identity prompt."""
    highlights = []
    task_nodes = self.experience_graph.find_by_type(NodeType.TASK)
    for task in task_nodes[-10:]:  # recent tasks
        for neighbor_id, edge in self.experience_graph._forward.get(task.id, []):
            neighbor = self.experience_graph.nodes.get(neighbor_id)
            if neighbor and neighbor.node_type == NodeType.FEEDBACK:
                if "成功" in neighbor.content:
                    highlights.append(task.content)
                    break
    return highlights[-5:]
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add anima/competence.py anima/experience_graph.py anima/persona.py tests/
git commit -m "feat: derive CompetenceEmbedding from graph topology, add experience highlights

- Competence scores now blend strategy success rates with graph density/concept count
- Identity prompt includes specific experience highlights, not just percentages
- Added get_topology_stats() to ExperienceGraph"
```

---

### Task 14: Update __init__.py and Final Integration

**Files:**
- Modify: `anima/__init__.py`
- Update: `demo.py` (fix double-recording, add semantic option)

- [ ] **Step 1: Update exports**

```python
# anima/__init__.py
"""
Anima - 有灵魂的AI Agent框架
"""

from .experience_graph import ExperienceGraph, NodeType, EdgeType
from .strategy import StrategyNetwork, TaskCategory, ActionType
from .competence import CompetenceEmbedding
from .persona import PersonaLayer
from .agent import AnimaAgent
from .embedding import EmbeddingProvider, MockEmbeddingProvider
from .extractor import ExperienceExtractor, MockExtractor
from .narrator import narrate_subgraph

try:
    from .embedding import AnthropicEmbeddingProvider
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
    "EmbeddingProvider", "MockEmbeddingProvider",
    "ExperienceExtractor", "MockExtractor",
    "narrate_subgraph",
]
```

- [ ] **Step 2: Add configure_semantic to AnimaAgent**

In `agent.py`:

```python
def configure_semantic(self, embedding_provider=None, extractor=None):
    """Enable semantic layer with embedding and LLM extraction."""
    self.persona.configure_semantic(embedding_provider, extractor)
    print(f"[Anima] 已启用语义层")
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add anima/__init__.py anima/agent.py
git commit -m "feat: export semantic layer APIs, add configure_semantic to AnimaAgent"
```

- [ ] **Step 5: Final commit — update CLAUDE.md**

Update CLAUDE.md with new architecture information (new files, semantic layer, jieba dependency, test commands).

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with Phase 2 architecture"
```

---

## Post-Plan: What Comes Next (Phase 3+)

These are **not part of this plan** but are the natural next steps once Phase 1+2 are solid:

1. **PATTERN Node Consolidation** — Periodically scan for repeated PROBLEM→SOLUTION pairs across tasks, create PATTERN nodes that abstract common strategies. Requires sufficient graph data to be meaningful.

2. **Robustness Against Feedback Noise** — Add anomaly detection: if a feedback reward is > 2 standard deviations from the running mean for that category, dampen its learning impact.

3. **Real LLM Integration Test** — Write a demo that actually calls Claude/GPT to process a task, using the PersonaLayer context. Measure whether two diverged agents produce qualitatively different outputs on the same task.

4. **Topology-Based Competence V2** — Causal chain depth, cross-domain bridge count, pattern abstraction ratio as competence signals.

5. **Agent Comparison Dashboard** — Visualize two agents' graph topologies side-by-side, showing structural differences.
