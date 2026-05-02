"""
经验图谱 (Experience Graph)

核心理念：Agent的记忆不是一个平面的数据库，而是一个有拓扑结构的图。
节点是经历过的事件，边是它们之间的关系。
新经验的加入会重组局部结构（学习），而不仅仅是"加一条记录"。
面对新任务时，通过激活扩散(spreading activation)找到相关经验。

这是Agent"想起什么"的机制——不同的图拓扑导致不同的激活模式，
从而让不同Agent面对同一问题时表现出不同的行为。
"""

import json
import time
import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class NodeType(Enum):
    TASK = "task"           # 一次具体的任务
    CONCEPT = "concept"     # 一个概念/知识点
    SKILL = "skill"         # 一个技能
    PROBLEM = "problem"     # 遇到的问题
    SOLUTION = "solution"   # 解决方案
    FEEDBACK = "feedback"   # 主人的反馈
    PATTERN = "pattern"     # 归纳出的模式（高阶节点）


class EdgeType(Enum):
    CAUSAL = "causal"           # 因果关系 A导致了B
    SIMILAR = "similar"         # 相似关系
    TEMPORAL = "temporal"       # 时间顺序
    COMPOSED_OF = "composed_of" # 组成关系
    SOLVED_BY = "solved_by"     # 问题-解决方案
    REQUIRES = "requires"       # 依赖关系
    CONFLICTS = "conflicts"     # 冲突关系
    REINFORCES = "reinforces"   # 强化关系


# 有向边：激活只沿 source→target 方向传播
DIRECTED_EDGE_TYPES = {
    EdgeType.CAUSAL,
    EdgeType.TEMPORAL,
    EdgeType.SOLVED_BY,
    EdgeType.REQUIRES,
    EdgeType.COMPOSED_OF,
}

# 无向边：激活双向传播
UNDIRECTED_EDGE_TYPES = {
    EdgeType.SIMILAR,
    EdgeType.CONFLICTS,
    EdgeType.REINFORCES,
}


@dataclass
class Node:
    id: str
    node_type: NodeType
    content: str                            # 节点的文本描述
    embedding: list = field(default_factory=list)  # 语义向量
    activation: float = 0.0                 # 当前激活值
    strength: float = 1.0                   # 节点强度（被激活越多越强）
    created_at: float = field(default_factory=time.time)
    last_activated: float = 0.0
    activation_count: int = 0               # 被激活的总次数
    metadata: dict = field(default_factory=dict)

    def decay(self, current_time: float, decay_rate: float = 0.001):
        """时间衰减——长期不被激活的节点强度会降低"""
        time_delta = current_time - self.last_activated
        if self.last_activated > 0:
            self.strength *= math.exp(-decay_rate * time_delta / 3600)  # 按小时衰减
            self.strength = max(0.01, self.strength)  # 不完全消失


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0                 # 连接强度
    created_at: float = field(default_factory=time.time)
    co_activation_count: int = 0        # 共同被激活的次数（赫布学习）


class ExperienceGraph:
    """
    经验图谱的核心实现。

    关键设计：
    1. 激活扩散(Spreading Activation) - 从种子节点开始，沿边扩散激活值
    2. 赫布学习(Hebbian Learning) - "一起激活的节点，连接更强"
    3. 结构重组(Structural Reorganization) - 新经验可能创建新的边或模式节点
    4. 遗忘机制(Forgetting) - 长期不被激活的节点和边会衰减
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._forward: dict[str, list[tuple[str, Edge]]] = {}   # source→target
        self._backward: dict[str, list[tuple[str, Edge]]] = {}  # target→source

    def _generate_id(self, content: str, node_type: str) -> str:
        raw = f"{node_type}:{content}:{time.time()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def add_node(self, node_type: NodeType, content: str,
                 embedding: list = None, metadata: dict = None) -> Node:
        """添加一个新的经验节点"""
        node_id = self._generate_id(content, node_type.value)
        node = Node(
            id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding or [],
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        self._forward.setdefault(node_id, [])
        self._backward.setdefault(node_id, [])
        return node

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType, weight: float = 1.0) -> Optional[Edge]:
        """添加一条关系边"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        # 检查是否已存在同类型的边（有向边只查 source→target）
        for _, edge in self._forward.get(source_id, []):
            if edge.target_id == target_id and edge.edge_type == edge_type:
                edge.weight = min(edge.weight + 0.1, 5.0)  # 强化已有边
                return edge
        # 无向边还需检查反向是否已存在
        if edge_type in UNDIRECTED_EDGE_TYPES:
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

    def spreading_activation(self, seed_ids: list[str],
                             initial_activation: float = 1.0,
                             decay_factor: float = 0.4,
                             max_depth: int = 3,
                             min_activation: float = 0.1) -> list[tuple[Node, float]]:
        """
        激活扩散算法——Agent"想起"相关经验的核心机制。

        从种子节点出发，沿着边向外扩散激活值。
        每经过一条边，激活值乘以 decay_factor * edge_weight * node_strength。
        最终返回所有被激活的节点及其激活值，按激活值排序。

        这个算法决定了Agent面对一个新任务时"想起"什么——
        不同的图拓扑会导致完全不同的激活结果。
        """
        # 重置所有激活值
        for node in self.nodes.values():
            node.activation = 0.0

        # 设置种子节点
        for seed_id in seed_ids:
            if seed_id in self.nodes:
                self.nodes[seed_id].activation = initial_activation

        # BFS扩散 — filter out non-existent seed ids
        valid_seeds = {sid for sid in seed_ids if sid in self.nodes}
        current_layer = set(valid_seeds)
        visited = set(valid_seeds)

        for depth in range(max_depth):
            next_layer = set()
            for node_id in current_layer:
                current_activation = self.nodes[node_id].activation
                if current_activation < min_activation:
                    continue

                for neighbor_id, edge in self._forward.get(node_id, []):
                    # 激活传播量 = 当前激活值 × 衰减因子 × 边权重 × 目标节点强度
                    propagated = (current_activation * decay_factor *
                                  edge.weight * self.nodes[neighbor_id].strength)

                    if propagated >= min_activation:
                        # 累加而非替换（多条路径可以汇聚），但设上限
                        self.nodes[neighbor_id].activation = min(
                            self.nodes[neighbor_id].activation + propagated,
                            initial_activation * 3.0  # 最多3倍初始值
                        )
                        next_layer.add(neighbor_id)

            visited.update(next_layer)
            current_layer = next_layer

        # 收集所有被激活的节点
        activated = []
        current_time = time.time()
        for node_id in visited:
            node = self.nodes[node_id]
            if node.activation >= min_activation:
                node.last_activated = current_time
                node.activation_count += 1
                activated.append((node, node.activation))

        activated.sort(key=lambda x: x[1], reverse=True)
        return activated

    def hebbian_update(self, activated_nodes: list[tuple[Node, float]],
                       learning_rate: float = 0.1):
        """
        赫布学习——"一起激活的节点，连接更强"

        每次任务完成后调用，根据共同被激活的节点来强化或创建边。
        这是图谱"自我重组"的关键机制。
        """
        activated_ids = {node.id for node, _ in activated_nodes}

        # 强化已有边
        for edge in self.edges:
            if edge.source_id in activated_ids and edge.target_id in activated_ids:
                edge.co_activation_count += 1
                edge.weight = min(edge.weight + learning_rate, 5.0)

        # 对于共同激活但没有直接边的高激活节点，考虑创建新边
        high_activation = [(n, a) for n, a in activated_nodes if a > 0.3]
        for i in range(len(high_activation)):
            for j in range(i + 1, len(high_activation)):
                node_i, act_i = high_activation[i]
                node_j, act_j = high_activation[j]

                # 检查是否已有边
                has_edge = False
                for _, edge in self._forward.get(node_i.id, []):
                    if edge.target_id == node_j.id or edge.source_id == node_j.id:
                        has_edge = True
                        break

                if not has_edge:
                    # 创建新的相似性边（这就是"结构重组"）
                    combined_activation = act_i * act_j
                    if combined_activation > 0.5:
                        self.add_edge(node_i.id, node_j.id,
                                      EdgeType.SIMILAR,
                                      weight=combined_activation)

    def find_by_embedding(self, query_embedding: list[float],
                          top_k: int = 5,
                          min_similarity: float = 0.1) -> list[tuple["Node", float]]:
        """使用余弦相似度查找与 query_embedding 最相似的节点。

        Find nodes most similar to query_embedding using cosine similarity.
        Only considers nodes that have embeddings stored; skips the rest.

        Args:
            query_embedding: 查询向量
            top_k: 返回前 K 个结果
            min_similarity: 相似度阈值，低于此值的节点不返回

        Returns:
            list of (Node, similarity_score) sorted by descending similarity
        """
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

    def find_by_type(self, node_type: NodeType) -> list[Node]:
        """按类型查找节点"""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def find_by_content(self, keywords: list[str]) -> list[Node]:
        """按关键词查找节点（简单文本匹配）"""
        results = []
        for node in self.nodes.values():
            score = sum(1 for kw in keywords if kw.lower() in node.content.lower())
            if score > 0:
                results.append((node, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in results]

    def extract_subgraph(self, activated: list[tuple[Node, float]],
                         max_nodes: int = 10) -> str:
        """
        将激活的子图转化为文本描述，用于注入到LLM的上下文中。
        这是经验图谱与大模型之间的接口。
        """
        top_nodes = activated[:max_nodes]
        if not top_nodes:
            return "没有找到相关经验。"

        lines = ["以下是与当前任务相关的历史经验（按相关度排序）：\n"]
        for i, (node, activation) in enumerate(top_nodes, 1):
            relevance = "高度相关" if activation > 0.5 else "中度相关" if activation > 0.2 else "可能相关"
            lines.append(f"{i}. [{node.node_type.value}] {node.content} ({relevance})")

            # 找到与此节点相关的边，提供关系上下文
            related = []
            for neighbor_id, edge in self._forward.get(node.id, []):
                neighbor = self.nodes.get(neighbor_id)
                if neighbor and any(n.id == neighbor_id for n, _ in top_nodes):
                    related.append(f"  → {edge.edge_type.value}: {neighbor.content}")
            for r in related[:3]:
                lines.append(r)

        return "\n".join(lines)

    def decay_all(self, decay_rate: float = 0.001):
        """对所有节点执行时间衰减——遗忘机制"""
        current_time = time.time()
        for node in self.nodes.values():
            node.decay(current_time, decay_rate)

        # 衰减边
        for edge in self.edges:
            if edge.co_activation_count == 0:
                edge.weight *= 0.99  # 从未共同激活的边缓慢衰减

    def get_topology_stats(self) -> dict:
        """Per-domain topology statistics for competence derivation."""
        domain_nodes = {}
        for node in self.nodes.values():
            domain = node.metadata.get("category")
            if node.node_type == NodeType.TASK and domain:
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
            concept_count = sum(
                1 for nid in node_ids
                if nid in self.nodes and self.nodes[nid].node_type == NodeType.CONCEPT
            )
            results[domain] = {
                "node_count": node_count,
                "edge_count": edge_count,
                "edge_density": edge_count / max(node_count, 1),
                "concept_count": concept_count,
            }
        return results

    def get_stats(self) -> dict:
        """返回图谱的统计信息"""
        type_counts = {}
        for node in self.nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": type_counts,
            "avg_strength": (sum(n.strength for n in self.nodes.values()) /
                             max(len(self.nodes), 1)),
            "avg_edge_weight": (sum(e.weight for e in self.edges) /
                                max(len(self.edges), 1)),
        }

    def to_dict(self) -> dict:
        """序列化（持久化存储）"""
        return {
            "nodes": {nid: {
                "id": n.id, "node_type": n.node_type.value,
                "content": n.content, "embedding": n.embedding,
                "strength": n.strength, "created_at": n.created_at,
                "last_activated": n.last_activated,
                "activation_count": n.activation_count,
                "metadata": n.metadata,
            } for nid, n in self.nodes.items()},
            "edges": [{
                "source_id": e.source_id, "target_id": e.target_id,
                "edge_type": e.edge_type.value, "weight": e.weight,
                "created_at": e.created_at,
                "co_activation_count": e.co_activation_count,
            } for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceGraph":
        """反序列化"""
        graph = cls()
        for nid, nd in data["nodes"].items():
            node = Node(
                id=nd["id"], node_type=NodeType(nd["node_type"]),
                content=nd["content"], embedding=nd.get("embedding", []),
                strength=nd.get("strength", 1.0),
                created_at=nd.get("created_at", 0),
                last_activated=nd.get("last_activated", 0),
                activation_count=nd.get("activation_count", 0),
                metadata=nd.get("metadata", {}),
            )
            graph.nodes[nid] = node
            graph._forward.setdefault(nid, [])
            graph._backward.setdefault(nid, [])

        for ed in data["edges"]:
            edge = Edge(
                source_id=ed["source_id"], target_id=ed["target_id"],
                edge_type=EdgeType(ed["edge_type"]), weight=ed["weight"],
                created_at=ed.get("created_at", 0),
                co_activation_count=ed.get("co_activation_count", 0),
            )
            graph.edges.append(edge)

            graph._forward.setdefault(edge.source_id, []).append(
                (edge.target_id, edge))
            graph._backward.setdefault(edge.target_id, []).append(
                (edge.source_id, edge))

            if edge.edge_type in UNDIRECTED_EDGE_TYPES:
                graph._forward.setdefault(edge.target_id, []).append(
                    (edge.source_id, edge))
                graph._backward.setdefault(edge.source_id, []).append(
                    (edge.target_id, edge))

        return graph
