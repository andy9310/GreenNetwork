# topology.py
import json
import random
from typing import Optional
import networkx as nx

class GraphGenerator:
    """
    負責根據設定產生一個「連通」的隨機拓樸。
    """

    def __init__(
        self,
        topo_config: Optional[str] = None,
        num_nodes: int = 40,
        num_edges: int = 61,
        seed: int = 0,
    ):
        self.topo_config = topo_config
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.seed = seed

        # 基本檢查
        if self.num_edges < self.num_nodes - 1:
            raise ValueError(
                f"num_edges={self.num_edges} 太少，至少要 num_nodes-1={self.num_nodes - 1} "
                "才有辦法產生連通圖"
            )

    def generate(self) -> nx.Graph:
        """
        產生一個 random connected graph：
        1. 先建立一棵 random spanning tree (N-1 edges) → 保證連通
        2. 再隨機加邊到 num_edges
        """
        rng = random.Random(self.seed)

        g = nx.Graph()
        g.add_nodes_from(range(self.num_nodes))

        # --- Step 1: random spanning tree ---
        nodes = list(g.nodes())
        rng.shuffle(nodes)

        # 把第 i 個 node 接到前面 [0..i-1] 任一個節點上 → 一定連通
        for i in range(1, self.num_nodes):
            u = nodes[i]
            v = rng.choice(nodes[:i])
            g.add_edge(u, v)

        # --- Step 2: 隨機補邊到 num_edges ---
        current_edges = g.number_of_edges()
        target_edges = self.num_edges

        while current_edges < target_edges:
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u == v:
                continue
            if g.has_edge(u, v):
                continue
            g.add_edge(u, v)
            current_edges += 1

        return g

