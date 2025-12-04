# envs/routing_env.py
import numpy as np
import networkx as nx
import random
import json
from typing import Dict, Any, Tuple
from env_util.graph import GraphGenerator 

class BoxSpec:
    def __init__(self, shape):
        self.shape = shape

class DiscreteSpec:
    def __init__(self, n: int):
        self.n = n

class SDNEnv:
    """
    - Traffic is generated between random source/destination pairs. (traffic matix)
    - Routing uses shortest paths over the topology (all links usable).
    - Link load / utilization is computed from flows.
    - Reward (shared by all agents) penalizes:
        * total energy consumption
        * delay (function of utilization)
        * overload (utilization > 1)
    """

    def __init__(
        self,
        num_flows_per_step: int = 5,
        base_capacity: float = 10.0,
        base_power: float = 1.0,
        base_delay: float = 1.0,
        w_energy: float = 0.1,
        w_delay: float = 1.0,
        w_overload: float = 2.0,
        max_steps: int = 50,
        seed: int = 0,
        topo_config: dict | None = None,  # <-- 新增：dict
        num_nodes: int = 40,              # <-- 新增：預設節點數
        num_edges: int = 61,              # <-- 新增：預設邊數
    ):
        self.graph_generator = GraphGenerator(
            topo_config=topo_config,
            seed=seed,
        )
        self.graph = self.graph_generator.generate()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.num_flows_per_step = num_flows_per_step
        self.base_capacity = base_capacity
        self.base_power = base_power
        self.base_delay = base_delay
        self.w_energy = w_energy
        self.w_delay = w_delay
        self.w_overload = w_overload
        self.max_steps = max_steps

        # nodes are labeled as integers in the graph
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)
        self.agent_ids = self.nodes  # use node ids directly as agent ids

        # precompute max degree for obs padding
        degrees = dict(self.graph.degree())
        self.max_degree = max(degrees.values())

        # action space: 
        # 1. open/close
        # 2. 
        self.act_dim = 3
        # observation dimension:
        #   deg_norm (1) + mode_norm (1) + local_load (1) +
        #   per-edge features: (util, overload_flag) * max_degree
        self.obs_dim = 3 + 2 * self.max_degree

        # environment state
        self.step_count = 0
        self.node_modes: Dict[int, int] = {}
        # edge keys are (min(u,v), max(u,v))
        self.edge_load: Dict[Tuple[int, int], float] = {}
        self.edge_capacity: Dict[Tuple[int, int], float] = {}

        # mode factors
        self.mode_capacity_factor = {0: 0.5, 1: 1.0, 2: 1.5}
        self.mode_power_factor = {0: 0.5, 1: 1.0, 2: 1.5}

    


    @property
    def possible_agents(self):
        return self.agent_ids

    def observation_space(self, agent_id):
        return BoxSpec((self.obs_dim,))

    def action_space(self, agent_id):
        return DiscreteSpec(self.act_dim)

    def reset(self) -> Dict[int, np.ndarray]:
        self.step_count = 0
        # all nodes start in normal mode
        self.node_modes = {i: 1 for i in self.nodes}
        # reset loads and capacities
        self.edge_load = {}
        self.edge_capacity = {}
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            self.edge_load[key] = 0.0
            # initial capacity when both modes are 1
            self.edge_capacity[key] = self.base_capacity * self.mode_capacity_factor[1]
        return self._get_obs_all()

    def step(self, actions: Dict[int, int]):
        """
        actions: dict {node_id: action_mode (0/1/2)}
        """
        self.step_count += 1

        # update node modes based on actions (clip to 0..2)
        for i in self.nodes:
            a = actions.get(i, 1)
            a = int(a)
            if a < 0 or a > 2:
                a = 1
            self.node_modes[i] = a

        # simulate traffic and link metrics
        total_energy, total_delay, total_overload = self._simulate_traffic_step()

        # compute global reward (negative cost)
        # normalize by number of edges to keep magnitude moderate
        num_edges = self.graph.number_of_edges()
        cost = (
            self.w_energy * total_energy
            + self.w_delay * total_delay
            + self.w_overload * total_overload
        ) / max(num_edges, 1)
        reward_global = -float(cost)

        rewards = {i: reward_global for i in self.nodes}

        done_flag = self.step_count >= self.max_steps
        dones = {i: done_flag for i in self.nodes}
        infos = {i: {} for i in self.nodes}

        next_obs = self._get_obs_all()
        return next_obs, rewards, dones, infos

    def _edge_key(self, u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _simulate_traffic_step(self):
        """
        Generate random flows, route them on shortest paths,
        compute link loads, utilization, energy, delay, overload.
        """
        # reset loads
        for key in self.edge_load.keys():
            self.edge_load[key] = 0.0

        # update capacities based on node modes
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            mode_u = self.node_modes[u]
            mode_v = self.node_modes[v]
            factor_u = self.mode_capacity_factor[mode_u]
            factor_v = self.mode_capacity_factor[mode_v]
            # use min of both ends to be conservative
            cap_factor = min(factor_u, factor_v)
            self.edge_capacity[key] = self.base_capacity * cap_factor

        # generate flows: random src/dst pairs with random demand
        flows = []
        for _ in range(self.num_flows_per_step):
            s = self.rng.choice(self.nodes)
            d = self.rng.choice(self.nodes)
            while d == s:
                d = self.rng.choice(self.nodes)
            # demand: positive, around base_capacity / 4
            demand = max(
                0.1,
                float(
                    self.np_rng.normal(self.base_capacity / 4.0,
                                       self.base_capacity / 10.0)
                ),
            )
            flows.append((s, d, demand))

        # route flows using shortest paths by hop count
        disconnect_penalty = 0.0
        for s, d, demand in flows:
            try:
                path = nx.shortest_path(self.graph, s, d)
            except nx.NetworkXNoPath:
                # if disconnected, add penalty
                disconnect_penalty += demand * 10.0
                continue

            # accumulate load on each edge along the path
            for u, v in zip(path[:-1], path[1:]):
                key = self._edge_key(u, v)
                self.edge_load[key] += demand

        total_energy = 0.0
        total_delay = 0.0
        total_overload = disconnect_penalty

        # compute per-edge metrics
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            load = self.edge_load[key]
            capacity = self.edge_capacity[key]
            if capacity <= 0.0:
                # if capacity is zero, treat as heavy overload
                util = 10.0
            else:
                util = load / capacity

            # overload when util > 1
            overload = max(0.0, util - 1.0)
            # simple convex delay: base_delay * (1 + util^2)
            delay = self.base_delay * (1.0 + util * util)

            mode_u = self.node_modes[u]
            mode_v = self.node_modes[v]
            power_factor = 0.5 * (
                self.mode_power_factor[mode_u] + self.mode_power_factor[mode_v]
            )
            # energy = base_power * factor * (1 + 0.5 * util)
            energy = self.base_power * power_factor * (1.0 + 0.5 * util)

            total_energy += energy
            total_delay += delay
            total_overload += overload * 5.0  # scale up overload cost
        return total_energy, total_delay, total_overload

    def _get_obs_all(self) -> Dict[int, np.ndarray]:
        return {i: self._get_obs_for_node(i) for i in self.nodes}

    def _get_obs_for_node(self, i: int) -> np.ndarray:
        neighbors = sorted(list(self.graph.neighbors(i)))
        deg_i = len(neighbors)
        deg_norm = deg_i / max(self.max_degree, 1)

        mode_i = self.node_modes.get(i, 1)
        mode_norm = mode_i / 2.0  # 0,0.5,1

        # Compute local aggregate load
        local_load = 0.0
        for j in neighbors:
            key = self._edge_key(i, j)
            local_load += self.edge_load.get(key, 0.0)
        # normalize by base_capacity * max_degree
        denom = self.base_capacity * max(self.max_degree, 1)
        local_load_norm = local_load / denom if denom > 0 else 0.0

        # per-edge features: (util, overload_flag) for up to max_degree neighbors
        edge_feats = np.zeros(2 * self.max_degree, dtype=np.float32)
        for idx, j in enumerate(neighbors):
            if idx >= self.max_degree:
                break
            key = self._edge_key(i, j)
            load = self.edge_load.get(key, 0.0)
            capacity = self.edge_capacity.get(key, self.base_capacity)
            util = load / capacity if capacity > 0 else 0.0
            util_clipped = float(min(util, 2.0))  # clip for stability
            overload_flag = 1.0 if util > 1.0 else 0.0

            edge_feats[2 * idx] = util_clipped
            edge_feats[2 * idx + 1] = overload_flag

        obs = np.concatenate(
            [
                np.array([deg_norm, mode_norm, local_load_norm], dtype=np.float32),
                edge_feats,
            ]
        )
        assert obs.shape[0] == self.obs_dim
        return obs
