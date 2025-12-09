# envs/env_dual_action.py
"""
SDN Environment with Dual Actions:
1. Edge Control: Open/Close edges (binary per edge)
2. Path Selection: Choose routing paths for flows (categorical per flow)
"""
import numpy as np
import networkx as nx
import random
import json
from typing import Dict, Any, Tuple, List, Set
from pathlib import Path

from env_util.graph import GraphGenerator
from env_util.path_utils import (
    compute_all_flow_paths, 
    route_flows_on_paths, 
    compute_edge_metrics
)


class BoxSpec:
    def __init__(self, shape):
        self.shape = shape


class DiscreteSpec:
    def __init__(self, n: int):
        self.n = n


class DualActionSDNEnv:
    """
    SDN Environment with dual action space:
    - Edge control: binary decision per edge (open/close)
    - Path selection: select from k-shortest paths per flow
    
    Reward penalizes:
    - Energy consumption (closed edges save energy)
    - Delay (function of utilization)
    - Overload (utilization > 1.0)
    - Disconnection (no valid path for flow)
    """

    def __init__(self, config_path: str = None):
        """
        Initialize environment from config file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "train_config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract config parameters
        self.num_flows_per_step = self.config.get('num_flows_per_step', 5)
        self.base_capacity = self.config.get('base_capacity', 10.0)
        self.base_power = self.config.get('base_power', 1.0)
        self.base_delay = self.config.get('base_delay', 1.0)
        self.w_energy = self.config.get('w_energy', 0.1)
        self.w_delay = self.config.get('w_delay', 1.0)
        self.w_overload = self.config.get('w_overload', 2.0)
        self.max_steps = self.config.get('max_steps', 50)
        self.k_paths = self.config.get('k_paths', 3)  # Number of alternative paths
        seed = self.config.get('graph_seed', 42)
        
        # Initialize RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        # Generate or load graph topology
        topo_config_path = Path(__file__).parent.parent / "config" / "topo_config.json"
        if topo_config_path.exists():
            with open(topo_config_path, 'r') as f:
                topo_config = json.load(f)
        else:
            topo_config = None
        
        self.graph_generator = GraphGenerator(
            topo_config=topo_config,
            seed=seed,
        )
        self.graph = self.graph_generator.generate()
        
        # Graph properties
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)
        self.edges = list(self.graph.edges())
        self.num_edges = len(self.edges)
        
        # Create edge index mapping
        self.edge_to_idx = {}
        self.idx_to_edge = {}
        for idx, (u, v) in enumerate(self.edges):
            edge_key = self._edge_key(u, v)
            self.edge_to_idx[edge_key] = idx
            self.idx_to_edge[idx] = edge_key
        
        # Compute max degree for observation padding
        degrees = dict(self.graph.degree())
        self.max_degree = max(degrees.values())
        
        # Action space dimensions
        self.edge_action_dim = self.num_edges  # Binary per edge
        self.flow_action_dim = self.num_flows_per_step * self.k_paths  # Categorical per flow
        
        # Observation dimension (global state for centralized control)
        # Global features: [num_edges] edge utilizations + [num_edges] edge states + [num_flows] flow demands (normalized)
        self.obs_dim = self.num_edges * 2 + self.num_flows_per_step
        
        # Environment state
        self.step_count = 0
        self.edge_states: Dict[Tuple[int, int], int] = {}  # 0=closed, 1=open
        self.edge_load: Dict[Tuple[int, int], float] = {}
        self.edge_capacity: Dict[Tuple[int, int], float] = {}
        self.current_flows: List[Tuple[int, int, float]] = []
        
        print(f"[DualActionSDNEnv] Initialized")
        print(f"  Nodes: {self.num_nodes}, Edges: {self.num_edges}")
        print(f"  Flows per step: {self.num_flows_per_step}, K-paths: {self.k_paths}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dims: edges={self.edge_action_dim}, flows={self.flow_action_dim}")

    def _edge_key(self, u: int, v: int) -> Tuple[int, int]:
        """Canonical edge key (smaller node first)"""
        return (u, v) if u < v else (v, u)

    def reset(self) -> np.ndarray:
        """
        Reset environment.
        Returns:
            Global observation (centralized)
        """
        self.step_count = 0
        
        # Initialize all edges as open with base capacity
        self.edge_states = {}
        self.edge_load = {}
        self.edge_capacity = {}
        for u, v in self.edges:
            key = self._edge_key(u, v)
            self.edge_states[key] = 1  # Open
            self.edge_load[key] = 0.0
            self.edge_capacity[key] = self.base_capacity
        
        # Generate initial flows
        self.current_flows = self._generate_flows()
        
        return self._get_observation()

    def _generate_flows(self) -> List[Tuple[int, int, float]]:
        """Generate random traffic flows"""
        flows = []
        for _ in range(self.num_flows_per_step):
            s = self.rng.choice(self.nodes)
            d = self.rng.choice(self.nodes)
            while d == s:
                d = self.rng.choice(self.nodes)
            
            # Demand: normal distribution around base_capacity / 4
            demand = max(
                0.1,
                float(
                    self.np_rng.normal(
                        self.base_capacity / 4.0,
                        self.base_capacity / 10.0
                    )
                ),
            )
            flows.append((s, d, demand))
        return flows

    def _get_observation(self) -> np.ndarray:
        """
        Get global observation (centralized controller view).
        
        Observation structure:
        - Edge utilizations: [num_edges]
        - Edge states (open/closed): [num_edges]
        - Flow demands (normalized): [num_flows]
        """
        # Edge utilizations
        edge_utils = []
        for idx in range(self.num_edges):
            edge_key = self.idx_to_edge[idx]
            load = self.edge_load.get(edge_key, 0.0)
            capacity = self.edge_capacity.get(edge_key, self.base_capacity)
            util = load / capacity if capacity > 0 else 0.0
            edge_utils.append(min(util, 2.0))  # Clip for stability
        
        # Edge states
        edge_states_list = []
        for idx in range(self.num_edges):
            edge_key = self.idx_to_edge[idx]
            state = self.edge_states.get(edge_key, 1)
            edge_states_list.append(float(state))
        
        # Flow demands (normalized by base_capacity)
        flow_demands = [demand / self.base_capacity for _, _, demand in self.current_flows]
        
        obs = np.concatenate([
            np.array(edge_utils, dtype=np.float32),
            np.array(edge_states_list, dtype=np.float32),
            np.array(flow_demands, dtype=np.float32),
        ])
        
        return obs

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step with dual actions.
        
        Args:
            actions: Dictionary with:
                - 'edge_control': [num_edges] binary array (0=close, 1=open)
                - 'path_selection': [num_flows] integer array (path indices 0 to k-1)
        
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # Parse actions
        edge_actions = actions['edge_control']  # [num_edges] binary
        path_actions = actions['path_selection']  # [num_flows] indices
        
        # Update edge states based on edge control actions
        for idx in range(self.num_edges):
            edge_key = self.idx_to_edge[idx]
            self.edge_states[edge_key] = int(edge_actions[idx])
        
        # Get closed edges
        closed_edges = {key for key, state in self.edge_states.items() if state == 0}
        
        # Compute k-shortest paths for all flows (considering closed edges)
        flow_paths = compute_all_flow_paths(
            self.graph, 
            self.current_flows, 
            k=self.k_paths,
            closed_edges=closed_edges
        )
        
        # Route flows based on path selection actions
        path_selections = {flow_id: int(path_actions[flow_id]) for flow_id in range(len(self.current_flows))}
        
        edge_loads, disconnect_penalty = route_flows_on_paths(
            self.current_flows,
            flow_paths,
            path_selections,
            self.edge_load
        )
        
        self.edge_load = edge_loads
        
        # Compute metrics
        total_energy, total_delay, total_overload = compute_edge_metrics(
            self.edge_load,
            self.edge_capacity,
            self.edge_states,
            self.base_power,
            self.base_delay
        )
        
        total_overload += disconnect_penalty
        
        # Compute reward (negative cost, normalized by number of edges)
        cost = (
            self.w_energy * total_energy +
            self.w_delay * total_delay +
            self.w_overload * total_overload
        ) / max(self.num_edges, 1)
        
        reward = -float(cost)
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Generate new flows for next step
        if not done:
            self.current_flows = self._generate_flows()
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Info
        info = {
            'energy': total_energy,
            'delay': total_delay,
            'overload': total_overload,
            'num_closed_edges': sum(1 for s in self.edge_states.values() if s == 0),
            'disconnect_penalty': disconnect_penalty,
        }
        
        return next_obs, reward, done, info

    @property
    def action_space(self):
        """Return action space specification"""
        return {
            'edge_control': self.edge_action_dim,
            'path_selection': (self.num_flows_per_step, self.k_paths)
        }
    
    @property
    def observation_space(self):
        """Return observation space specification"""
        return BoxSpec((self.obs_dim,))
