# envs/routing_env.py
import numpy as np
import networkx as nx
import random
import json
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
from env_util.graph import GraphGenerator
from clustering.cluster import RegionClusterer 

class BoxSpec:
    def __init__(self, shape):
        self.shape = shape

class DiscreteSpec:
    def __init__(self, n: int):
        self.n = n

class SDNEnv:
    """
    Cluster-based Multi-Agent SDN Environment with Action Masking
    
    - Agents are CLUSTERS of nodes (not individual nodes)
    - Each agent controls all nodes in its cluster
    - Action dimension is fixed based on max cluster size (with masking)
    - Traffic is generated between random source/destination pairs
    - Routing uses shortest paths over the topology
    - Reward (shared by all agents) penalizes:
        * total energy consumption
        * delay (function of utilization)
        * overload (utilization > 1)
    """

    def __init__(self, config_path: Optional[str] = None, num_clusters: Optional[int] = None):
        """
        Args:
            config_path: Path to configuration file
            num_clusters: Number of clusters for agents (if None, uses config or defaults to num_nodes/4)
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "train_config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract parameters
        self.num_flows_per_step = self.config.get('num_flows_per_step', 5)
        self.base_capacity = self.config.get('base_capacity', 10.0)
        self.base_power = self.config.get('base_power', 1.0)
        self.base_delay = self.config.get('base_delay', 1.0)
        self.w_energy = self.config.get('w_energy', 0.1)
        self.w_delay = self.config.get('w_delay', 1.0)
        self.w_overload = self.config.get('w_overload', 2.0)
        self.max_steps = self.config.get('max_steps', 50)
        seed = self.config.get('graph_seed', 42)
        
        # Initialize RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        # Generate graph topology
        topo_config_path = Path(__file__).parent.parent / "config" / "topo_config.json"
        if topo_config_path.exists():
            with open(topo_config_path, 'r') as f:
                topo_config = json.load(f)
        else:
            topo_config = None
        
        self.graph_generator = GraphGenerator(topo_config=topo_config, seed=seed)
        self.graph = self.graph_generator.generate()

        # Graph properties
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)
        degrees = dict(self.graph.degree())
        self.max_degree = max(degrees.values())
        
        # Perform clustering to create agents
        if num_clusters is None:
            num_clusters = self.config.get('num_clusters', max(1, self.num_nodes // 4))
        
        print(f"[SDNEnv] Clustering {self.num_nodes} nodes into {num_clusters} clusters...")
        clusterer = RegionClusterer(
            num_regions=num_clusters,
            random_state=seed,
            use_degree=True,
            use_local_density=True,
            use_traffic=False,
            ensure_connectivity=True  # 保证每个cluster是连通子图
        )
        self.cluster_mapping = clusterer.cluster(self.graph, traffic_profiles=None)
        
        # Group nodes by cluster
        self.cluster_to_nodes: Dict[int, List[int]] = {}
        for node, cluster_id in self.cluster_mapping.items():
            if cluster_id not in self.cluster_to_nodes:
                self.cluster_to_nodes[cluster_id] = []
            self.cluster_to_nodes[cluster_id].append(node)
        
        # Verify connectivity
        all_connected = True
        for cluster_id, nodes in self.cluster_to_nodes.items():
            subgraph = self.graph.subgraph(nodes)
            if not nx.is_connected(subgraph):
                print(f"  ⚠️ Warning: Cluster {cluster_id} is not connected!")
                all_connected = False
        
        if all_connected:
            print(f"  ✅ All {len(self.cluster_to_nodes)} clusters are connected!")
        
        # Agent IDs are cluster IDs
        self.agent_ids = sorted(list(self.cluster_to_nodes.keys()))
        self.num_agents = len(self.agent_ids)
        
        # Action space: each node can take 3 actions (low/normal/high power mode)
        self.act_dim_per_node = 3
        
        # Find max cluster size for fixed action dimension
        self.cluster_sizes = {cid: len(nodes) for cid, nodes in self.cluster_to_nodes.items()}
        self.max_cluster_size = max(self.cluster_sizes.values())
        self.min_cluster_size = min(self.cluster_sizes.values())
        
        # Fixed action dimension (max_cluster_size * actions_per_node)
        self.max_action_dim = self.max_cluster_size * self.act_dim_per_node
        
        # Observation dimension:
        # [cluster_features] + [node_features for each node in cluster (padded)] + [action_mask]
        # Cluster features: avg_degree, num_nodes, total_load
        # Per-node features: degree_norm, mode, local_load, neighbor_utils (padded to max_degree)
        self.cluster_feature_dim = 3
        self.node_feature_dim = 3 + 2 * self.max_degree  # deg_norm, mode, local_load, (util, overload)*max_degree
        self.obs_dim = (
            self.cluster_feature_dim +
            self.max_cluster_size * self.node_feature_dim +
            self.max_action_dim  # action mask
        )
        
        print(f"[SDNEnv] Created {self.num_agents} cluster agents")
        print(f"  Cluster sizes: min={self.min_cluster_size}, max={self.max_cluster_size}, avg={self.num_nodes/self.num_agents:.1f}")
        print(f"  Action dim per agent: {self.max_action_dim} (with masking)")
        print(f"  Observation dim: {self.obs_dim}")

        # Environment state
        self.step_count = 0
        self.node_modes: Dict[int, int] = {}  # {node_id: mode (0/1/2)}
        self.edge_load: Dict[Tuple[int, int], float] = {}
        self.edge_capacity: Dict[Tuple[int, int], float] = {}
        
        # Mode factors for capacity and power
        self.mode_capacity_factor = {0: 0.5, 1: 1.0, 2: 1.5}
        self.mode_power_factor = {0: 0.5, 1: 1.0, 2: 1.5}

    def get_cluster_info(self, cluster_id: int) -> Dict[str, Any]:
        """Get information about a specific cluster"""
        return {
            'cluster_id': cluster_id,
            'nodes': self.cluster_to_nodes[cluster_id],
            'size': len(self.cluster_to_nodes[cluster_id]),
            'action_dim': len(self.cluster_to_nodes[cluster_id]) * self.act_dim_per_node
        }
        
    @property
    def possible_agents(self):
        return self.agent_ids

    def observation_space(self, agent_id):
        """Observation space includes action mask"""
        return BoxSpec((self.obs_dim,))

    def action_space(self, agent_id):
        """Fixed action dimension for all agents (use masking for smaller clusters)"""
        # For discrete actions: each node in cluster chooses from 3 modes
        # Total actions: max_cluster_size positions, each with 3 choices
        # We use a flattened representation: action_idx = node_idx * 3 + mode_choice
        return DiscreteSpec(self.max_action_dim)
    
    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """
        Get action mask for a specific agent.
        Returns binary mask: 1 for valid actions, 0 for invalid (padded) actions.
        
        Args:
            agent_id: Cluster ID
        
        Returns:
            mask: Binary array of shape (max_action_dim,)
        """
        cluster_size = len(self.cluster_to_nodes[agent_id])
        valid_actions = cluster_size * self.act_dim_per_node
        
        mask = np.zeros(self.max_action_dim, dtype=np.float32)
        mask[:valid_actions] = 1.0
        return mask

    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations for all agents"""
        self.step_count = 0
        
        # Initialize all nodes to mode 1 (normal)
        self.node_modes = {n: 1 for n in self.nodes}
        
        # Reset edge loads and capacities
        self.edge_load = {}
        self.edge_capacity = {}
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            self.edge_load[key] = 0.0
            self.edge_capacity[key] = self.base_capacity * self.mode_capacity_factor[1]
        
        return self._get_obs_all()

    def step(self, actions: Dict[int, int]):
        """
        Execute one step with cluster-level actions.
        
        Args:
            actions: dict {cluster_id: action_idx}
                    action_idx is a flat index from 0 to max_action_dim-1
                    
        Returns:
            observations: dict {cluster_id: obs_vector}
            rewards: dict {cluster_id: reward}
            dones: dict {cluster_id: done_flag}
            infos: dict {cluster_id: info_dict}
        """
        self.step_count += 1
        
        # Decode actions: convert flat action index to node modes
        for cluster_id, action_idx in actions.items():
            cluster_nodes = self.cluster_to_nodes[cluster_id]
            cluster_size = len(cluster_nodes)
            
            # Decode action: which node and which mode
            # action_idx = node_position * act_dim_per_node + mode_choice
            node_position = action_idx // self.act_dim_per_node
            mode_choice = action_idx % self.act_dim_per_node
            
            # Only apply if within valid range (due to masking, should always be valid)
            if node_position < cluster_size:
                node_id = cluster_nodes[node_position]
                self.node_modes[node_id] = mode_choice
        
        # Update edge capacities based on node modes
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            mode_u = self.node_modes.get(u, 1)
            mode_v = self.node_modes.get(v, 1)
            # Use minimum capacity factor of both endpoints
            cap_factor = min(self.mode_capacity_factor[mode_u], self.mode_capacity_factor[mode_v])
            self.edge_capacity[key] = self.base_capacity * cap_factor
        
        # Simulate traffic and compute metrics
        total_energy, total_delay, total_overload = self._simulate_traffic_step()
        
        # Compute global reward (negative cost)
        num_edges = self.graph.number_of_edges()
        cost = (
            self.w_energy * total_energy +
            self.w_delay * total_delay +
            self.w_overload * total_overload
        ) / max(num_edges, 1)
        reward_global = -float(cost)
        
        # All agents share the same global reward
        rewards = {cid: reward_global for cid in self.agent_ids}
        
        # Check if episode is done
        done_flag = self.step_count >= self.max_steps
        dones = {cid: done_flag for cid in self.agent_ids}
        infos = {
            cid: {
                'total_energy': total_energy,
                'total_delay': total_delay,
                'total_overload': total_overload,
                'cost': cost
            } for cid in self.agent_ids
        }
        
        # Get next observations
        next_obs = self._get_obs_all()
        return next_obs, rewards, dones, infos

    def _get_obs_all(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents (clusters)"""
        return {cid: self._get_obs(cid) for cid in self.agent_ids}
    
    def _get_obs(self, cluster_id: int) -> np.ndarray:
        """
        Get observation for a specific cluster agent.
        Observation includes:
        1. Cluster-level features
        2. Per-node features (padded to max_cluster_size)
        3. Action mask
        """
        cluster_nodes = self.cluster_to_nodes[cluster_id]
        cluster_size = len(cluster_nodes)
        
        # 1. Cluster-level features
        avg_degree = np.mean([self.graph.degree(n) for n in cluster_nodes])
        avg_degree_norm = avg_degree / max(self.num_nodes - 1, 1)
        
        total_cluster_load = sum(
            self.edge_load.get(self._edge_key(u, v), 0.0)
            for n in cluster_nodes
            for u, v in [(n, nb) for nb in self.graph.neighbors(n)]
        ) / 2  # Divide by 2 to avoid double counting
        total_cluster_load_norm = total_cluster_load / max(self.base_capacity * len(cluster_nodes), 1)
        
        cluster_features = np.array([
            avg_degree_norm,
            cluster_size / self.num_nodes,  # normalized cluster size
            total_cluster_load_norm
        ], dtype=np.float32)
        
        # 2. Per-node features (padded)
        node_features_list = []
        for i in range(self.max_cluster_size):
            if i < cluster_size:
                node_id = cluster_nodes[i]
                node_feat = self._get_node_features(node_id)
            else:
                # Padding for smaller clusters
                node_feat = np.zeros(self.node_feature_dim, dtype=np.float32)
            node_features_list.append(node_feat)
        
        node_features = np.concatenate(node_features_list)
        
        # 3. Action mask
        action_mask = self.get_action_mask(cluster_id)
        
        # Concatenate all features
        obs = np.concatenate([cluster_features, node_features, action_mask])
        return obs.astype(np.float32)
    
    def _get_node_features(self, node_id: int) -> np.ndarray:
        """
        Get features for a single node.
        Features: [degree_norm, mode_norm, local_load_norm, neighbor_utils (padded to max_degree)]
        """
        # Degree normalized
        degree = self.graph.degree(node_id)
        degree_norm = degree / max(self.num_nodes - 1, 1)
        
        # Current mode normalized
        mode = self.node_modes.get(node_id, 1)
        mode_norm = mode / 2.0  # modes are 0, 1, 2
        
        # Local load: sum of loads on adjacent edges
        neighbors = list(self.graph.neighbors(node_id))
        local_load = sum(
            self.edge_load.get(self._edge_key(node_id, nb), 0.0)
            for nb in neighbors
        )
        local_load_norm = local_load / max(self.base_capacity * len(neighbors), 1) if neighbors else 0.0
        
        # Per-neighbor edge features (util, overload_flag) - padded to max_degree
        edge_features = []
        for i in range(self.max_degree):
            if i < len(neighbors):
                nb = neighbors[i]
                key = self._edge_key(node_id, nb)
                load = self.edge_load.get(key, 0.0)
                capacity = self.edge_capacity.get(key, self.base_capacity)
                util = load / capacity if capacity > 0 else 0.0
                util_clipped = float(min(util, 2.0))  # Clip for stability
                overload_flag = 1.0 if util > 1.0 else 0.0
                edge_features.extend([util_clipped, overload_flag])
            else:
                # Padding
                edge_features.extend([0.0, 0.0])
        
        features = np.array(
            [degree_norm, mode_norm, local_load_norm] + edge_features,
            dtype=np.float32
        )
        return features
    
    def _simulate_traffic_step(self) -> Tuple[float, float, float]:
        """
        Simulate traffic for one step and compute metrics.
        
        Returns:
            total_energy, total_delay, total_overload
        """
        # Generate random traffic flows
        flows = []
        for _ in range(self.num_flows_per_step):
            src = self.rng.choice(self.nodes)
            dst = self.rng.choice(self.nodes)
            while dst == src:
                dst = self.rng.choice(self.nodes)
            
            # Random demand
            demand = max(
                0.1,
                float(self.np_rng.normal(self.base_capacity / 4.0, self.base_capacity / 10.0))
            )
            flows.append((src, dst, demand))
        
        # Reset edge loads
        for key in self.edge_load:
            self.edge_load[key] = 0.0
        
        # Route flows using shortest paths and accumulate loads
        for src, dst, demand in flows:
            try:
                path = nx.shortest_path(self.graph, src, dst)
                # Add demand to each edge in path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = self._edge_key(u, v)
                    self.edge_load[key] = self.edge_load.get(key, 0.0) + demand
            except nx.NetworkXNoPath:
                # No path available, skip this flow
                pass
        
        # Compute metrics
        total_energy = 0.0
        total_delay = 0.0
        total_overload = 0.0
        
        for (u, v) in self.graph.edges():
            key = self._edge_key(u, v)
            load = self.edge_load.get(key, 0.0)
            capacity = self.edge_capacity.get(key, self.base_capacity)
            
            # Utilization
            util = load / capacity if capacity > 0 else 0.0
            
            # Overload penalty
            overload = max(0.0, util - 1.0)
            total_overload += overload * 5.0  # Penalty factor
            
            # Delay increases with utilization (convex)
            delay = self.base_delay * (1.0 + util * util)
            total_delay += delay
            
            # Energy: based on node modes
            mode_u = self.node_modes.get(u, 1)
            mode_v = self.node_modes.get(v, 1)
            power_factor = 0.5 * (self.mode_power_factor[mode_u] + self.mode_power_factor[mode_v])
            energy = self.base_power * power_factor * (1.0 + 0.5 * util)
            total_energy += energy
        
        return total_energy, total_delay, total_overload
    
    def _edge_key(self, u: int, v: int) -> Tuple[int, int]:
        """Canonical edge key (smaller node first)"""
        return (u, v) if u < v else (v, u)
