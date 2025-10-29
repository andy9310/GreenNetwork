import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from dataclasses import dataclass
import random
import math

from cluster import dynamic_clustering
from algorithm import DeterministicLinkManager  # New import

@dataclass
class Flow:
    s: int
    t: int
    size: float
    ttl: int  # steps remaining

class SDNEnv:
    """
    Simplified SDN environment for hierarchical MARL with dynamic clustering.
    - Graph with capacities and base delays.
    - Stochastic flows; routing via k-shortest paths with congestion delay.
    - Actions: per-cluster utilization thresholds + inter-cluster min edges to keep.
    - Deterministic refinement: protect high-priority flows by force-keeping links on their paths.
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed", 42))
        np.random.seed(cfg.get("seed", 42))
        self.n = cfg.get("num_nodes", 0)
        self.num_edges = cfg.get("num_edges", 0)
        self.num_regions = cfg["num_regions"]
        self.num_hosts = cfg["num_hosts"]
        self.energy_on = cfg["energy_per_link_on"]
        self.energy_sleep = cfg["energy_per_link_sleep"]
        self.k_paths = cfg["routing_k_paths"]
        self.cong_factor = cfg["congestion_delay_factor"]
        self.recluster_every = cfg["recluster_every_steps"]
        self.cluster_bins = np.array(cfg["cluster_threshold_bins"], dtype=float)
        self.inter_keep_opts = cfg["inter_cluster_keep_min"]

        # NEW: Traffic load management
        self.traffic_load_mode = cfg.get("traffic_load_mode", "low")
        self.traffic_modes = cfg.get("traffic_modes", {})
        self.current_traffic_config = self.traffic_modes.get(self.traffic_load_mode, {})
        
        # Calculate target utilization for network
        self.target_util_range = self.current_traffic_config.get("target_utilization_range", [0.2, 0.4])
        self.flow_intensity_multiplier = self.current_traffic_config.get("flow_intensity_multiplier", 1.0)
        
        # Track network capacity for flow sizing
        self._calculate_network_capacity()

        # Initialize deterministic link manager
        algorithm_type = cfg.get("deactivation_algorithm", "greedy")
        self.link_manager = DeterministicLinkManager(algorithm_type)

        # Clustering configuration
        self.no_clustering = cfg.get("no_clustering", False)
        self.adaptive_clustering = cfg.get("adaptive_clustering", False)
        self.clustering_method = cfg.get("clustering_method", "silhouette")
        self.clustering_k_range = cfg.get("clustering_k_range", [2, 10])
        self.dp_means_lambda = cfg.get("dp_means_lambda", None)
        
        # Handle no-clustering mode (single big cluster)
        if self.no_clustering:
            self.num_clusters = 1  # Single cluster for entire network
            self.max_clusters = 1
            self.adaptive_clustering = False  # Disable adaptive clustering
            print("🔄 No-clustering mode: Treating entire network as single cluster")
        else:
            # If adaptive, use max_clusters for action/obs space; else use fixed k
            if self.adaptive_clustering:
                self.max_clusters = self.clustering_k_range[1]
                self.num_clusters = self.max_clusters  # Start with max for space definition
            else:
                self.num_clusters = cfg.get("num_clusters", self.num_regions)
                self.max_clusters = self.num_clusters

        # Build or load topology
        graphml_path = cfg.get("graphml_path")
        if graphml_path:
            self._load_topology_from_graphml(graphml_path, use_geo_delay=cfg.get("use_geo_delay", True))
        else:
            self._build_topology()
        self._assign_regions_and_hosts()
        self._time = 0
        self._flows: List[Flow] = []
        self._cluster_map: Dict[int, int] = {}
        self._actual_num_clusters = self.num_clusters  # Track actual clusters after clustering
        self._last_action = None
        
        # Clustering statistics tracking
        self.clustering_stats = {
            'cluster_count_history': [],
            'reclustering_events': 0,
            'total_reclustering_time': 0.0,
            'clustering_method_used': self.clustering_method if not self.no_clustering else "no_clustering"
        }

        # Observation / action sizes (use max_clusters for consistent dimensions)
        self.action_n = self.max_clusters * len(self.cluster_bins) + len(self.inter_keep_opts)  # thresholds for each cluster + one global inter-keep choice
        self.obs_dim = self.max_clusters * 4 + 3  # [traffic_in, traffic_out, active_links, mean_util]*K + inter_summary(3)
        
        # Store original dimensions for agent compatibility
        self._original_obs_dim = self.obs_dim
        self._original_action_n = self.action_n

    def _build_topology(self):
        # Random geometric graph for spatial flavor
        pos = {i: (np.random.rand(), np.random.rand()) for i in range(self.n)}
        G = nx.gnm_random_graph(self.n, self.num_edges, seed=self.cfg.get("seed", 42))
        # Ensure connectivity
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps)-1):
                a = random.choice(list(comps[i]))
                b = random.choice(list(comps[i+1]))
                G.add_edge(a, b)
        for u, v in G.edges():
            cap = max(0.5, np.random.normal(self.cfg["edge_capacity_mean"], self.cfg["edge_capacity_std"]))
            delay = self.cfg["edge_base_delay_ms"] * (1.0 + 0.5*np.random.rand())
            G[u][v]["capacity"] = cap
            G[u][v]["delay_ms"] = delay
            G[u][v]["active"] = 1
        self.G_full = G
        self._calculate_network_capacity()

    def _load_topology_from_graphml(self, path: str, use_geo_delay: bool = True):
        Gx = nx.read_graphml(path)
        if isinstance(Gx, (nx.MultiGraph, nx.MultiDiGraph)):
            Gx = nx.Graph(Gx)
        # Build mapping to consecutive int ids and extract positions if present
        nodes = list(Gx.nodes())
        idx_map = {n: i for i, n in enumerate(nodes)}
        pos = {}
        for n, data in Gx.nodes(data=True):
            lon = None
            lat = None
            for k, v in data.items():
                lk = str(k).lower()
                if lon is None and lk in ("lon", "longitude", "x"):
                    try:
                        lon = float(v)
                    except Exception:
                        lon = None
                if lat is None and lk in ("lat", "latitude", "y"):
                    try:
                        lat = float(v)
                    except Exception:
                        lat = None
            if lon is not None and lat is not None:
                pos[idx_map[n]] = (lon, lat)

        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        for u, v in Gx.edges():
            ui = idx_map[u]; vi = idx_map[v]
            G.add_edge(ui, vi)

        # Capacities
        for u, v in G.edges():
            cap = max(0.5, np.random.normal(self.cfg.get("edge_capacity_mean", 5.0), self.cfg.get("edge_capacity_std", 1.5)))
            G[u][v]["capacity"] = cap
            G[u][v]["active"] = 1

        # Delays
        if use_geo_delay and len(pos) >= 2:
            ms_per_km = self.cfg.get("ms_per_km", 0.005)
            def haversine(p1, p2):
                R = 6371.0
                lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
                lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                return R * c
            for u, v in G.edges():
                if u in pos and v in pos:
                    dist_km = haversine(pos[u], pos[v])
                    G[u][v]["delay_ms"] = max(0.1, dist_km * ms_per_km)
                else:
                    G[u][v]["delay_ms"] = self.cfg.get("edge_base_delay_ms", 2.0)
        else:
            for u, v in G.edges():
                G[u][v]["delay_ms"] = self.cfg.get("edge_base_delay_ms", 2.0)

        self.G_full = G
        self.n = G.number_of_nodes()
        self.num_edges = G.number_of_edges()
        # Spatial region assignment for traffic scheduling
        if len(pos) == self.n and self.num_regions > 1:
            xs = np.array([pos[i][0] for i in range(self.n)])
            order = np.argsort(xs)
            per = max(1, self.n // self.num_regions)
            region_of = {}
            for rank, node in enumerate(order):
                region_of[node] = min(rank // per, self.num_regions - 1)
            self.region_of = region_of
        self._calculate_network_capacity()

    def _assign_regions_and_hosts(self):
        if not hasattr(self, 'region_of'):
            per = self.n // self.num_regions
            self.region_of = {}
            for i in range(self.n):
                self.region_of[i] = min(i // per, self.num_regions-1)

        # choose hosts from nodes with degree>=1
        candidates = [i for i in self.G_full.nodes() if self.G_full.degree(i) > 0]
        self.hosts = random.sample(candidates, k=min(self.num_hosts, len(candidates)))
        # map hosts to edge nodes (themselves for this abstraction)
        self.host_nodes = self.hosts

    def reset(self) -> np.ndarray:
        # Reset edge states active=1
        for u, v in self.G_full.edges():
            self.G_full[u][v]["active"] = 1
        self._time = 0
        self._flows = []
        self._clusterize()
        return self._observe()

    def _clusterize(self):
        import time
        start_time = time.time()
        
        # Handle no-clustering mode (single big cluster)
        if self.no_clustering:
            # Assign all nodes to cluster 0 (single big cluster)
            self._cluster_map = {i: 0 for i in range(self.n)}
            self._actual_num_clusters = 1
            clustering_time = time.time() - start_time
            
            # Track statistics
            self.clustering_stats['cluster_count_history'].append(1)
            self.clustering_stats['total_reclustering_time'] += clustering_time
            return
        
        # Build a quick traffic matrix + svc share snapshot for clustering
        n = self.n
        tm = np.zeros((n, n), dtype=float)
        
        # Build actual traffic matrix from current flows if available
        if len(self._flows) > 0:
            for flow in self._flows:
                tm[flow.s, flow.t] += flow.size
        
        # Use adaptive clustering if enabled
        if self.adaptive_clustering:
            k_range = tuple(self.clustering_k_range)
            self._cluster_map = dynamic_clustering(
                self.G_full, tm, 
                k=None,  # Let it auto-determine
                method=self.clustering_method,
                k_range=k_range,
                lambda_param=self.dp_means_lambda,
                seed=self.cfg.get("seed", 42)
            )
            # Update actual cluster count
            self._actual_num_clusters = len(set(self._cluster_map.values()))
            
            # CRITICAL FIX: Cap the number of clusters to max_clusters to prevent dimension mismatch
            if self._actual_num_clusters > self.max_clusters:
                print(f"⚠️  Warning: DP-means created {self._actual_num_clusters} clusters, capping to {self.max_clusters}")
                # Reassign excess clusters to existing ones
                unique_clusters = list(set(self._cluster_map.values()))
                if len(unique_clusters) > self.max_clusters:
                    # Keep the first max_clusters clusters, reassign the rest
                    keep_clusters = unique_clusters[:self.max_clusters]
                    excess_clusters = unique_clusters[self.max_clusters:]
                    
                    # Reassign excess clusters to the last kept cluster
                    for node, cluster in self._cluster_map.items():
                        if cluster in excess_clusters:
                            self._cluster_map[node] = keep_clusters[-1]  # Assign to last kept cluster
                    
                    self._actual_num_clusters = self.max_clusters
        else:
            # Fixed k clustering
            self._cluster_map = dynamic_clustering(
                self.G_full, tm, 
                k=self.num_clusters, 
                seed=self.cfg.get("seed", 42)
            )
            self._actual_num_clusters = self.num_clusters
        
        # Track clustering statistics
        clustering_time = time.time() - start_time
        self.clustering_stats['cluster_count_history'].append(self._actual_num_clusters)
        self.clustering_stats['total_reclustering_time'] += clustering_time

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Action is a flat index -> thresholds per cluster + one inter-keep option"""
        thresholds, inter_keep = self._decode_action(action)
        
        # Use the new deterministic algorithm system
        self._apply_enhanced_action(thresholds, inter_keep)

        # Generate flows for this step
        self._generate_new_flows()

        # Route flows and compute latency/energy
        latency_ms, utilization = self._route_and_measure()

        energy = self._energy_cost()
        # Reward: energy saving positive, latency & SLA viol negative (weighted)
        base_all_on = self.energy_on * self.G_full.number_of_edges()
        energy_saving = (base_all_on - energy) / base_all_on
        
        # Balanced penalty for overload
        # Strong enough to matter, but allows agent to learn trade-offs
        latency_penalty = 0.001 * latency_ms
        
        # EXTREME overload penalty - overload should NEVER happen in real networks
        util_stats = self.get_current_utilization_stats()
        avg_util_pct = util_stats["average_utilization"]  # Now uses ACTUAL uncapped value
        overload_penalty = 0.0
        
        if avg_util_pct > 100.0:
            # Catastrophic overload (>100%) - this should never happen
            # Quadratic penalty based on ACTUAL overload amount
            excess = avg_util_pct - 100.0
            overload_penalty = 10.0 + 0.1 * (excess ** 2)  # Base 10 + quadratic growth
        elif avg_util_pct > 80.0:
            # High utilization (80-100%) - exponential penalty
            excess = avg_util_pct - 80.0
            overload_penalty = 0.1 * (excess ** 1.5)  # Exponential growth: 0.2 at 84%, 0.9 at 90%, 2.8 at 100%
        
        penalty = latency_penalty + overload_penalty
        reward = energy_saving - penalty

        self._time += 1
        if self._time % self.recluster_every == 0:
            old_cluster_count = self._actual_num_clusters
            self._clusterize()
            if not self.no_clustering and self._actual_num_clusters != old_cluster_count:
                self.clustering_stats['reclustering_events'] += 1

        obs = self._observe(utilization=utilization)
        done = self._time >= self.cfg["max_steps_per_episode"]
        
        # Calculate active links count
        active_links_count = sum(1 for _, _, d in self.G_full.edges(data=True) if d["active"] == 1)
        
        info = {
            "energy": energy, 
            "latency_ms": latency_ms, 
            "energy_saving": energy_saving,
            "active_links": active_links_count
        }
        self._last_action = action
        return obs, reward, done, info

    def _decode_action(self, a: int):
        binsK = len(self.cluster_bins)
        # thresholds per cluster chosen by taking a in base-binsK
        thresholds_idx = []
        for _ in range(self.max_clusters):
            thresholds_idx.append(a % binsK)
            a //= binsK
        inter_idx = a % len(self.inter_keep_opts)
        thresholds = [float(self.cluster_bins[i]) for i in thresholds_idx]
        
        # If using adaptive clustering, only use thresholds for actual clusters
        if self.adaptive_clustering:
            thresholds = thresholds[:self._actual_num_clusters]
        
        return thresholds, self.inter_keep_opts[inter_idx]

    def _apply_enhanced_action(self, thresholds: List[float], inter_keep_min: int):
        """Use the new deterministic algorithm system"""
        # Apply the complete deterministic deactivation strategy
        self.link_manager.apply_complete_deactivation_strategy(
            graph=self.G_full,
            region_of=self.region_of,
            thresholds=thresholds,
            inter_keep_min=inter_keep_min,
            flows=self._flows
        )

    def _generate_new_flows(self):
        """Enhanced flow generation with traffic load control and inter-region alternation"""
        current_step = self._time
        peaks = self.cfg["peaks"]
        host_by_region = {r: [] for r in range(self.num_regions)}
        
        for h in self.hosts:
            host_by_region[self.region_of[h]].append(h)
        
        # Get current traffic mode settings
        traffic_config = self.current_traffic_config
        
        # Determine rotating peak region
        period = self.cfg.get("peak_rotation_period", self.cfg["max_steps_per_episode"])
        rotating_peak = (current_step // max(1, period)) % self.num_regions

        for r in range(self.num_regions):
            lo, hi = peaks[f"region_{r}"]
            in_peak = (current_step % self.cfg["max_steps_per_episode"]) in range(lo, hi)
            if r == rotating_peak:
                in_peak = True
            
            # Determine flow probability based on traffic mode
            if in_peak:
                base_prob = traffic_config.get("peak_flow_probability", 0.3)
            else:
                base_prob = traffic_config.get("offpeak_flow_probability", 0.1)
            
            # Apply intensity multiplier
            flow_prob = base_prob * self.flow_intensity_multiplier
            
            num_potential_flows = max(1, int(flow_prob * len(host_by_region[r]) / 2))
            
            for _ in range(num_potential_flows):
                if random.random() < flow_prob and len(host_by_region[r]) >= 2:
                    s, t = random.sample(host_by_region[r], 2)
                    
                    # Calculate flow size based on target utilization and path length
                    flow_size = self._calculate_adaptive_flow_size(s, t, in_peak)
                    
                    # Longer TTL for more realistic flows
                    ttl = random.randint(5, 15)
                    
                    self._flows.append(Flow(s=s, t=t, size=flow_size, ttl=ttl))

            # Inter-region flows biased to/from rotating peak
            other_regions = [rr for rr in range(self.num_regions) if rr != r]
            if other_regions and host_by_region[r]:
                target_r = rotating_peak if r != rotating_peak else self.rng.choice(other_regions)
                src_hosts = host_by_region[r]
                dst_hosts = host_by_region[target_r]
                if dst_hosts:
                    inter_prob = flow_prob * 0.5
                    inter_trials = max(1, int(inter_prob * (len(src_hosts)+len(dst_hosts)) / 4))
                    for _ in range(inter_trials):
                        if random.random() < inter_prob:
                            s = self.rng.choice(src_hosts)
                            t = self.rng.choice(dst_hosts)
                            flow_size = self._calculate_adaptive_flow_size(s, t, r == rotating_peak)
                            ttl = random.randint(5, 15)
                            self._flows.append(Flow(s=s, t=t, size=flow_size, ttl=ttl))

        # Remove expired flows
        for f in self._flows:
            f.ttl -= 1
        self._flows = [f for f in self._flows if f.ttl > 0]

    def _shortest_path_active(self, s: int, t: int):
        # Use only active edges
        H = nx.Graph()
        for u, v, data in self.G_full.edges(data=True):
            if data["active"] == 1:
                H.add_edge(u, v, delay_ms=data["delay_ms"], capacity=data["capacity"])
        if not H.has_node(s) or not H.has_node(t):
            return None
        try:
            return nx.shortest_path(H, s, t, weight="delay_ms")
        except nx.NetworkXNoPath:
            return None

    def _route_and_measure(self) -> Tuple[float, float, Dict[Tuple[int,int], float]]:
        # Simple routing: route each flow along current shortest path (active edges only)
        # accumulate per-edge utilization and compute per-flow latency
        
        # Reset all edge utilizations to 0 before measuring
        for u, v in self.G_full.edges():
            self.G_full[u][v]["utilization"] = 0.0
        
        util = {(u, v): 0.0 for u, v in self.G_full.edges()}
        total_latency_ms = 0.0
        total_flows = len(self._flows)

        # prebuild active subgraph
        H = nx.Graph()
        for u, v, data in self.G_full.edges(data=True):
            if data["active"] == 1:
                H.add_edge(u, v, delay_ms=data["delay_ms"], capacity=data["capacity"])

        for f in self._flows:
            try:
                path = nx.shortest_path(H, f.s, f.t, weight="delay_ms")
            except Exception:
                path = None
            if path is None:
                # penalize unserved flow: large latency
                total_latency_ms += 50.0
                continue

            # base latency
            base = 0.0
            min_cap = float("inf")
            edges = []
            for u, v in zip(path, path[1:]):
                if u > v: u, v = v, u
                d = self.G_full[u][v]
                base += d["delay_ms"]
                min_cap = min(min_cap, d["capacity"])
                edges.append((u, v))

            # add congestion penalty based on temp utilization
            # we approximate utilization by incrementing edge loads by flow "size"
            # Normalize flow size to capacity units (assuming capacity is in Mbps and size in bytes)
            # Convert bytes to Mbits: size_bytes / (1024*1024/8) = size_bytes / 131072
            # Then normalize by a time unit (assume 1ms transmission time)
            flow_load = f.size / 1000.0  # Simplified: treat as load units
            edge_delay = 0.0
            for (u, v) in edges:
                util[(u, v)] += flow_load
                cap = self.G_full[u][v]["capacity"]
                ufrac = util[(u, v)] / max(1e-6, cap)
                if ufrac <= 1.0:
                    extra = 0.0
                else:
                    extra = self.cong_factor * (ufrac - 1.0) * self.G_full[u][v]["delay_ms"]
                edge_delay += extra

            flow_latency = base + edge_delay
            total_latency_ms += flow_latency

        # Calculate average latency
        avg_latency = total_latency_ms / max(1, total_flows)
        
        # normalize util to fraction and store in graph
        util_frac = {}
        for (u, v), load in util.items():
            cap = self.G_full[u][v]["capacity"]
            util_frac[(u, v)] = load / max(cap, 1e-6)
            # Store utilization in graph for later retrieval
            self.G_full[u][v]["utilization"] = util_frac[(u, v)]
        return avg_latency, util_frac

    def _energy_cost(self) -> float:
        on = 0
        off = 0
        for _, _, d in self.G_full.edges(data=True):
            if d["active"] == 1:
                on += 1
            else:
                off += 1
        return on * self.energy_on + off * self.energy_sleep

    def _observe(self, utilization: Dict[Tuple[int,int], float] = None) -> np.ndarray:
        # Aggregate features per cluster (use actual clusters from _cluster_map)
        K_actual = self._actual_num_clusters
        K_max = self.max_clusters
        feats = []
        G = self.G_full
        
        # Compute features for actual clusters
        for c in range(K_actual):
            nodes = [i for i in range(self.n) if self._cluster_map.get(i, 0) == c]
            if len(nodes) == 0:
                # Empty cluster - use zeros
                feats.extend([0.0] * 6)
                continue
                
            sub = G.subgraph(nodes).copy()
            active_links = sum(1 for _,_,d in sub.edges(data=True) if d["active"] == 1)
            mean_util = 0.0
            m = 0
            if utilization is not None:
                for u, v in sub.edges():
                    u_, v_ = (u, v) if u < v else (v, u)
                    if (u_, v_) in utilization:
                        mean_util += utilization[(u_, v_)]
                        m += 1
            mean_util = (mean_util / m) if m > 0 else 0.0
            # traffic summaries (approximate by number of active flows involving nodes in c)
            tin = sum(1 for f in self._flows if f.t in nodes)
            tout = sum(1 for f in self._flows if f.s in nodes)
            feats.extend([float(tin), float(tout), float(active_links), float(mean_util)])

        # Pad with zeros if using adaptive clustering and K_actual < K_max
        if self.adaptive_clustering and K_actual < K_max:
            padding = [0.0] * 4 * (K_max - K_actual)
            feats.extend(padding)

        # Inter-cluster summary: number of active edges between different clusters
        inter_active = 0
        inter_total = 0
        for u, v, d in G.edges(data=True):
            if self._cluster_map.get(u, 0) != self._cluster_map.get(v, 0):
                inter_total += 1
                if d["active"] == 1:
                    inter_active += 1
        inter_ratio = (inter_active / inter_total) if inter_total > 0 else 1.0
        # time-of-day enc (phase)
        phase = (self._time % self.cfg["max_steps_per_episode"]) / max(1, self.cfg["max_steps_per_episode"]-1)
        last_a = float(self._last_action if self._last_action is not None else 0) / float(max(1, self.action_n-1))
        feats.extend([inter_ratio, phase, last_a])
        return np.array(feats, dtype=np.float32)

    def _calculate_network_capacity(self):
        """Calculate total network capacity to size flows appropriately"""
        self.total_network_capacity = 0
        self.avg_edge_capacity = 0
        
        if hasattr(self, 'G_full'):
            capacities = [data['capacity'] for _, _, data in self.G_full.edges(data=True)]
            self.total_network_capacity = sum(capacities)
            self.avg_edge_capacity = np.mean(capacities) if capacities else 5.0
        else:
            self.avg_edge_capacity = self.cfg.get("edge_capacity_mean", 5.0)
    
    def get_clustering_statistics(self):
        """Get clustering statistics for analysis"""
        stats = self.clustering_stats.copy()
        stats.update({
            'current_cluster_count': self._actual_num_clusters,
            'avg_cluster_count': np.mean(stats['cluster_count_history']) if stats['cluster_count_history'] else 1,
            'cluster_count_std': np.std(stats['cluster_count_history']) if stats['cluster_count_history'] else 0,
            'total_nodes': self.n,
            'nodes_per_cluster': self.n / self._actual_num_clusters if self._actual_num_clusters > 0 else self.n,
            'is_no_clustering': self.no_clustering
        })
        return stats
            
    def _calculate_adaptive_flow_size(self, s: int, t: int, in_peak: bool) -> float:
        """Calculate flow size to achieve target utilization"""
        target_min, target_max = self.target_util_range
        
        # Estimate path length (simplified)
        try:
            path_length = len(nx.shortest_path(self.G_full, s, t)) - 1
        except:
            path_length = 3  # Default assumption
        
        # Base flow size to achieve target utilization per edge on path
        # Target utilization per flow = (target_total_util / expected_flows_per_edge)
        expected_concurrent_flows = len(self._flows) / max(1, self.G_full.number_of_edges())
        target_util_per_flow = np.random.uniform(target_min, target_max) / max(1, expected_concurrent_flows)
        
        # Flow size = target_utilization * average_capacity * scaling_factor
        base_size = target_util_per_flow * self.avg_edge_capacity * 1000  # Convert to bytes
        
        # Add randomness and peak/off-peak variation
        if in_peak:
            size_multiplier = random.uniform(1.2, 2.0)  # Larger flows during peak
        else:
            size_multiplier = random.uniform(0.5, 1.2)  # Smaller flows off-peak
        
        final_size = base_size * size_multiplier
        
        # Ensure reasonable bounds
        min_size = self.cfg.get("flow_size_bytes_min", 1000)
        max_size = self.cfg.get("flow_size_bytes_max", 50000)
        
        return max(min_size, min(max_size, final_size))

    def _assign_priority_by_size_and_mode(self, flow_size: float, in_peak: bool) -> int:
        """Assign priority based on flow size and current conditions"""
        # Larger flows get higher priority (lower numbers)
        if flow_size > 30000:
            priority_options = [1, 2] if in_peak else [1, 2, 3]
        elif flow_size > 15000:
            priority_options = [2, 3, 4] if in_peak else [3, 4]
        elif flow_size > 5000:
            priority_options = [3, 4, 5]
        else:
            priority_options = [4, 5, 6]
            
        return random.choice(priority_options)

    def set_traffic_load_mode(self, mode: str):
        """Change traffic load mode during runtime"""
        if mode in self.traffic_modes:
            self.traffic_load_mode = mode
            self.current_traffic_config = self.traffic_modes[mode]
            self.target_util_range = self.current_traffic_config.get("target_utilization_range", [0.2, 0.4])
            self.flow_intensity_multiplier = self.current_traffic_config.get("flow_intensity_multiplier", 1.0)
            print(f"🔄 Traffic load mode changed to: {mode} ({self.current_traffic_config.get('description', '')})")
        else:
            print(f"❌ Unknown traffic mode: {mode}. Available modes: {list(self.traffic_modes.keys())}")

    def get_current_utilization_stats(self):
        """Get current network utilization statistics"""
        total_util = 0
        active_edges = 0
        max_util = 0
        overloaded_edges = 0
        
        for u, v, data in self.G_full.edges(data=True):
            if data.get("active", 1) == 1:
                util = data.get("utilization", 0.0)
                total_util += util
                max_util = max(max_util, util)
                active_edges += 1
                if util > 1.0:
                    overloaded_edges += 1
        
        # Calculate ACTUAL utilization without capping (for penalties/rewards)
        avg_util_actual = (total_util / max(1, active_edges)) * 100
        max_util_actual = max_util * 100
        
        # For display, cap at 100% but keep actual values for calculations
        avg_util_display = min(100.0, avg_util_actual)
        max_util_display = min(100.0, max_util_actual)
        
        return {
            "average_utilization": avg_util_actual,  # ACTUAL value for calculations
            "average_utilization_display": avg_util_display,  # Capped for display
            "max_utilization": max_util_actual,  # ACTUAL value
            "max_utilization_display": max_util_display,  # Capped for display
            "total_utilization": total_util,
            "active_edges": active_edges,
            "overloaded_edges": overloaded_edges,
            "target_range": self.target_util_range,
            "current_mode": self.traffic_load_mode,
            "overloaded": max_util > 1.0
        }

    # Convenience for baselines
    def set_all_active(self):
        for u, v in self.G_full.edges():
            self.G_full[u][v]["active"] = 1

    def sleep_by_threshold(self, thr: float):
        G = self.G_full
        deg = {i: max(1, G.degree(i)) for i in G.nodes()}
        for u, v in G.edges():
            proxy_util = 0.5*(1.0/deg[u] + 1.0/deg[v])
            G[u][v]["active"] = 1 if proxy_util >= thr else 0