import json
import random
from typing import Optional
import networkx as nx
# region_clustering.py
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import networkx as nx

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class Cluster:

    def __init__(self, topo_info):
        self.topo_info = topo_info
        """
        topo info contain:
        1. nodes and edges
        2. 
        """
    def start_cluster(self):
        return cluster_id_mapping

# region_clustering_dp.py (or put inside your existing region_clustering.py)
from typing import Dict, Optional
import numpy as np
import networkx as nx

from region_clustering import RegionClusterer, TrafficProfile  # reuse feature builder


class Clusterer():
    """
    Dynamic cluster count using DP-means with connectivity constraint.
    Instead of specifying num_regions, you specify lambda_ (cluster radius penalty).
    Algorithm:
      - Start with 1 cluster centered at first point.
      - For each point:
          * If min distance^2 > lambda_ -> create new cluster centered at that point
          * Else assign to nearest cluster (with connectivity check)
      - Update centers as mean of assigned points; repeat until convergence or max_iters.
      - Post-process to ensure all clusters are connected subgraphs.
    """

    def __init__(
        self,
        lambda_: float,
        max_iters: int = 100,
        random_state: int = 0,
        use_degree: bool = True,
        use_local_density: bool = True,
        use_traffic: bool = True,
        ensure_connectivity: bool = True,  # 
    ):
        self.lambda_ = float(lambda_)
        self.max_iters = max_iters
        self.ensure_connectivity = ensure_connectivity
        self.use_degree = use_degree
        self.use_local_density = use_local_density
        self.use_traffic = use_traffic
        self.random_state = random_state

    def _compute_structural_features(
        self, g: nx.Graph, nodes: List[int]
    ) -> np.ndarray:
        """
        :
        結構特徵：
        - normalized degree: deg / (N-1)
        - local edge density: neighbors 子圖的 edge density
        """
        N = len(nodes)
        degrees = dict(g.degree())
        feats = []

        for n in nodes:
            row = []

            if self.use_degree:
                deg = degrees.get(n, 0)
                deg_norm = deg / max(N - 1, 1)
                row.append(deg_norm)

            if self.use_local_density:
                nbrs = list(g.neighbors(n))
                d = len(nbrs)
                if d <= 1:
                    local_density = 0.0
                else:
                    sub = g.subgraph(nbrs)
                    e_sub = sub.number_of_edges()
                    max_possible = d * (d - 1) / 2
                    local_density = e_sub / max(max_possible, 1)
                row.append(local_density)

            feats.append(row)

        return np.array(feats, dtype=float)
        
    def cluster(
        self,
        g: nx.Graph,
        traffic_profiles: Optional[Dict[int, TrafficProfile]] = None,
    ) -> Dict[int, int]:
        """
        Returns: {node_id: region_id} with dynamic number of regions.
        """
        nodes, X = self.build_feature_matrix(g, traffic_profiles)
        n_samples, n_features = X.shape
        if n_samples == 0:
            return {}

        # --- initialize: first point becomes first cluster ---
        centers = [X[0].copy()]      # list of cluster centers
        labels = np.zeros(n_samples, dtype=int)

        for it in range(self.max_iters):
            changed = False

            # Assignment step
            for i in range(n_samples):
                x = X[i]
                # compute squared distances to existing centers
                dists = np.array([np.sum((x - c) ** 2) for c in centers])
                j_min = int(np.argmin(dists))
                d_min = dists[j_min]

                if d_min > self.lambda_:
                    # create new cluster
                    centers.append(x.copy())
                    new_label = len(centers) - 1
                    if labels[i] != new_label:
                        labels[i] = new_label
                        changed = True
                else:
                    # assign to nearest
                    if labels[i] != j_min:
                        labels[i] = j_min
                        changed = True

            # Update step
            new_centers = []
            for k in range(len(centers)):
                mask = labels == k
                if not np.any(mask):
                    # cluster k has no points; skip (could delete, but easier: keep old center)
                    new_centers.append(centers[k])
                    continue
                new_centers.append(X[mask].mean(axis=0))
            centers = new_centers

            if not changed:
                # assignments stable, stop
                break

        # Map back to node -> region_id
        node_to_region = {n: int(labels[i]) for i, n in enumerate(nodes)}
        
        # Post-process to ensure connectivity if requested
        if self.ensure_connectivity:
            node_to_region = self._fix_connectivity(g, node_to_region)
        
        return node_to_region
    
    def _fix_connectivity(self, g: nx.Graph, partition: Dict[int, int]) -> Dict[int, int]:
        """
        修复partition使每个cluster都是连通子图
        
        策略:
        - 对每个cluster，找出所有连通分量
        - 保留最大的连通分量在原cluster
        - 将小的连通分量分配给相邻cluster或创建新cluster
        """
        fixed_partition = {}
        next_cluster_id = max(partition.values()) + 1
        
        # 按cluster分组节点
        cluster_to_nodes = {}
        for node, cid in partition.items():
            if cid not in cluster_to_nodes:
                cluster_to_nodes[cid] = []
            cluster_to_nodes[cid].append(node)
        
        # 处理每个cluster
        for cluster_id, nodes in cluster_to_nodes.items():
            # 提取子图
            subgraph = g.subgraph(nodes)
            
            # 找出连通分量
            connected_components = list(nx.connected_components(subgraph))
            
            if len(connected_components) == 1:
                # 已经连通，直接保留
                for node in nodes:
                    fixed_partition[node] = cluster_id
            else:
                # 不连通，需要拆分
                # 保留最大的连通分量
                largest_component = max(connected_components, key=len)
                for node in largest_component:
                    fixed_partition[node] = cluster_id
                
                # 其他节点：分配给邻居最多的cluster
                for component in connected_components:
                    if component == largest_component:
                        continue
                    
                    component_nodes = list(component)
                    
                    # 统计相邻cluster
                    neighbor_clusters = {}
                    for node in component_nodes:
                        for neighbor in g.neighbors(node):
                            if neighbor in fixed_partition:
                                ncid = fixed_partition[neighbor]
                                neighbor_clusters[ncid] = neighbor_clusters.get(ncid, 0) + 1
                    
                    if neighbor_clusters:
                        # 分配给最常见的相邻cluster
                        best_cluster = max(neighbor_clusters, key=neighbor_clusters.get)
                        for node in component_nodes:
                            fixed_partition[node] = best_cluster
                    else:
                        # 没有相邻cluster，创建新的
                        for node in component_nodes:
                            fixed_partition[node] = next_cluster_id
                        next_cluster_id += 1
        
        return fixed_partition
    
    def build_feature_matrix(
        self,
        g: nx.Graph,
        traffic_profiles: Optional[Dict[int, TrafficProfile]] = None,
    ):
        """
        构建特征矩阵
        """
        nodes = list(g.nodes())
        if not nodes:
            return [], np.array([])
        
        structural = self._compute_structural_features(g, nodes)
        
        if self.use_traffic and traffic_profiles is not None:
            traffic = self._compute_traffic_features(nodes, traffic_profiles)
            feats = np.concatenate([structural, traffic], axis=1)
        else:
            feats = structural
        
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return nodes, feats
    
    def _compute_traffic_features(
        self,
        nodes: List[int],
        traffic_profiles: Dict[int, TrafficProfile],
    ) -> np.ndarray:
        """提取traffic特征"""
        baselines = np.array([traffic_profiles[n].baseline for n in nodes], dtype=float)
        amplitudes = np.array([traffic_profiles[n].amplitude for n in nodes], dtype=float)
        
        def norm(x: np.ndarray) -> np.ndarray:
            xmax = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
            return x / xmax
        
        feats = np.stack([norm(baselines), norm(amplitudes)], axis=1)
        return feats


@dataclass
class TrafficProfile:
    """描述一個節點 / 模式的 traffic 參數."""
    baseline: float
    amplitude: float
    period: float
    phase: float
    noise_std: float = 0.0   # epsilon 的標準差之類


class RegionClusterer:
    """
    把拓樸 + (選擇性) traffic 參數變成 feature，然後做 KMeans clustering，
    把每個 node 分成 num_regions 個 region。
    
    现在支持连通性约束：ensure_connectivity=True 时保证每个region是连通子图
    """

    def __init__(
        self,
        num_regions: int,
        random_state: int = 0,
        use_degree: bool = True,
        use_local_density: bool = True,
        use_traffic: bool = True,
        ensure_connectivity: bool = False,  # 新增：保证连通性
    ):
        self.num_regions = num_regions
        self.random_state = random_state
        self.use_degree = use_degree
        self.use_local_density = use_local_density
        self.use_traffic = use_traffic
        self.ensure_connectivity = ensure_connectivity

    # ---------- feature construction ----------

    

    def _compute_traffic_features(
        self,
        nodes: List[int],
        traffic_profiles: Dict[int, TrafficProfile],
    ) -> np.ndarray:
        """
        Traffic heterogeneity 特徵：
        - baseline, amplitude, period, phase, noise_std
        全部做簡單 normalization（除以 max absolute value），
        避免某一個 scale 太大主導 clustering。
        """
        baselines = np.array([traffic_profiles[n].baseline for n in nodes], dtype=float)
        amplitudes = np.array([traffic_profiles[n].amplitude for n in nodes], dtype=float)
        periods = np.array([traffic_profiles[n].period for n in nodes], dtype=float)
        phases = np.array([traffic_profiles[n].phase for n in nodes], dtype=float)
        noise = np.array([traffic_profiles[n].noise_std for n in nodes], dtype=float)

        def norm(x: np.ndarray) -> np.ndarray:
            xmax = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
            return x / xmax

        feats = np.stack(
            [
                norm(baselines),
                norm(amplitudes),
                norm(periods),
                norm(phases),
                norm(noise),
            ],
            axis=1,
        )
        return feats

    def build_feature_matrix(
        self,
        g: nx.Graph,
        traffic_profiles: Optional[Dict[int, TrafficProfile]] = None,
    ):
        """
        輸出：
        - nodes: 節點順序 list
        - X: feature matrix, shape [num_nodes, feature_dim]
        """
        nodes = list(g.nodes())
        if not nodes:
            raise ValueError("Graph has no nodes")

        structural = self._compute_structural_features(g, nodes)

        if self.use_traffic and traffic_profiles is not None:
            traffic = self._compute_traffic_features(nodes, traffic_profiles)
            feats = np.concatenate([structural, traffic], axis=1)
        else:
            feats = structural

        # 保險：把 NaN / inf 變 0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return nodes, feats

    # ---------- clustering ----------

    def cluster(
        self,
        g: nx.Graph,
        traffic_profiles: Optional[Dict[int, TrafficProfile]] = None,
    ) -> Dict[int, int]:
        """
        回傳: {node_id: region_id}
        如果ensure_connectivity=True，保证每个region是连通子图
        """
        if KMeans is None:
            raise ImportError(
                "scikit-learn is required for RegionClusterer.cluster().\n"
                "pip install scikit-learn 或 uv add scikit-learn"
            )

        nodes, X = self.build_feature_matrix(g, traffic_profiles)
        kmeans = KMeans(
            n_clusters=self.num_regions,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)
        node_to_region = {n: int(label) for n, label in zip(nodes, labels)}
        
        # Post-process to ensure connectivity if requested
        if self.ensure_connectivity:
            node_to_region = self._fix_connectivity(g, node_to_region)
        
        return node_to_region
    
    def _fix_connectivity(self, g: nx.Graph, partition: Dict[int, int]) -> Dict[int, int]:
        """
        修复partition使每个cluster都是连通子图
        （与Clusterer类中的方法相同）
        """
        fixed_partition = {}
        next_cluster_id = max(partition.values()) + 1
        
        cluster_to_nodes = {}
        for node, cid in partition.items():
            if cid not in cluster_to_nodes:
                cluster_to_nodes[cid] = []
            cluster_to_nodes[cid].append(node)
        
        for cluster_id, nodes in cluster_to_nodes.items():
            subgraph = g.subgraph(nodes)
            connected_components = list(nx.connected_components(subgraph))
            
            if len(connected_components) == 1:
                for node in nodes:
                    fixed_partition[node] = cluster_id
            else:
                largest_component = max(connected_components, key=len)
                for node in largest_component:
                    fixed_partition[node] = cluster_id
                
                for component in connected_components:
                    if component == largest_component:
                        continue
                    
                    component_nodes = list(component)
                    neighbor_clusters = {}
                    for node in component_nodes:
                        for neighbor in g.neighbors(node):
                            if neighbor in fixed_partition:
                                ncid = fixed_partition[neighbor]
                                neighbor_clusters[ncid] = neighbor_clusters.get(ncid, 0) + 1
                    
                    if neighbor_clusters:
                        best_cluster = max(neighbor_clusters, key=neighbor_clusters.get)
                        for node in component_nodes:
                            fixed_partition[node] = best_cluster
                    else:
                        for node in component_nodes:
                            fixed_partition[node] = next_cluster_id
                        next_cluster_id += 1
        
        return fixed_partition
    
    def _compute_structural_features(
        self, g: nx.Graph, nodes: List[int]
    ) -> np.ndarray:
        """
        结构特征：degree 和 local density
        """
        N = len(nodes)
        degrees = dict(g.degree())
        feats = []

        for n in nodes:
            row = []

            if self.use_degree:
                deg = degrees.get(n, 0)
                deg_norm = deg / max(N - 1, 1)
                row.append(deg_norm)

            if self.use_local_density:
                nbrs = list(g.neighbors(n))
                d = len(nbrs)
                if d <= 1:
                    local_density = 0.0
                else:
                    sub = g.subgraph(nbrs)
                    e_sub = sub.number_of_edges()
                    max_possible = d * (d - 1) / 2
                    local_density = e_sub / max(max_possible, 1)
                row.append(local_density)

            feats.append(row)

        return np.array(feats, dtype=float)


def assign_modes_by_region(
    region_labels: Dict[int, int],
    region_modes: Dict[int, TrafficProfile],
) -> Dict[int, TrafficProfile]:
    """
    「region means the same cluster's modes are in the same region」用這個函式實作：
    給每個 region 一組模式（baseline/amplitude/period/phase），
    同一個 region 的所有 node 都 share 同一個 TrafficProfile。
    """
    node_modes: Dict[int, TrafficProfile] = {}
    for node, region in region_labels.items():
        if region not in region_modes:
            raise KeyError(f"region {region} has no TrafficProfile in region_modes")
        node_modes[node] = region_modes[region]
    return node_modes
