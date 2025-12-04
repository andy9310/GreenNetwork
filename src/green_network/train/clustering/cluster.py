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
    Dynamic cluster count using DP-means.
    Instead of specifying num_regions, you specify lambda_ (cluster radius penalty).
    Algorithm:
      - Start with 1 cluster centered at first point.
      - For each point:
          * If min distance^2 > lambda_ -> create new cluster centered at that point
          * Else assign to nearest cluster
      - Update centers as mean of assigned points; repeat until convergence or max_iters.
    """

    def __init__(
        self,
        lambda_: float,
        max_iters: int = 100,
        random_state: int = 0,
        use_degree: bool = True,
        use_local_density: bool = True,
        use_traffic: bool = True,
    ):
        self.lambda_ = float(lambda_)
        self.max_iters = max_iters

    def _compute_structural_features(
        self, g: nx.Graph, nodes: List[int]
    ) -> np.ndarray:
        """
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
        return node_to_region


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
    """

    def __init__(
        self,
        num_regions: int,
        random_state: int = 0,
        use_degree: bool = True,
        use_local_density: bool = True,
        use_traffic: bool = True,
    ):
        self.num_regions = num_regions
        self.random_state = random_state
        self.use_degree = use_degree
        self.use_local_density = use_local_density
        self.use_traffic = use_traffic

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
        return {n: int(label) for n, label in zip(nodes, labels)}


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
