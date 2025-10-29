from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import itertools
import math
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Metrics on induced subgraphs / ego-subgraphs
# ============================================================

def _subgraph_from_nodes(G: nx.Graph, nodes: Iterable) -> nx.Graph:
    H = G.subgraph(nodes).copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    return H

def algebraic_connectivity(H: nx.Graph) -> float:
    n = H.number_of_nodes()
    if n < 2:
        return 0.0
    if not nx.is_connected(H):
        return 0.0
    L = nx.laplacian_matrix(H).astype(float).toarray()
    vals = np.linalg.eigvalsh(L)
    vals = np.clip(vals, 0.0, None)
    return float(vals[1])

def regional_metrics(
    G: nx.Graph,
    regions: Optional[Mapping[str, Iterable]] = None,
    region_attr: Optional[str] = "region",
) -> Dict[str, Dict[str, float]]:
    if regions is None:
        buckets: Dict[str, List] = {}
        for u, data in G.nodes(data=True):
            rid = data.get(region_attr, None)
            if rid is None:
                raise ValueError(
                    f"Node {u!r} has no region attribute '{region_attr}'. "
                    f"Either provide 'regions=' or set this attribute."
                )
            buckets.setdefault(rid, []).append(u)
        regions = buckets

    out: Dict[str, Dict[str, float]] = {}
    for rk, nodes in regions.items():
        H = _subgraph_from_nodes(G, nodes)
        n = H.number_of_nodes()
        m = H.number_of_edges()
        if n <= 1:
            density = 0.0
            avg_deg = 0.0
            a2 = 0.0
        else:
            density = 2.0 * m / (n * (n - 1))
            avg_deg = 2.0 * m / n
            a2 = algebraic_connectivity(H)
        out[rk] = {
            "n": float(n),
            "m": float(m),
            "density": float(density),
            "avg_degree": float(avg_deg),
            "algebraic_connectivity": float(a2),
        }
    return out

# ============================================================
# Temporal traffic model (kept for completeness)
# ============================================================

@dataclass
class RegionTrafficParams:
    lambda_bar: float
    delta: float
    omega: float
    phi: float
    noise_std: float = 0.0

def simulate_regional_lambda(
    params: Mapping[str, RegionTrafficParams],
    t: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    lam: Dict[str, np.ndarray] = {}
    for rk, p in params.items():
        base = p.lambda_bar + p.delta * np.sin(p.omega * t + p.phi)
        noise = rng.normal(0.0, p.noise_std, size=len(t)) if p.noise_std > 0 else 0.0
        series = base + noise
        lam[rk] = np.clip(series, 0.0, None)
    return lam

def default_g(lhs: float, rhs: float, mode: str = "mean") -> float:
    if mode == "mean":
        return 0.5 * (lhs + rhs)
    if mode == "product":
        return lhs * rhs
    if mode == "min":
        return min(lhs, rhs)
    if mode == "max":
        return max(lhs, rhs)
    raise ValueError(f"Unknown mode for g: {mode!r}")

def pick_node_pairs_across_regions(
    G: nx.Graph,
    regions: Mapping[str, Iterable],
    max_pairs_per_region_pair: int = 20,
    allow_intra_region: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int, str, str]]:
    if rng is None:
        rng = np.random.default_rng()
    regions_list: Dict[str, List[int]] = {rk: list(nodes) for rk, nodes in regions.items()}
    pairs: List[Tuple[int, int, str, str]] = []

    region_pairs = list(itertools.product(regions_list.keys(), regions_list.keys()))
    if not allow_intra_region:
        region_pairs = [(k, l) for (k, l) in region_pairs if k != l]

    for k, l in region_pairs:
        srcs = regions_list[k]
        dsts = regions_list[l]
        if len(srcs) == 0 or len(dsts) == 0:
            continue
        num = min(max_pairs_per_region_pair, len(srcs) * len(dsts))
        if len(srcs) * len(dsts) <= num:
            combos = list(itertools.product(srcs, dsts))
            rng.shuffle(combos)
            combos = combos[:num]
        else:
            combos = []
            seen = set()
            while len(combos) < num:
                s = int(rng.choice(srcs))
                t = int(rng.choice(dsts))
                if (s, t) not in seen and s != t:
                    combos.append((s, t))
                    seen.add((s, t))
        for s, t in combos:
            if G.has_node(s) and G.has_node(t):
                pairs.append((s, t, k, l))
    return pairs

def generate_flow_demands_over_time(
    G: nx.Graph,
    regions: Mapping[str, Iterable],
    t: np.ndarray,
    lambda_series: Mapping[str, np.ndarray],
    base_demand: float = 1.0,
    scale: float = 1.0,
    g_mode: str = "mean",
    max_pairs_per_region_pair: int = 20,
    allow_intra_region: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, int, str, str]], np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    pairs = pick_node_pairs_across_regions(
        G, regions,
        max_pairs_per_region_pair=max_pairs_per_region_pair,
        allow_intra_region=allow_intra_region,
        rng=rng,
    )
    F = len(pairs)
    T = len(t)
    D = np.zeros((T, F), dtype=float)

    def g_fun(a, b):
        return default_g(a, b, mode=g_mode)

    for j, (_s, _t, reg_s, reg_t) in enumerate(pairs):
        lam_s = lambda_series[reg_s]
        lam_t = lambda_series[reg_t]
        d = base_demand * scale * np.vectorize(g_fun)(lam_s, lam_t)
        D[:, j] = np.clip(d, 0.0, None)
    return pairs, D

# ============================================================
# Node-level features & DP-means with quality-driven λ selection
# ============================================================

def _ego_metrics(G: nx.Graph, u: int, radius: int = 2) -> Tuple[float, float, float]:
    H = nx.ego_graph(G, u, radius=radius, undirected=True).copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    n = H.number_of_nodes()
    m = H.number_of_edges()
    if n <= 1:
        density = 0.0
        avg_deg = 0.0
        a2 = 0.0
    else:
        density = 2.0 * m / (n * (n - 1))
        avg_deg = 2.0 * m / n
        if nx.is_connected(H):
            L = nx.laplacian_matrix(H).astype(float).toarray()
            vals = np.linalg.eigvalsh(L)
            vals = np.clip(vals, 0.0, None)
            a2 = float(vals[1])
        else:
            a2 = 0.0
    return density, avg_deg, a2

def build_node_feature_matrix(
    G: nx.Graph, radius: int = 2, nodes: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, List[int]]:
    if nodes is None:
        order = list(G.nodes())
    else:
        order = list(nodes)
    X = np.zeros((len(order), 3), dtype=float)
    for i, u in enumerate(order):
        X[i, :] = _ego_metrics(G, u, radius=radius)
    return X, order

def _zscore(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)

def dp_means(
    X: np.ndarray,
    lam: float,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N, D = X.shape
    centers = [X.mean(axis=0)]
    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        changed = False
        for i in range(N):
            x = X[i]
            dists = np.linalg.norm(x - np.vstack(centers), axis=1)**2
            j = int(np.argmin(dists))
            if dists[j] > lam:
                centers.append(x.copy())
                new_label = len(centers) - 1
                if labels[i] != new_label:
                    labels[i] = new_label
                    changed = True
            else:
                if labels[i] != j:
                    labels[i] = j
                    changed = True

        K = len(centers)
        new_centers = []
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(X[idx].mean(axis=0))
        new_centers_arr = np.vstack(new_centers)
        if (not changed) and np.allclose(np.vstack(centers), new_centers_arr):
            centers = new_centers
            break
        centers = new_centers

    return labels, np.vstack(centers)

def dp_means_with_history(
    X: np.ndarray,
    lam: float,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    N, D = X.shape
    centers = [X.mean(axis=0)]
    labels = np.zeros(N, dtype=int)
    labels_hist: List[np.ndarray] = [labels.copy()]
    centers_hist: List[np.ndarray] = [np.vstack(centers).copy()]

    for _ in range(max_iter):
        changed = False
        for i in range(N):
            x = X[i]
            dists = np.linalg.norm(x - np.vstack(centers), axis=1)**2
            j = int(np.argmin(dists))
            if dists[j] > lam:
                centers.append(x.copy())
                new_label = len(centers) - 1
                if labels[i] != new_label:
                    labels[i] = new_label
                    changed = True
            else:
                if labels[i] != j:
                    labels[i] = j
                    changed = True

        K = len(centers)
        new_centers = []
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(X[idx].mean(axis=0))
        new_centers_arr = np.vstack(new_centers)
        labels_hist.append(labels.copy())
        centers_hist.append(new_centers_arr.copy())
        if (not changed) and np.allclose(np.vstack(centers), new_centers_arr):
            centers = new_centers
            break
        centers = new_centers

    return labels_hist, centers_hist

def _extract_or_compute_positions(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    # Try to read lon/lat-like attributes; else fall back to spring_layout
    pos_str: Dict[str, Tuple[float, float]] = {}
    for n, data in G.nodes(data=True):
        lon = None
        lat = None
        for k in list(data.keys()):
            lk = str(k).lower()
            if lon is None and lk in ("lon", "longitude", "x"):
                try:
                    lon = float(data[k])
                except Exception:
                    lon = None
            if lat is None and lk in ("lat", "latitude", "y"):
                try:
                    lat = float(data[k])
                except Exception:
                    lat = None
        if lon is not None and lat is not None:
            pos_str[str(n)] = (lon, lat)

    # Map to integer nodes if needed
    pos: Dict[int, Tuple[float, float]] = {}
    if len(pos_str) > 0:
        # Normalize to [0,1] for nicer aspect
        xs = np.array([p[0] for p in pos_str.values()])
        ys = np.array([p[1] for p in pos_str.values()])
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)
        for n in G.nodes():
            key = str(n)
            if key in pos_str:
                x, y = pos_str[key]
                pos[n] = ((x - x_min)/x_span, (y - y_min)/y_span)
    if len(pos) < G.number_of_nodes():
        spring = nx.spring_layout(G, seed=42)
        for n in G.nodes():
            if n not in pos:
                pos[n] = tuple(spring[n])
    return pos

def animate_clustering_evolution(
    G: nx.Graph,
    lam: float,
    radius: int = 2,
    normalize: bool = True,
    out_path: str = "train/topologies/cluster_evolution.mp4",
    fps: int = 2,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X, order = build_node_feature_matrix(G, radius=radius)
    if normalize:
        X = _zscore(X)
    labels_hist, _ = dp_means_with_history(X, lam=lam, seed=42)
    pos = _extract_or_compute_positions(G)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_axis_off()

    def draw_frame(labels: np.ndarray):
        ax.clear()
        ax.set_axis_off()
        # Build color list per node order
        unique = np.unique(labels)
        color_map = plt.cm.get_cmap('tab20', len(unique))
        node_colors = {}
        for idx, k in enumerate(unique):
            node_list = [order[i] for i in range(len(order)) if labels[i] == k]
            for n in node_list:
                node_colors[n] = color_map(idx)
        colors = [node_colors.get(n, (0.7, 0.7, 0.7, 1.0)) for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=1.0, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=120, edgecolors='black', linewidths=0.5)
        ax.set_title(f"DP-means clustering evolution (λ={lam:.4g})")

    def update(frame_idx):
        draw_frame(labels_hist[min(frame_idx, len(labels_hist)-1)])
        return []

    anim = FuncAnimation(fig, update, frames=len(labels_hist), interval=1000//fps)
    # Save MP4 if possible; fallback to GIF
    try:
        anim.save(out_path, fps=fps, dpi=120)
        plt.close(fig)
        return out_path
    except Exception:
        alt = os.path.splitext(out_path)[0] + ".gif"
        anim.save(alt, fps=fps, dpi=120)
        plt.close(fig)
        return alt

def save_clustering_snapshots(
    G: nx.Graph,
    lam: float,
    radius: int = 2,
    normalize: bool = True,
    out_dir: str = "train/topologies/cluster_frames",
    every: int = 1,
    max_frames: Optional[int] = None,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    X, order = build_node_feature_matrix(G, radius=radius)
    if normalize:
        X = _zscore(X)
    labels_hist, _ = dp_means_with_history(X, lam=lam, seed=42)
    pos = _extract_or_compute_positions(G)

    frame_indices = list(range(0, len(labels_hist), max(1, every)))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    saved: List[str] = []
    for idx in frame_indices:
        labels = labels_hist[idx]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_axis_off()
        unique = np.unique(labels)
        color_map = plt.cm.get_cmap('tab20', len(unique))
        node_colors = {}
        for cidx, k in enumerate(unique):
            node_list = [order[i] for i in range(len(order)) if labels[i] == k]
            for n in node_list:
                node_colors[n] = color_map(cidx)
        colors = [node_colors.get(n, (0.7, 0.7, 0.7, 1.0)) for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=1.0, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=120, edgecolors='black', linewidths=0.5)
        ax.set_title(f"DP-means evolution frame {idx+1}/{len(labels_hist)} (λ={lam:.4g})")
        path = os.path.join(out_dir, f"frame_{idx:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved.append(path)
    return saved

# ---------- Quality target: minimize intra, maximize inter, balance sizes ----------

def _cluster_stats(X: np.ndarray, labels: np.ndarray):
    unique = np.unique(labels)
    K = len(unique)
    centers = np.array([X[labels == k].mean(axis=0) for k in unique])
    sizes = np.array([np.sum(labels == k) for k in unique])
    # Intra-region variance
    intra = 0.0
    for k in unique:
        grp = X[labels == k]
        diff = grp - centers[k]
        intra += (diff**2).sum() / max(len(grp), 1)
    intra /= max(K, 1)
    # Inter-region separation
    if K <= 1:
        inter = 0.0
    else:
        inter = 0.0
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                inter += np.linalg.norm(centers[i] - centers[j])**2
        inter /= (K * (K - 1))
    balance = float(np.std(sizes)) if K > 0 else 0.0
    return dict(K=K, centers=centers, sizes=sizes, intra=intra, inter=inter, balance=balance)

def clustering_quality_score(intra: float, inter: float, balance: float, balance_penalty: float = 0.1) -> float:
    return inter / (intra + 1e-9) - balance_penalty * balance

def cluster_by_dpmeans_quality(
    G: nx.Graph,
    radius: int = 2,
    normalize: bool = True,
    seed: int = 42,
    lam_grid: Optional[Sequence[float]] = None,
    balance_penalty: float = 0.1,
    min_clusters: int = 2,
    max_clusters: Optional[int] = None,
) -> Tuple[Dict[str, set], Dict[int, str], Dict[str, Dict[str, float]], Dict[str, float], Dict[int, str]]:
    """
    Quality-driven dynamic clustering:
      - builds features X (δ, d̄, a₂) from ego-subgraphs
      - sweeps λ values for DP-means
      - picks the clustering that maximizes Q = inter/(intra+eps) - λ_bal * std(|V_k|)

    Returns:
      regions_map:   {"R1": {...}, "R2": {...}, ...}
      node2region:   {node: "Rk"}
      region_metrics: per-region induced-subgraph metrics
      quality_info:  {"best_lambda": ..., "Q": ..., "intra": ..., "inter": ..., "balance": ..., "K": ...}
      labels_named:  {node: "Rk"} (same as node2region; returned separately for convenience)
    """
    X, order = build_node_feature_matrix(G, radius=radius)
    if normalize:
        X = _zscore(X)

    # Build a default λ grid based on data spread if not provided
    if lam_grid is None:
        # Use a log-spaced grid between small and large squared-distance scales
        # Compute pairwise distances to set rough bounds
        if len(X) > 1:
            dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)**2
            dists = dists[np.triu_indices(len(X), k=1)]
            dmin = max(np.percentile(dists, 5), 1e-4)
            dmax = max(np.percentile(dists, 90), dmin * 10)
        else:
            dmin, dmax = 0.1, 10.0
        lam_grid = np.unique(np.geomspace(dmin, dmax, num=12))
    best = dict(Q=-np.inf, lam=None, labels=None, stats=None)

    for lam in lam_grid:
        labels, _ = dp_means(X, lam=lam, seed=seed)
        stats = _cluster_stats(X, labels)
        K = stats["K"]
        if K < min_clusters:
            continue
        if max_clusters is not None and K > max_clusters:
            continue
        Q = clustering_quality_score(stats["intra"], stats["inter"], stats["balance"], balance_penalty)
        if Q > best["Q"]:
            best.update(Q=Q, lam=lam, labels=labels, stats=stats)

    if best["labels"] is None:
        # Fallback: force a single run (no constraints)
        lam = lam_grid[len(lam_grid)//2]
        labels, _ = dp_means(X, lam=lam, seed=seed)
        stats = _cluster_stats(X, labels)
        best.update(Q=clustering_quality_score(stats["intra"], stats["inter"], stats["balance"], balance_penalty),
                    lam=lam, labels=labels, stats=stats)

    labels = best["labels"]
    unique = np.unique(labels)
    kmap = {k: f"R{idx+1}" for idx, k in enumerate(unique)}

    regions_map: Dict[str, set] = {kmap[k]: set() for k in unique}
    node2region: Dict[int, str] = {}
    labels_named: Dict[int, str] = {}
    for node, lab in zip(order, labels):
        rk = kmap[lab]
        regions_map[rk].add(node)
        node2region[node] = rk
        labels_named[node] = rk

    region_summary = regional_metrics(G, regions=regions_map)

    quality_info = {
        "best_lambda": float(best["lam"]),
        "Q": float(best["Q"]),
        "intra": float(best["stats"]["intra"]),
        "inter": float(best["stats"]["inter"]),
        "balance": float(best["stats"]["balance"]),
        "K": int(best["stats"]["K"]),
    }
    return regions_map, node2region, region_summary, quality_info, labels_named

# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    # Load GEANT topology from GraphML
    graphml_path = "train/topologies/Geant2012.graphml"
    G_raw = nx.read_graphml(graphml_path)
    # Ensure simple undirected graph
    G = nx.Graph(G_raw)
    G.remove_edges_from(nx.selfloop_edges(G))
    # Relabel nodes to consecutive integers for consistency
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Quality-driven dynamic clustering
    regions_map, node2region, summary, qinfo, labels_named = cluster_by_dpmeans_quality(
        G,
        radius=2,
        normalize=True,
        seed=42,
        lam_grid=None,          # auto grid
        balance_penalty=0.1,    # tune if you want more balanced clusters
        min_clusters=2,
        max_clusters=8,
    )

    print(f"Best λ = {qinfo['best_lambda']:.4f} | K={qinfo['K']} | Q={qinfo['Q']:.3f} "
          f"(inter={qinfo['inter']:.3f}, intra={qinfo['intra']:.3f}, balance={qinfo['balance']:.3f})")

    print("\nRegions (sizes):")
    for rk, nodes in regions_map.items():
        print(f"  {rk}: |V|={len(nodes)}")

    print("\nPer-region induced-subgraph metrics (δ, d̄, a₂):")
    for rk, m in summary.items():
        print(f"{rk}: density={m['density']:.3f}, avg_deg={m['avg_degree']:.3f}, a2={m['algebraic_connectivity']:.3f}")
    print("finish clustering")
    # Save static clustering snapshots across DP-means iterations using best λ
    saved_frames = save_clustering_snapshots(
        G,
        lam=qinfo['best_lambda'],
        radius=2,
        normalize=True,
        out_dir="train/topologies/cluster_frames",
        every=1,
        max_frames=None,
    )
    if saved_frames:
        print(f"Saved {len(saved_frames)} frames to train/topologies/cluster_frames (e.g., {os.path.basename(saved_frames[0])} ... {os.path.basename(saved_frames[-1])})")

    # (Optional) Build temporal traffic based on discovered regions
    day_seconds = 24 * 3600.0
    omega = 2.0 * math.pi / day_seconds
    region_keys = list(regions_map.keys())
    params = {
        rk: RegionTrafficParams(
            lambda_bar=10.0 - 1.0 * (i % 3),
            delta=3.0 + 0.5 * (i % 2),
            omega=omega,
            phi=2.0 * math.pi * i / max(1, len(region_keys)),
            noise_std=0.5
        )
        for i, rk in enumerate(region_keys)
    }
    t = np.arange(0.0, 6 * 3600.0 + 1, 300.0)
    lam_series = simulate_regional_lambda(params, t)

    pairs, D = generate_flow_demands_over_time(
        G=G,
        regions=regions_map,
        t=t,
        lambda_series=lam_series,
        base_demand=1.0,
        scale=1.0,
        g_mode="mean",
        max_pairs_per_region_pair=6,
        allow_intra_region=False,
    )
    print(f"\nGenerated {len(pairs)} cross-region flows over {len(t)} timesteps. D shape={D.shape}")


