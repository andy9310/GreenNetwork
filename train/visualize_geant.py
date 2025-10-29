import os
import urllib.request
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

GEANT_URL = "https://topology-zoo.org/files/Geant2012.graphml"
LOCAL_DIR = os.path.join("train", "topologies")
LOCAL_GRAPHML = os.path.join(LOCAL_DIR, "Geant2012.graphml")
OUTPUT_IMG = os.path.join(LOCAL_DIR, "geant2012_regions.png")


def ensure_download(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"Downloading {url} -> {dst} ...")
        urllib.request.urlretrieve(url, dst)
        print("Download complete.")


def load_graph(path: str) -> nx.Graph:
    # NetworkX will read as MultiGraph sometimes; convert to simple Graph
    G = nx.read_graphml(path)
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        G = nx.Graph(G)
    return G


def extract_positions(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    # Internet Topology Zoo commonly uses 'Longitude' and 'Latitude'
    for n, data in G.nodes(data=True):
        lon = None
        lat = None
        # Try common keys, case-insensitive
        for k in list(data.keys()):
            lk = k.lower()
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
            pos[n] = (lon, lat)
    return pos


def plot_geant(G: nx.Graph,
               pos: Dict[str, Tuple[float, float]],
               save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalize coordinates for nicer aspect (optional): keep lon/lat aspect ratio
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    norm_pos = {n: ((pos[n][0] - x_min) / x_span, (pos[n][1] - y_min) / y_span) for n in G.nodes() if n in pos}

    plt.figure(figsize=(12, 9))

    # Draw all edges in light gray
    nx.draw_networkx_edges(G, norm_pos, edge_color='lightgray', width=1.2, alpha=0.8)

    # Draw all nodes in a single style
    nx.draw_networkx_nodes(
        G,
        norm_pos,
        nodelist=list(norm_pos.keys()),
        node_color="#4ECDC4",
        node_shape='o',
        edgecolors='black',
        linewidths=1.2,
        alpha=0.95,
        node_size=300,
    )

    # Labels
    nx.draw_networkx_labels(G, norm_pos, font_size=7, font_weight='bold')

    plt.title('GEANT (2012) Topology', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def main():
    ensure_download(GEANT_URL, LOCAL_GRAPHML)
    G = load_graph(LOCAL_GRAPHML)

    pos = extract_positions(G)
    if len(pos) < G.number_of_nodes():
        # Fallback layout if some nodes missing coordinates
        print("Some nodes missing coordinates; using spring_layout for them.")
        spring = nx.spring_layout(G, seed=42)
        for n in G.nodes():
            if n not in pos:
                pos[n] = spring[n]

    out = plot_geant(G, pos, OUTPUT_IMG)
    print(f"Saved GEANT topology visualization to: {out}")


if __name__ == "__main__":
    main()
