import os
import random
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make_region_nodes(G: nx.Graph, start_id: int, count: int, region: str,
                      shape: str, base_pos: Tuple[float, float]) -> Tuple[List[int], Dict[int, Tuple[float, float]]]:
    """
    Create nodes for a region with a specified geometric pattern and return node IDs and positions.

    shape options: 'circle', 'grid', 'star', 'cluster'
    base_pos is the (x, y) offset for the region's center.
    """
    nodes = list(range(start_id, start_id + count))
    pos: Dict[int, Tuple[float, float]] = {}

    rng = np.random.default_rng(42 + start_id)

    if shape == 'circle':
        r = 1.5
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for i, a in enumerate(angles):
            x = base_pos[0] + r * np.cos(a)
            y = base_pos[1] + r * np.sin(a)
            pos[nodes[i]] = (x, y)
        # Connect in a ring
        for i in range(count):
            G.add_edge(nodes[i], nodes[(i + 1) % count])

    elif shape == 'grid':
        # Closest to a 3x4 grid for 10 nodes: use 3 rows, 4 cols and leave 2 spots empty
        rows, cols = 3, 4
        spacing = 1.2
        coords = []
        for r in range(rows):
            for c in range(cols):
                coords.append((c * spacing, r * spacing))
        # pick 10 positions out of 12 to place nodes (keep grid shape)
        chosen_idx = list(range(len(coords)))
        rng.shuffle(chosen_idx)
        chosen_idx = sorted(chosen_idx[:count])
        for i, node in enumerate(nodes):
            cx, cy = coords[chosen_idx[i]]
            pos[node] = (base_pos[0] + cx - spacing, base_pos[1] + cy - spacing)
        # Connect grid neighbors if both exist
        pos_to_node = {pos[n]: n for n in nodes}
        def find_node_at(x, y):
            # inverse lookup with tolerance
            for n, (px, py) in pos.items():
                if abs(px - x) < 1e-6 and abs(py - y) < 1e-6:
                    return n
            return None
        # build a virtual grid adjacency
        placed = {(round((px - base_pos[0] + spacing)/spacing, 3),
                   round((py - base_pos[1] + spacing)/spacing, 3)): n for n, (px, py) in pos.items()}
        for (gx, gy), n in placed.items():
            for dx, dy in [(1, 0), (0, 1)]:
                m = placed.get((gx + dx, gy + dy))
                if m is not None:
                    G.add_edge(n, m)

    elif shape == 'star':
        # one hub + (count-1) leaves
        hub = nodes[0]
        pos[hub] = (base_pos[0], base_pos[1])
        r = 1.8
        leaves = nodes[1:]
        angles = np.linspace(0, 2 * np.pi, len(leaves), endpoint=False)
        for i, node in enumerate(leaves):
            x = base_pos[0] + r * np.cos(angles[i])
            y = base_pos[1] + r * np.sin(angles[i])
            pos[node] = (x, y)
            G.add_edge(hub, node)

    elif shape == 'cluster':
        # Gaussian blob and connect via k-NN inside region
        center = np.array(base_pos)
        pts = center + rng.normal(scale=0.6, size=(count, 2))
        for i, node in enumerate(nodes):
            pos[node] = (float(pts[i, 0]), float(pts[i, 1]))
        # k-NN connections
        k = 3
        for i in range(count):
            dists = []
            for j in range(count):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                dists.append((d, nodes[j]))
            dists.sort()
            for _, nbr in dists[:k]:
                G.add_edge(nodes[i], nbr)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    # annotate region attribute
    for n in nodes:
        G.nodes[n]['region'] = region
        G.nodes[n]['shape'] = shape

    return nodes, pos


def build_topology() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """Build a 40-node topology with 4 regions of distinct shapes and a few inter-region links."""
    G = nx.Graph()
    total_nodes = 40
    per_region = 10

    # Layout regions in quadrants
    centers = {
        'Region-A': (-5.0, 5.0),   # circle
        'Region-B': (5.0, 5.0),    # grid
        'Region-C': (-5.0, -5.0),  # star
        'Region-D': (5.0, -5.0),   # cluster
    }
    shapes = {
        'Region-A': 'circle',
        'Region-B': 'grid',
        'Region-C': 'star',
        'Region-D': 'cluster',
    }

    pos: Dict[int, Tuple[float, float]] = {}

    current_id = 0
    region_nodes: Dict[str, List[int]] = {}

    for region_name in ['Region-A', 'Region-B', 'Region-C', 'Region-D']:
        nodes, p = make_region_nodes(
            G, current_id, per_region, region_name, shapes[region_name], centers[region_name]
        )
        region_nodes[region_name] = nodes
        pos.update(p)
        current_id += per_region

    # Inter-region links (pick a few representative nodes)
    rng = random.Random(123)
    def pick(nodes, k):
        return rng.sample(nodes, k)

    # Connect each pair of neighboring regions with 2 links
    inter_pairs = [
        ('Region-A', 'Region-B'),
        ('Region-A', 'Region-C'),
        ('Region-B', 'Region-D'),
        ('Region-C', 'Region-D'),
    ]
    for r1, r2 in inter_pairs:
        sources = pick(region_nodes[r1], 2)
        targets = pick(region_nodes[r2], 2)
        for s, t in zip(sources, targets):
            G.add_edge(s, t)
            G.edges[s, t]['inter_region'] = True

    # Mark remaining edges as intra-region
    for u, v in G.edges():
        if 'inter_region' not in G.edges[u, v]:
            G.edges[u, v]['inter_region'] = False

    assert G.number_of_nodes() == total_nodes
    return G, pos


def plot_topology(G: nx.Graph, pos: Dict[int, Tuple[float, float]], save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Colors and markers per region
    region_styles = {
        'Region-A': {'color': '#4ECDC4', 'marker': 'o', 'label': 'Region-A (circle)'},
        'Region-B': {'color': '#FF6B6B', 'marker': 's', 'label': 'Region-B (grid)'},
        'Region-C': {'color': '#45B7D1', 'marker': '^', 'label': 'Region-C (star)'},
        'Region-D': {'color': '#E9C46A', 'marker': 'D', 'label': 'Region-D (cluster)'},
    }

    plt.figure(figsize=(12, 10))

    # Draw edges: intra light, inter bold
    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('inter_region', False)]
    inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('inter_region', False)]

    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color='lightgray', width=1.0, alpha=0.7)
    if inter_edges:
        nx.draw_networkx_edges(G, pos, edgelist=inter_edges, edge_color='black', width=2.2, alpha=0.9, style='solid')

    # Draw nodes per region with distinct markers/colors
    for region, style in region_styles.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('region') == region]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=style['color'],
            node_shape=style['marker'],
            edgecolors='black',
            linewidths=1.2,
            alpha=0.95,
            label=style['label'],
            node_size=300,
        )

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')

    plt.title('40-node Topology with Region-specific Shapes', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), framealpha=0.95)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def main():
    G, pos = build_topology()
    out = plot_topology(G, pos, save_path=os.path.join('train', 'topologies', 'topology_40_nodes.png'))
    print(f"Saved topology figure to: {out}")


if __name__ == '__main__':
    main()
