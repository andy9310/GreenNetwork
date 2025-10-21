"""
生成和可視化骨幹網路拓樸
用於驗證拓樸A和拓樸B的結構差異
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, List
import random


def generate_core_edge_topology(num_nodes: int, num_edges: int, seed: int = 42) -> nx.Graph:
    """
    生成核心-邊緣架構拓樸 (拓樸A)
    
    Args:
        num_nodes: 總節點數 (80)
        num_edges: 目標連結數 (200)
        seed: 隨機種子
    
    Returns:
        NetworkX圖
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 核心和邊緣節點分配
    core_nodes = 20
    edge_nodes = num_nodes - core_nodes
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # 標記節點類型
    for i in range(core_nodes):
        G.nodes[i]['type'] = 'core'
        G.nodes[i]['region'] = 0  # 核心區域
    
    for i in range(core_nodes, num_nodes):
        G.nodes[i]['type'] = 'edge'
        # 邊緣節點分配到region 1或2
        G.nodes[i]['region'] = 1 if i < (core_nodes + edge_nodes // 2) else 2
    
    edges_added = 0
    
    # 1. 核心節點間高度互連 (60%連通性)
    core_pairs = [(i, j) for i in range(core_nodes) for j in range(i+1, core_nodes)]
    core_target = int(len(core_pairs) * 0.6)
    core_edges = random.sample(core_pairs, min(core_target, len(core_pairs)))
    G.add_edges_from(core_edges)
    edges_added += len(core_edges)
    
    # 2. 每個邊緣節點連接到核心 (至少1個，平均2-3個)
    for edge_node in range(core_nodes, num_nodes):
        num_core_connections = random.randint(1, 3)
        core_targets = random.sample(range(core_nodes), num_core_connections)
        for core_target in core_targets:
            if not G.has_edge(edge_node, core_target):
                G.add_edge(edge_node, core_target)
                edges_added += 1
    
    # 3. 邊緣節點間稀疏連接 (15%連通性)
    edge_node_list = list(range(core_nodes, num_nodes))
    edge_pairs = [(i, j) for i in edge_node_list for j in edge_node_list if i < j]
    edge_target = int(len(edge_pairs) * 0.15)
    edge_edges = random.sample(edge_pairs, min(edge_target, len(edge_pairs)))
    
    for u, v in edge_edges:
        if edges_added >= num_edges:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    # 4. 確保連通性
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node_a = random.choice(list(components[i]))
            node_b = random.choice(list(components[i + 1]))
            G.add_edge(node_a, node_b)
            edges_added += 1
    
    # 5. 調整到目標邊數
    while edges_added < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    print(f"✅ 核心-邊緣拓樸生成完成:")
    print(f"   - 節點數: {G.number_of_nodes()}")
    print(f"   - 連結數: {G.number_of_edges()}")
    print(f"   - 核心節點: {core_nodes}")
    print(f"   - 邊緣節點: {edge_nodes}")
    print(f"   - 平均度數: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    return G


def generate_distributed_mesh_topology(num_nodes: int, num_edges: int, seed: int = 123) -> nx.Graph:
    """
    生成分散式網狀架構拓樸 (拓樸B)
    
    Args:
        num_nodes: 總節點數 (80)
        num_edges: 目標連結數 (200)
        seed: 隨機種子
    
    Returns:
        NetworkX圖
    """
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # 將節點分配到3個區域
    nodes_per_region = num_nodes // 3
    
    for i in range(num_nodes):
        if i < nodes_per_region:
            G.nodes[i]['region'] = 0  # 北部
            G.nodes[i]['type'] = 'edge'
        elif i < 2 * nodes_per_region:
            G.nodes[i]['region'] = 1  # 中央
            G.nodes[i]['type'] = 'hub'
        else:
            G.nodes[i]['region'] = 2  # 南部
            G.nodes[i]['type'] = 'edge'
    
    edges_added = 0
    
    # 1. 每個區域內建立網狀連接
    for region in range(3):
        region_nodes = [n for n in G.nodes() if G.nodes[n]['region'] == region]
        
        # 區域內連接密度
        if region == 1:  # 中央區域密度較低
            density = 0.3
        else:  # 邊緣區域密度較高
            density = 0.4
        
        region_pairs = [(i, j) for i in region_nodes for j in region_nodes if i < j]
        target_edges = int(len(region_pairs) * density)
        selected_edges = random.sample(region_pairs, min(target_edges, len(region_pairs)))
        
        G.add_edges_from(selected_edges)
        edges_added += len(selected_edges)
    
    # 2. 建立跨區域連結 (15條專用連結)
    region_0_nodes = [n for n in G.nodes() if G.nodes[n]['region'] == 0]
    region_1_nodes = [n for n in G.nodes() if G.nodes[n]['region'] == 1]
    region_2_nodes = [n for n in G.nodes() if G.nodes[n]['region'] == 2]
    
    # Region 0 <-> Region 1
    for _ in range(5):
        u = random.choice(region_0_nodes)
        v = random.choice(region_1_nodes)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    # Region 1 <-> Region 2
    for _ in range(5):
        u = random.choice(region_1_nodes)
        v = random.choice(region_2_nodes)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    # Region 0 <-> Region 2 (直接連接)
    for _ in range(5):
        u = random.choice(region_0_nodes)
        v = random.choice(region_2_nodes)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    # 3. 確保連通性
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node_a = random.choice(list(components[i]))
            node_b = random.choice(list(components[i + 1]))
            G.add_edge(node_a, node_b)
            edges_added += 1
    
    # 4. 調整到目標邊數
    while edges_added < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    print(f"✅ 分散式網狀拓樸生成完成:")
    print(f"   - 節點數: {G.number_of_nodes()}")
    print(f"   - 連結數: {G.number_of_edges()}")
    print(f"   - 平均度數: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    return G


def visualize_topology(G: nx.Graph, title: str, save_path: str = None):
    """
    可視化拓樸結構
    
    Args:
        G: NetworkX圖
        title: 圖表標題
        save_path: 儲存路徑
    """
    plt.figure(figsize=(16, 8))
    
    # 使用spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # 根據區域和類型設定顏色
    node_colors = []
    for node in G.nodes():
        region = G.nodes[node].get('region', 0)
        node_type = G.nodes[node].get('type', 'normal')
        
        if node_type == 'core':
            node_colors.append('#FF6B6B')  # 紅色 - 核心
        elif region == 0:
            node_colors.append('#4ECDC4')  # 青色 - Region 0
        elif region == 1:
            node_colors.append('#45B7D1')  # 藍色 - Region 1
        else:
            node_colors.append('#96CEB4')  # 綠色 - Region 2
    
    # 繪製網路
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 拓樸圖已儲存至: {save_path}")
    
    plt.show()


def analyze_topology_properties(G: nx.Graph, name: str):
    """
    分析拓樸特性
    
    Args:
        G: NetworkX圖
        name: 拓樸名稱
    """
    print(f"\n📊 {name} 拓樸分析:")
    print(f"{'='*60}")
    
    # 基本統計
    print(f"節點數: {G.number_of_nodes()}")
    print(f"連結數: {G.number_of_edges()}")
    print(f"平均度數: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"網路密度: {nx.density(G):.4f}")
    
    # 連通性
    print(f"是否連通: {'是' if nx.is_connected(G) else '否'}")
    if nx.is_connected(G):
        print(f"平均最短路徑長度: {nx.average_shortest_path_length(G):.2f}")
        print(f"網路直徑: {nx.diameter(G)}")
    
    # 中心性
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    print(f"最大度中心性: {max(degree_centrality.values()):.4f}")
    print(f"平均度中心性: {np.mean(list(degree_centrality.values())):.4f}")
    print(f"最大介數中心性: {max(betweenness_centrality.values()):.4f}")
    
    # 區域統計
    regions = {}
    for node in G.nodes():
        region = G.nodes[node].get('region', 0)
        if region not in regions:
            regions[region] = []
        regions[region].append(node)
    
    print(f"\n區域分佈:")
    for region, nodes in sorted(regions.items()):
        region_subgraph = G.subgraph(nodes)
        internal_edges = region_subgraph.number_of_edges()
        print(f"  Region {region}: {len(nodes)} 節點, {internal_edges} 內部連結")
    
    # 度數分佈
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\n度數分佈:")
    print(f"  最小度數: {min(degrees)}")
    print(f"  最大度數: {max(degrees)}")
    print(f"  平均度數: {np.mean(degrees):.2f}")
    print(f"  度數標準差: {np.std(degrees):.2f}")
    
    print(f"{'='*60}\n")


def compare_topologies(G_A: nx.Graph, G_B: nx.Graph):
    """
    比較兩個拓樸的差異
    
    Args:
        G_A: 拓樸A
        G_B: 拓樸B
    """
    print(f"\n🔍 拓樸比較:")
    print(f"{'='*60}")
    
    # 度數分佈比較
    degrees_A = [G_A.degree(n) for n in G_A.nodes()]
    degrees_B = [G_B.degree(n) for n in G_B.nodes()]
    
    print(f"平均度數差異: {abs(np.mean(degrees_A) - np.mean(degrees_B)):.2f}")
    print(f"度數標準差 - A: {np.std(degrees_A):.2f}, B: {np.std(degrees_B):.2f}")
    
    # 中心性比較
    bc_A = list(nx.betweenness_centrality(G_A).values())
    bc_B = list(nx.betweenness_centrality(G_B).values())
    
    print(f"介數中心性集中度 - A: {np.std(bc_A):.4f}, B: {np.std(bc_B):.4f}")
    print(f"  (較高的標準差表示流量更集中)")
    
    # 路徑長度比較
    if nx.is_connected(G_A) and nx.is_connected(G_B):
        apl_A = nx.average_shortest_path_length(G_A)
        apl_B = nx.average_shortest_path_length(G_B)
        print(f"平均最短路徑長度 - A: {apl_A:.2f}, B: {apl_B:.2f}")
    
    print(f"{'='*60}\n")


def main():
    """主函數"""
    print("🚀 開始生成骨幹網路拓樸...\n")
    
    # 參數設定
    num_nodes = 80
    num_edges = 200
    
    # 生成拓樸A (核心-邊緣)
    print("📍 生成拓樸A (核心-邊緣架構)...")
    G_A = generate_core_edge_topology(num_nodes, num_edges, seed=42)
    
    # 生成拓樸B (分散式網狀)
    print("\n📍 生成拓樸B (分散式網狀架構)...")
    G_B = generate_distributed_mesh_topology(num_nodes, num_edges, seed=123)
    
    # 分析拓樸特性
    analyze_topology_properties(G_A, "拓樸A (核心-邊緣)")
    analyze_topology_properties(G_B, "拓樸B (分散式網狀)")
    
    # 比較拓樸
    compare_topologies(G_A, G_B)
    
    # 可視化
    print("📊 生成拓樸可視化...")
    visualize_topology(G_A, "拓樸A: 核心-邊緣架構 (Core-Edge)", 
                      "experiment/results/topology_A_visualization.png")
    visualize_topology(G_B, "拓樸B: 分散式網狀架構 (Distributed Mesh)", 
                      "experiment/results/topology_B_visualization.png")
    
    # 繪製度數分佈比較
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    degrees_A = [G_A.degree(n) for n in G_A.nodes()]
    plt.hist(degrees_A, bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('拓樸A 度數分佈', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    degrees_B = [G_B.degree(n) for n in G_B.nodes()]
    plt.hist(degrees_B, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('拓樸B 度數分佈', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("experiment/results/degree_distribution_comparison.png", dpi=300, bbox_inches='tight')
    print("📊 度數分佈比較圖已儲存")
    plt.show()
    
    print("\n✅ 所有拓樸生成和分析完成!")
    print("📁 結果已儲存至 experiment/results/")


if __name__ == "__main__":
    main()
