"""
测试连通性约束的聚类算法
"""
import networkx as nx
import matplotlib.pyplot as plt
from cluster import RegionClusterer, Clusterer
from connected_cluster import ConnectedClusterer


def verify_all_clusters_connected(g: nx.Graph, partition: dict) -> bool:
    """验证所有clusters是否连通"""
    cluster_to_nodes = {}
    for node, cid in partition.items():
        if cid not in cluster_to_nodes:
            cluster_to_nodes[cid] = []
        cluster_to_nodes[cid].append(node)
    
    all_connected = True
    for cluster_id, nodes in cluster_to_nodes.items():
        subgraph = g.subgraph(nodes)
        is_connected = nx.is_connected(subgraph)
        status = "✅" if is_connected else "❌"
        print(f"  Cluster {cluster_id}: {len(nodes)} nodes - {status} {'Connected' if is_connected else 'NOT Connected'}")
        if not is_connected:
            all_connected = False
            components = list(nx.connected_components(subgraph))
            print(f"    → 分成 {len(components)} 个连通分量，大小: {[len(c) for c in components]}")
    
    return all_connected


def test_region_clusterer():
    """测试 RegionClusterer (KMeans)"""
    print("\n" + "="*70)
    print("测试 RegionClusterer (KMeans-based)")
    print("="*70)
    
    # 创建测试图
    g = nx.karate_club_graph()
    print(f"图: {g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边")
    
    # 测试1: 不保证连通性
    print("\n【测试1】 ensure_connectivity=False (默认)")
    clusterer = RegionClusterer(
        num_regions=4,
        random_state=42,
        ensure_connectivity=False
    )
    partition = clusterer.cluster(g)
    print(f"生成 {len(set(partition.values()))} 个clusters")
    verify_all_clusters_connected(g, partition)
    
    # 测试2: 保证连通性
    print("\n【测试2】 ensure_connectivity=True")
    clusterer = RegionClusterer(
        num_regions=4,
        random_state=42,
        ensure_connectivity=True  # 开启连通性约束
    )
    partition = clusterer.cluster(g)
    print(f"生成 {len(set(partition.values()))} 个clusters")
    is_all_connected = verify_all_clusters_connected(g, partition)
    
    if is_all_connected:
        print("\n✅ 所有clusters都是连通的!")
    else:
        print("\n❌ 仍有不连通的clusters")


def test_dp_means_clusterer():
    """测试 Clusterer (DP-means)"""
    print("\n" + "="*70)
    print("测试 Clusterer (DP-means)")
    print("="*70)
    
    g = nx.karate_club_graph()
    print(f"图: {g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边")
    
    # 测试1: 不保证连通性
    print("\n【测试1】 ensure_connectivity=False")
    clusterer = Clusterer(
        lambda_=0.5,
        max_iters=50,
        random_state=42,
        ensure_connectivity=False
    )
    partition = clusterer.cluster(g)
    print(f"生成 {len(set(partition.values()))} 个clusters (动态数量)")
    verify_all_clusters_connected(g, partition)
    
    # 测试2: 保证连通性
    print("\n【测试2】 ensure_connectivity=True")
    clusterer = Clusterer(
        lambda_=0.5,
        max_iters=50,
        random_state=42,
        ensure_connectivity=True  # 开启连通性约束
    )
    partition = clusterer.cluster(g)
    print(f"生成 {len(set(partition.values()))} 个clusters (动态数量)")
    is_all_connected = verify_all_clusters_connected(g, partition)
    
    if is_all_connected:
        print("\n✅ 所有clusters都是连通的!")
    else:
        print("\n❌ 仍有不连通的clusters")


def test_connected_clusterer():
    """测试 ConnectedClusterer (天然保证连通)"""
    print("\n" + "="*70)
    print("测试 ConnectedClusterer (天然连通)")
    print("="*70)
    
    g = nx.karate_club_graph()
    print(f"图: {g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边")
    
    for method in ["seeded_growth", "postprocess"]:
        print(f"\n【方法】 {method}")
        clusterer = ConnectedClusterer(
            num_clusters=4,
            method=method,
            random_state=42
        )
        partition = clusterer.cluster(g)
        print(f"生成 {len(set(partition.values()))} 个clusters")
        is_all_connected = verify_all_clusters_connected(g, partition)
        
        if is_all_connected:
            print(f"✅ {method} 方法保证了连通性!")
        else:
            print(f"❌ {method} 方法未能保证连通性")


def visualize_clustering(g: nx.Graph, partition: dict, title: str = "Clustering"):
    """可视化聚类结果"""
    import matplotlib.pyplot as plt
    
    # 使用spring layout
    pos = nx.spring_layout(g, seed=42)
    
    # 为每个cluster分配颜色
    num_clusters = len(set(partition.values()))
    colors = plt.cm.tab10(range(num_clusters))
    
    # 绘制
    plt.figure(figsize=(10, 8))
    for cluster_id in set(partition.values()):
        nodes_in_cluster = [n for n, c in partition.items() if c == cluster_id]
        nx.draw_networkx_nodes(
            g, pos,
            nodelist=nodes_in_cluster,
            node_color=[colors[cluster_id]],
            node_size=300,
            label=f"Cluster {cluster_id} ({len(nodes_in_cluster)} nodes)"
        )
    
    nx.draw_networkx_edges(g, pos, alpha=0.3)
    nx.draw_networkx_labels(g, pos, font_size=8)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"clustering_{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    print(f"图片保存为: clustering_{title.replace(' ', '_')}.png")
    plt.close()


def comprehensive_test():
    """综合测试"""
    print("\n" + "="*70)
    print("🧪 综合测试：对比不同方法")
    print("="*70)
    
    g = nx.karate_club_graph()
    
    methods = [
        ("KMeans (无约束)", RegionClusterer(num_regions=4, ensure_connectivity=False)),
        ("KMeans (有约束)", RegionClusterer(num_regions=4, ensure_connectivity=True)),
        ("DP-means (无约束)", Clusterer(lambda_=0.5, ensure_connectivity=False)),
        ("DP-means (有约束)", Clusterer(lambda_=0.5, ensure_connectivity=True)),
        ("Seeded Growth", ConnectedClusterer(num_clusters=4, method="seeded_growth")),
    ]
    
    results = []
    for name, clusterer in methods:
        print(f"\n【{name}】")
        partition = clusterer.cluster(g)
        num_clusters = len(set(partition.values()))
        is_connected = verify_all_clusters_connected(g, partition)
        
        results.append({
            'method': name,
            'num_clusters': num_clusters,
            'all_connected': is_connected
        })
    
    # 总结
    print("\n" + "="*70)
    print("📊 结果总结")
    print("="*70)
    print(f"{'方法':<20} {'Cluster数':<12} {'全部连通'}")
    print("-"*70)
    for r in results:
        status = "✅ 是" if r['all_connected'] else "❌ 否"
        print(f"{r['method']:<20} {r['num_clusters']:<12} {status}")


if __name__ == "__main__":
    # 运行所有测试
    test_region_clusterer()
    test_dp_means_clusterer()
    test_connected_clusterer()
    comprehensive_test()
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)
    print("\n总结：")
    print("1. RegionClusterer 和 Clusterer 现在都支持 ensure_connectivity 参数")
    print("2. 设置 ensure_connectivity=True 可以保证每个cluster是连通子图")
    print("3. ConnectedClusterer 提供额外的方法（seeded_growth, postprocess）")
    print("4. 推荐：对于需要连通性的场景，使用 ensure_connectivity=True")
