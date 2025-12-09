"""
Connected Graph Clustering - 保证每个cluster是连通子图
提供多种方法确保cluster内节点互相连接
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Optional
from collections import deque


class ConnectedClusterer:
    """
    基于图连通性的聚类，保证每个cluster形成连通子图
    
    方法1: Community Detection (Louvain算法)
    方法2: Post-processing修复 (将不连通的cluster拆分)
    方法3: Constrained growth (从种子节点生长，只添加相邻节点)
    """
    
    def __init__(
        self,
        num_clusters: int,
        method: str = "louvain",  # "louvain", "postprocess", "seeded_growth"
        random_state: int = 42
    ):
        """
        Args:
            num_clusters: 目标cluster数量 (某些方法可能产生不同数量)
            method: 聚类方法
            random_state: 随机种子
        """
        self.num_clusters = num_clusters
        self.method = method
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def cluster(self, g: nx.Graph) -> Dict[int, int]:
        """
        执行聚类，保证连通性
        
        Args:
            g: NetworkX图
            
        Returns:
            {node_id: cluster_id} 映射
        """
        if self.method == "louvain":
            return self._cluster_louvain(g)
        elif self.method == "postprocess":
            return self._cluster_with_postprocess(g)
        elif self.method == "seeded_growth":
            return self._cluster_seeded_growth(g)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    # ============= 方法1: Louvain社区检测 =============
    def _cluster_louvain(self, g: nx.Graph) -> Dict[int, int]:
        """
        使用Louvain算法进行社区检测
        优点: 自动发现自然的连通社区
        缺点: cluster数量不可控
        """
        try:
            import community  # python-louvain
            # Louvain算法
            partition = community.best_partition(g, random_state=self.random_state)
            
            # 如果cluster数量不匹配，进行合并或拆分
            num_found = len(set(partition.values()))
            if num_found > self.num_clusters:
                partition = self._merge_clusters(g, partition, self.num_clusters)
            elif num_found < self.num_clusters:
                partition = self._split_clusters(g, partition, self.num_clusters)
            
            return partition
            
        except ImportError:
            print("Warning: python-louvain not installed. Using Girvan-Newman instead.")
            return self._cluster_girvan_newman(g)
    
    def _cluster_girvan_newman(self, g: nx.Graph) -> Dict[int, int]:
        """
        使用Girvan-Newman算法（NetworkX内置）
        通过移除边的betweenness centrality来分割社区
        """
        from networkx.algorithms import community as nx_comm
        
        # 使用Girvan-Newman算法
        comp = nx_comm.girvan_newman(g)
        
        # 迭代直到达到目标cluster数量
        communities = None
        for communities in comp:
            if len(communities) >= self.num_clusters:
                break
        
        # 转换为dict格式
        node_to_cluster = {}
        for cluster_id, community in enumerate(communities):
            for node in community:
                node_to_cluster[node] = cluster_id
        
        return node_to_cluster
    
    # ============= 方法2: Post-processing修复 =============
    def _cluster_with_postprocess(self, g: nx.Graph) -> Dict[int, int]:
        """
        先用KMeans聚类，然后修复不连通的clusters
        
        流程:
        1. 基于特征的聚类 (KMeans)
        2. 检查每个cluster的连通性
        3. 将不连通的cluster拆分成多个连通分量
        4. 重新分配cluster ID
        """
        # Step 1: 初始聚类 (使用简单的度数作为特征)
        nodes = list(g.nodes())
        degrees = np.array([g.degree(n) for n in nodes]).reshape(-1, 1)
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
            labels = kmeans.fit_predict(degrees)
            initial_partition = {n: int(labels[i]) for i, n in enumerate(nodes)}
        except ImportError:
            # Fallback: 随机分配
            initial_partition = {n: self.rng.randint(0, self.num_clusters) for n in nodes}
        
        # Step 2: 修复连通性
        return self._fix_connectivity(g, initial_partition)
    
    def _fix_connectivity(self, g: nx.Graph, partition: Dict[int, int]) -> Dict[int, int]:
        """
        修复partition使每个cluster都是连通的
        
        策略:
        - 对每个cluster，找出所有连通分量
        - 保留最大的连通分量
        - 将其他节点重新分配到相邻的cluster
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
                    
                    # 为这个连通分量分配新的cluster ID
                    # 或者分配给相邻cluster中节点最多的那个
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
    
    # ============= 方法3: Seeded Growth =============
    def _cluster_seeded_growth(self, g: nx.Graph) -> Dict[int, int]:
        """
        从种子节点开始生长，每次只添加相邻节点
        保证每个cluster天然连通
        
        算法:
        1. 随机选择k个种子节点（尽量分散）
        2. 从种子节点开始BFS/贪心生长
        3. 每次将未分配的相邻节点加入当前cluster
        4. 使用某种策略决定优先级（例如degree, 特征相似度等）
        """
        nodes = list(g.nodes())
        n = len(nodes)
        
        # Step 1: 选择种子节点（尽量分散）
        seeds = self._select_dispersed_seeds(g, self.num_clusters)
        
        # Step 2: 初始化
        partition = {}
        for i, seed in enumerate(seeds):
            partition[seed] = i
        
        # Step 3: 贪心生长
        # 维护每个cluster的边界节点
        cluster_frontiers = {i: {seeds[i]} for i in range(self.num_clusters)}
        
        unassigned = set(nodes) - set(seeds)
        
        while unassigned:
            # 找出所有可扩展的边界
            candidates = []  # (node, cluster_id, priority)
            
            for cluster_id, frontier in cluster_frontiers.items():
                for node in frontier:
                    for neighbor in g.neighbors(node):
                        if neighbor in unassigned:
                            # 计算优先级 (例如：度数越高越优先)
                            priority = g.degree(neighbor)
                            candidates.append((neighbor, cluster_id, priority))
            
            if not candidates:
                # 没有可扩展的，剩余节点形成孤立cluster
                for node in unassigned:
                    partition[node] = self.num_clusters
                    self.num_clusters += 1
                break
            
            # 选择优先级最高的
            candidates.sort(key=lambda x: -x[2])
            node, cluster_id, _ = candidates[0]
            
            # 分配节点
            partition[node] = cluster_id
            cluster_frontiers[cluster_id].add(node)
            unassigned.remove(node)
        
        return partition
    
    def _select_dispersed_seeds(self, g: nx.Graph, k: int) -> List[int]:
        """
        选择k个尽量分散的种子节点
        使用最远点采样策略
        """
        nodes = list(g.nodes())
        if k >= len(nodes):
            return nodes
        
        seeds = []
        
        # 第一个种子：随机选择或选择中心性最高的
        degrees = dict(g.degree())
        first_seed = max(nodes, key=lambda n: degrees[n])
        seeds.append(first_seed)
        
        # 后续种子：选择距离已有种子最远的节点
        for _ in range(k - 1):
            max_min_dist = -1
            best_node = None
            
            for node in nodes:
                if node in seeds:
                    continue
                
                # 计算到最近种子的距离
                min_dist = float('inf')
                for seed in seeds:
                    try:
                        dist = nx.shortest_path_length(g, node, seed)
                        min_dist = min(min_dist, dist)
                    except nx.NetworkXNoPath:
                        pass
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_node = node
            
            if best_node is not None:
                seeds.append(best_node)
            else:
                # Fallback: 随机选择
                remaining = [n for n in nodes if n not in seeds]
                seeds.append(self.rng.choice(remaining))
        
        return seeds
    
    # ============= 辅助方法 =============
    def _merge_clusters(self, g: nx.Graph, partition: Dict[int, int], target_k: int) -> Dict[int, int]:
        """合并最相似的clusters直到达到目标数量"""
        current_clusters = set(partition.values())
        
        while len(current_clusters) > target_k:
            # 找出最应该合并的两个cluster（例如：边界连接最多的）
            best_pair = None
            max_connections = -1
            
            for c1 in current_clusters:
                for c2 in current_clusters:
                    if c1 >= c2:
                        continue
                    
                    # 计算两个cluster之间的边数
                    nodes_c1 = [n for n, c in partition.items() if c == c1]
                    nodes_c2 = [n for n, c in partition.items() if c == c2]
                    
                    connections = sum(
                        1 for n1 in nodes_c1 for n2 in g.neighbors(n1) if n2 in nodes_c2
                    )
                    
                    if connections > max_connections:
                        max_connections = connections
                        best_pair = (c1, c2)
            
            if best_pair is None:
                break
            
            # 合并
            c1, c2 = best_pair
            for node, cid in partition.items():
                if cid == c2:
                    partition[node] = c1
            
            current_clusters.remove(c2)
        
        # 重新编号
        cluster_mapping = {old: new for new, old in enumerate(sorted(current_clusters))}
        return {n: cluster_mapping[c] for n, c in partition.items()}
    
    def _split_clusters(self, g: nx.Graph, partition: Dict[int, int], target_k: int) -> Dict[int, int]:
        """拆分最大的clusters直到达到目标数量"""
        # TODO: 实现拆分逻辑（可以用spectral bisection）
        return partition
    
    def verify_connectivity(self, g: nx.Graph, partition: Dict[int, int]) -> bool:
        """
        验证partition中每个cluster是否连通
        
        Returns:
            True if all clusters are connected
        """
        cluster_to_nodes = {}
        for node, cid in partition.items():
            if cid not in cluster_to_nodes:
                cluster_to_nodes[cid] = []
            cluster_to_nodes[cid].append(node)
        
        for cluster_id, nodes in cluster_to_nodes.items():
            subgraph = g.subgraph(nodes)
            if not nx.is_connected(subgraph):
                print(f"❌ Cluster {cluster_id} is NOT connected!")
                return False
        
        print(f"✅ All {len(cluster_to_nodes)} clusters are connected!")
        return True


# ============= 使用示例 =============
if __name__ == "__main__":
    # 创建测试图
    g = nx.karate_club_graph()
    
    print("="*60)
    print("测试连通性聚类算法")
    print("="*60)
    
    for method in ["seeded_growth", "postprocess"]:
        print(f"\n方法: {method}")
        clusterer = ConnectedClusterer(num_clusters=4, method=method)
        partition = clusterer.cluster(g)
        
        print(f"生成了 {len(set(partition.values()))} 个clusters")
        clusterer.verify_connectivity(g, partition)
        
        # 显示cluster大小
        cluster_sizes = {}
        for cid in partition.values():
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
        print(f"Cluster sizes: {dict(sorted(cluster_sizes.items()))}")
