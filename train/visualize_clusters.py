"""
Cluster Visualization Tool for SDN Network Training

Provides two types of visualizations:
1. Network Topology with Cluster Coloring
2. Feature Space Visualization (Dimensionality Reduction)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ClusterVisualizer:
    """Visualize network clusters in topology and feature space"""
    
    def __init__(self, save_dir="cluster_visualizations"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save visualization images
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color palette for clusters (supports up to 20 clusters)
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
            '#E63946', '#A8DADC', '#457B9D', '#F1FAEE', '#E76F51',
            '#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'
        ]
    
    def visualize_network_topology(self, 
                                   G: nx.Graph, 
                                   cluster_map: Dict[int, int],
                                   episode: int,
                                   active_edges: Optional[set] = None,
                                   title_suffix: str = "",
                                   layout: str = 'spring') -> str:
        """
        Visualize network topology with cluster coloring
        
        Args:
            G: NetworkX graph
            cluster_map: Dict mapping node_id -> cluster_id
            episode: Current episode number
            active_edges: Set of active edges (u, v)
            title_suffix: Additional text for title
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'spectral')
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Get number of clusters
        num_clusters = len(set(cluster_map.values()))
        
        # Generate layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Prepare node colors based on clusters
        node_colors = [self.colors[cluster_map.get(node, 0) % len(self.colors)] 
                      for node in G.nodes()]
        
        # Draw edges
        if active_edges is not None:
            # Draw inactive edges (gray, thin)
            inactive_edges = [(u, v) for u, v in G.edges() 
                            if (u, v) not in active_edges and (v, u) not in active_edges]
            nx.draw_networkx_edges(G, pos, edgelist=inactive_edges,
                                 edge_color='lightgray', width=0.5, alpha=0.3, ax=ax)
            
            # Separate inter-cluster and intra-cluster active edges
            intra_cluster_edges = []
            inter_cluster_edges = []
            
            for u, v in active_edges:
                if cluster_map.get(u, 0) == cluster_map.get(v, 0):
                    intra_cluster_edges.append((u, v))
                else:
                    inter_cluster_edges.append((u, v))
            
            # Draw intra-cluster edges (green, medium)
            if intra_cluster_edges:
                nx.draw_networkx_edges(G, pos, edgelist=intra_cluster_edges,
                                     edge_color='green', width=1.5, alpha=0.6, ax=ax,
                                     label='Intra-cluster (active)')
            
            # Draw inter-cluster edges (red, thick)
            if inter_cluster_edges:
                nx.draw_networkx_edges(G, pos, edgelist=inter_cluster_edges,
                                     edge_color='red', width=2.5, alpha=0.8, ax=ax,
                                     label='Inter-cluster (active)')
        else:
            # Draw all edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5, ax=ax)
        
        # Draw nodes with cluster colors
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, alpha=0.9, 
                              edgecolors='black', linewidths=1.5, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold', ax=ax)
        
        # Create legend for clusters
        legend_elements = []
        for cluster_id in range(num_clusters):
            color = self.colors[cluster_id % len(self.colors)]
            cluster_nodes = [n for n, c in cluster_map.items() if c == cluster_id]
            label = f'Cluster {cluster_id} ({len(cluster_nodes)} nodes)'
            legend_elements.append(mpatches.Patch(color=color, label=label))
        
        # Add edge type legend if active edges provided
        if active_edges is not None:
            legend_elements.append(mpatches.Patch(color='green', label='Intra-cluster links'))
            legend_elements.append(mpatches.Patch(color='red', label='Inter-cluster links'))
            legend_elements.append(mpatches.Patch(color='lightgray', label='Inactive links'))
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                 framealpha=0.9, bbox_to_anchor=(1.02, 1))
        
        # Title and formatting
        title = f'Network Topology - Episode {episode}\n'
        title += f'{num_clusters} Clusters, {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges'
        if active_edges:
            title += f'\nActive: {len(active_edges)}/{G.number_of_edges()} links'
        if title_suffix:
            title += f'\n{title_suffix}'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.save_dir}/topology_ep{episode}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename
    
    def visualize_feature_space(self,
                               X: np.ndarray,
                               cluster_map: Dict[int, int],
                               episode: int,
                               method: str = 'pca',
                               title_suffix: str = "") -> str:
        """
        Visualize clusters in reduced feature space
        
        Args:
            X: Feature matrix (n_samples, n_features)
            cluster_map: Dict mapping node_id -> cluster_id
            episode: Current episode number
            method: Dimensionality reduction method ('pca' or 'tsne')
            title_suffix: Additional text for title
            
        Returns:
            Path to saved figure
        """
        n_samples = X.shape[0]
        num_clusters = len(set(cluster_map.values()))
        
        # Convert cluster_map to labels array
        labels = np.array([cluster_map.get(i, 0) for i in range(n_samples)])
        
        # Dimensionality reduction
        if method.lower() == 'tsne':
            if n_samples < 30:
                perplexity = max(5, n_samples // 3)
            else:
                perplexity = 30
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_reduced = reducer.fit_transform(X)
            method_name = 't-SNE'
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = 'PCA'
            variance_explained = reducer.explained_variance_ratio_
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each cluster
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            cluster_points = X_reduced[mask]
            
            if len(cluster_points) > 0:
                color = self.colors[cluster_id % len(self.colors)]
                
                # Scatter plot
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=color, label=f'Cluster {cluster_id} ({mask.sum()} nodes)',
                          s=150, alpha=0.7, edgecolors='black', linewidths=1.5)
                
                # Add cluster center
                center = cluster_points.mean(axis=0)
                ax.scatter(center[0], center[1], c=color, marker='X',
                          s=400, edgecolors='black', linewidths=2, zorder=10)
                
                # Add node labels
                for i, (x, y) in enumerate(cluster_points):
                    node_id = np.where(mask)[0][i]
                    ax.annotate(str(node_id), (x, y), fontsize=7, ha='center', va='center',
                               fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                       edgecolor='black', alpha=0.8))
        
        # Legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Title
        title = f'Feature Space Visualization ({method_name}) - Episode {episode}\n'
        title += f'{num_clusters} Clusters, {n_samples} Nodes'
        if method.lower() == 'pca':
            title += f'\nVariance Explained: {variance_explained[0]:.1%} (PC1), {variance_explained[1]:.1%} (PC2)'
        if title_suffix:
            title += f'\n{title_suffix}'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.save_dir}/features_{method.lower()}_ep{episode}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename
    
    def visualize_both(self,
                      G: nx.Graph,
                      X: np.ndarray,
                      cluster_map: Dict[int, int],
                      episode: int,
                      active_edges: Optional[set] = None,
                      clustering_method: str = "",
                      num_clusters: int = 0) -> Tuple[str, str]:
        """
        Create both topology and feature space visualizations
        
        Args:
            G: NetworkX graph
            X: Feature matrix
            cluster_map: Dict mapping node_id -> cluster_id
            episode: Current episode number
            active_edges: Set of active edges
            clustering_method: Name of clustering method used
            num_clusters: Number of clusters
            
        Returns:
            Tuple of (topology_path, feature_path)
        """
        # Create title suffix
        suffix = ""
        if clustering_method:
            suffix += f"Method: {clustering_method}"
        if num_clusters > 0:
            suffix += f", K={num_clusters}"
        
        # Generate both visualizations
        topo_path = self.visualize_network_topology(
            G, cluster_map, episode, active_edges, title_suffix=suffix
        )
        
        feature_path = self.visualize_feature_space(
            X, cluster_map, episode, method='pca', title_suffix=suffix
        )
        
        return topo_path, feature_path
    
    def create_comparison_plot(self,
                              G: nx.Graph,
                              X: np.ndarray,
                              cluster_map: Dict[int, int],
                              episode: int,
                              active_edges: Optional[set] = None) -> str:
        """
        Create a single figure with both topology and feature space side-by-side
        
        Args:
            G: NetworkX graph
            X: Feature matrix
            cluster_map: Dict mapping node_id -> cluster_id
            episode: Current episode number
            active_edges: Set of active edges
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(20, 9))
        
        # Get number of clusters
        num_clusters = len(set(cluster_map.values()))
        n_samples = X.shape[0]
        labels = np.array([cluster_map.get(i, 0) for i in range(n_samples)])
        
        # ============ LEFT: Network Topology ============
        ax1 = plt.subplot(1, 2, 1)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors
        node_colors = [self.colors[cluster_map.get(node, 0) % len(self.colors)] 
                      for node in G.nodes()]
        
        # Draw edges
        if active_edges:
            inactive = [(u, v) for u, v in G.edges() 
                       if (u, v) not in active_edges and (v, u) not in active_edges]
            nx.draw_networkx_edges(G, pos, edgelist=inactive,
                                 edge_color='lightgray', width=0.5, alpha=0.3, ax=ax1)
            
            intra = [(u, v) for u, v in active_edges 
                    if cluster_map.get(u, 0) == cluster_map.get(v, 0)]
            inter = [(u, v) for u, v in active_edges 
                    if cluster_map.get(u, 0) != cluster_map.get(v, 0)]
            
            if intra:
                nx.draw_networkx_edges(G, pos, edgelist=intra,
                                     edge_color='green', width=1.5, alpha=0.6, ax=ax1)
            if inter:
                nx.draw_networkx_edges(G, pos, edgelist=inter,
                                     edge_color='red', width=2.5, alpha=0.8, ax=ax1)
        else:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5, ax=ax1)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=200, alpha=0.9, 
                              edgecolors='black', linewidths=1.5, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=5, font_weight='bold', ax=ax1)
        
        ax1.set_title(f'Network Topology\n{num_clusters} Clusters, {G.number_of_nodes()} Nodes',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # ============ RIGHT: Feature Space (PCA) ============
        ax2 = plt.subplot(1, 2, 2)
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Plot clusters
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            cluster_points = X_pca[mask]
            
            if len(cluster_points) > 0:
                color = self.colors[cluster_id % len(self.colors)]
                ax2.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=color, label=f'Cluster {cluster_id}',
                          s=120, alpha=0.7, edgecolors='black', linewidths=1.5)
                
                # Cluster center
                center = cluster_points.mean(axis=0)
                ax2.scatter(center[0], center[1], c=color, marker='X',
                          s=300, edgecolors='black', linewidths=2, zorder=10)
        
        ax2.legend(loc='best', fontsize=9)
        ax2.set_title(f'Feature Space (PCA)\nVariance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('PC1', fontsize=10)
        ax2.set_ylabel('PC2', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Overall title
        fig.suptitle(f'Cluster Visualization - Episode {episode}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        filename = f'{self.save_dir}/comparison_ep{episode}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename


# ============================================
# Convenience Functions
# ============================================

def visualize_clusters_from_env(env, episode: int, save_dir: str = "cluster_visualizations"):
    """
    Visualize clusters directly from SDNEnv instance
    
    Args:
        env: SDNEnv instance
        episode: Current episode number
        save_dir: Directory to save visualizations
        
    Returns:
        Tuple of (topology_path, feature_path, comparison_path)
    """
    from cluster import featureize_graph
    
    visualizer = ClusterVisualizer(save_dir=save_dir)
    
    # Get cluster mapping
    cluster_map = env.region_of
    
    # Get feature matrix
    traffic_in = env.traffic_matrix.sum(axis=0)
    traffic_out = env.traffic_matrix.sum(axis=1)
    X = featureize_graph(env.G_full, traffic_in, traffic_out)
    
    # Get active edges
    active_edges = set()
    for u, v, data in env.G_full.edges(data=True):
        if data.get('active', 1) == 1:
            active_edges.add((u, v))
    
    # Get clustering stats
    stats = env.get_clustering_statistics()
    method = stats.get('clustering_method_used', 'unknown')
    num_clusters = stats.get('current_cluster_count', 0)
    
    # Create all visualizations
    topo_path, feature_path = visualizer.visualize_both(
        env.G_full, X, cluster_map, episode, active_edges, method, num_clusters
    )
    
    comparison_path = visualizer.create_comparison_plot(
        env.G_full, X, cluster_map, episode, active_edges
    )
    
    return topo_path, feature_path, comparison_path


if __name__ == "__main__":
    # Example usage
    print("Cluster Visualization Tool")
    print("=" * 60)
    print("\nUsage in training script:")
    print("```python")
    print("from visualize_clusters import visualize_clusters_from_env")
    print("")
    print("# In your training loop:")
    print("if (ep + 1) % 10 == 0:  # Visualize every 10 episodes")
    print("    topo, feat, comp = visualize_clusters_from_env(env, ep + 1)")
    print("    print(f'Visualizations saved: {comp}')")
    print("```")
