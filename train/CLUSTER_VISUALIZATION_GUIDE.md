# 集群可視化指南 (Cluster Visualization Guide)

## 📊 概述

這個工具提供兩種集群可視化方式：
1. **網絡拓撲圖** - 顯示節點的集群分配和連接關係
2. **特徵空間圖** - 顯示降維後的節點分布

---

## 🎨 可視化類型

### 1. 網絡拓撲圖 (Network Topology)

**特點：**
- 每個集群用不同顏色標示
- 綠色邊：集群內部連接（intra-cluster）
- 紅色邊：集群之間連接（inter-cluster）
- 灰色邊：未啟用的連接
- 顯示節點編號和集群統計

**輸出文件：** `cluster_visualizations/topology_ep{episode}.png`

### 2. 特徵空間圖 (Feature Space)

**特點：**
- 使用 PCA 或 t-SNE 降維到 2D
- 每個點代表一個節點
- 相同顏色的點屬於同一集群
- X 標記表示集群中心
- 顯示方差解釋比例（PCA）

**輸出文件：** `cluster_visualizations/features_pca_ep{episode}.png`

### 3. 對比圖 (Comparison Plot)

**特點：**
- 左右並排顯示拓撲圖和特徵空間圖
- 方便同時觀察兩種視角

**輸出文件：** `cluster_visualizations/comparison_ep{episode}.png`

---

## 🚀 使用方法

### 方法 1: 在訓練過程中自動生成（已整合）

訓練腳本已經整合了可視化功能，每 10 個 episode 自動生成：

```bash
python train.py config.json low
```

**輸出示例：**
```
Episode  10/200 | Reward:   123.45 | ...
📊 Generating cluster visualizations for episode 10...
   ✅ Topology: cluster_visualizations/topology_ep10.png
   ✅ Features: cluster_visualizations/features_pca_ep10.png
   ✅ Comparison: cluster_visualizations/comparison_ep10.png
```

### 方法 2: 手動調用

```python
from visualize_clusters import visualize_clusters_from_env

# 在訓練循環中
if (ep + 1) % 10 == 0:
    topo_path, feat_path, comp_path = visualize_clusters_from_env(env, ep + 1)
    print(f"Saved: {comp_path}")
```

### 方法 3: 使用 ClusterVisualizer 類

```python
from visualize_clusters import ClusterVisualizer
from cluster import featureize_graph

visualizer = ClusterVisualizer(save_dir="my_visualizations")

# 準備數據
cluster_map = env.region_of  # {node_id: cluster_id}
traffic_in = env.traffic_matrix.sum(axis=0)
traffic_out = env.traffic_matrix.sum(axis=1)
X = featureize_graph(env.G_full, traffic_in, traffic_out, env.svc_class_share)

# 獲取活躍邊
active_edges = {(u, v) for u, v, d in env.G_full.edges(data=True) if d.get('active', 1) == 1}

# 生成拓撲圖
topo_path = visualizer.visualize_network_topology(
    env.G_full, cluster_map, episode=10, active_edges=active_edges
)

# 生成特徵空間圖（PCA）
feat_path = visualizer.visualize_feature_space(
    X, cluster_map, episode=10, method='pca'
)

# 生成特徵空間圖（t-SNE）
tsne_path = visualizer.visualize_feature_space(
    X, cluster_map, episode=10, method='tsne'
)

# 生成對比圖
comp_path = visualizer.create_comparison_plot(
    env.G_full, X, cluster_map, episode=10, active_edges=active_edges
)
```

---

## 🎯 參數說明

### `visualize_network_topology()`

```python
visualize_network_topology(
    G,                    # NetworkX 圖
    cluster_map,          # {node_id: cluster_id}
    episode,              # Episode 編號
    active_edges=None,    # 活躍邊的集合 {(u,v), ...}
    title_suffix="",      # 標題附加文字
    layout='spring'       # 布局算法: 'spring', 'kamada_kawai', 'circular', 'spectral'
)
```

**布局算法選擇：**
- `spring`: 力導向布局（默認，適合大多數情況）
- `kamada_kawai`: 基於距離的布局（更均勻）
- `circular`: 圓形布局（適合小型網絡）
- `spectral`: 譜布局（基於圖拉普拉斯矩陣）

### `visualize_feature_space()`

```python
visualize_feature_space(
    X,                    # 特徵矩陣 (n_samples, n_features)
    cluster_map,          # {node_id: cluster_id}
    episode,              # Episode 編號
    method='pca',         # 降維方法: 'pca' 或 'tsne'
    title_suffix=""       # 標題附加文字
)
```

**降維方法比較：**

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **PCA** | 快速、可解釋、線性 | 可能無法捕捉非線性結構 | 快速分析、大數據集 |
| **t-SNE** | 保留局部結構、視覺效果好 | 慢、隨機性、難解釋 | 詳細分析、小數據集 |

---

## 📁 輸出文件結構

```
cluster_visualizations/
├── topology_ep10.png          # 拓撲圖
├── topology_ep20.png
├── features_pca_ep10.png      # PCA 特徵空間圖
├── features_pca_ep20.png
├── features_tsne_ep10.png     # t-SNE 特徵空間圖（如果生成）
├── comparison_ep10.png        # 對比圖
└── comparison_ep20.png
```

---

## 🎨 顏色方案

工具支持最多 20 個集群，使用以下顏色：

```python
colors = [
    '#FF6B6B',  # 紅色系
    '#4ECDC4',  # 青色系
    '#45B7D1',  # 藍色系
    '#FFA07A',  # 橙色系
    '#98D8C8',  # 綠色系
    # ... 等 20 種顏色
]
```

---

## 📊 圖表解讀

### 拓撲圖解讀

**節點：**
- 顏色 = 集群分配
- 編號 = 節點 ID
- 黑色邊框 = 所有節點

**邊：**
- 🟢 綠色粗線 = 集群內部活躍連接
- 🔴 紅色粗線 = 集群之間活躍連接
- ⚪ 灰色細線 = 未啟用連接

**圖例：**
- 顯示每個集群的節點數量
- 顯示活躍連接數量和比例

### 特徵空間圖解讀

**點的位置：**
- 距離近 = 特徵相似
- 同色點 = 同一集群
- X 標記 = 集群中心

**PCA 軸：**
- PC1 (橫軸) = 第一主成分（最大方差方向）
- PC2 (縱軸) = 第二主成分（次大方差方向）
- 方差解釋比例 = 該軸保留的信息量

**好的集群特徵：**
- ✅ 同色點聚集在一起（高內聚）
- ✅ 不同色點分離（低耦合）
- ❌ 顏色混雜 = 集群質量差

---

## 🔍 實際案例分析

### 案例 1: DP-means 自適應集群

```
Episode 50:
- 集群數量: 7
- 方法: dp_means_adaptive
- 拓撲: 清晰的集群邊界
- 特徵空間: PCA 解釋 65% 方差
```

**觀察：**
- 集群 0-2: 高流量節點（特徵空間右上）
- 集群 3-4: 中等流量節點（中間）
- 集群 5-6: 低流量節點（左下）
- 集群間連接: 5-8 條紅色邊（符合配置）

### 案例 2: 固定 K=3 集群

```
Episode 50:
- 集群數量: 3
- 方法: kmeans (k=3)
- 拓撲: 集群大小不均
- 特徵空間: 部分重疊
```

**觀察：**
- 集群 0: 80 個節點（過大）
- 集群 1: 15 個節點
- 集群 2: 5 個節點（過小）
- 特徵空間有重疊 → 可能需要更多集群

---

## 🛠️ 自定義和擴展

### 修改顏色方案

```python
visualizer = ClusterVisualizer()
visualizer.colors = ['#FF0000', '#00FF00', '#0000FF', ...]  # 自定義顏色
```

### 修改節點大小

編輯 `visualize_clusters.py`:
```python
# Line ~100
nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                      node_size=500,  # 改大節點
                      alpha=0.9, ...)
```

### 添加更多布局算法

```python
# 在 visualize_network_topology() 中添加
elif layout == 'hierarchical':
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
```

### 保存高分辨率圖片

```python
plt.savefig(filename, dpi=300, bbox_inches='tight')  # 300 DPI
```

---

## 📈 性能考慮

### 計算時間

| 網絡大小 | 拓撲圖 | PCA | t-SNE | 總時間 |
|---------|--------|-----|-------|--------|
| 50 nodes | ~0.5s | ~0.1s | ~2s | ~2.6s |
| 100 nodes | ~1s | ~0.2s | ~5s | ~6.2s |
| 200 nodes | ~2s | ~0.3s | ~15s | ~17.3s |
| 500 nodes | ~5s | ~0.5s | ~60s | ~65.5s |

**建議：**
- 小網絡 (< 100 nodes): 使用 t-SNE
- 大網絡 (> 200 nodes): 只用 PCA
- 訓練時: 每 10-20 episodes 生成一次

### 內存使用

- 拓撲圖: ~50MB per image
- 特徵空間圖: ~30MB per image
- 建議定期清理舊圖片

---

## 🐛 常見問題

### Q1: 圖片太小，看不清節點編號

**解決：**
```python
visualizer = ClusterVisualizer()
# 在 visualize_network_topology() 中
nx.draw_networkx_labels(G, pos, font_size=10)  # 增大字體
```

### Q2: t-SNE 報錯 "perplexity too large"

**原因：** 節點數太少  
**解決：** 自動處理（代碼已包含），或使用 PCA

### Q3: 顏色不夠用（超過 20 個集群）

**解決：**
```python
import matplotlib.cm as cm
visualizer.colors = [cm.tab20(i) for i in range(20)]  # 使用 matplotlib 調色板
```

### Q4: 拓撲圖節點重疊

**解決：** 嘗試不同布局
```python
visualizer.visualize_network_topology(G, cluster_map, ep, layout='kamada_kawai')
```

### Q5: 可視化失敗但不影響訓練

**原因：** 已用 try-except 包裹  
**檢查：** 查看錯誤消息，通常是缺少依賴

---

## 📦 依賴項

確保安裝以下包：

```bash
pip install networkx matplotlib scikit-learn numpy
```

**可選（用於更好的布局）：**
```bash
pip install pygraphviz  # 需要 Graphviz
```

---

## 🎓 進階用法

### 生成動畫（集群演化）

```python
import imageio

# 收集所有 episode 的圖片
images = []
for ep in range(10, 201, 10):
    img_path = f'cluster_visualizations/comparison_ep{ep}.png'
    images.append(imageio.imread(img_path))

# 生成 GIF
imageio.mimsave('cluster_evolution.gif', images, duration=0.5)
```

### 批量分析多次訓練

```python
import glob
import pandas as pd

# 分析所有訓練運行
runs = glob.glob('cluster_visualizations_run*')
for run_dir in runs:
    visualizer = ClusterVisualizer(save_dir=run_dir)
    # ... 分析邏輯
```

### 導出集群統計

```python
def export_cluster_stats(env, episode):
    stats = env.get_clustering_statistics()
    cluster_sizes = {}
    for node, cluster in env.region_of.items():
        cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
    
    df = pd.DataFrame({
        'episode': episode,
        'num_clusters': stats['current_cluster_count'],
        'method': stats['clustering_method_used'],
        'cluster_sizes': [cluster_sizes]
    })
    df.to_csv(f'cluster_stats_ep{episode}.csv', index=False)
```

---

## ✅ 檢查清單

使用可視化工具前確認：

- [ ] 已安裝所有依賴 (`pip install -r requirements.txt`)
- [ ] 訓練腳本已導入 `visualize_clusters`
- [ ] `cluster_visualizations/` 目錄有寫入權限
- [ ] 配置中啟用了集群 (`no_clustering: false`)
- [ ] Episode 數量足夠（至少 10 個以上）

---

## 📚 參考資料

- NetworkX 布局算法: https://networkx.org/documentation/stable/reference/drawing.html
- PCA 原理: https://scikit-learn.org/stable/modules/decomposition.html#pca
- t-SNE 原理: https://scikit-learn.org/stable/modules/manifold.html#t-sne

---

**最後更新:** 2025-10-13  
**版本:** 1.0  
**作者:** Research-GreenNetwork Team
