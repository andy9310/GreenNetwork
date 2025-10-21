# 🎨 集群可視化快速入門

## 📦 安裝依賴

```bash
pip install networkx matplotlib scikit-learn numpy pandas
```

---

## 🚀 快速開始

### 1️⃣ 在訓練中自動生成（推薦）

訓練腳本已經整合了可視化功能，直接運行即可：

```bash
python train.py config.json low
```

**每 10 個 episode 自動生成：**
- 網絡拓撲圖
- 特徵空間圖（PCA）
- 對比圖

**輸出位置：** `cluster_visualizations/`

---

### 2️⃣ 測試可視化功能

```bash
# 基本測試
python test_visualization.py

# 比較不同集群方法
python test_visualization.py --compare
```

**輸出位置：** `test_visualizations/`

---

## 📊 生成的圖表

### 拓撲圖示例
```
cluster_visualizations/topology_ep10.png
```
- 🔵🟢🔴 不同顏色 = 不同集群
- 🟢 綠色邊 = 集群內連接
- 🔴 紅色邊 = 集群間連接
- ⚪ 灰色邊 = 未啟用連接

### 特徵空間圖示例
```
cluster_visualizations/features_pca_ep10.png
```
- 每個點 = 一個節點
- 相同顏色 = 同一集群
- X 標記 = 集群中心
- 距離近 = 特徵相似

### 對比圖示例
```
cluster_visualizations/comparison_ep10.png
```
- 左側：網絡拓撲
- 右側：特徵空間
- 方便同時觀察

---

## 💻 手動調用

### 方法 1: 從環境直接生成

```python
from visualize_clusters import visualize_clusters_from_env

# 在訓練循環中
topo_path, feat_path, comp_path = visualize_clusters_from_env(env, episode=10)
print(f"已保存: {comp_path}")
```

### 方法 2: 使用 ClusterVisualizer 類

```python
from visualize_clusters import ClusterVisualizer
from cluster import featureize_graph

# 創建可視化器
visualizer = ClusterVisualizer(save_dir="my_visualizations")

# 準備數據
cluster_map = env.region_of
traffic_in = env.traffic_matrix.sum(axis=0)
traffic_out = env.traffic_matrix.sum(axis=1)
X = featureize_graph(env.G_full, traffic_in, traffic_out, env.svc_class_share)

# 獲取活躍邊
active_edges = {(u, v) for u, v, d in env.G_full.edges(data=True) 
                if d.get('active', 1) == 1}

# 生成拓撲圖
topo_path = visualizer.visualize_network_topology(
    env.G_full, cluster_map, episode=10, active_edges=active_edges
)

# 生成特徵空間圖
feat_path = visualizer.visualize_feature_space(
    X, cluster_map, episode=10, method='pca'
)

# 生成對比圖
comp_path = visualizer.create_comparison_plot(
    env.G_full, X, cluster_map, episode=10, active_edges=active_edges
)
```

---

## 🎯 常用場景

### 場景 1: 訓練時每 N 個 episode 可視化

```python
# 在 train.py 中（已整合）
if (ep + 1) % 10 == 0:
    visualize_clusters_from_env(env, ep + 1)
```

### 場景 2: 比較不同集群方法

```python
methods = ['dp_means_adaptive', 'silhouette', 'kmeans']

for method in methods:
    cfg['clustering_method'] = method
    env = SDNEnv(cfg)
    env.reset()
    
    visualize_clusters_from_env(
        env, episode=0, 
        save_dir=f"visualizations/{method}"
    )
```

### 場景 3: 生成訓練演化動畫

```python
import imageio

# 收集所有圖片
images = []
for ep in range(10, 201, 10):
    img = imageio.imread(f'cluster_visualizations/comparison_ep{ep}.png')
    images.append(img)

# 生成 GIF
imageio.mimsave('cluster_evolution.gif', images, duration=0.5)
```

---

## 🎨 自定義選項

### 改變布局算法

```python
visualizer.visualize_network_topology(
    G, cluster_map, episode, 
    layout='kamada_kawai'  # 或 'circular', 'spectral'
)
```

### 使用 t-SNE 代替 PCA

```python
visualizer.visualize_feature_space(
    X, cluster_map, episode, 
    method='tsne'  # 更好的視覺效果，但較慢
)
```

### 自定義保存目錄

```python
visualizer = ClusterVisualizer(save_dir="custom_dir")
```

---

## 📁 文件結構

```
train/
├── visualize_clusters.py              # 可視化工具
├── test_visualization.py              # 測試腳本
├── CLUSTER_VISUALIZATION_GUIDE.md     # 詳細指南
├── VISUALIZATION_QUICKSTART.md        # 本文件
└── cluster_visualizations/            # 輸出目錄
    ├── topology_ep10.png
    ├── topology_ep20.png
    ├── features_pca_ep10.png
    ├── features_pca_ep20.png
    ├── comparison_ep10.png
    └── comparison_ep20.png
```

---

## 🔍 圖表解讀

### 好的集群特徵

✅ **拓撲圖：**
- 同色節點聚集在一起
- 集群內連接（綠色）多於集群間連接（紅色）
- 集群大小相對均衡

✅ **特徵空間圖：**
- 同色點形成明顯的群組
- 不同群組之間有清晰分離
- 集群中心（X）位於群組中央

### 需要改進的情況

❌ **拓撲圖：**
- 同色節點分散
- 紅色邊（集群間）過多
- 某些集群過大或過小

❌ **特徵空間圖：**
- 不同顏色的點混雜
- 群組邊界不清晰
- 集群中心偏離群組

---

## 🐛 常見問題

### Q: 圖片生成失敗

**檢查：**
1. 是否安裝了所有依賴？
2. 目錄是否有寫入權限？
3. 查看錯誤消息

### Q: 節點編號看不清

**解決：**
- 使用對比圖（較大）
- 或修改 `visualize_clusters.py` 中的 `font_size`

### Q: t-SNE 太慢

**解決：**
- 使用 PCA（默認）
- 或減少節點數量

### Q: 顏色不夠用

**解決：**
- 工具支持最多 20 個集群
- 如需更多，修改 `colors` 列表

---

## 📊 性能參考

| 網絡大小 | 生成時間 | 文件大小 |
|---------|---------|---------|
| 50 nodes | ~2s | ~2MB |
| 100 nodes | ~6s | ~3MB |
| 200 nodes | ~17s | ~5MB |

---

## 📚 更多資源

- **詳細指南：** `CLUSTER_VISUALIZATION_GUIDE.md`
- **代碼文檔：** `visualize_clusters.py` 中的 docstrings
- **測試腳本：** `test_visualization.py`

---

## ✅ 檢查清單

開始前確認：

- [ ] 已安裝依賴（matplotlib, networkx, sklearn）
- [ ] 配置啟用集群（`no_clustering: false`）
- [ ] 有足夠的磁盤空間（每張圖 2-5MB）
- [ ] 目錄有寫入權限

---

## 🎯 下一步

1. **運行測試：** `python test_visualization.py`
2. **開始訓練：** `python train.py config.json low`
3. **查看結果：** 打開 `cluster_visualizations/` 中的圖片
4. **分析集群：** 根據圖表調整集群參數

---

**最後更新：** 2025-10-13  
**快速問題？** 查看 `CLUSTER_VISUALIZATION_GUIDE.md`
