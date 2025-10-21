# 骨幹網路實驗快速參考

## 🎯 實驗目標
比較兩種骨幹網路拓樸在**相同規模、相同平均流量**但**不同形狀和流量分佈**下的表現

---

## 📊 配置對比表

| 項目 | 拓樸A (核心-邊緣) | 拓樸B (分散式網狀) |
|------|------------------|-------------------|
| **配置檔案** | `backbone_topology_A.json` | `backbone_topology_B.json` |
| **Seed** | 42 | 123 |
| **節點數** | 80 | 80 |
| **連結數** | 200 | 200 |
| **主機數** | 60 | 60 |
| **區域數** | 3 | 3 |
| **平均流量** | 40% (0.3-0.5) | 40% (0.3-0.5) |
| | | |
| **拓樸類型** | 集中式核心骨幹 | 分散式區域網狀 |
| **核心節點** | 20個 (25%) | 無明確核心 |
| **邊緣節點** | 60個 (75%) | 80個分散 |
| **核心連通性** | 60% | N/A |
| **網狀密度** | N/A | 40% |
| | | |
| **高流量區域** | Region 0 (核心) | Region 0, 2 (邊緣) |
| **核心流量權重** | 3.0 | 1.0 |
| **邊緣流量權重** | 1.0 | 2.5 |
| **核心高峰時段** | 20-100步 (長) | 140-200步 (短) |
| **邊緣高峰時段** | 0-60, 120-180步 | 0-70, 70-140步 |

---

## 🚀 快速開始

### 1️⃣ 生成和查看拓樸
```bash
cd train
python experiment/generate_backbone_topologies.py
```
**輸出**: 拓樸可視化圖、度數分佈、結構分析

### 2️⃣ 訓練拓樸A
```bash
python train.py --config experiment/configs/backbone_topology_A.json
```

### 3️⃣ 訓練拓樸B
```bash
python train.py --config experiment/configs/backbone_topology_B.json
```

### 4️⃣ 可視化結果
```bash
python visualize_clusters.py --result_dir experiment/results/backbone_topology_A
python visualize_clusters.py --result_dir experiment/results/backbone_topology_B
```

---

## 📈 關鍵差異

### 拓樸結構差異
```
拓樸A (核心-邊緣):
    [邊緣] ← → [核心] ← → [邊緣]
     稀疏      密集      稀疏
     
拓樸B (分散式網狀):
    [區域0] ← → [區域1] ← → [區域2]
     網狀        網狀        網狀
```

### 流量分佈差異
```
拓樸A: 核心高流量 (3.0x)，邊緣低流量 (1.0x)
  Region 0 (核心): ████████████ (高)
  Region 1 (東部): ████ (低)
  Region 2 (西部): ████ (低)

拓樸B: 邊緣高流量 (2.5x)，中央低流量 (1.0x)
  Region 0 (北部): ██████████ (高)
  Region 1 (中央): ████ (低)
  Region 2 (南部): ██████████ (高)
```

### 流量時序差異
```
拓樸A: 核心持續高峰
  0────60───100──120──180──200 (步)
  [R1高] [R0高────] [R2高]

拓樸B: 邊緣輪流高峰
  0────70───────140──200 (步)
  [R0高] [R2高────] [R1高]
```

---

## 🎯 預期發現

### 能耗效率
- **拓樸A**: 核心集中管理 → 可能更高效
- **拓樸B**: 分散架構 → 可能能耗較高

### 延遲性能
- **拓樸A**: 跨區域經核心 → 路徑短但可能擁塞
- **拓樸B**: 多路徑選擇 → 路徑長但負載分散

### 負載平衡
- **拓樸A**: 核心高、邊緣低 → 不均勻
- **拓樸B**: 分散均勻 → 較平衡

### 聚類效果
- **拓樸A**: 結構清晰 → 聚類穩定
- **拓樸B**: 網狀結構 → 可能頻繁重聚類

---

## 📊 評估指標清單

### 必看指標
- ✅ 總能耗 (Total Energy)
- ✅ 平均延遲 (Avg Latency)
- ✅ SLA違反率 (SLA Violation %)
- ✅ 活躍連結數 (Active Links)

### 進階指標
- 📊 連結利用率分佈
- 📊 區域間負載差異
- 📊 聚類數量變化
- 📊 重新聚類頻率

---

## 📁 檔案位置

```
train/experiment/configs/
├── backbone_topology_A.json          ← 拓樸A配置
├── backbone_topology_B.json          ← 拓樸B配置
├── backbone_comparison_config.json   ← 對比配置
├── BACKBONE_EXPERIMENT_README.md     ← 詳細說明
├── 實驗配置說明.md                    ← 中文說明
└── QUICK_REFERENCE.md                ← 本文檔

train/experiment/
├── generate_backbone_topologies.py   ← 拓樸生成腳本
└── results/                          ← 結果輸出目錄
```

---

## ⚡ 常用命令

```bash
# 快速測試 (100 episodes)
python train.py --config experiment/configs/backbone_topology_A.json --episodes 100

# 完整訓練 (2000 episodes)
python train.py --config experiment/configs/backbone_topology_A.json

# 查看訓練進度
tail -f experiment/results/backbone_topology_A/training.log

# 生成可視化
python visualize_clusters.py --result_dir experiment/results/backbone_topology_A

# 比較兩個實驗結果
python experiment/analyze_results.py \
  --exp1 experiment/results/backbone_topology_A \
  --exp2 experiment/results/backbone_topology_B
```

---

## 🔧 參數調整建議

### 如果想增加流量負載:
```json
"target_utilization_range": [0.5, 0.7]  // 從 [0.3, 0.5] 改為 [0.5, 0.7]
"flow_intensity_multiplier": 3.0        // 從 2.5 改為 3.0
```

### 如果想改變高峰時段:
```json
"peaks": {
  "region_0": [30, 110],  // 調整時間範圍
  "region_1": [10, 70],
  "region_2": [130, 190]
}
```

### 如果想調整聚類:
```json
"clustering_k_range": [2, 6],     // 從 [3, 8] 改為 [2, 6]
"dp_means_lambda": 3.0,           // 從 2.5 改為 3.0
"recluster_every_steps": 20       // 從 30 改為 20
```

---

## 💡 實驗技巧

1. **先可視化**: 運行 `generate_backbone_topologies.py` 確認拓樸正確
2. **快速測試**: 先用100 episodes測試，確認無誤再完整訓練
3. **監控資源**: 注意記憶體和CPU使用率
4. **保存結果**: 定期備份 `experiment/results/` 目錄
5. **對比分析**: 使用相同的評估指標比較兩個拓樸

---

## 📞 問題排查

### 記憶體不足
- 減少 `buffer_size`: 從 100000 改為 50000
- 減少 `batch_size`: 從 64 改為 32

### 訓練太慢
- 減少 `episodes`: 從 2000 改為 1000
- 減少 `max_steps_per_episode`: 從 200 改為 100

### 拓樸生成失敗
- 檢查 `num_edges` 是否合理 (不能超過 n*(n-1)/2)
- 確認 `seed` 設定正確

---

**快速參考完成！開始您的實驗吧！** 🚀
