# 🌿 GreenNetwork Colab Training

**Energy-Aware SDN Routing with Deep Reinforcement Learning**

Train a DQN agent with adaptive clustering to optimize energy consumption in Software-Defined Networks while maintaining QoS constraints.

---

## 🚀 Quick Start (Google Colab)

### **Option 1: One-Click Launch**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Research-GreenNetwork/blob/main/colab-training/GreenNetwork_Colab_Training.ipynb)

### **Option 2: Manual Setup**

1. **Upload to Google Drive**
   ```bash
   # Clone this repository
   git clone https://github.com/YOUR_USERNAME/Research-GreenNetwork.git
   cd Research-GreenNetwork/colab-training
   ```

2. **Open in Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → Upload → Select `GreenNetwork_Colab_Training.ipynb`

3. **Run All Cells**
   - Runtime → Run all (or Ctrl+F9)
   - Training will start automatically with GPU acceleration

---

## 📋 Features

### **🎯 Core Capabilities**
- ✅ **DQN with Adaptive Clustering**: DP-means algorithm for dynamic network partitioning
- ✅ **Energy Optimization**: Smart link deactivation to reduce power consumption
- ✅ **SLA-Aware Routing**: Maintains latency constraints for different traffic priorities
- ✅ **Traffic Load Modes**: Configurable low/high traffic scenarios
- ✅ **Real-time Visualization**: Live training metrics and network statistics

### **🔧 Colab-Specific Features**
- 🚀 **GPU Acceleration**: Automatic CUDA detection and usage
- 💾 **Google Drive Integration**: Save models and results to your Drive
- 📊 **Interactive Dashboards**: Real-time plots with matplotlib/plotly
- ⚡ **Fast Setup**: One-click installation of all dependencies
- 🎛️ **Configurable Scenarios**: Pre-built configs for different network sizes

---

## 🏗️ Repository Structure

```
colab-training/
├── GreenNetwork_Colab_Training.ipynb    # 🎯 Main training notebook
├── README.md                            # 📖 This file
├── requirements.txt                     # 📦 Python dependencies
│
├── configs/                             # ⚙️ Network configurations
│   ├── tiny_network.json               #   20 nodes (quick test)
│   ├── small_network.json              #   50 nodes (5 min training)
│   ├── medium_network.json             #   100 nodes (15 min training)
│   └── large_network.json              #   200 nodes (full experiment)
│
└── src/                                 # 🔬 Source code
    ├── env.py                          #   SDN environment
    ├── agent.py                        #   DQN agent
    ├── cluster.py                      #   Clustering algorithms
    ├── algorithm.py                    #   Link deactivation strategies
    └── utils.py                        #   Visualization & helpers
```

---

## 🎓 Training Scenarios

### **1. Quick Test (2-5 minutes)**
```python
config = "configs/tiny_network.json"
episodes = 100
# 20 nodes, 40 edges, 8 hosts
```

### **2. Small Network (5-10 minutes)**
```python
config = "configs/small_network.json"
episodes = 500
# 50 nodes, 100 edges, 20 hosts
```

### **3. Medium Network (15-30 minutes)**
```python
config = "configs/medium_network.json"
episodes = 1000
# 100 nodes, 500 edges, 40 hosts
```

### **4. Large Network (1-2 hours)**
```python
config = "configs/large_network.json"
episodes = 2000
# 200 nodes, 2000 edges, 80 hosts
```

---

## 📊 Metrics Tracked

### **Energy Efficiency**
- Energy saving percentage (vs. all-links-on baseline)
- Active links count
- Power consumption per episode

### **Quality of Service**
- Average end-to-end latency (ms)
- SLA violation rate (%)
- Throughput (flows/second)

### **Network Performance**
- Link utilization (average & max)
- Overloaded edges count
- Cluster count (adaptive clustering)

### **Training Progress**
- Episode rewards
- Loss values
- Epsilon decay (exploration rate)

---

## 🔬 Algorithm Overview

### **Our Method: DQN with Adaptive Clustering**

1. **Network Clustering** (DP-means)
   - Dynamically partition network into clusters
   - Adapt cluster count based on traffic patterns
   - Feature-based clustering (topology + traffic + service classes)

2. **Hierarchical Action Space**
   - Per-cluster utilization thresholds (9 bins)
   - Inter-cluster link preservation (min edges to keep)
   - Total actions: `9^K × 4` where K = max clusters

3. **Link Deactivation Strategy**
   - Greedy or priority-based algorithms
   - SLA-aware constraint checking
   - Connectivity preservation

4. **Reward Function**
   ```python
   reward = energy_saving - (latency_penalty + sla_penalty + overload_penalty)
   ```

---

## ⚙️ Configuration Options

### **Network Topology**
```json
{
  "num_nodes": 100,
  "num_edges": 500,
  "num_hosts": 40,
  "num_regions": 3
}
```

### **Training Hyperparameters**
```json
{
  "episodes": 1000,
  "max_steps_per_episode": 200,
  "batch_size": 256,
  "lr": 0.0003,
  "gamma": 0.95,
  "epsilon_start": 1.0,
  "epsilon_end": 0.05
}
```

### **Traffic Load Modes**
```json
{
  "traffic_load_mode": "low",  // or "high"
  "target_utilization_range": [0.2, 0.4],  // low: 20-40%, high: 60-80%
  "flow_intensity_multiplier": 2.0
}
```

### **Clustering Settings**
```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means_adaptive",
  "clustering_k_range": [2, 10],
  "no_clustering": false  // Set true to disable clustering
}
```

---

## 📈 Expected Results

| Metric | Our Method (DQN+Clustering) | Baseline (All Links On) |
|--------|----------------------------|------------------------|
| **Energy Saving** | 70-80% ✅ | 0% |
| **Avg Latency** | 10-15ms ✅ | 8-12ms |
| **SLA Violations** | 5-10% ✅ | 2-5% |
| **Computation Time** | 0.1-1s per step | <0.01s |
| **Scalability** | Good (up to 200 nodes) | Excellent |

---

## 🛠️ Troubleshooting

### **Out of Memory Error**
```python
# Reduce network size or batch size
config["num_nodes"] = 50
config["batch_size"] = 128
```

### **Training Too Slow**
```python
# Enable GPU runtime
# Runtime → Change runtime type → GPU (T4)

# Or reduce episodes
config["episodes"] = 500
```

### **Poor Convergence**
```python
# Adjust learning rate
config["lr"] = 0.001  # Increase for faster learning

# Increase exploration
config["epsilon_decay_steps"] = 10000  # Slower decay
```

### **High SLA Violations**
```python
# Increase SLA penalty weight in env.py
sla_penalty = 0.1 * sla_viol_pct  # Increase from 0.05

# Or disable aggressive link deactivation
config["cluster_threshold_bins"] = [0.3, 0.4, 0.5, 0.6, 0.7]
```

---

## 📚 References

1. **Energy-Aware Routing in SDN**
   - Paper: "Energy-aware routing algorithms in Software-Defined Networks"
   - Heuristic-based approach with utilization thresholds

2. **RL-Based Energy Routing**
   - Paper: "Reinforcement Learning and Energy-Aware Routing"
   - Tabular Q-learning without clustering

3. **DP-means Clustering**
   - Non-parametric k-means extension
   - Automatically determines cluster count

---

## 🤝 Contributing

Found a bug or want to improve the code? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- NetworkX for graph algorithms
- PyTorch for deep learning
- Google Colab for free GPU resources
- scikit-learn for clustering metrics

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

**Happy Training! 🚀**
