# 🚀 Quick Start Guide

Get started with GreenNetwork training in Google Colab in under 5 minutes!

---

## ⚡ Super Quick Start (1 minute)

1. **Open the notebook in Colab**
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Research-GreenNetwork/blob/main/colab-training/GreenNetwork_Colab_Training.ipynb)

2. **Run all cells**
   - Click `Runtime` → `Run all` (or press `Ctrl+F9`)
   - Training will start automatically!

3. **Watch the magic happen** ✨
   - Real-time plots update every 10 episodes
   - Models auto-save to Google Drive
   - GPU acceleration enabled by default

---

## 📝 Step-by-Step Guide (5 minutes)

### Step 1: Upload to Colab

**Option A: Direct Upload**
```bash
# Download this repository
git clone https://github.com/YOUR_USERNAME/Research-GreenNetwork.git
cd Research-GreenNetwork/colab-training
```

Then:
1. Go to [Google Colab](https://colab.research.google.com/)
2. `File` → `Upload notebook`
3. Select `GreenNetwork_Colab_Training.ipynb`

**Option B: From GitHub**
1. Go to [Google Colab](https://colab.research.google.com/)
2. `File` → `Open notebook` → `GitHub` tab
3. Enter: `YOUR_USERNAME/Research-GreenNetwork`
4. Select `colab-training/GreenNetwork_Colab_Training.ipynb`

---

### Step 2: Enable GPU (Recommended)

1. Click `Runtime` → `Change runtime type`
2. Select `GPU` (T4 or better)
3. Click `Save`

**Why GPU?** Training is 10-50× faster with GPU acceleration!

---

### Step 3: Run the Notebook

**Execute cells in order:**

1. **Setup & Installation** (Cell 1-2)
   - Installs dependencies
   - Takes ~30 seconds

2. **Import Libraries** (Cell 3)
   - Loads all necessary modules

3. **Environment Setup** (Cell 4-5)
   - Detects GPU
   - Mounts Google Drive (optional)

4. **Configuration** (Cell 6-7)
   - Choose network size: `tiny`, `small`, `medium`, or `large`
   - Default: `small_network` (5-10 min training)

5. **Initialize** (Cell 8)
   - Creates environment and agent

6. **Training** (Cell 9) 🎯
   - **This is the main training loop**
   - Live plots update automatically
   - Can be interrupted anytime (Runtime → Interrupt)

7. **Results** (Cells 10-12)
   - View summary statistics
   - Interactive Plotly dashboard
   - Clustering analysis

8. **Evaluation** (Cell 13)
   - Test the trained model
   - Greedy policy (no exploration)

9. **Export** (Cell 14)
   - Download results to your computer

---

## 🎯 Training Scenarios

### Quick Test (2-5 minutes)
```python
CONFIG_NAME = 'tiny_network'
# 20 nodes, 40 edges, 100 episodes
```

### Small Network (5-10 minutes) ⭐ **Recommended**
```python
CONFIG_NAME = 'small_network'
# 50 nodes, 100 edges, 500 episodes
```

### Medium Network (15-30 minutes)
```python
CONFIG_NAME = 'medium_network'
# 100 nodes, 500 edges, 1000 episodes
```

### Large Network (1-2 hours)
```python
CONFIG_NAME = 'large_network'
# 200 nodes, 2000 edges, 2000 episodes
```

---

## 🎛️ Customization

### Change Traffic Load
```python
# In Cell 7 (Custom Configuration)
config['traffic_load_mode'] = 'high'  # or 'low'
```

### Disable Clustering
```python
config['no_clustering'] = True
```

### Adjust Learning Rate
```python
config['lr'] = 0.001  # Default: 0.0003
```

### Change Episodes
```python
config['episodes'] = 1000  # Adjust training length
```

---

## 📊 What to Expect

### Training Progress

You'll see real-time plots showing:
- **Reward**: Should increase over time
- **Energy Saving**: Target 70-80%
- **Latency**: Should stay low (10-15ms)
- **SLA Violations**: Target <10%
- **Active Links**: Decreases as agent learns to save energy
- **Utilization**: Should stay in target range (20-40% for low traffic)

### Convergence

- **Tiny network**: Converges in ~50 episodes
- **Small network**: Converges in ~200-300 episodes
- **Medium network**: Converges in ~500-700 episodes
- **Large network**: Converges in ~1000-1500 episodes

### Good Results

✅ Energy saving: 70-80%
✅ SLA violations: <10%
✅ Latency: 10-15ms
✅ Reward: Positive and increasing

---

## 🐛 Troubleshooting

### "Out of Memory" Error

**Solution 1**: Reduce network size
```python
CONFIG_NAME = 'tiny_network'  # or 'small_network'
```

**Solution 2**: Reduce batch size
```python
config['batch_size'] = 128  # Default: 256
```

**Solution 3**: Enable GPU
- Runtime → Change runtime type → GPU

---

### Training Too Slow

**Solution 1**: Enable GPU (if not already)
- Runtime → Change runtime type → GPU (T4)

**Solution 2**: Reduce episodes
```python
config['episodes'] = 500  # Reduce from 1000
```

**Solution 3**: Use smaller network
```python
CONFIG_NAME = 'small_network'
```

---

### Poor Performance (Low Energy Saving)

**Solution 1**: Train longer
```python
config['episodes'] = 2000  # Increase episodes
```

**Solution 2**: Adjust exploration
```python
config['epsilon_decay_steps'] = 10000  # Slower decay
```

**Solution 3**: Increase learning rate
```python
config['lr'] = 0.001  # Faster learning
```

---

### High SLA Violations

**Solution 1**: Increase SLA penalty (in `src/env.py`)
```python
sla_penalty = 0.1 * sla_viol_pct  # Increase from 0.05
```

**Solution 2**: Use less aggressive thresholds
```python
config['cluster_threshold_bins'] = [0.3, 0.4, 0.5, 0.6, 0.7]
```

---

## 💾 Saving & Loading Models

### Auto-Save

Models are automatically saved:
- **Best model**: `best_model_{CONFIG_NAME}.pth` (highest reward)
- **Checkpoints**: `checkpoint_{CONFIG_NAME}_ep{N}.pth` (every N episodes)

### Save Location

- **With Google Drive**: `/content/drive/MyDrive/GreenNetwork_Models/`
- **Without Google Drive**: `./models/`

### Load a Model

```python
# Load best model
agent.load('path/to/best_model_small_network.pth', map_location=device)
```

---

## 📈 Monitoring Training

### Live Plots

Plots update every 10 episodes automatically. To change frequency:
```python
visualizer = ColabVisualizer(update_every=5)  # Update every 5 episodes
```

### Progress Bar

Shows real-time metrics:
- Current reward
- Energy saving %
- SLA violations %
- Exploration rate (ε)

### TensorBoard (Advanced)

Add to notebook:
```python
%load_ext tensorboard
%tensorboard --logdir ./logs
```

---

## 🎓 Next Steps

After your first successful training:

1. **Compare traffic modes**: Try `'low'` vs `'high'` traffic
2. **Test without clustering**: Set `no_clustering = True`
3. **Experiment with algorithms**: Change `deactivation_algorithm` to `'greedy'`
4. **Scale up**: Try larger networks
5. **Hyperparameter tuning**: Adjust learning rate, epsilon decay, etc.

---

## 📚 Additional Resources

- **Main README**: `../README.md`
- **Experiment Guide**: `../train/experiment/README.md`
- **Source Code**: `src/` directory
- **Configurations**: `configs/` directory

---

## 🤝 Need Help?

- **GitHub Issues**: [Open an issue](https://github.com/YOUR_USERNAME/Research-GreenNetwork/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/Research-GreenNetwork/discussions)
- **Email**: your.email@example.com

---

## 🎉 Success Checklist

- [ ] Notebook opens in Colab
- [ ] GPU is enabled
- [ ] Dependencies install successfully
- [ ] Training starts without errors
- [ ] Live plots appear
- [ ] Models save to Google Drive
- [ ] Energy saving reaches 70%+
- [ ] SLA violations stay below 10%

**All checked? Congratulations! You're ready to experiment! 🚀**

---

**Happy Training!** 🌿
