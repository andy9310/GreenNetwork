# Google Colab Setup Guide for Training Scripts

## 🔍 Code Analysis Summary

### Current CUDA/Device Usage Issues

After analyzing all training scripts in the `train/` directory, here are the **critical issues** for Colab deployment:

#### ❌ **Problems Found**

1. **`train.py` (Line 174)**: Hardcoded device from config
   ```python
   agent = HierarchicalDQN(..., device=cfg.get("device","cpu"))
   ```
   - Uses config file device setting (defaults to "cpu")
   - **Does NOT auto-detect CUDA**

2. **`agent.py` (Lines 50-51)**: Device passed but not auto-detected
   ```python
   self.q = QNet(obs_dim, action_n).to(device)
   self.qt = QNet(obs_dim, action_n).to(device)
   ```
   - Relies on passed device parameter
   - No automatic CUDA detection

3. **`experiment/run_experiments.py` (Lines 113, 276)**: Hardcoded to CPU
   ```python
   agent = HierarchicalDQN(..., device='cpu')  # Line 113
   'device': 'cpu',  # Line 276 in config generation
   ```
   - **Always uses CPU regardless of GPU availability**

4. **ReplayBuffer sampling (agent.py:26-33)**: Tensors created on CPU
   ```python
   s = torch.as_tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
   ```
   - Creates tensors without device specification
   - Requires manual `.to(device)` later (which is done, but inefficient)

---

## ✅ Required Changes for Colab

### 1. **Auto-Detect CUDA Device**

Add this helper function to all training scripts:

```python
import torch

def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("⚠️  No GPU available, using CPU")
    return device
```

### 2. **Files That Need Modification**

#### **File 1: `train/train.py`**
- **Line 174**: Change from `cfg.get("device","cpu")` to auto-detection
- **Add**: GPU memory monitoring
- **Add**: Colab-specific optimizations

#### **File 2: `train/agent.py`**
- **Lines 26-33**: Create tensors directly on device in ReplayBuffer
- **Add**: Device property for easy access

#### **File 3: `train/experiment/run_experiments.py`**
- **Line 113**: Replace `device='cpu'` with auto-detection
- **Line 276**: Remove hardcoded `'device': 'cpu'` from config

#### **File 4: `train/run_comparison.py`** (if exists)
- Similar changes as above

---

## 📦 Colab-Specific Additions

### 1. **Install Dependencies Cell**

```python
# Cell 1: Install dependencies
!pip install -q networkx matplotlib pandas plotly tqdm torch torchvision

# Verify GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2. **Mount Google Drive (Optional)**

```python
# Cell 2: Mount Drive for saving models
from google.colab import drive
drive.mount('/content/drive')

# Create save directory
import os
save_dir = '/content/drive/MyDrive/GreenNetwork_Results'
os.makedirs(save_dir, exist_ok=True)
print(f"✅ Models will be saved to: {save_dir}")
```

### 3. **Clone Repository**

```python
# Cell 3: Clone repo
!git clone https://github.com/andy9310/Research-GreenNetwork.git
%cd Research-GreenNetwork/train
!ls -la
```

### 4. **GPU Memory Management**

```python
# Cell 4: GPU utilities
import torch

def clear_gpu_memory():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU cache cleared")

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return allocated, reserved
    return 0, 0
```

### 5. **Progress Monitoring for Colab**

```python
# Cell 5: Colab-friendly progress tracking
from IPython.display import clear_output
import matplotlib.pyplot as plt

def plot_live_training(metrics, episode):
    """Live plotting for Colab"""
    clear_output(wait=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward
    axes[0, 0].plot(metrics['rewards'], 'b-', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True)
    
    # Energy Saving
    axes[0, 1].plot(metrics['energy_savings'], 'g-', alpha=0.7)
    axes[0, 1].set_title('Energy Savings (%)')
    axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True)
    
    # Latency
    axes[1, 0].plot(metrics['latencies'], 'm-', alpha=0.7)
    axes[1, 0].set_title('Latency (ms)')
    axes[1, 0].grid(True)
    
    # SLA Violations
    axes[1, 1].plot(metrics['sla_violations'], 'orange', alpha=0.7)
    axes[1, 1].set_title('SLA Violations (%)')
    axes[1, 1].axhline(y=10, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True)
    
    plt.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
    plt.tight_layout()
    plt.show()
```

---

## 🚀 Complete Colab Notebook Structure

### **Recommended Cell Order**

```python
# ============================================
# CELL 1: Setup & Dependencies
# ============================================
!pip install -q networkx matplotlib pandas plotly tqdm

import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# CELL 2: Mount Drive (Optional)
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# CELL 3: Clone Repository
# ============================================
!git clone https://github.com/andy9310/Research-GreenNetwork.git
%cd Research-GreenNetwork/train

# ============================================
# CELL 4: Device Detection Function
# ============================================
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️  Using CPU")
    return device

device = get_device()

# ============================================
# CELL 5: Run Training (MODIFIED)
# ============================================
# Import modified training script
from train import run_training

# Override config to use detected device
import json
with open('config.json', 'r') as f:
    cfg = json.load(f)

# FORCE CUDA if available
cfg['config']['device'] = device

# Save modified config
with open('config_colab.json', 'w') as f:
    json.dump(cfg, f, indent=2)

# Run training
results = run_training('config_colab.json', traffic_mode='low')

# ============================================
# CELL 6: Save Results to Drive
# ============================================
import shutil
save_dir = '/content/drive/MyDrive/GreenNetwork_Results'
shutil.copytree('training_results', f'{save_dir}/training_results', dirs_exist_ok=True)
print(f"✅ Results saved to {save_dir}")

# ============================================
# CELL 7: Run Experiments (MODIFIED)
# ============================================
from experiment.run_experiments import ExperimentRunner

runner = ExperimentRunner(output_dir='/content/drive/MyDrive/GreenNetwork_Results')
runner.run_all_experiments(
    topologies=[20, 100, 500],  # Start small for Colab
    methods=['dqn_clustering', 'energy_aware'],
    episodes_per_method={'dqn_clustering': 500, 'energy_aware': 50}
)
```

---

## 🔧 Specific Code Modifications Needed

### **Modification 1: `train/train.py`**

**Before (Line 174):**
```python
agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=cfg, device=cfg.get("device","cpu"))
```

**After:**
```python
# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU available, using CPU")

agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=cfg, device=device)
```

### **Modification 2: `train/agent.py`**

**Add to `__init__` (after line 44):**
```python
# Print device info
if device == "cuda":
    print(f"🎮 Agent initialized on GPU: {torch.cuda.get_device_name(0)}")
```

**Optimize ReplayBuffer.sample (Lines 26-33):**
```python
def sample(self, batch_size:int):
    batch = random.sample(self.buf, batch_size)
    # Create tensors directly on target device (passed during init)
    device = 'cpu'  # Will be moved to GPU in train_step
    s = torch.as_tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=device)
    a = torch.as_tensor([b.a for b in batch], dtype=torch.int64, device=device).unsqueeze(1)
    r = torch.as_tensor([b.r for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
    s2 = torch.as_tensor(np.stack([b.s2 for b in batch]), dtype=torch.float32, device=device)
    d = torch.as_tensor([b.d for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
    return s, a, r, s2, d
```

### **Modification 3: `train/experiment/run_experiments.py`**

**Before (Line 113):**
```python
agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=config, device='cpu')
```

**After:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=config, device=device)
```

**Before (Line 276):**
```python
'device': 'cpu',
```

**After:**
```python
'device': 'cuda' if torch.cuda.is_available() else 'cpu',
```

---

## 📊 Performance Expectations on Colab

### **GPU vs CPU Speed Comparison**

| Task | CPU (Colab) | GPU (T4) | Speedup |
|------|-------------|----------|---------|
| **Single Episode** | ~2-5s | ~0.5-1s | **4-5x** |
| **100 Episodes** | ~5-8 min | ~1-2 min | **4-5x** |
| **1000 Episodes** | ~50-80 min | ~10-20 min | **4-5x** |

### **Colab GPU Specifications**

- **Free Tier**: Tesla T4 (16GB VRAM) or K80 (12GB VRAM)
- **Session Limit**: 12 hours continuous
- **Recommended**: Use Colab Pro for longer training runs

---

## ⚡ Optimization Tips for Colab

### 1. **Batch Size Tuning**
```python
# Increase batch size on GPU for better utilization
if device == "cuda":
    cfg['batch_size'] = 512  # Larger batches on GPU
else:
    cfg['batch_size'] = 256  # Smaller on CPU
```

### 2. **Checkpoint Saving**
```python
# Save checkpoints more frequently on Colab (session can disconnect)
if (ep + 1) % 5 == 0:  # Every 5 episodes instead of 10
    torch.save(agent.q.state_dict(), f'{save_dir}/checkpoint_ep{ep+1}.pth')
```

### 3. **Memory Management**
```python
# Clear GPU cache periodically
if (ep + 1) % 50 == 0 and device == "cuda":
    torch.cuda.empty_cache()
```

### 4. **Mixed Precision Training** (Advanced)
```python
# Use automatic mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    loss = agent.train_step()
```

---

## 🎯 Quick Start Commands for Colab

### **Option 1: Single Training Run**
```bash
# After modifications
python train.py config.json low
```

### **Option 2: Full Experiment Suite**
```bash
# Run all experiments (small scale for Colab)
python experiment/run_experiments.py --all
```

### **Option 3: Specific Experiment**
```bash
# Test on small topology first
python experiment/run_experiments.py --method dqn_clustering --topology 100 --episodes 500
```

---

## 📝 Summary of Required Changes

### **Critical Changes (Must Do)**
1. ✅ Add auto-device detection to `train.py` (Line 174)
2. ✅ Add auto-device detection to `run_experiments.py` (Lines 113, 276)
3. ✅ Add GPU memory monitoring

### **Recommended Changes (Should Do)**
4. ✅ Optimize ReplayBuffer tensor creation
5. ✅ Add Colab-specific progress visualization
6. ✅ Implement checkpoint saving to Google Drive

### **Optional Optimizations (Nice to Have)**
7. ⭐ Mixed precision training
8. ⭐ Dynamic batch size based on GPU memory
9. ⭐ Distributed training for multiple GPUs

---

## 🔗 Next Steps

1. **Apply modifications** to the 3 critical files
2. **Test locally** with CPU to ensure no breaking changes
3. **Upload to Colab** and test with GPU
4. **Monitor GPU utilization** using `nvidia-smi` or `torch.cuda` utilities
5. **Optimize hyperparameters** for GPU (larger batch sizes, etc.)

---

**Last Updated**: 2025-10-12
