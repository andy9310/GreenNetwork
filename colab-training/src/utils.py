"""
Utility functions for Colab training visualization and monitoring
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ColabVisualizer:
    """Real-time visualization for Colab training"""
    
    def __init__(self, update_every=10):
        self.update_every = update_every
        self.metrics = {
            'episode': [],
            'reward': [],
            'loss': [],
            'energy_saving': [],
            'latency': [],
            'sla_violations': [],
            'active_links': [],
            'utilization': [],
            'cluster_count': []
        }
        
    def log(self, episode, reward, loss, energy_saving, latency, sla_viol, 
            active_links, utilization, cluster_count):
        """Log metrics for an episode"""
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['loss'].append(loss if loss is not None else 0.0)
        self.metrics['energy_saving'].append(energy_saving * 100)
        self.metrics['latency'].append(latency)
        self.metrics['sla_violations'].append(sla_viol)
        self.metrics['active_links'].append(active_links)
        self.metrics['utilization'].append(utilization)
        self.metrics['cluster_count'].append(cluster_count)
        
    def plot_live(self, episode):
        """Plot live training progress"""
        if episode % self.update_every != 0:
            return
            
        clear_output(wait=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16, fontweight='bold')
        
        # Reward
        axes[0, 0].plot(self.metrics['episode'], self.metrics['reward'], 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('Episode Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy Saving
        axes[0, 1].plot(self.metrics['episode'], self.metrics['energy_saving'], 'g-', alpha=0.7, linewidth=2)
        axes[0, 1].set_title('Energy Saving (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Energy Saving %')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Target: 70%')
        axes[0, 1].legend()
        
        # Latency
        axes[0, 2].plot(self.metrics['episode'], self.metrics['latency'], 'm-', alpha=0.7, linewidth=2)
        axes[0, 2].set_title('Average Latency (ms)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Latency (ms)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # SLA Violations
        axes[1, 0].plot(self.metrics['episode'], self.metrics['sla_violations'], 'orange', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('SLA Violations (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('SLA Violation %')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Target: <10%')
        axes[1, 0].legend()
        
        # Active Links
        axes[1, 1].plot(self.metrics['episode'], self.metrics['active_links'], 'purple', alpha=0.7, linewidth=2)
        axes[1, 1].set_title('Active Links', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Active Links')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Utilization
        axes[1, 2].plot(self.metrics['episode'], self.metrics['utilization'], 'cyan', alpha=0.7, linewidth=2)
        axes[1, 2].set_title('Network Utilization (%)', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Utilization %')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhspan(20, 40, alpha=0.2, color='green', label='Target Range')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_interactive(self):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Reward', 'Energy Saving (%)', 'Latency (ms)', 
                          'SLA Violations (%)', 'Active Links', 'Utilization (%)'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Reward
        fig.add_trace(
            go.Scatter(x=self.metrics['episode'], y=self.metrics['reward'], 
                      mode='lines', name='Reward', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Energy Saving
        fig.add_trace(
            go.Scatter(x=self.metrics['episode'], y=self.metrics['energy_saving'], 
                      mode='lines', name='Energy Saving', line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2)
        
        # Latency
        fig.add_trace(
            go.Scatter(x=self.metrics['episode'], y=self.metrics['latency'], 
                      mode='lines', name='Latency', line=dict(color='magenta', width=2)),
            row=2, col=1
        )
        
        # SLA Violations
        fig.add_trace(
            go.Scatter(x=self.metrics['episode'], y=self.metrics['sla_violations'], 
                      mode='lines', name='SLA Violations', line=dict(color='orange', width=2)),
            row=2, col=2
        )
        fig.add_hline(y=10, line_dash="dash", line_color="red", opacity=0.5, row=2, col=2)
        
        # Active Links
        fig.add_trace(
            go.Scatter(x=self.metrics['episode'], y=self.metrics['active_links'], 
                      mode='lines', name='Active Links', line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # Utilization
        fig.add_trace(
            go.Scatter(x=self.metrics['episode'], y=self.metrics['utilization'], 
                      mode='lines', name='Utilization', line=dict(color='cyan', width=2)),
            row=3, col=2
        )
        
        fig.update_layout(height=900, showlegend=False, title_text="Training Dashboard")
        fig.show()
        
    def get_summary(self):
        """Get training summary statistics"""
        if len(self.metrics['episode']) == 0:
            return "No data yet"
            
        recent_window = min(50, len(self.metrics['reward']))
        
        summary = {
            'Total Episodes': len(self.metrics['episode']),
            'Avg Reward (Last 50)': np.mean(self.metrics['reward'][-recent_window:]),
            'Avg Energy Saving (Last 50)': np.mean(self.metrics['energy_saving'][-recent_window:]),
            'Avg Latency (Last 50)': np.mean(self.metrics['latency'][-recent_window:]),
            'Avg SLA Violations (Last 50)': np.mean(self.metrics['sla_violations'][-recent_window:]),
            'Best Reward': max(self.metrics['reward']),
            'Best Energy Saving': max(self.metrics['energy_saving']),
        }
        
        return pd.DataFrame([summary]).T.rename(columns={0: 'Value'})
    
    def save_to_csv(self, filepath):
        """Save metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)
        print(f"✅ Metrics saved to {filepath}")


def setup_colab_environment():
    """Setup Colab environment with necessary configurations"""
    import os
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except:
        IN_COLAB = False
        print("⚠️  Not running in Colab")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("⚠️  No GPU available, using CPU")
    
    return device, IN_COLAB


def mount_google_drive():
    """Mount Google Drive for saving models"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        save_dir = '/content/drive/MyDrive/GreenNetwork_Models'
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ Google Drive mounted. Models will be saved to: {save_dir}")
        return save_dir
    except:
        print("⚠️  Could not mount Google Drive. Using local storage.")
        return './models'


def create_progress_bar(total, desc="Training"):
    """Create a tqdm progress bar"""
    from tqdm.notebook import tqdm
    return tqdm(total=total, desc=desc, ncols=100)


def print_training_header(config):
    """Print formatted training configuration"""
    print("=" * 80)
    print("🌿 GreenNetwork Training - Energy-Aware SDN Routing".center(80))
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Network: {config['num_nodes']} nodes, {config['num_edges']} edges, {config['num_hosts']} hosts")
    print(f"   Episodes: {config['episodes']}, Steps/Episode: {config['max_steps_per_episode']}")
    print(f"   Traffic Mode: {config['traffic_load_mode']}")
    print(f"   Clustering: {config['clustering_method']} (K range: {config['clustering_k_range']})")
    print(f"   Device: {config['device']}")
    print(f"   Batch Size: {config['batch_size']}, Learning Rate: {config['lr']}")
    print("=" * 80 + "\n")


def print_episode_summary(episode, total_episodes, metrics):
    """Print formatted episode summary"""
    print(f"\n{'='*80}")
    print(f"Episode {episode}/{total_episodes} Summary".center(80))
    print(f"{'='*80}")
    print(f"  Reward:          {metrics['reward']:8.2f}")
    print(f"  Energy Saving:   {metrics['energy_saving']*100:6.1f}%")
    print(f"  Latency:         {metrics['latency']:6.2f} ms")
    print(f"  SLA Violations:  {metrics['sla_violations']:6.1f}%")
    print(f"  Active Links:    {metrics['active_links']:4d}")
    print(f"  Utilization:     {metrics['utilization']:6.1f}%")
    print(f"  Cluster Count:   {metrics['cluster_count']:4d}")
    print(f"{'='*80}\n")
