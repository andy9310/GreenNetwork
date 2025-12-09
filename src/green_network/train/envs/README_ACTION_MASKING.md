# Cluster-Based Multi-Agent Environment with Action Masking

## Overview

This implementation provides a **cluster-based multi-agent SDN environment** where:
- **Agents are clusters of nodes** (not individual nodes)
- Each cluster has a **variable number of nodes**
- **Fixed action dimension** across all agents (based on max cluster size)
- **Action masking** handles variable cluster sizes elegantly

## Key Features

### 1. Automatic Clustering
```python
env = SDNEnv(num_clusters=10)  # Creates 10 cluster agents
```

- Uses **KMeans clustering** based on:
  - Node degree
  - Local network density
  - Optional: traffic patterns
- Configurable via `num_clusters` parameter

### 2. Fixed Action Dimension with Masking

**Problem:** Clusters have different sizes (e.g., Cluster A has 5 nodes, Cluster B has 12 nodes)

**Solution:** 
- All agents have action space size = `max_cluster_size × 3`
- Smaller clusters use **action masking** to disable unused actions
- Training algorithms (PPO/MAPPO) apply mask during action selection

**Example:**
```python
# Cluster 0: 5 nodes → 15 valid actions (5 × 3)
# Cluster 1: 12 nodes → 36 valid actions (12 × 3)
# Both have action_space = DiscreteSpec(36) with masking

action_mask = env.get_action_mask(cluster_id)
# For Cluster 0: [1,1,1,...,1,0,0,...,0]  (15 ones, 21 zeros)
# For Cluster 1: [1,1,1,...,1,1,1,...,1]  (36 ones)
```

### 3. Action Space Design

Each cluster agent controls multiple nodes. The action is **flattened**:

```
action_idx = node_position × 3 + mode_choice

where:
- node_position ∈ [0, max_cluster_size-1]
- mode_choice ∈ {0, 1, 2}  (low/normal/high power mode)
```

**Example for a cluster with 3 nodes:**
- Actions 0-2: Node 0 (modes 0, 1, 2)
- Actions 3-5: Node 1 (modes 0, 1, 2)  
- Actions 6-8: Node 2 (modes 0, 1, 2)
- Actions 9+: Masked (invalid)

### 4. Observation Space

Each agent receives:

```python
observation = [
    cluster_features,      # [3] - avg_degree, size, total_load
    node_features,         # [max_cluster_size × node_feat_dim] - padded
    action_mask           # [max_action_dim] - binary mask
]
```

**Node features** (per node):
- Normalized degree
- Current power mode (0/1/2)
- Local link load
- Neighbor edge utilizations (padded to max_degree)

## Usage Guide

### Basic Usage

```python
from env import SDNEnv

# Initialize environment
env = SDNEnv(num_clusters=8)

# Reset
obs = env.reset()  # Returns: Dict[cluster_id, observation]

# Get action masks
action_masks = {cid: env.get_action_mask(cid) for cid in env.agent_ids}

# Select actions (must respect masks!)
actions = {}
for agent_id in env.agent_ids:
    mask = action_masks[agent_id]
    valid_actions = np.where(mask > 0)[0]
    action = np.random.choice(valid_actions)
    actions[agent_id] = int(action)

# Step
next_obs, rewards, dones, infos = env.step(actions)
```

### Integration with MAPPO/PPO

```python
import torch
from torch.distributions import Categorical

class MaskedPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, action_mask):
        """
        Args:
            obs: [batch, obs_dim] - includes action mask at the end
            action_mask: [batch, action_dim] - binary mask
        
        Returns:
            action, log_prob
        """
        logits = self.network(obs)  # [batch, action_dim]
        
        # Apply mask: set invalid actions to -inf
        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e9
        
        # Create distribution
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob


# Training loop
env = SDNEnv(num_clusters=10)
policy = MaskedPolicy(env.obs_dim, env.max_action_dim)

for episode in range(num_episodes):
    obs = env.reset()
    
    for step in range(max_steps):
        # Extract action masks from observations or get separately
        action_masks = {aid: env.get_action_mask(aid) for aid in env.agent_ids}
        
        # Convert to tensors
        obs_batch = torch.stack([torch.FloatTensor(obs[aid]) for aid in env.agent_ids])
        mask_batch = torch.stack([torch.FloatTensor(action_masks[aid]) for aid in env.agent_ids])
        
        # Select actions
        actions, log_probs = policy(obs_batch, mask_batch)
        
        # Convert back to dict
        action_dict = {aid: int(actions[i]) for i, aid in enumerate(env.agent_ids)}
        
        # Step
        next_obs, rewards, dones, infos = env.step(action_dict)
        
        # Store and update...
```

### Extracting Action Mask from Observation

The action mask is **included at the end** of each observation:

```python
def extract_features_and_mask(obs_vec, env):
    """
    Split observation into features and mask
    """
    mask_start = env.obs_dim - env.max_action_dim
    features = obs_vec[:mask_start]
    action_mask = obs_vec[mask_start:]
    return features, action_mask

# Usage
obs = env.reset()
for agent_id in env.agent_ids:
    features, mask = extract_features_and_mask(obs[agent_id], env)
    # Use features for value/policy network
    # Use mask for action sampling
```

## Configuration

Edit `config/train_config.json`:

```json
{
    "num_clusters": 10,          // Number of cluster agents
    "max_steps": 50,             // Episode length
    "num_flows_per_step": 5,     // Traffic flows per step
    "base_capacity": 10.0,       // Base link capacity
    "base_power": 1.0,           // Base power consumption
    "base_delay": 1.0,           // Base delay
    "w_energy": 0.1,             // Energy weight in reward
    "w_delay": 1.0,              // Delay weight in reward
    "w_overload": 2.0            // Overload penalty weight
}
```

## API Reference

### SDNEnv

**Constructor:**
```python
SDNEnv(config_path=None, num_clusters=None)
```

**Key Methods:**

```python
# Reset environment
obs = env.reset() -> Dict[int, np.ndarray]

# Get action mask for an agent
mask = env.get_action_mask(agent_id) -> np.ndarray

# Step environment
next_obs, rewards, dones, infos = env.step(actions: Dict[int, int])

# Get cluster information
info = env.get_cluster_info(cluster_id) -> Dict[str, Any]
```

**Key Attributes:**

```python
env.agent_ids              # List of cluster IDs
env.num_agents             # Number of clusters
env.max_action_dim         # Fixed action dimension
env.obs_dim                # Observation dimension
env.cluster_to_nodes       # Dict mapping cluster_id -> list of node_ids
env.max_cluster_size       # Size of largest cluster
env.cluster_sizes          # Dict mapping cluster_id -> size
```

## Example Output

```
[SDNEnv] Clustering 40 nodes into 10 clusters...
[SDNEnv] Created 10 cluster agents
  Cluster sizes: min=2, max=6, avg=4.0
  Action dim per agent: 18 (with masking)
  Observation dim: 543

Cluster 0:
  Nodes: [0, 5, 12, 23]
  Size: 4
  Valid actions: 12 out of 18

Cluster 1:
  Nodes: [1, 8, 15, 19, 27, 35]
  Size: 6
  Valid actions: 18 out of 18
```

## Advantages of This Approach

1. **Compatible with standard MARL libraries** (RLlib, MAPPO implementations)
2. **No custom action space definitions** needed
3. **Simple to implement** action masking in policy networks
4. **Efficient**: Fixed-size tensors for batching
5. **Flexible**: Works with any clustering algorithm
6. **Scalable**: Handles large networks with variable cluster sizes

## Common Pitfalls

❌ **Don't:** Sample actions without checking the mask
```python
# BAD - may select invalid action
action = np.random.randint(0, env.max_action_dim)
```

✅ **Do:** Filter with mask first
```python
# GOOD - only selects valid actions
mask = env.get_action_mask(agent_id)
valid_actions = np.where(mask > 0)[0]
action = np.random.choice(valid_actions)
```

❌ **Don't:** Forget to apply mask in policy network
```python
# BAD - logits include invalid actions
dist = Categorical(logits=logits)
```

✅ **Do:** Mask logits before distribution
```python
# GOOD - invalid actions have ~0 probability
masked_logits = logits.clone()
masked_logits[action_mask == 0] = -1e9
dist = Categorical(logits=masked_logits)
```

## Testing

Run the example script to verify the implementation:

```bash
cd src/green_network/train/envs
python example_usage.py
```

This will demonstrate:
- Random policy with action masking
- Observation structure
- Integration patterns for training

## Next Steps

1. **Integrate with your MAPPO agent**: Modify `agent.select_actions()` to accept and use action masks
2. **Add mask to buffer**: Store action masks in your replay buffer for training
3. **Test with different cluster counts**: Experiment with `num_clusters` parameter
4. **Visualize clusters**: Use `visualize_clusters.py` to see the clustering results

## Questions?

Check `example_usage.py` for concrete usage patterns and integration examples.
