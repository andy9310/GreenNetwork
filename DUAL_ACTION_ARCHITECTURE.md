# Dual-Action SDN Architecture

## Overview

This document describes the dual-action reinforcement learning architecture for Software-Defined Networking (SDN) optimization. The system uses **two types of actions** to jointly optimize network performance and energy consumption:

1. **Edge Control**: Binary decisions to open/close network links
2. **Path Selection**: Routing decisions for traffic flows

---

## Architecture Components

### 1. Dual-Action Policy Network

**File**: `src/green_network/train/network/policy.py`

The `DualActionPolicy` class implements a multi-head neural network with:

```python
class DualActionPolicy(nn.Module):
    def __init__(self, obs_dim, num_edges, num_flows, k_paths=3):
        # Shared feature extractor
        self.shared_net = nn.Sequential(...)
        
        # Head 1: Edge control (binary per edge)
        self.edge_control_head = nn.Sequential(...)
        
        # Head 2: Path selection (categorical per flow)
        self.path_select_head = nn.Sequential(...)
```

**Key Features**:
- **Shared backbone**: Learns common representations
- **Edge control head**: Outputs binary decisions (Bernoulli distribution)
- **Path selection head**: Outputs categorical decisions (Categorical distribution)
- **Independent sampling**: Actions from both heads sampled independently

**Action Output**:
```python
edge_actions, path_actions, log_probs = policy.act(obs)
# edge_actions: [batch, num_edges] - binary (0/1)
# path_actions: [batch, num_flows] - indices (0 to k-1)
# log_probs: dict with edge, path, and total log probabilities
```

---

### 2. Dual-Action Environment

**File**: `src/green_network/train/envs/env_dual_action.py`

The `DualActionSDNEnv` implements the environment with dual action space:

#### Action Space

```python
actions = {
    'edge_control': [0/1] * num_edges,      # Binary per edge
    'path_selection': [0 to k-1] * num_flows # Path index per flow
}
```

#### Observation Space

Global centralized observation includes:
- **Edge utilizations**: `[num_edges]` - current load/capacity ratio per edge
- **Edge states**: `[num_edges]` - binary open/closed status
- **Flow demands**: `[num_flows]` - normalized traffic demands

Total dimension: `2 * num_edges + num_flows`

#### Step Function

```python
next_obs, reward, done, info = env.step(actions)
```

**Process**:
1. Update edge states from `edge_control` actions
2. Compute k-shortest paths for each flow (avoiding closed edges)
3. Route flows on selected paths from `path_selection` actions
4. Compute edge loads, utilization, and metrics
5. Calculate reward based on energy, delay, and overload

---

### 3. Path Computation Utilities

**File**: `src/green_network/train/envs/env_util/path_utils.py`

Key functions:

#### `compute_k_shortest_paths`
```python
paths = compute_k_shortest_paths(graph, source, target, k=3, closed_edges=None)
# Returns: List of k shortest paths (each path is list of nodes)
```

#### `compute_all_flow_paths`
```python
flow_paths = compute_all_flow_paths(graph, flows, k=3, closed_edges=None)
# Returns: Dict mapping flow_id -> list of k paths
```

#### `route_flows_on_paths`
```python
edge_loads, disconnect_penalty = route_flows_on_paths(
    flows, flow_paths, path_selections, num_edges_dict
)
# Returns: Edge loads and penalty for disconnected flows
```

#### `compute_edge_metrics`
```python
total_energy, total_delay, total_overload = compute_edge_metrics(
    edge_loads, edge_capacities, edge_states, base_power, base_delay
)
# Returns: Aggregated metrics for reward computation
```

---

## Reward Function

The reward is designed to balance multiple objectives:

```python
cost = (
    w_energy * total_energy +      # Energy consumption
    w_delay * total_delay +        # Network delay
    w_overload * total_overload    # Capacity violations
) / num_edges

reward = -cost  # Maximize reward = minimize cost
```

### Components:

1. **Energy**: 
   - Closed edges save power (no base power consumption)
   - Open edges consume `base_power * (1 + 0.5 * utilization)`
   
2. **Delay**: 
   - Convex function of utilization: `base_delay * (1 + util²)`
   - Increases rapidly with congestion
   
3. **Overload**:
   - Penalty when `utilization > 1.0`
   - Heavy penalty for using closed edges
   - Severe penalty for disconnected flows (no valid path)

### Trade-offs:

- **Close edges** → Save energy BUT may increase delay (longer paths) or cause overload
- **Keep edges open** → More routing options BUT higher energy cost
- **Select paths** → Balance between shortest path (low delay) and load distribution (avoid overload)

---

## Configuration Parameters

**File**: `src/green_network/train/config/train_config.json`

### Environment Parameters
```json
{
    "num_flows_per_step": 5,      // Number of traffic flows per timestep
    "k_paths": 3,                  // Number of alternative paths per flow
    "max_steps": 50,               // Episode length
    
    "base_capacity": 10.0,         // Base link capacity
    "base_power": 1.0,             // Base power consumption
    "base_delay": 1.0,             // Base link delay
    
    "w_energy": 0.1,               // Energy weight in reward
    "w_delay": 1.0,                // Delay weight in reward
    "w_overload": 2.0              // Overload weight in reward
}
```

### Training Parameters
```json
{
    "num_episodes": 1000,          // Total training episodes
    "rollout_steps": 512,          // Steps per rollout
    "lr": 0.0003,                  // Learning rate
    "ppo_epochs": 10,              // PPO update epochs
    "gamma": 0.99,                 // Discount factor
    "lam": 0.95                    // GAE lambda
}
```

---

## Usage Examples

### 1. Basic Training

```python
from envs.env_dual_action import DualActionSDNEnv
from network.policy import DualActionPolicy

# Create environment
env = DualActionSDNEnv(config_path="config/train_config.json")

# Create policy
policy = DualActionPolicy(
    obs_dim=env.obs_dim,
    num_edges=env.num_edges,
    num_flows=env.num_flows_per_step,
    k_paths=env.k_paths
)

# Training loop
for episode in range(num_episodes):
    obs = env.reset()
    
    for step in range(rollout_steps):
        # Sample actions
        edge_actions, path_actions, log_probs = policy.act(obs_tensor)
        
        # Execute in environment
        actions = {
            'edge_control': edge_actions.numpy(),
            'path_selection': path_actions.numpy()
        }
        next_obs, reward, done, info = env.step(actions)
        
        # Store and update...
```

### 2. Run Training Script

```bash
# Basic training
python src/green_network/train/trainer/train_dual_action.py

# Run demonstration mode
python src/green_network/train/trainer/train_dual_action.py --demo
```

### 3. Custom Configuration

Modify `config/train_config.json`:

```json
{
    "k_paths": 5,          // Try 5 alternative paths
    "w_energy": 0.5,       // Increase energy importance
    "w_overload": 5.0,     // Heavily penalize overload
    "num_flows_per_step": 10  // More traffic load
}
```

---

## Key Design Decisions

### 1. Centralized vs Distributed Control

**Current**: Centralized controller
- Single agent controls all edges and routes all flows
- Global observation of network state
- Easier to train, better coordination

**Alternative**: Distributed per-node agents
- Each node controls incident edges
- Local observations only
- More scalable but harder coordination

### 2. K-Shortest Paths

**Why k paths?**
- Provides routing flexibility
- Agent learns to balance load across paths
- Handles edge closures gracefully

**Path selection**:
- Computed dynamically each step
- Considers closed edges
- Falls back to shortest path if alternatives unavailable

### 3. Action Space Design

**Edge Control**: Binary (Bernoulli)
- Simple: open or close
- Energy savings from closing edges
- Risk of disconnection

**Path Selection**: Categorical per flow
- Independent decision per flow
- K options per flow
- Learned routing policy

---

## Challenges and Solutions

### Challenge 1: Disconnection

**Problem**: Closing too many edges disconnects the network

**Solutions**:
1. Heavy penalty for disconnected flows (10x demand)
2. K-shortest paths provide alternatives
3. Agent learns to keep critical edges open

### Challenge 2: Action Space Size

**Problem**: Large action space (num_edges + num_flows * k_paths)

**Solutions**:
1. Independent factorization (Bernoulli + Categorical)
2. Shared feature extraction
3. Efficient path computation caching

### Challenge 3: Non-Stationary Flows

**Problem**: Traffic changes every step

**Solutions**:
1. Include flow demands in observation
2. Learn generalizable routing policy
3. Sufficient rollout steps for diverse scenarios

---

## Performance Metrics

Track these during training:

```python
info = {
    'energy': total_energy,              # Total power consumption
    'delay': total_delay,                # Total network delay
    'overload': total_overload,          # Capacity violations
    'num_closed_edges': num_closed,      # Energy-saving actions
    'disconnect_penalty': penalty,       // Connectivity failures
}
```

**Good performance indicators**:
- Decreasing total cost (increasing reward)
- Low disconnect penalty (< 0.1)
- Balanced energy-delay tradeoff
- 20-40% edges closed (energy savings)
- Overload < 1.0 (no capacity violations)

---

## Future Enhancements

1. **Attention Mechanisms**: For better flow-to-path matching
2. **Graph Neural Networks**: Leverage graph structure in policy
3. **Hierarchical Actions**: First close edges, then route flows
4. **Multi-Agent**: Distributed per-node controllers
5. **Dynamic Weights**: Adaptive reward weights based on network conditions
6. **Transfer Learning**: Pre-train on smaller networks

---

## References

- PPO Algorithm: Schulman et al., "Proximal Policy Optimization Algorithms"
- K-Shortest Paths: Yen's Algorithm (NetworkX implementation)
- SDN Energy Optimization: Various research papers on green networking

---

## Troubleshooting

### Issue: High disconnect penalty

**Symptoms**: `info['disconnect_penalty']` is large
**Cause**: Too many edges closed
**Fix**: 
- Increase `w_overload` to penalize disconnection more
- Reduce entropy coefficient to encourage consistency
- Warm-start with all edges open

### Issue: No energy savings

**Symptoms**: All edges remain open
**Cause**: Energy weight too low
**Fix**:
- Increase `w_energy` in config
- Reduce `base_power` to make savings more significant
- Add explicit reward bonus for closing edges

### Issue: Training instability

**Symptoms**: Reward variance very high
**Cause**: Action space exploration
**Fix**:
- Reduce learning rate
- Increase `clip_ratio` for more conservative updates
- Add value function clipping
- Use learning rate scheduling

---

For questions or issues, please refer to the code comments or create an issue in the repository.
