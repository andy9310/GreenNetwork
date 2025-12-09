# Quick Start: Dual-Action SDN

## What Are the Two Actions?

Your SDN agent makes **two simultaneous decisions**:

### 1. Edge Control (Link Management)
- **What**: Decide which network links to keep open or close
- **Why**: Closed links save energy but may affect routing
- **Action Format**: Binary array `[0/1]` of length `num_edges`
  - `0` = Close the link (save energy)
  - `1` = Keep the link open (allow traffic)

**Example**: With 10 edges
```python
edge_actions = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
# Edges 2, 4, 5, 9 are closed
# Edges 0, 1, 3, 6, 7, 8 are open
```

### 2. Path Selection (Traffic Routing)
- **What**: For each traffic flow, choose which path to use
- **Why**: Different paths have different delays and loads
- **Action Format**: Integer array `[0 to k-1]` of length `num_flows`
  - Each value selects one of k pre-computed shortest paths

**Example**: With 5 flows and k=3 paths per flow
```python
path_actions = [0, 2, 1, 0, 2]
# Flow 0: Use path 0 (shortest path)
# Flow 1: Use path 2 (third alternative)
# Flow 2: Use path 1 (second alternative)
# Flow 3: Use path 0
# Flow 4: Use path 2
```

---

## How It Works

```
┌─────────────────────────────────────────┐
│  1. Observe Network State               │
│     - Edge utilizations                 │
│     - Edge states (open/closed)         │
│     - Flow demands                      │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  2. Agent Decides                       │
│     ┌──────────────┐  ┌──────────────┐ │
│     │ Edge Control │  │Path Selection│ │
│     │ [0,1,1,0...] │  │ [2,0,1,0...] │ │
│     └──────────────┘  └──────────────┘ │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  3. Environment Executes                │
│     a) Update edge states               │
│     b) Compute k-shortest paths         │
│     c) Route flows on selected paths    │
│     d) Calculate loads & metrics        │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  4. Compute Reward                      │
│     reward = -(w₁*energy + w₂*delay +   │
│               w₃*overload)              │
└─────────────────────────────────────────┘
```

---

## Running the Code

### 1. Demo Mode (See It In Action)

```bash
cd src/green_network/train
python trainer/train_dual_action.py --demo
```

**What it shows**:
- Creates environment and policy
- Samples actions
- Shows edge control decisions
- Shows path selection decisions
- Executes one step
- Displays reward breakdown

### 2. Full Training

```bash
python trainer/train_dual_action.py
```

**What it does**:
- Trains for 1000 episodes (configurable)
- Logs every 10 episodes
- Shows reward, energy, delay, overload

---

## Key Files

### Created for You:

1. **`network/policy.py`** - `DualActionPolicy` class
   - Multi-head neural network
   - Outputs both action types
   
2. **`envs/env_dual_action.py`** - `DualActionSDNEnv` class
   - Handles dual actions
   - Computes paths and routes flows
   
3. **`envs/env_util/path_utils.py`** - Path computation utilities
   - K-shortest paths algorithm
   - Flow routing logic
   - Metric computation
   
4. **`trainer/train_dual_action.py`** - Training script
   - Complete example
   - Demo mode included

### Configuration:

Edit `config/train_config.json`:

```json
{
    "k_paths": 3,              // How many path options per flow
    "num_flows_per_step": 5,   // How many flows to route
    "w_energy": 0.1,           // Energy importance
    "w_delay": 1.0,            // Delay importance
    "w_overload": 2.0          // Overload penalty
}
```

---

## Understanding the Output

### During Training:

```
Episode 10/1000
  Total Reward: -12.345
  Avg Energy: 25.4
  Avg Delay: 15.2
  Avg Overload: 2.1
  Avg Closed Edges: 8.5/61
```

**Interpreting**:
- **Reward**: Negative cost (higher is better, target: > -10)
- **Energy**: Power consumption (lower is better)
- **Delay**: Network latency (lower is better)
- **Overload**: Capacity violations (target: near 0)
- **Closed Edges**: Energy savings (typically 20-40% closed)

### Good Training Progress:

```
Episode 0    → Reward: -25.0  (random policy)
Episode 100  → Reward: -15.0  (learning...)
Episode 500  → Reward: -8.5   (good!)
Episode 1000 → Reward: -7.2   (converged)
```

---

## Common Pitfalls

### ❌ Problem: All edges stay open
**Cause**: Energy weight too low  
**Fix**: Increase `w_energy` in config

### ❌ Problem: High disconnect penalty
**Cause**: Too many edges closed  
**Fix**: Increase `w_overload` or reduce `w_energy`

### ❌ Problem: Training very slow
**Cause**: Large network + many flows  
**Fix**: Start with smaller network (20 nodes, 3 flows)

---

## Customization Ideas

### 1. Change Path Options
```json
"k_paths": 5  // More routing flexibility
```

### 2. Adjust Trade-offs
```json
"w_energy": 0.5,    // More energy-conscious
"w_delay": 0.5,     // Less delay-sensitive
"w_overload": 10.0  // Strictly avoid overload
```

### 3. Scale Up
```json
"num_flows_per_step": 20,  // More traffic
"num_nodes": 100,           // Larger network
"rollout_steps": 1024       // Longer episodes
```

---

## Next Steps

1. **Run demo** to see actions in action
2. **Train baseline** model with default config
3. **Experiment** with reward weights
4. **Visualize** results (plot reward over time)
5. **Implement PPO** updates (currently just data collection)

---

## Architecture Diagram

```
                    Observation (Network State)
                              │
                              ▼
                    ┌─────────────────┐
                    │  Shared Network │
                    │   (256 units)   │
                    └────────┬────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
      ┌─────────────────┐    ┌─────────────────┐
      │ Edge Control    │    │ Path Selection  │
      │ Head (Binary)   │    │ Head (Cat.)     │
      └────────┬────────┘    └────────┬────────┘
               │                      │
               ▼                      ▼
         [0,1,1,0,...]          [2,0,1,0,...]
         Edge Actions           Path Actions
```

---

## Quick Reference

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `DualActionPolicy` | Observation | Edge + Path actions | Agent brain |
| `DualActionSDNEnv` | Actions | Obs, Reward, Done | Simulator |
| `compute_k_shortest_paths` | Graph, Flow | K paths | Path options |
| `route_flows_on_paths` | Flows, Paths, Actions | Edge loads | Traffic routing |

---

## Help & Debugging

Run with more logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check shapes:
```python
print(f"Obs shape: {obs.shape}")           # Should be [obs_dim]
print(f"Edge actions: {edge_actions.shape}") # Should be [num_edges]
print(f"Path actions: {path_actions.shape}") # Should be [num_flows]
```

Validate actions:
```python
assert edge_actions.min() >= 0 and edge_actions.max() <= 1
assert path_actions.min() >= 0 and path_actions.max() < k_paths
```

---

**Ready to start?** Run `python trainer/train_dual_action.py --demo` to see it in action!
