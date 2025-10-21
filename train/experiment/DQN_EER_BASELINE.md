# DQN-based Energy-Efficient Routing (DQN-EER) Baseline

## Overview

This baseline implements the **DQN-based energy-efficient routing algorithm** from the paper "DQN-based energy-efficient routing algorithm in software-defined networks". It uses Deep Q-Networks (DQN) to learn optimal link activation policies that minimize energy consumption while maintaining Quality of Service (QoS) constraints.

## Paper Reference

- **Title**: DQN-based energy-efficient routing algorithm in software-defined networks
- **Location**: `paper/DQN-based_energy-efficient_routing_algorithm_in_so (1) (1).pdf`
- **Key Contribution**: Applies deep reinforcement learning to SDN routing for energy efficiency

## Algorithm Description

### Core Concept

The DQN-EER algorithm treats energy-efficient routing as a reinforcement learning problem where:
- **State**: Network conditions (link utilization, flow statistics, topology features)
- **Action**: Link activation configuration (which links to keep active/deactivate)
- **Reward**: Energy savings minus penalties for SLA violations and high latency

### Key Features

1. **Deep Neural Network**: Uses a multi-layer feedforward network to approximate Q-values
   - Input: State features (10-dimensional vector)
   - Hidden layers: 2 layers with 128 neurons each
   - Output: Q-values for each possible action

2. **Experience Replay**: Stores transitions in a replay buffer for stable learning
   - Buffer size: 10,000 transitions
   - Batch size: 32 samples per training step

3. **Target Network**: Maintains a separate target network for stable Q-value estimation
   - Updated every 10 training steps
   - Reduces moving target problem

4. **Epsilon-Greedy Exploration**: Balances exploration and exploitation
   - Initial epsilon: 1.0 (full exploration)
   - Final epsilon: 0.01 (mostly exploitation)
   - Decay rate: 0.995 per episode

### State Representation

The state vector contains 10 features:
1. **Active link ratio**: Fraction of currently active links
2. **Number of flows**: Normalized count of active flows
3. **Average flow size**: Normalized average flow size
4. **Average utilization**: Mean link utilization
5. **Maximum utilization**: Peak link utilization
6. **Minimum utilization**: Lowest link utilization
7. **Average degree**: Mean node degree (normalized)
8. **Total links**: Total number of links (normalized)
9. **Active links**: Number of active links (normalized)
10. **Utilized links**: Number of links with traffic (normalized)

### Action Space

Actions represent different link activation configurations:
- **Action dimension**: 100 (configurable based on network size)
- **Action mapping**: Each action index maps to a specific number of links to keep active
- **Range**: From 20% minimum to 100% of all links

### Reward Function

```
reward = energy_saving - 2.0 × sla_violations - 0.1 × latency
```

Where:
- `energy_saving`: Percentage of energy saved compared to all-links-on baseline
- `sla_violations`: Percentage of flows violating SLA latency thresholds
- `latency`: Average end-to-end latency in milliseconds

## Implementation Details

### File Structure

```
baselines/
├── dqn_energy_routing.py      # Main DQN-EER implementation
└── __init__.py                # Module exports

test_dqn_standalone.py         # Standalone test script
```

### Class: `DQNEnergyRouting`

#### Initialization Parameters

```python
DQNEnergyRouting(
    state_dim=10,              # State feature dimension
    action_dim=100,            # Number of actions
    learning_rate=0.001,       # Learning rate for optimizer
    discount_factor=0.95,      # Discount factor (gamma)
    epsilon_start=1.0,         # Initial exploration rate
    epsilon_end=0.01,          # Minimum exploration rate
    epsilon_decay=0.995,       # Epsilon decay rate per episode
    batch_size=32,             # Training batch size
    buffer_size=10000,         # Replay buffer capacity
    target_update_freq=10,     # Target network update frequency
    hidden_dims=[128, 128]     # Hidden layer dimensions
)
```

#### Key Methods

1. **`get_state(graph, flows)`**
   - Extracts state features from network
   - Returns: 10-dimensional numpy array

2. **`select_links(graph, flows, traffic_matrix)`**
   - Selects which links to keep active
   - Uses epsilon-greedy policy
   - Returns: (active_links, metrics)

3. **`store_transition(state, action, reward, next_state, done)`**
   - Stores experience in replay buffer

4. **`train_step()`**
   - Performs one training iteration
   - Samples batch from replay buffer
   - Updates Q-network using TD learning
   - Returns: training loss

5. **`get_stats()`**
   - Returns algorithm statistics
   - Includes computation time, loss, epsilon, buffer size

## Usage

### Standalone Testing

```bash
# Test on 100-link network with 200 episodes (150 training + 50 evaluation)
python test_dqn_standalone.py --links 100 --episodes 200 --train-episodes 150

# Test on larger network
python test_dqn_standalone.py --links 500 --episodes 300 --train-episodes 250

# Quick test on small network
python test_dqn_standalone.py --links 20 --episodes 100 --train-episodes 80
```

### Integration with Experiment Runner

```bash
# Run DQN-EER in comparison experiments
python run_experiments.py --method dqn_eer --topology 500 --episodes 2000

# Run all baselines including DQN-EER
python run_experiments.py --all
```

### Programmatic Usage

```python
from baselines import DQNEnergyRouting
from env import SDNEnv

# Initialize environment
env = SDNEnv(config)

# Initialize DQN-EER
dqn = DQNEnergyRouting(
    state_dim=10,
    action_dim=100,
    learning_rate=0.001
)

# Training loop
for episode in range(num_episodes):
    obs = env.reset()
    prev_state = None
    prev_action = None
    
    for step in range(max_steps):
        # Get state
        state = dqn.get_state(env.G_full, env._flows)
        
        # Select links
        active_links, metrics = dqn.select_links(env.G_full, env._flows, None)
        
        # Apply to environment
        for u, v in env.G_full.edges():
            is_active = (u, v) in active_links or (v, u) in active_links
            env.G_full[u][v]['active'] = 1 if is_active else 0
        
        # Measure performance
        latency, sla_viol, _ = env._route_and_measure()
        energy = env._energy_cost()
        
        # Calculate reward
        base_energy = env.energy_on * env.G_full.number_of_edges()
        energy_saving = (base_energy - energy) / base_energy * 100
        reward = energy_saving - 2.0 * sla_viol - 0.1 * latency
        
        # Store transition and train
        if prev_state is not None:
            done = (step == max_steps - 1)
            dqn.store_transition(prev_state, prev_action, reward, state, done)
            loss = dqn.train_step()
        
        prev_state = state
        prev_action = metrics['action_idx']
        
        # Generate new flows
        env._generate_new_flows()

# Get statistics
stats = dqn.get_stats()
print(f"Average computation time: {stats['avg_computation_time']:.6f}s")
print(f"Average training loss: {stats['avg_training_loss']:.6f}")
print(f"Final epsilon: {stats['epsilon']:.4f}")
```

## Performance Characteristics

### Expected Results

Based on the paper and typical DQN performance:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Energy Saving** | 65-75% | Better than tabular RL, slightly less than clustering approach |
| **Latency** | 10-18 ms | Low latency due to learned policies |
| **SLA Violations** | 8-12% | Moderate violations, improves with training |
| **Computation Time** | 0.05-0.5 s | Faster than tabular RL, slower than heuristic |
| **Scalability** | Good | Handles large networks via function approximation |

### Training Characteristics

- **Convergence**: Typically converges after 100-150 episodes
- **Exploration phase**: First 50-80 episodes (high epsilon)
- **Exploitation phase**: After 150+ episodes (low epsilon)
- **Stability**: Experience replay and target network ensure stable learning

### Comparison with Other Baselines

| Feature | EAR (Heuristic) | RL-ER (Tabular) | DQN-EER (Deep) | Our Method |
|---------|----------------|-----------------|----------------|------------|
| Learning | No | Yes | Yes | Yes |
| Function Approx | N/A | No | Yes | Yes |
| Scalability | Excellent | Poor | Good | Excellent |
| Energy Saving | Medium | Medium-High | High | Highest |
| Computation | Fastest | Slowest | Medium | Medium |

## Advantages

1. **Function Approximation**: Handles continuous state spaces better than tabular methods
2. **Scalability**: Works well on large networks without state space explosion
3. **Learning**: Adapts to traffic patterns through experience
4. **Stability**: Experience replay and target network ensure stable convergence
5. **Generalization**: Can generalize to unseen network conditions

## Limitations

1. **Training Time**: Requires 150+ episodes to converge
2. **Hyperparameter Sensitivity**: Performance depends on learning rate, epsilon decay, etc.
3. **No Clustering**: Treats network as single entity (no decomposition)
4. **Simplified Network**: Uses numpy implementation (not PyTorch/TensorFlow)
5. **Action Space**: Limited to predefined action configurations

## Future Improvements

1. **PyTorch Implementation**: Replace numpy with PyTorch for GPU acceleration
2. **Prioritized Experience Replay**: Sample important transitions more frequently
3. **Dueling DQN**: Separate value and advantage streams
4. **Multi-step Returns**: Use n-step TD learning
5. **Distributed Training**: Parallel experience collection (A3C/IMPALA)
6. **Hierarchical Actions**: Combine with clustering for better scalability

## References

1. Original paper: "DQN-based energy-efficient routing algorithm in software-defined networks"
2. DQN paper: Mnih et al., "Playing Atari with Deep Reinforcement Learning", 2013
3. Double DQN: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2015

## Troubleshooting

### Issue: Slow convergence

**Solution**: 
- Increase learning rate (e.g., 0.001 → 0.005)
- Decrease epsilon decay (e.g., 0.995 → 0.99)
- Increase batch size (e.g., 32 → 64)

### Issue: Unstable training

**Solution**:
- Decrease learning rate (e.g., 0.001 → 0.0005)
- Increase target network update frequency (e.g., 10 → 20)
- Increase replay buffer size (e.g., 10000 → 50000)

### Issue: Poor performance

**Solution**:
- Check reward function scaling
- Verify state normalization
- Ensure sufficient training episodes (>150)
- Adjust reward penalties for SLA violations

## Contact

For questions or issues with the DQN-EER baseline implementation, please refer to the main experiment README or contact the research team.
