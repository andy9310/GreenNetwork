"""
Standalone test script for DQN-based Energy-Efficient Routing (DQN-EER) baseline
Run this to get individual metrics for DQN-EER algorithm
"""

import sys
import json
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import SDNEnv
from baselines.dqn_energy_routing import DQNEnergyRouting


def load_config(n_links=100):
    """Load configuration from JSON file"""
    config_file = Path(__file__).parent / f"configs/topology_{n_links}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Available configs: 20, 100, 500, 1000, 2000 links"
        )
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def test_dqn_eer(n_links=100, episodes=200, train_episodes=150):
    """
    Test DQN-based Energy-Efficient Routing algorithm
    
    Args:
        n_links: Network size (number of links)
        episodes: Total number of episodes (training + evaluation)
        train_episodes: Number of training episodes
    
    Returns:
        dict: Performance metrics
    """
    print("=" * 80)
    print("DQN-BASED ENERGY-EFFICIENT ROUTING (DQN-EER) - STANDALONE TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Network Size: {n_links} links")
    print(f"  Training Episodes: {train_episodes}")
    print(f"  Evaluation Episodes: {episodes - train_episodes}")
    print(f"  Total Episodes: {episodes}")
    print("\n" + "-" * 80)
    
    # Initialize
    config = load_config(n_links)
    env = SDNEnv(config)
    
    # Initialize DQN-EER with appropriate dimensions
    state_dim = 10  # State feature dimension
    action_dim = min(100, n_links)  # Action space (number of link configurations)
    
    dqn = DQNEnergyRouting(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_size=10000,
        target_update_freq=10
    )
    
    # Metrics storage
    energy_savings = []
    latencies = []
    sla_violations = []
    active_links_list = []
    training_losses = []
    
    print(f"\n🚀 Running {episodes} episodes (Training: {train_episodes}, Eval: {episodes-train_episodes})...\n")
    
    for ep in range(episodes):
        obs = env.reset()
        
        ep_energy = []
        ep_latency = []
        ep_sla = []
        ep_active = []
        ep_losses = []
        
        is_training = ep < train_episodes
        phase = "TRAIN" if is_training else "EVAL"
        
        prev_state = None
        prev_action = None
        
        for t in range(config['max_steps_per_episode']):
            # Get current state
            state = dqn.get_state(env.G_full, env._flows)
            
            # Get active links from DQN-EER algorithm
            active_links, metrics = dqn.select_links(env.G_full, env._flows, None)
            action_idx = metrics['action_idx']
            
            # Apply link activation to environment
            for u, v in env.G_full.edges():
                is_active = (u, v) in active_links or (v, u) in active_links
                env.G_full[u][v]['active'] = 1 if is_active else 0
            
            # Measure performance
            latency, _ = env._route_and_measure()
            energy = env._energy_cost()
            
            # Calculate energy saving
            base_all_on = env.energy_on * env.G_full.number_of_edges()
            energy_saving = (base_all_on - energy) / base_all_on * 100 if base_all_on > 0 else 0
            
            # Calculate reward (for training)
            # Reward = energy_saving - penalty for high latency
            reward = energy_saving - 0.1 * latency
            
            # Store transition and train (only during training phase)
            if is_training and prev_state is not None:
                done = (t == config['max_steps_per_episode'] - 1)
                dqn.store_transition(prev_state, prev_action, reward, state, done)
                
                # Train the network
                loss = dqn.train_step()
                if loss is not None:
                    ep_losses.append(loss)
            
            # Update previous state and action
            prev_state = state
            prev_action = action_idx
            
            ep_energy.append(energy_saving)
            ep_latency.append(latency)
            ep_active.append(len(active_links))
            
            # Generate new flows for next step
            env._generate_new_flows()
        
        # Store episode averages
        energy_savings.append(np.mean(ep_energy))
        latencies.append(np.mean(ep_latency))
        active_links_list.append(np.mean(ep_active))
        if ep_losses:
            training_losses.append(np.mean(ep_losses))
        
        # Progress update
        if (ep + 1) % 10 == 0 or ep == 0 or ep == train_episodes:
            recent_energy = np.mean(energy_savings[-10:])
            recent_latency = np.mean(latencies[-10:])
            recent_loss = np.mean(training_losses[-10:]) if training_losses else 0
            print(f"  [{phase}] Episode {ep+1:3d}/{episodes}: "
                  f"Energy={recent_energy:5.1f}%, "
                  f"Latency={recent_latency:5.2f}ms, "
                  f"Loss={recent_loss:.4f}, "
                  f"ε={dqn.epsilon:.3f}")
    
    # Get algorithm statistics
    stats = dqn.get_stats()
    
    # Calculate final metrics (average of evaluation episodes only)
    eval_start = train_episodes
    eval_energy = energy_savings[eval_start:]
    eval_latency = latencies[eval_start:]
    eval_active = active_links_list[eval_start:]
    
    results = {
        'energy_saving': np.mean(eval_energy),
        'energy_saving_std': np.std(eval_energy),
        'latency': np.mean(eval_latency),
        'latency_std': np.std(eval_latency),
        'computation_time': stats['avg_computation_time'],
        'avg_training_loss': stats['avg_training_loss'],
        'avg_active_links': np.mean(eval_active),
        'total_links': env.G_full.number_of_edges(),
        'final_epsilon': stats['epsilon'],
        'buffer_size': stats['buffer_size'],
        'update_count': stats['update_count']
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (averaged over evaluation episodes)")
    print("=" * 80)
    print(f"\n📊 Performance Metrics:")
    print(f"  Energy Saving:       {results['energy_saving']:.2f}% ± {results['energy_saving_std']:.2f}%")
    print(f"  Latency:             {results['latency']:.2f} ± {results['latency_std']:.2f} ms")
    print(f"  Computation Time:    {results['computation_time']:.6f} seconds")
    print(f"\n🔗 Link Statistics:")
    print(f"  Total Links:         {results['total_links']}")
    print(f"  Avg Active Links:    {results['avg_active_links']:.1f}")
    print(f"  Avg Deactivated:     {results['total_links'] - results['avg_active_links']:.1f} "
          f"({(1 - results['avg_active_links']/results['total_links'])*100:.1f}%)")
    print(f"\n🤖 Learning Statistics:")
    print(f"  Final Epsilon:       {results['final_epsilon']:.4f}")
    print(f"  Avg Training Loss:   {results['avg_training_loss']:.6f}")
    print(f"  Buffer Size:         {results['buffer_size']}")
    print(f"  Network Updates:     {results['update_count']}")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DQN-based Energy-Efficient Routing baseline')
    parser.add_argument('--links', type=int, default=100, 
                        help='Number of links (20, 100, 500, 1000, 2000)')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Total number of episodes')
    parser.add_argument('--train-episodes', type=int, default=150,
                        help='Number of training episodes')
    
    args = parser.parse_args()
    
    results = test_dqn_eer(
        n_links=args.links,
        episodes=args.episodes,
        train_episodes=args.train_episodes
    )
    
    # Save results to file
    import json
    output_file = f"results/dqn_eer_standalone_{args.links}links.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
