# train_dual_action.py
"""
Training script for dual-action SDN environment
"""
import torch
import json
import os
import logging
from pathlib import Path

from envs.env_dual_action import DualActionSDNEnv
from network.policy import DualActionPolicy
from network.value import Value


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("DualActionTraining")


def train_dual_action(config_path: str = None, device: str = "cpu"):
    """
    Train the dual-action SDN agent
    
    Args:
        config_path: Path to training configuration JSON
        device: Device to train on ('cpu' or 'cuda')
    """
    logger = setup_logger()
    logger.info("="*60)
    logger.info("Starting Dual-Action SDN Training")
    logger.info("="*60)
    
    # Create environment
    logger.info("Initializing environment...")
    env = DualActionSDNEnv(config_path=config_path)
    
    # Get dimensions
    obs_dim = env.obs_dim
    num_edges = env.num_edges
    num_flows = env.num_flows_per_step
    k_paths = env.k_paths
    
    logger.info(f"Environment specs:")
    logger.info(f"  Observation dim: {obs_dim}")
    logger.info(f"  Num edges: {num_edges}")
    logger.info(f"  Num flows: {num_flows}")
    logger.info(f"  K-paths: {k_paths}")
    
    # Create policy and value networks
    logger.info("Creating networks...")
    policy = DualActionPolicy(
        obs_dim=obs_dim,
        num_edges=num_edges,
        num_flows=num_flows,
        k_paths=k_paths
    ).to(device)
    
    value_fn = Value(state_dim=obs_dim).to(device)
    
    # Optimizer
    lr = env.config.get('lr', 3e-4)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=lr
    )
    
    logger.info(f"Networks initialized with lr={lr}")
    logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"Value parameters: {sum(p.numel() for p in value_fn.parameters()):,}")
    
    # Training parameters
    num_episodes = env.config.get('num_episodes', 100)
    rollout_steps = env.config.get('rollout_steps', 512)
    
    logger.info("="*60)
    logger.info("Starting training loop")
    logger.info(f"Episodes: {num_episodes}, Rollout steps: {rollout_steps}")
    logger.info("="*60)
    
    # Training loop
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_info = {
            'energy': 0.0,
            'delay': 0.0,
            'overload': 0.0,
            'num_closed_edges': 0.0,
        }
        
        for step in range(rollout_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Sample actions from policy
            with torch.no_grad():
                edge_actions, path_actions, log_probs = policy.act(obs_tensor)
            
            # Convert to numpy for environment
            edge_actions_np = edge_actions.cpu().numpy()[0]  # [num_edges]
            path_actions_np = path_actions.cpu().numpy()[0]  # [num_flows]
            
            actions = {
                'edge_control': edge_actions_np,
                'path_selection': path_actions_np
            }
            
            # Step environment
            next_obs, reward, done, info = env.step(actions)
            
            episode_reward += reward
            for key in episode_info:
                if key in info:
                    episode_info[key] += info[key]
            
            obs = next_obs
            
            if done:
                break
        
        # Log episode statistics
        if episode % 10 == 0:
            avg_energy = episode_info['energy'] / (step + 1)
            avg_delay = episode_info['delay'] / (step + 1)
            avg_overload = episode_info['overload'] / (step + 1)
            avg_closed = episode_info['num_closed_edges'] / (step + 1)
            
            logger.info(f"Episode {episode}/{num_episodes}")
            logger.info(f"  Total Reward: {episode_reward:.3f}")
            logger.info(f"  Avg Energy: {avg_energy:.3f}")
            logger.info(f"  Avg Delay: {avg_delay:.3f}")
            logger.info(f"  Avg Overload: {avg_overload:.3f}")
            logger.info(f"  Avg Closed Edges: {avg_closed:.1f}/{num_edges}")
        
        # TODO: Implement PPO update here
        # For now, this is just data collection demonstration
    
    logger.info("="*60)
    logger.info("Training completed")
    logger.info("="*60)
    
    return policy, value_fn


def demonstrate_action_selection():
    """
    Demonstrate how to use the dual-action policy
    """
    logger = setup_logger()
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATION: Dual-Action Selection")
    logger.info("="*60)
    
    # Create a simple environment
    env = DualActionSDNEnv()
    obs = env.reset()
    
    # Create policy
    policy = DualActionPolicy(
        obs_dim=env.obs_dim,
        num_edges=env.num_edges,
        num_flows=env.num_flows_per_step,
        k_paths=env.k_paths
    )
    
    # Sample observation
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    logger.info(f"\nObservation shape: {obs_tensor.shape}")
    logger.info(f"Observation: {obs_tensor[0, :10].numpy()}... (showing first 10)")
    
    # Get actions
    with torch.no_grad():
        edge_actions, path_actions, log_probs = policy.act(obs_tensor)
    
    logger.info(f"\n--- Edge Control Actions ---")
    logger.info(f"Shape: {edge_actions.shape}")
    logger.info(f"Values (first 10 edges): {edge_actions[0, :10].int().numpy()}")
    logger.info(f"Num closed edges: {(edge_actions[0] == 0).sum().item()}/{env.num_edges}")
    
    logger.info(f"\n--- Path Selection Actions ---")
    logger.info(f"Shape: {path_actions.shape}")
    logger.info(f"Values (all flows): {path_actions[0].int().numpy()}")
    logger.info(f"(Each value is path index 0-{env.k_paths-1})")
    
    logger.info(f"\n--- Log Probabilities ---")
    logger.info(f"Edge log prob: {log_probs['edge_log_probs'].item():.4f}")
    logger.info(f"Path log prob: {log_probs['path_log_probs'].item():.4f}")
    logger.info(f"Total log prob: {log_probs['total_log_probs'].item():.4f}")
    
    # Execute step
    actions_dict = {
        'edge_control': edge_actions[0].numpy(),
        'path_selection': path_actions[0].numpy()
    }
    
    next_obs, reward, done, info = env.step(actions_dict)
    
    logger.info(f"\n--- Environment Response ---")
    logger.info(f"Reward: {reward:.4f}")
    logger.info(f"Energy: {info['energy']:.4f}")
    logger.info(f"Delay: {info['delay']:.4f}")
    logger.info(f"Overload: {info['overload']:.4f}")
    logger.info(f"Disconnect penalty: {info['disconnect_penalty']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Demonstration completed successfully!")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demonstration
        demonstrate_action_selection()
    else:
        # Run training
        try:
            policy, value_fn = train_dual_action(device=device)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
