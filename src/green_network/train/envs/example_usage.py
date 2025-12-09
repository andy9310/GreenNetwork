"""
Example: How to use the cluster-based SDN environment with action masking
"""
import numpy as np
from env import SDNEnv

def example_random_policy_with_masking():
    """Example of using action masking with a random policy"""
    
    # Initialize environment with 8 clusters
    env = SDNEnv(num_clusters=8)
    
    print("\n" + "="*60)
    print("ENVIRONMENT INFO")
    print("="*60)
    print(f"Number of agents (clusters): {env.num_agents}")
    print(f"Agent IDs: {env.agent_ids}")
    print(f"Max action dimension: {env.max_action_dim}")
    print(f"Observation dimension: {env.obs_dim}")
    
    # Show cluster information
    print("\n" + "="*60)
    print("CLUSTER DETAILS")
    print("="*60)
    for cid in env.agent_ids[:3]:  # Show first 3 clusters
        info = env.get_cluster_info(cid)
        print(f"\nCluster {cid}:")
        print(f"  Nodes: {info['nodes']}")
        print(f"  Size: {info['size']}")
        print(f"  Valid actions: {info['action_dim']} out of {env.max_action_dim}")
    
    # Reset environment
    obs = env.reset()
    
    print("\n" + "="*60)
    print("RUNNING EPISODE WITH ACTION MASKING")
    print("="*60)
    
    # Run one episode
    total_rewards = {cid: 0.0 for cid in env.agent_ids}
    done = False
    step = 0
    
    while not done and step < 10:  # Run for 10 steps
        step += 1
        
        # For each agent, select action using mask
        actions = {}
        for agent_id in env.agent_ids:
            # Get action mask
            action_mask = env.get_action_mask(agent_id)
            
            # Get valid action indices
            valid_actions = np.where(action_mask > 0)[0]
            
            # Randomly select from valid actions
            action = np.random.choice(valid_actions)
            actions[agent_id] = int(action)
        
        # Step environment
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Accumulate rewards
        for agent_id in env.agent_ids:
            total_rewards[agent_id] += rewards[agent_id]
        
        done = dones[env.agent_ids[0]]  # All agents have same done flag
        
        if step == 1:  # Show first step details
            print(f"\nStep {step}:")
            print(f"  Sample action for cluster 0: {actions[env.agent_ids[0]]}")
            print(f"  Reward: {rewards[env.agent_ids[0]]:.4f}")
            print(f"  Energy: {infos[env.agent_ids[0]]['total_energy']:.2f}")
            print(f"  Delay: {infos[env.agent_ids[0]]['total_delay']:.2f}")
            print(f"  Overload: {infos[env.agent_ids[0]]['total_overload']:.2f}")
        
        obs = next_obs
    
    print(f"\nCompleted {step} steps")
    avg_reward = sum(total_rewards.values()) / len(total_rewards)
    print(f"Average total reward: {avg_reward:.4f}")


def example_policy_with_masking_integration():
    """
    Example showing how to integrate action masking in a policy network.
    This is pseudocode for how your MAPPO agent would handle masking.
    """
    print("\n" + "="*60)
    print("PSEUDOCODE: POLICY NETWORK WITH MASKING")
    print("="*60)
    
    code = """
# In your policy network forward pass:
class MaskedPolicy(nn.Module):
    def forward(self, obs, action_mask):
        # obs includes the action mask already, or pass separately
        
        # Get logits from network
        logits = self.policy_network(obs)  # shape: [batch, max_action_dim]
        
        # Apply mask: set invalid actions to -inf
        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e9
        
        # Create distribution (softmax will make -inf -> 0 probability)
        dist = torch.distributions.Categorical(logits=masked_logits)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob

# In your training loop:
for episode in range(num_episodes):
    obs = env.reset()
    
    for step in range(max_steps):
        # Get action masks
        action_masks = {aid: env.get_action_mask(aid) for aid in env.agent_ids}
        
        # Select actions with masking
        actions, log_probs = agent.select_actions(obs, action_masks)
        
        # Step environment
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Store experience
        agent.store_step(obs, actions, log_probs, rewards, dones, action_masks)
        
        obs = next_obs
        if all(dones.values()):
            break
    
    # Update policy
    agent.update()
"""
    print(code)


def example_observation_structure():
    """Show the structure of observations"""
    env = SDNEnv(num_clusters=5)
    obs = env.reset()
    
    print("\n" + "="*60)
    print("OBSERVATION STRUCTURE")
    print("="*60)
    
    agent_id = env.agent_ids[0]
    obs_vec = obs[agent_id]
    
    print(f"\nObservation for agent {agent_id}:")
    print(f"  Total size: {len(obs_vec)}")
    print(f"  Expected size: {env.obs_dim}")
    
    # Parse observation components
    offset = 0
    
    # Cluster features
    cluster_feat = obs_vec[offset:offset + env.cluster_feature_dim]
    offset += env.cluster_feature_dim
    print(f"\n  Cluster features (size={env.cluster_feature_dim}):")
    print(f"    avg_degree_norm: {cluster_feat[0]:.4f}")
    print(f"    cluster_size_norm: {cluster_feat[1]:.4f}")
    print(f"    total_load_norm: {cluster_feat[2]:.4f}")
    
    # Node features (padded)
    node_feats_size = env.max_cluster_size * env.node_feature_dim
    node_feats = obs_vec[offset:offset + node_feats_size]
    offset += node_feats_size
    print(f"\n  Node features (size={node_feats_size}):")
    print(f"    {env.max_cluster_size} nodes × {env.node_feature_dim} features per node")
    
    # First node example
    first_node_feat = node_feats[:env.node_feature_dim]
    print(f"    First node features: degree={first_node_feat[0]:.4f}, mode={first_node_feat[1]:.4f}, load={first_node_feat[2]:.4f}")
    
    # Action mask
    action_mask = obs_vec[offset:offset + env.max_action_dim]
    valid_count = int(action_mask.sum())
    print(f"\n  Action mask (size={env.max_action_dim}):")
    print(f"    Valid actions: {valid_count}/{env.max_action_dim}")
    print(f"    Mask preview: {action_mask[:10]}...")


if __name__ == "__main__":
    # Run examples
    example_random_policy_with_masking()
    example_observation_structure()
    example_policy_with_masking_integration()
