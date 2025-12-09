# networks/policy.py
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

## Dual-action policy: edge control + path selection
class DualActionPolicy(nn.Module): 
    """
    Multi-head policy for SDN control:
    - Head 1: Edge control (binary per edge - open/close)
    - Head 2: Path selection (categorical per flow - choose from k paths)
    """
    def __init__(self, obs_dim: int, num_edges: int, num_flows: int, k_paths: int = 3):
        super().__init__()
        self.num_edges = num_edges
        self.num_flows = num_flows
        self.k_paths = k_paths
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Head 1: Edge control (binary for each edge)
        self.edge_control_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_edges),  # Output logits for each edge
        )
        
        # Head 2: Path selection (categorical for each flow)
        self.path_select_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_flows * k_paths),  # Output logits for each flow's path options
        )

    def forward(self, obs):
        """
        Returns:
            edge_logits: [batch, num_edges] - logits for edge open/close
            path_logits: [batch, num_flows, k_paths] - logits for path selection
        """
        features = self.shared_net(obs)
        
        # Edge control logits
        edge_logits = self.edge_control_head(features)  # [batch, num_edges]
        
        # Path selection logits
        path_logits = self.path_select_head(features)  # [batch, num_flows * k_paths]
        path_logits = path_logits.view(-1, self.num_flows, self.k_paths)  # [batch, num_flows, k_paths]
        
        return edge_logits, path_logits

    def act(self, obs, deterministic=False):
        """
        Sample actions from both heads
        Returns:
            edge_actions: [batch, num_edges] - binary (0/1)
            path_actions: [batch, num_flows] - indices (0 to k_paths-1)
            log_probs: dict with 'edge' and 'path' log probabilities
        """
        edge_logits, path_logits = self.forward(obs)
        
        # Sample edge control actions (Bernoulli distribution)
        edge_probs = torch.sigmoid(edge_logits)
        edge_dist = Bernoulli(probs=edge_probs)
        if deterministic:
            edge_actions = (edge_probs > 0.5).float()
        else:
            edge_actions = edge_dist.sample()
        edge_log_probs = edge_dist.log_prob(edge_actions).sum(dim=-1)  # [batch]
        
        # Sample path selection actions (Categorical distribution per flow)
        path_actions = []
        path_log_probs_list = []
        for flow_idx in range(self.num_flows):
            flow_logits = path_logits[:, flow_idx, :]  # [batch, k_paths]
            flow_dist = Categorical(logits=flow_logits)
            if deterministic:
                flow_action = flow_logits.argmax(dim=-1)
            else:
                flow_action = flow_dist.sample()
            flow_log_prob = flow_dist.log_prob(flow_action)
            path_actions.append(flow_action)
            path_log_probs_list.append(flow_log_prob)
        
        path_actions = torch.stack(path_actions, dim=-1)  # [batch, num_flows]
        path_log_probs = torch.stack(path_log_probs_list, dim=-1).sum(dim=-1)  # [batch]
        
        # Total log probability
        total_log_prob = edge_log_probs + path_log_probs
        
        return edge_actions, path_actions, {
            'edge_log_probs': edge_log_probs,
            'path_log_probs': path_log_probs,
            'total_log_probs': total_log_prob
        }


# Backward compatible simple policy
class Policy(nn.Module): 
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.logits = nn.Linear(128, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        logits = self.logits(x)
        return logits

    def act(self, obs):
        """obs: torch tensor [N, obs_dim]"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, dist
