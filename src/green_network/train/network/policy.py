# networks/policy.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
