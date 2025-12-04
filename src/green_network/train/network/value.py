# networks/value.py
import torch
import torch.nn as nn

class CentralValue(nn.Module):
    """
    Centralized critic that can see concatenated observations
    of all agents, or a global state.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        # state: [batch, state_dim]
        return self.net(state).squeeze(-1)  # [batch]
