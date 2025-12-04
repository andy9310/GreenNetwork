# buffer/rollout_buffer.py
import torch

class RolloutBuffer:
    def __init__(self, rollout_steps, num_agents, obs_dim, device="cpu"):
        self.T = rollout_steps
        self.N = num_agents
        self.obs = torch.zeros(self.T, self.N, obs_dim, device=device)
        self.actions = torch.zeros(self.T, self.N, dtype=torch.long, device=device)
        self.logps = torch.zeros(self.T, self.N, device=device)
        self.rewards = torch.zeros(self.T, self.N, device=device)
        self.dones = torch.zeros(self.T, self.N, device=device)
        self.values = torch.zeros(self.T, self.N, device=device)
        self.ptr = 0
        self.device = device

    def store(self, t, agent_idx, obs, action, logp, reward, done, value):
        self.obs[t, agent_idx] = torch.as_tensor(obs, device=self.device)
        self.actions[t, agent_idx] = action
        self.logps[t, agent_idx] = logp
        self.rewards[t, agent_idx] = reward
        self.dones[t, agent_idx] = float(done)
        self.values[t, agent_idx] = value

    def get_flat(self):
        # flatten time and agents: [T*N, ...]
        T, N = self.T, self.N
        obs = self.obs.reshape(T * N, -1)
        actions = self.actions.reshape(T * N)
        logps = self.logps.reshape(T * N)
        rewards = self.rewards.reshape(T * N)
        dones = self.dones.reshape(T * N)
        values = self.values.reshape(T * N)
        return obs, actions, logps, rewards, dones, values
