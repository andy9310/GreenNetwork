# agents/mappo_agent.py
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config.mappo_config import MAPPOConfig
from networks.policy import CategoricalPolicy
from networks.value import CentralValue
from buffer.rollout_buffer import RolloutBuffer

class Agent:
    def __init__(self, obs_dim, act_dim, num_agents, config: MAPPOConfig, device="cpu"):
        self.cfg = config
        self.device = device

        # Shared policy for all agents
        self.policy = CategoricalPolicy(obs_dim, act_dim).to(device)
        # Centralized critic (state_dim = num_agents * obs_dim here)
        self.value_fn = CentralValue(state_dim=num_agents * obs_dim).to(device)

        self.optim = Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()),
            lr=self.cfg.lr,
        )

        self.num_agents = num_agents
        self.buffer = RolloutBuffer(
            rollout_steps=self.cfg.rollout_steps,
            num_agents=num_agents,
            obs_dim=obs_dim,
            device=device,
        )

    def select_actions(self, obs_dict):
        """
        obs_dict: {agent_id: np.ndarray(obs_dim)}
        Returns: actions_dict, logps_dict
        """
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.stack(
            [torch.as_tensor(obs_dict[a], device=self.device, dtype=torch.float32)
             for a in agent_ids],
            dim=0,  # [N, obs_dim]
        )
        actions, logps, _ = self.policy.act(obs_tensor)
        actions_np = actions.cpu().numpy()
        logps_np = logps.detach().cpu().numpy()

        actions_dict = {a: actions_np[i] for i, a in enumerate(agent_ids)}
        logps_dict = {a: logps_np[i] for i, a in enumerate(agent_ids)}
        return actions_dict, logps_dict

    def compute_values(self, obs_dict):
        # Centralized critic uses concatenated obs of all agents
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.stack(
            [torch.as_tensor(obs_dict[a], device=self.device, dtype=torch.float32)
             for a in agent_ids],
            dim=0,
        )  # [N, obs_dim]
        joint_state = obs_tensor.view(1, -1)   # [1, N*obs_dim]
        value = self.value_fn(joint_state)     # [1]
        # broadcast same value to all agents for simplicity
        return {a: value.item() for a in agent_ids}

    def store_step(self, t, obs_dict, actions_dict, logps_dict, rewards_dict, dones_dict):
        values_dict = self.compute_values(obs_dict)
        for idx, agent_id in enumerate(sorted(obs_dict.keys())):
            self.buffer.store(
                t=t,
                agent_idx=idx,
                obs=obs_dict[agent_id],
                action=actions_dict[agent_id],
                logp=logps_dict[agent_id],
                reward=rewards_dict[agent_id],
                done=dones_dict[agent_id],
                value=values_dict[agent_id],
            )

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        rewards, values, dones: [T*N] flattened.
        For simplicity we treat it as a single chain; in practice
        you may want per-agent/per-env GAE with masks.
        """
        T_N = rewards.shape[0]
        adv = torch.zeros(T_N, device=self.device)
        gae = 0.0
        for t in reversed(range(T_N)):
            mask = 1.0 - dones[t]
            next_value = last_value if t == T_N - 1 else values[t+1]
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * mask * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def update(self):
        obs, actions, old_logps, rewards, dones, values = self.buffer.get_flat()

        # last_value: simple approximation → 0; in a real impl, use bootstrap from last obs
        last_value = torch.zeros(1, device=self.device)
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = obs.size(0)
        for _ in range(self.cfg.ppo_epochs):
            idx = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = idx[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values_old = values[mb_idx]

                # Policy loss
                logits = self.policy(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(mb_actions)
                ratio = torch.exp(new_logps - mb_old_logps)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.cfg.clip_ratio,
                                            1+self.cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                joint_states = mb_obs.view(mb_obs.size(0), -1)  # not perfect but ok in toy
                # In real code, joint_states should be built properly
                values_pred = self.value_fn(joint_states)
                value_loss = F.mse_loss(values_pred, mb_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optim.step()
