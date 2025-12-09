# train_mappo.py
import torch
import json
import os
import logging
from pathlib import Path
from utils.logger import get_training_logger
from agents.agent import Agent
from envs.env import SDNEnv

def train():
    # Setup logger
    logger = get_training_logger()
    logger.info("Starting Green Network Training")
    with open("../config/train_config.json", 'r') as f:
        train_config = json.load(f)

    logger.info(f"Configuration loaded: {train_config.num_episodes} episodes, device={train_config.device}")
    logger.info(f"Initializing SDN environment...")
    env = SDNEnv()
    
    logger.info("Environment created successfully")
    logger.info(f"  Actual nodes: {env.num_nodes}")
    logger.info(f"  Actual edges: {env.graph.number_of_edges()}")
    logger.info(f"  Max degree: {env.max_degree}")
    logger.info(f"  Observation dim: {env.obs_dim}")
    
    agent_ids = env.agents
    num_agents = len(agent_ids)

    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    act_dim = env.action_space(agent_ids[0]).n

    logger.info(f"Initializing MAPPO agent...")
    logger.info(f"  Num agents: {num_agents}")
    logger.info(f"  Obs dim: {obs_dim}")
    logger.info(f"  Act dim: {act_dim}")
    logger.info(f"  Device: {device}")
    
    agent = Agent(obs_dim, act_dim, num_agents, cfg, device=device)
    logger.info("Agent initialized")
    logger.info("Starting training loop")

    
    # for ep in range(num_episodes):
    #     obs = env.reset()
    #     ep_return = {a: 0.0 for a in agent_ids}

    #     for t in range(cfg.rollout_steps):
    #         # obs: {agent_id: obs_vector}
            
    #         actions, logps = agent.select_actions(obs)
    #         next_obs, rewards, dones, infos = env.step(actions)
    #         ## store in buffer
    #         agent.store_step(
    #             t,
    #             obs_dict=obs,
    #             actions_dict=actions,
    #             logps_dict=logps,
    #             rewards_dict=rewards,
    #             dones_dict=dones,
    #         )

    #         for a in agent_ids:
    #             ep_return[a] += rewards[a]

    #         obs = next_obs
    #     agent.update()
    #     avg_return = sum(ep_return.values()) / len(ep_return)
    #     # Log every 10 episodes
    #     if ep % 10 == 0:
    #         logger.info(f"Episode {ep}/{num_episodes} | Avg Return: {avg_return:.3f}")
        
    #     # Log every episode to file (less verbose for console)
    #     if ep % 1 == 0:
    #         logger.debug(f"EP {ep} | Return: {avg_return:.3f}")
    
    # logger.info("Training completed successfully")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train()
   
