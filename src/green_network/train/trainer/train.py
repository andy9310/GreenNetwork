# train_mappo.py
import torch
import json
import os
import logging
from pathlib import Path

# Import training utilities
try:
    from utils.logger import get_training_logger
except ImportError:
    # Fallback if utils not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    def get_training_logger(log_dir=None):
        return logging.getLogger("GreenNetwork.Train")

try:
    from config.mappo_config import Config
except ImportError:
    # Fallback config
    class Config:
        def __init__(self):
            self.num_nodes = 40
            self.num_edges = 61
            self.graph_method = "random_tree_plus"
            self.graph_seed = 42
            self.max_steps = 50
            self.num_flows_per_step = 5
            self.rollout_steps = 512
            self.gamma = 0.99
            self.lam = 0.95
            self.lr = 3e-4
            self.clip_ratio = 0.2
            self.entropy_coef = 0.01
            self.value_coef = 0.5
            self.max_grad_norm = 0.5
            self.ppo_epochs = 10
            self.minibatch_size = 1024
            self.shared_policy = True

from agents.agent import Agent
from envs.env import SDNEnv


def train(num_episodes=1000, device="cpu", topo_file="topo.json", log_dir="logs"):
    # Setup logger
    logger = get_training_logger(log_dir=log_dir)
    logger.info("="*60)
    logger.info("Starting Green Network Training")
    logger.info("="*60)
    
    cfg = Config()
    logger.info(f"Configuration loaded: {num_episodes} episodes, device={device}")
    
    # Read topology configuration from JSON file
    topo_config = None
    if os.path.exists(topo_file):
        logger.info(f"Loading topology from '{topo_file}'...")
        try:
            with open(topo_file, 'r') as f:
                topo_config = json.load(f)
            logger.info(f"✓ Topology loaded: {topo_config.get('num_nodes', 'N/A')} nodes, {topo_config.get('num_edges', 'N/A')} edges")
        except Exception as e:
            logger.error(f"Failed to load topology file: {e}")
            logger.warning("Falling back to random generation")
            topo_config = None
    else:
        logger.warning(f"Topology file '{topo_file}' not found")
        logger.info("Using random topology generation")
    
    # Create environment
    logger.info(f"Initializing SDN environment...")
    logger.info(f"  Target: {cfg.num_nodes} nodes, {cfg.num_edges} edges")
    logger.info(f"  Method: {cfg.graph_method if hasattr(cfg, 'graph_method') else 'random_tree_plus'}")
    logger.info(f"  Seed: {cfg.graph_seed if hasattr(cfg, 'graph_seed') else 42}")
    
    env = SDNEnv(
        topo_config=topo_config,
        num_nodes=cfg.num_nodes if hasattr(cfg, 'num_nodes') else 40,
        num_edges=cfg.num_edges if hasattr(cfg, 'num_edges') else 61,
        max_steps=cfg.max_steps if hasattr(cfg, 'max_steps') else 50,
        num_flows_per_step=cfg.num_flows_per_step if hasattr(cfg, 'num_flows_per_step') else 5,
        seed=cfg.graph_seed if hasattr(cfg, 'graph_seed') else 42
    )
    
    logger.info("✓ Environment created successfully")
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
    logger.info("✓ Agent initialized")

    logger.info("="*60)
    logger.info("Starting training loop")
    logger.info("="*60)
    
    for ep in range(num_episodes):
        obs = env.reset()
        ep_return = {a: 0.0 for a in agent_ids}

        for t in range(cfg.rollout_steps):
            # obs: {agent_id: obs_vector}
            # 1) choose actions
            actions, logps = agent.select_actions(obs)

            # 2) step env
            next_obs, rewards, dones, infos = env.step(actions)

            # 3) store in buffer
            agent.store_step(
                t,
                obs_dict=obs,
                actions_dict=actions,
                logps_dict=logps,
                rewards_dict=rewards,
                dones_dict=dones,
            )

            for a in agent_ids:
                ep_return[a] += rewards[a]

            obs = next_obs

            # if all done, you may break early
            if all(dones.values()):
                break

        agent.update()

        avg_return = sum(ep_return.values()) / len(ep_return)
        
        # Log every 10 episodes
        if ep % 10 == 0:
            logger.info(f"Episode {ep}/{num_episodes} | Avg Return: {avg_return:.3f}")
        
        # Log every episode to file (less verbose for console)
        if ep % 1 == 0:
            logger.debug(f"EP {ep} | Return: {avg_return:.3f}")
    
    logger.info("="*60)
    logger.info("Training completed successfully")
    logger.info("="*60)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup logging
    logger = get_training_logger()
    logger.info(f"Detected device: {device}")
    
    try:
        train(device=device)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
