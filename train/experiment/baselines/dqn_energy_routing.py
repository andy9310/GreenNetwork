"""
DQN-based Energy-Efficient Routing (DQN-EER) Baseline
Based on: "DQN-based energy-efficient routing algorithm in software-defined networks"

Deep Q-Network approach for learning energy-efficient routing policies in SDN
Key differences from RL-ER (tabular Q-learning):
- Uses deep neural network for function approximation
- Experience replay for stable learning
- Target network for improved convergence
- Handles continuous state spaces better
"""

import numpy as np
import networkx as nx
import time
from collections import deque
import random


class DQNEnergyRouting:
    """
    DQN-based energy-efficient routing algorithm for SDN
    
    The algorithm learns to select which links to keep active to minimize
    energy consumption while maintaining QoS constraints.
    """
    
    def __init__(self, 
                 state_dim=10,
                 action_dim=100,
                 learning_rate=0.001,
                 discount_factor=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 batch_size=32,
                 buffer_size=10000,
                 target_update_freq=10,
                 hidden_dims=[128, 128]):
        """
        Args:
            state_dim: Dimension of state representation
            action_dim: Number of possible actions (links to manage)
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            target_update_freq: Frequency of target network updates
            hidden_dims: Hidden layer dimensions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_dims = hidden_dims
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Networks (using numpy for simplicity, can be replaced with PyTorch)
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()
        
        # Training statistics
        self.computation_times = []
        self.training_losses = []
        self.update_count = 0
        self.episode_count = 0
        
    def _build_network(self):
        """Build a simple neural network using numpy"""
        # Initialize weights for a simple feedforward network
        network = {
            'w1': np.random.randn(self.state_dim, self.hidden_dims[0]) * 0.1,
            'b1': np.zeros(self.hidden_dims[0]),
            'w2': np.random.randn(self.hidden_dims[0], self.hidden_dims[1]) * 0.1,
            'b2': np.zeros(self.hidden_dims[1]),
            'w3': np.random.randn(self.hidden_dims[1], self.action_dim) * 0.1,
            'b3': np.zeros(self.action_dim)
        }
        return network
    
    def _forward(self, state, network):
        """Forward pass through the network"""
        # Layer 1
        z1 = np.dot(state, network['w1']) + network['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = np.dot(a1, network['w2']) + network['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output layer
        q_values = np.dot(a2, network['w3']) + network['b3']
        
        return q_values
    
    def _update_target_network(self):
        """Copy weights from Q-network to target network"""
        for key in self.q_network:
            self.target_network[key] = self.q_network[key].copy()
    
    def get_state(self, graph, flows):
        """
        Extract state representation from network
        
        Args:
            graph: NetworkX graph
            flows: List of active flows
            
        Returns:
            state: Numpy array of state features
        """
        # State features:
        # 1. Network-level features
        total_links = graph.number_of_edges()
        active_links = sum(1 for _, _, d in graph.edges(data=True) if d.get('active', 1) == 1)
        active_ratio = active_links / total_links if total_links > 0 else 0
        
        # 2. Flow statistics
        num_flows = len(flows)
        avg_flow_size = np.mean([f.size for f in flows]) if flows else 0
        
        # 3. Link utilization statistics
        utilizations = []
        for u, v, d in graph.edges(data=True):
            if d.get('active', 1) == 1:
                util = d.get('utilization', 0)
                utilizations.append(util)
        
        avg_util = np.mean(utilizations) if utilizations else 0
        max_util = np.max(utilizations) if utilizations else 0
        min_util = np.min(utilizations) if utilizations else 0
        
        # 4. Topology features
        avg_degree = np.mean([d for _, d in graph.degree()])
        
        # Combine into state vector (pad or truncate to state_dim)
        state = np.array([
            active_ratio,
            num_flows / 100.0,  # Normalize
            avg_flow_size / 1000.0,  # Normalize
            avg_util,
            max_util,
            min_util,
            avg_degree / 10.0,  # Normalize
            total_links / 1000.0,  # Normalize
            active_links / 1000.0,  # Normalize
            len(utilizations) / 1000.0  # Normalize
        ])
        
        # Pad or truncate to state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        
        return state
    
    def select_links(self, graph, flows, traffic_matrix=None):
        """
        Select which links to keep active using DQN policy
        
        Args:
            graph: NetworkX graph
            flows: List of active flows
            traffic_matrix: Traffic demand matrix (optional)
            
        Returns:
            active_links: Set of (u, v) tuples for active links
            metrics: Dict of performance metrics
        """
        start_time = time.time()
        
        # Get current state
        state = self.get_state(graph, flows)
        
        # Get Q-values for all actions
        q_values = self._forward(state, self.q_network)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action: select random subset of links
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action: select action with highest Q-value
            action_idx = np.argmax(q_values)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Convert action index to link activation pattern
        active_links = self._action_to_links(action_idx, graph)
        
        # Ensure connectivity
        active_links = self._ensure_connectivity(graph, active_links)
        
        comp_time = time.time() - start_time
        self.computation_times.append(comp_time)
        
        # Calculate metrics
        n_edges = graph.number_of_edges()
        metrics = {
            'active_links': len(active_links),
            'energy_saving': (n_edges - len(active_links)) / n_edges * 100 if n_edges > 0 else 0,
            'computation_time': comp_time,
            'epsilon': self.epsilon,
            'action_idx': action_idx
        }
        
        return active_links, metrics
    
    def _action_to_links(self, action_idx, graph):
        """
        Convert action index to set of active links
        
        Strategy: Use action_idx to determine activation threshold
        """
        edges = list(graph.edges())
        n_edges = len(edges)
        
        # Map action_idx to number of links to keep active
        # action_idx in [0, action_dim-1] -> keep [min_links, max_links] active
        min_links = max(int(n_edges * 0.2), 10)  # At least 20% or 10 links
        max_links = n_edges
        
        # Linear mapping
        keep_ratio = action_idx / max(self.action_dim - 1, 1)
        n_keep = int(min_links + keep_ratio * (max_links - min_links))
        n_keep = max(min_links, min(n_keep, max_links))
        
        # Rank links by importance (degree centrality + utilization)
        link_scores = []
        node_degrees = dict(graph.degree())
        
        for u, v in edges:
            # Importance score
            degree_score = (node_degrees[u] + node_degrees[v]) / (2 * graph.number_of_nodes())
            util_score = graph[u][v].get('utilization', 0.5)
            score = 0.6 * degree_score + 0.4 * util_score
            link_scores.append(((u, v), score))
        
        # Sort by score and select top n_keep
        link_scores.sort(key=lambda x: x[1], reverse=True)
        active_links = set([link for link, _ in link_scores[:n_keep]])
        
        return active_links
    
    def _ensure_connectivity(self, graph, active_links):
        """Ensure network remains connected"""
        # Build active subgraph
        H = nx.Graph()
        H.add_nodes_from(graph.nodes())
        H.add_edges_from(active_links)
        
        # If not connected, add bridges
        if not nx.is_connected(H):
            components = list(nx.connected_components(H))
            
            # Connect components
            for i in range(len(components) - 1):
                comp1 = components[i]
                comp2 = components[i + 1]
                
                # Find shortest bridge
                for n1 in comp1:
                    for n2 in comp2:
                        if graph.has_edge(n1, n2):
                            active_links.add((n1, n2) if n1 < n2 else (n2, n1))
                            break
                    else:
                        continue
                    break
        
        return active_links
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        Perform one training step using experience replay
        
        Returns:
            loss: Training loss (None if not enough samples)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Compute target Q-values
        next_q_values = np.array([self._forward(s, self.target_network) for s in next_states])
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute current Q-values
        current_q_values = np.array([self._forward(s, self.q_network) for s in states])
        
        # Compute loss (MSE between predictions and targets)
        loss = 0
        for i in range(self.batch_size):
            action = actions[i]
            target = targets[i]
            prediction = current_q_values[i, action]
            loss += (prediction - target) ** 2
        
        loss /= self.batch_size
        self.training_losses.append(loss)
        
        # Simplified weight update (approximate gradient descent)
        # In a full implementation, this would use proper backpropagation
        # For now, we do a simple update on the output layer
        for i in range(self.batch_size):
            action = actions[i]
            target = targets[i]
            prediction = current_q_values[i, action]
            
            # Gradient of MSE loss w.r.t. prediction
            grad_output = 2 * (prediction - target) / self.batch_size
            
            # Update output layer weights (simplified)
            # w3 has shape (hidden_dim, action_dim)
            # We update the column corresponding to the action
            # Using a small learning rate to avoid instability
            update_scale = self.lr * grad_output * 0.01  # Scale down for stability
            self.q_network['w3'][:, action] -= update_scale
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss
    
    def get_stats(self):
        """Get algorithm statistics"""
        return {
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0,
            'avg_training_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'update_count': self.update_count,
            'total_decisions': len(self.computation_times)
        }
    
    def save_model(self, filepath):
        """Save model weights"""
        np.savez(filepath, 
                 q_network=self.q_network,
                 target_network=self.target_network,
                 epsilon=self.epsilon)
    
    def load_model(self, filepath):
        """Load model weights"""
        data = np.load(filepath, allow_pickle=True)
        self.q_network = data['q_network'].item()
        self.target_network = data['target_network'].item()
        self.epsilon = float(data['epsilon'])
