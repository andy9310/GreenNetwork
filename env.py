import gym
import numpy as np
import networkx as nx
from gym import spaces


class NetworkEnv(gym.Env):
    """
    A custom environment representing a network of N nodes,
    each with up to 4 interfaces. The agent can decide which links
    to disable, and the traffic is then rerouted over the remaining links.
    
    Observation:
        - A vector of length E (number of links) describing the utilization
          of each link and whether it is open/closed.
    
    Action:
        - A binary vector of length E. 0 = close this link, 1 = keep it open.
    
    Reward:
        - Negative of the number of overloaded links, i.e. reward = -X,
          where X = number of links exceeding their capacity.
          Alternatively, you could define reward = -(sum of how much
          each link is over capacity).
          
    Done:
        - After a fixed number of steps (e.g., self.max_steps),
          or you could define your own termination condition.
    """

    def __init__(
        self, 
        num_nodes=6, 
        max_interfaces=4, 
        max_capacity=100, 
        max_steps=10, 
        seed=None
    ):
        super(NetworkEnv, self).__init__()
        
        self.num_nodes = num_nodes
        self.max_interfaces = max_interfaces
        self.max_capacity = max_capacity
        self.max_steps = max_steps
        self.current_step = 0
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random traffic matrix
        # traffic[i, j] = traffic from node i to node j
        self.traffic = None
        
        # We will represent the network as a Graph using networkx for convenience
        # The graph edges store capacities
        self.graph = nx.Graph()
        
        self._build_topology()
        
        # Number of edges
        self.num_edges = self.graph.number_of_edges()
        
        # Define observation space:
        # One possible approach:
        #   obs[i] = [usage_of_edge_i / capacity_of_edge_i, is_edge_open]
        # So each edge yields 2 numbers, total length = 2 * num_edges.
        low = np.array([0.0, 0.0] * self.num_edges, dtype=np.float32)
        high = np.array([np.inf, 1.0] * self.num_edges, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Define action space:
        # A binary vector for each edge: 0 -> close, 1 -> open
        self.action_space = spaces.MultiBinary(self.num_edges)
        
        # We will maintain a list of edges for indexing
        # e.g. edge_list[i] = (u, v)
        self.edge_list = list(self.graph.edges(data=False))
        
        # Track current usage of each edge; usage[i] is the traffic currently on edge i
        self.usage = np.zeros(self.num_edges, dtype=float)
        
    def _build_topology(self):
        """
        Create a random network topology with constraints:
          - Up to max_interfaces edges per node
          - Each edge has a random capacity (for demonstration, keep it constant or random)
        """
        self.graph.clear()
        
        # Add all nodes
        for n in range(self.num_nodes):
            self.graph.add_node(n)
        
        # Randomly connect nodes, ensuring at most 'max_interfaces' connections per node
        # A straightforward approach:
        #   1) Connect each node to 1 random neighbor to ensure connectivity (if you wish).
        #   2) Then add additional edges at random, ensuring the interface limit.
        
        # Step 1: Connect a chain or random spanning tree to keep graph (somewhat) connected
        #         This is optional, depending on how you want your random network.
        all_nodes = list(range(self.num_nodes))
        np.random.shuffle(all_nodes)
        for i in range(len(all_nodes) - 1):
            u = all_nodes[i]
            v = all_nodes[i + 1]
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, capacity=self.max_capacity)
        
        # Step 2: Attempt to add edges at random
        possible_edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if not self.graph.has_edge(i, j):
                    possible_edges.append((i, j))
        
        np.random.shuffle(possible_edges)
        for (u, v) in possible_edges:
            # Check the interface limit
            if self.graph.degree[u] < self.max_interfaces and self.graph.degree[v] < self.max_interfaces:
                self.graph.add_edge(u, v, capacity=self.max_capacity)
        
        # Now we have a random topology with each node having up to max_interfaces edges.
    
    def reset(self):
        """
        Reset the environment:
          - Generate a random traffic matrix
          - Mark all links as open
          - Zero out link usage
          - Return initial observation
        """
        self.current_step = 0
        
        # Generate random traffic matrix
        # For demonstration, let traffic[i, i] = 0
        # and each i->j (i != j) has some random traffic, up to e.g. 50
        self.traffic = np.random.randint(low=0, high=50, size=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            self.traffic[i, i] = 0
        
        # Reset usage on each edge to 0
        self.usage = np.zeros(self.num_edges, dtype=float)
        
        # By default, all edges are open to start
        # We'll incorporate that into usage array dimension for convenience. 
        # But we do need a separate structure if you want to track "open/closed" distinctly.
        # For demonstration, we'll just keep a “link_open” array:
        self.link_open = np.ones(self.num_edges, dtype=int)  # 1=open, 0=closed
        
        # Re-route traffic with all links open
        self._update_link_usage()
        
        return self._get_observation()

    def _get_observation(self):
        """
        Construct the observation vector:
           For each edge i:
             obs[2*i    ] = usage_i / capacity_i
             obs[2*i + 1] = link_open_i
        """
        obs = []
        for i, (u, v) in enumerate(self.edge_list):
            cap = self.graph[u][v]['capacity']
            usage_ratio = self.usage[i] / cap if cap > 0 else 0.0
            is_open = self.link_open[i]
            obs.append(usage_ratio)
            obs.append(is_open)
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """
        Take an action (binary vector of length num_edges):
          - Close or open each link based on action bits.
          - Reroute traffic on the resulting network.
          - Compute the reward as negative of the number of overloaded links.
          - Return (observation, reward, done, info)
        """
        self.current_step += 1
        
        # Apply action to links
        self.link_open = action.copy()
        
        # Recompute link usage with the new open/closed configuration
        self._update_link_usage()
        
        # Calculate how many links are overloaded
        overloaded_links = self._count_overloaded_links()
        
        # For example, let reward = - (number of overloaded links)
        reward = -float(overloaded_links)
        
        # Check if done
        done = (self.current_step >= self.max_steps)
        
        # (Optional) info dict
        info = {
            'overloaded_links': overloaded_links
        }
        
        return self._get_observation(), reward, done, info

    def _update_link_usage(self):
        """
        Re-route traffic in the network with the current open/closed state of edges,
        then compute usage on each link.
        
        For simplicity, we:
          1) Build a subgraph of only the open edges.
          2) For each (i, j) where traffic > 0, we find a path in the subgraph.
             (Here, we pick a shortest path if it exists.)
          3) Add that traffic amount to all edges on that path.
        """
        # Reset usage
        self.usage = np.zeros(self.num_edges, dtype=float)
        
        # Build subgraph with open edges only
        G_open = nx.Graph()
        G_open.add_nodes_from(self.graph.nodes(data=True))
        
        for i, (u, v) in enumerate(self.edge_list):
            if self.link_open[i] == 1:
                # Add the edge with the same capacity attribute
                capacity = self.graph[u][v]['capacity']
                G_open.add_edge(u, v, capacity=capacity)
        
        # For each traffic demand (i, j), route along subgraph
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                traffic_demand = self.traffic[i, j]
                if traffic_demand > 0 and i != j:
                    # Try to find a path in the G_open
                    # Here we use a shortest_path by hop-count for demonstration
                    try:
                        path = nx.shortest_path(G_open, source=i, target=j)
                        # Convert path to edges, add usage
                        for k in range(len(path) - 1):
                            u = path[k]
                            v = path[k + 1]
                            # Find edge index in edge_list
                            edge_idx = self._edge_index(u, v)
                            if edge_idx is not None:
                                self.usage[edge_idx] += traffic_demand
                    except nx.NetworkXNoPath:
                        # If there's no path, that traffic is essentially dropped (or 0 usage)
                        pass
    
    def _edge_index(self, u, v):
        """
        Return the index of edge (u, v) or (v, u) in self.edge_list
        (since undirected).
        """
        # We know edge_list is a list of (n1, n2) with n1<n2 if stored in default networkx manner,
        # but let's just check both directions to be safe.
        for i, (x, y) in enumerate(self.edge_list):
            if (x == u and y == v) or (x == v and y == u):
                return i
        return None
    
    def _count_overloaded_links(self):
        """
        Count how many links exceed their capacity.
        """
        overloaded = 0
        for i, (u, v) in enumerate(self.edge_list):
            cap = self.graph[u][v]['capacity']
            if self.usage[i] > cap:
                overloaded += 1
        return overloaded

    def render(self, mode='human'):
        """
        Print out the current usage of each link, or do something more advanced if desired.
        """
        print("Current step:", self.current_step)
        print("Link usage / capacity (open=1/closed=0):")
        for i, (u, v) in enumerate(self.edge_list):
            cap = self.graph[u][v]['capacity']
            usage = self.usage[i]
            is_open = self.link_open[i]
            print(f"Edge {u}-{v} | Open={is_open} | Usage={usage:.2f} / {cap}")

    def close(self):
        pass
