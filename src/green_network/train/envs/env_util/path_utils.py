"""
Utility functions for path computation in SDN environment
"""
import networkx as nx
from typing import List, Tuple, Dict, Set
import numpy as np


def compute_k_shortest_paths(graph: nx.Graph, source: int, target: int, k: int = 3, 
                             closed_edges: Set[Tuple[int, int]] = None) -> List[List[int]]:
    """
    Compute k-shortest simple paths from source to target.
    Excludes paths that use closed edges.
    
    Args:
        graph: NetworkX graph
        source: Source node
        target: Target node
        k: Number of paths to find
        closed_edges: Set of edges (u, v) that are closed (both orders)
    
    Returns:
        List of paths, where each path is a list of node indices
    """
    if closed_edges is None:
        closed_edges = set()
    
    # Create a view of the graph without closed edges
    def edge_filter(u, v):
        edge_key = (min(u, v), max(u, v))
        return edge_key not in closed_edges
    
    subgraph = nx.subgraph_view(graph, filter_edge=edge_filter)
    
    try:
        # Use nx.shortest_simple_paths which is a generator
        paths = []
        for path in nx.shortest_simple_paths(subgraph, source, target):
            paths.append(path)
            if len(paths) >= k:
                break
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # No path exists
        return []


def compute_all_flow_paths(graph: nx.Graph, flows: List[Tuple[int, int, float]], 
                           k: int = 3, closed_edges: Set[Tuple[int, int]] = None) -> Dict[int, List[List[int]]]:
    """
    Compute k-shortest paths for all flows.
    
    Args:
        graph: NetworkX graph
        flows: List of (source, dest, demand) tuples
        k: Number of alternative paths per flow
        closed_edges: Set of closed edges
    
    Returns:
        Dictionary mapping flow_id to list of k paths
    """
    flow_paths = {}
    for flow_id, (src, dst, demand) in enumerate(flows):
        paths = compute_k_shortest_paths(graph, src, dst, k, closed_edges)
        
        # If fewer than k paths found, pad with empty paths or repeat last path
        if len(paths) < k:
            if len(paths) > 0:
                # Repeat the shortest path if not enough alternatives
                while len(paths) < k:
                    paths.append(paths[0])
            else:
                # No path at all - use empty list
                paths = [[]] * k
        
        flow_paths[flow_id] = paths
    
    return flow_paths


def route_flows_on_paths(flows: List[Tuple[int, int, float]], 
                         flow_paths: Dict[int, List[List[int]]],
                         path_selections: Dict[int, int],
                         num_edges_dict: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], float]:
    """
    Route flows on selected paths and compute edge loads.
    
    Args:
        flows: List of (source, dest, demand) tuples
        flow_paths: Dictionary mapping flow_id to list of available paths
        path_selections: Dictionary mapping flow_id to selected path index (0 to k-1)
        num_edges_dict: Dictionary with all edge keys initialized
    
    Returns:
        Dictionary mapping edge_key to total load
    """
    edge_loads = {key: 0.0 for key in num_edges_dict.keys()}
    disconnect_penalty = 0.0
    
    for flow_id, (src, dst, demand) in enumerate(flows):
        path_idx = path_selections.get(flow_id, 0)
        paths = flow_paths.get(flow_id, [])
        
        if path_idx >= len(paths) or len(paths[path_idx]) == 0:
            # Invalid path or no path available
            disconnect_penalty += demand * 10.0
            continue
        
        selected_path = paths[path_idx]
        
        # Add flow demand to all edges along the path
        for u, v in zip(selected_path[:-1], selected_path[1:]):
            edge_key = (min(u, v), max(u, v))
            edge_loads[edge_key] += demand
    
    return edge_loads, disconnect_penalty


def compute_edge_metrics(edge_loads: Dict[Tuple[int, int], float],
                         edge_capacities: Dict[Tuple[int, int], float],
                         edge_states: Dict[Tuple[int, int], int],
                         base_power: float,
                         base_delay: float) -> Tuple[float, float, float]:
    """
    Compute total energy, delay, and overload from edge loads.
    
    Args:
        edge_loads: Edge loads
        edge_capacities: Edge capacities
        edge_states: Edge states (0=closed, 1=open)
        base_power: Base power consumption
        base_delay: Base delay
    
    Returns:
        (total_energy, total_delay, total_overload)
    """
    total_energy = 0.0
    total_delay = 0.0
    total_overload = 0.0
    
    for edge_key, load in edge_loads.items():
        capacity = edge_capacities.get(edge_key, 1.0)
        is_open = edge_states.get(edge_key, 1)
        
        if not is_open:
            # Closed edge: no power, but if it has load (shouldn't happen), heavy penalty
            if load > 0:
                total_overload += load * 100.0  # Severe penalty for using closed edge
            continue
        
        # Compute utilization
        if capacity <= 0.0:
            util = 10.0 if load > 0 else 0.0
        else:
            util = load / capacity
        
        # Overload penalty
        overload = max(0.0, util - 1.0)
        total_overload += overload * 5.0
        
        # Delay increases with utilization (convex)
        delay = base_delay * (1.0 + util * util)
        total_delay += delay
        
        # Energy: open edges consume base power + load-dependent power
        energy = base_power * (1.0 + 0.5 * util)
        total_energy += energy
    
    return total_energy, total_delay, total_overload
