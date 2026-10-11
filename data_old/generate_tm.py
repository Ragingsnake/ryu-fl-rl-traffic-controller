import itertools
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx
import numpy as np


def load_topology(topo_path: str) -> dict:
    """
    Load the topology from a JSON file.

    Args:
        topo_path: Path to the JSON topology file.

    Returns:
        Dictionary containing nodes and links.
    """
    with open(topo_path) as f:
        return json.load(f)


def build_graph(topology: dict) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from the topology dict.

    Args:
        topology: Dictionary containing 'nodes' and 'links'.

    Returns:
        A NetworkX DiGraph object.
    """
    G = nx.DiGraph()
    for node in topology["nodes"]:
        G.add_node(node["id"], **node)

    for link in topology["links"]:
        G.add_edge(link["source"], link["target"], capacity=link["capacity_bps"], delay=link["delay_ms"], weight=1)
        # Assuming bidirectional links as specified in spec
        G.add_edge(link["target"], link["source"], capacity=link["capacity_bps"], delay=link["delay_ms"], weight=1)

    return G


def generate_base_tms(G: nx.DiGraph) -> np.ndarray:
    """
    Generate the base Traffic Matrices (TMs) using the Gravity Model.

    Args:
        G: NetworkX DiGraph of the topology.

    Returns:
        A 14x14 numpy array representing the base TM.
    """
    num_nodes = G.number_of_nodes()
    base_tm = np.zeros((num_nodes, num_nodes))

    # Precompute shortest path hop distances
    paths = dict(nx.shortest_path_length(G, weight=None))  # Hop count distance

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            pop_i = G.nodes[i]["population_weight"]
            pop_j = G.nodes[j]["population_weight"]
            dist_ij = paths[i][j]
            # Gravity model: TM_ij = alpha * (pop_i * pop_j) / dist_ij^beta (beta=1.0)
            base_tm[i, j] = (pop_i * pop_j) / dist_ij

    return base_tm


def scale_alpha(G: nx.DiGraph, base_tm: np.ndarray, target_utilization: float = 0.85) -> float:
    """
    Find the alpha scaling factor so that the peak traffic creates ~target_utilization on busiest link
    using OSPF shortest path routing.

    Args:
        G: NetworkX DiGraph.
        base_tm: Unscaled base traffic matrix.
        target_utilization: Target maximum link utilization (0-1).

    Returns:
        Calculated alpha value.
    """
    num_nodes = G.number_of_nodes()
    link_loads = dict.fromkeys(G.edges(), 0.0)

    # Route all traffic along shortest paths to find max load
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            # Shortest path by hop count
            path = nx.shortest_path(G, source=i, target=j, weight=None)
            demand = base_tm[i, j]
            for u, v in itertools.pairwise(path):
                link_loads[(u, v)] += demand

    max_load_ratio = 0.0
    for edge in G.edges():
        capacity = G.edges[edge]["capacity"]
        if link_loads[edge] > 0:
            ratio = link_loads[edge] / capacity
            max_load_ratio = max(max_load_ratio, ratio)

    # We want max_load_ratio * alpha * peak_multiplier = target_utilization
    # Peak multiplier from temporal pattern: 1 + A_daily + A_weekly + noise = 1 + 0.3 + 0.1 + 0.05 = 1.45
    peak_multiplier = 1.45

    alpha = target_utilization / (max_load_ratio * peak_multiplier)
    print(f"Calculated alpha: {alpha:.2f} to hit ~{target_utilization * 100}% max link utilization at peak.")
    return alpha


def generate_temporal_tms(base_tm: np.ndarray, alpha: float, num_timesteps: int = 8064) -> np.ndarray:
    """
    Generate the temporal traffic matrices over all timesteps.

    Args:
        base_tm: Unscaled base traffic matrix.
        alpha: Scaling factor.
        num_timesteps: Number of timesteps to generate (8064 for 4 weeks).

    Returns:
        Numpy array of shape (num_timesteps, num_nodes, num_nodes).
    """
    num_nodes = base_tm.shape[0]
    tms = np.zeros((num_timesteps, num_nodes, num_nodes))

    scaled_base = base_tm * alpha

    T_day = 288
    T_week = 2016
    A_daily = 0.3
    A_weekly = 0.1
    sigma = 0.05

    np.random.seed(42)  # For reproducibility

    for t in range(num_timesteps):
        daily_comp = A_daily * np.sin(2 * np.pi * t / T_day)
        weekly_comp = A_weekly * np.sin(2 * np.pi * t / T_week)
        noise = np.random.normal(0, sigma, (num_nodes, num_nodes))

        # Avoid negative traffic demands
        multiplier = np.clip(1 + daily_comp + weekly_comp + noise, 0, None)
        tms[t] = scaled_base * multiplier

        # Zero out diagonal
        np.fill_diagonal(tms[t], 0)

    return tms


def main():
    """Main execution function for generating synthetic TMs."""
    print("Starting Traffic Matrix generation...")
    base_path = Path(__file__).parent
    topo_path = base_path / "nsfnet_topology.json"
    out_dir = base_path / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    topology = load_topology(str(topo_path))
    G = build_graph(topology)

    print("Building base gravity model...")
    base_tm = generate_base_tms(G)

    print("Scaling to target capacity...")
    alpha = scale_alpha(G, base_tm, target_utilization=0.85)

    print("Generating temporal traffic matrices (8064 timesteps)...")
    tms = generate_temporal_tms(base_tm, alpha, num_timesteps=8064)

    out_file = out_dir / "traffic_matrices.npy"
    np.save(str(out_file), tms)
    print(f"Generated TMs saved to: {out_file}")
    print(f"TM shape: {tms.shape}, Mean demand: {np.mean(tms):.2f} bps, Max demand: {np.max(tms):.2f} bps")


if __name__ == "__main__":
    main()
