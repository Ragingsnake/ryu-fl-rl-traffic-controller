import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np


def load_topology(topo_path: str) -> dict:
    """Load topology from JSON file."""
    with open(topo_path) as f:
        return json.load(f)


def build_graph(topology: dict) -> nx.DiGraph:
    """Build NetworkX DiGraph from topology."""
    G = nx.DiGraph()
    for node in topology["nodes"]:
        G.add_node(node["id"], **node)

    for link in topology["links"]:
        G.add_edge(link["source"], link["target"], capacity=link["capacity_bps"])
        G.add_edge(link["target"], link["source"], capacity=link["capacity_bps"])

    return G


def get_sorted_edges(G: nx.DiGraph) -> list[tuple[int, int]]:
    """Return a lexicographically sorted list of directed edges to ensure stable indexing."""
    return sorted(G.edges())


def compute_link_utilization(tms: np.ndarray, G: nx.DiGraph, edges: list[tuple[int, int]]) -> np.ndarray:
    """
    Compute link utilization time series from TMs using shortest-path routing.

    Args:
        tms: Traffic matrices of shape (timesteps, num_nodes, num_nodes)
        G: Topology graph
        edges: Sorted list of directed edges

    Returns:
        Link utilization array of shape (timesteps, num_edges)
    """
    timesteps, num_nodes, _ = tms.shape
    num_edges = len(edges)
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}

    link_utils = np.zeros((timesteps, num_edges))

    print("Precomputing shortest paths...")
    # Precompute all-pairs shortest paths (hop count)
    shortest_paths = dict(nx.all_pairs_shortest_path(G))

    print("Routing traffic and computing utilizations...")
    for t in range(timesteps):
        if t > 0 and t % 1000 == 0:
            print(f"Processed {t}/{timesteps} timesteps...")

        loads = np.zeros(num_edges)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                demand = tms[t, i, j]
                if demand == 0:
                    continue

                path = shortest_paths[i][j]
                for u, v in itertools.pairwise(path):
                    idx = edge_to_idx[(u, v)]
                    loads[idx] += demand

        for i, edge in enumerate(edges):
            capacity = G.edges[edge]["capacity"]
            # Compute utilization (clipped to avoid negative, but leave upper bound for now, ideally <=1.0)
            link_utils[t, i] = max(0.0, loads[i] / capacity)

    return link_utils


def create_windows(data: np.ndarray, T: int = 12, H: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series forecasting.

    Args:
        data: Array of shape (timesteps, num_features)
        T: Lookback window size
        H: Prediction horizon

    Returns:
        X: shape (samples, T, num_features)
        Y: shape (samples, H, num_features)
    """
    X, Y = [], []
    num_samples = len(data) - T - H + 1

    for i in range(num_samples):
        X.append(data[i : i + T])
        Y.append(data[i + T : i + T + H])

    return np.array(X), np.array(Y)


def get_domain_edge_indices(G: nx.DiGraph, edges: list[tuple[int, int]], domain_label: str) -> list[int]:
    """
    Get the indices of edges visible to a specific domain.
    An edge is visible if at least one of its endpoints is in the domain.
    """
    domain_nodes = {n for n, attr in G.nodes(data=True) if attr.get("domain") == domain_label}

    indices = []
    for i, (u, v) in enumerate(edges):
        if u in domain_nodes or v in domain_nodes:
            indices.append(i)

    return indices


def main():
    print("Starting data preprocessing...")
    base_path = Path(__file__).parent
    topo_path = base_path / "nsfnet_topology.json"
    tm_path = base_path / "generated" / "traffic_matrices.npy"
    out_dir = base_path / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    topology = load_topology(str(topo_path))
    G = build_graph(topology)
    edges = get_sorted_edges(G)
    tms = np.load(str(tm_path))

    print(f"Loaded TMs of shape: {tms.shape}")

    # 2. Compute Link Utilizations
    link_utils = compute_link_utilization(tms, G, edges)
    print(f"Link utilizations shape: {link_utils.shape}, Max utilization: {np.max(link_utils):.2f}")

    # Normalize TMs (demands divided by global max)
    tm_max = np.max(tms)
    tms_norm = tms / tm_max if tm_max > 0 else tms

    # Save global arrays
    np.save(str(out_dir / "global_link_utils.npy"), link_utils)
    np.save(str(out_dir / "global_demands_norm.npy"), tms_norm)

    # 3. Create sliding windows (global)
    T, H = 12, 3
    X_global, Y_global = create_windows(link_utils, T, H)
    print(f"Generated global windows - X: {X_global.shape}, Y: {Y_global.shape}")

    # Train/Val/Test splits bounds
    # Test offset due to sliding window:
    # train = 0 to 5760 (exclusive)
    # val = 5760 to 6912
    # test = 6912 to end
    split_train = 5760
    split_val = 6912

    # 4. FL Data Distribution (Domain splitting)
    domains = ["A", "B", "C"]

    # Map domain letters to numeric IDs for FL client compatibility
    domain_to_id = {"A": 0, "B": 1, "C": 2}

    for domain in domains:
        print(f"Processing Domain {domain}...")
        domain_idx = get_domain_edge_indices(G, edges, domain)
        print(f"Domain {domain} monitors {len(domain_idx)} edges.")

        # Filter features for domain
        X_domain = X_global[:, :, domain_idx]
        Y_domain = Y_global[:, :, domain_idx]

        # Split data
        max_train = min(split_train, len(X_domain))
        max_val = min(split_val, len(X_domain))

        X_train = X_domain[:max_train]
        Y_train = Y_domain[:max_train]

        X_val = X_domain[max_train:max_val]
        Y_val = Y_domain[max_train:max_val]

        X_test = X_domain[max_val:]
        Y_test = Y_domain[max_val:]

        # Save using numeric domain IDs that FL client.py expects
        # Format: domain_{id}_train_X.npy (flat in preprocessed/)
        did = domain_to_id[domain]

        np.save(str(out_dir / f"domain_{did}_train_X.npy"), X_train.astype(np.float32))
        np.save(str(out_dir / f"domain_{did}_train_Y.npy"), Y_train.astype(np.float32))
        np.save(str(out_dir / f"domain_{did}_val_X.npy"), X_val.astype(np.float32))
        np.save(str(out_dir / f"domain_{did}_val_Y.npy"), Y_val.astype(np.float32))
        np.save(str(out_dir / f"domain_{did}_test_X.npy"), X_test.astype(np.float32))
        np.save(str(out_dir / f"domain_{did}_test_Y.npy"), Y_test.astype(np.float32))

        # Also save the edge indices mapping for global inference later
        with open(str(out_dir / f"domain_{did}_edge_indices.json"), "w") as f:
            json.dump(domain_idx, f)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
