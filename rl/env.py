import json
import os
from collections import deque
from itertools import islice

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from gymnasium import spaces

from fl.model import create_model

# Constants
CAPACITY_MBPS = 10.0
MAX_QUEUE_DELAY_MS = 50.0
MAX_UTIL_CLIP = 2.0
EPISODE_LENGTH = 100
LOOKBACK_WINDOW = 12
PREDICTION_HORIZON = 3
MAX_F_IN = 7


class DomainPredictor:
    def __init__(self, fl_model_path: str, data_dir: str = "data/preprocessed"):
        self.models = {}
        self.domain_indices = {}
        self.active = False
        self.max_f_in = MAX_F_IN

        if not os.path.exists(fl_model_path):
            print(f"FL model not found at {fl_model_path}")
            return

        try:
            state_dict = torch.load(fl_model_path, map_location="cpu")
            # Dynamically infer f_in from model weights
            if "lstm.weight_ih_l0" in state_dict:
                self.max_f_in = state_dict["lstm.weight_ih_l0"].shape[1]
            elif "fc.weight" in state_dict:
                self.max_f_in = state_dict["fc.weight"].shape[0] // (3 * 64)

            self.active = True

            for domain_id in [0, 1, 2]:
                idx_path = os.path.join(data_dir, f"domain_{domain_id}_edge_indices.json")
                if os.path.exists(idx_path):
                    with open(idx_path) as f:
                        self.domain_indices[domain_id] = json.load(f)

                    model = create_model(f_in=self.max_f_in)
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models[domain_id] = model
                else:
                    print(f"Warning: Domain {domain_id} indices not found.")
                    self.active = False
        except (RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError) as e:
            print(f"Failed to load FL model or domain info: {e}")
            self.active = False

    def predict(self, history_buffer: np.ndarray, num_edges: int) -> np.ndarray:
        if not self.active:
            return np.zeros(num_edges, dtype=np.float32)

        pred = np.zeros(num_edges, dtype=np.float32)
        with torch.no_grad():
            for domain_id, indices in self.domain_indices.items():
                domain_hist = history_buffer[:, indices]  # shape: (12, F_in)
                f_in = len(indices)
                if f_in < self.max_f_in:
                    pad = np.zeros((history_buffer.shape[0], self.max_f_in - f_in), dtype=np.float32)
                    domain_hist = np.concatenate([domain_hist, pad], axis=1)
                elif f_in > self.max_f_in:
                    domain_hist = domain_hist[:, : self.max_f_in]

                x = torch.tensor(domain_hist, dtype=torch.float32).unsqueeze(0)  # (1, 12, max_f_in)
                y = self.models[domain_id](x)  # (1, 3, max_f_in)

                # We want t+1 prediction -> y[0, 0, :min(f_in, self.max_f_in)]
                valid_len = min(f_in, self.max_f_in)
                domain_pred = y[0, 0, :valid_len].numpy()

                # Map back to full edge array.
                # Note: multiple domains might predict the same edge (cross-domain).
                # We can average them, or just let the last one overwrite. Averaging is safer.
                for i, idx in enumerate(indices):
                    if pred[idx] == 0:
                        pred[idx] = domain_pred[i]
                    else:
                        pred[idx] = (pred[idx] + domain_pred[i]) / 2.0

        return pred


class NSFNETRoutingEnv(gym.Env):
    """
    Lightweight Network Simulator (Gym Environment) for RL Traffic Engineering.
    Spec: §6
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, topo_path: str, tm_path: str, fl_model_path: str | None = None):
        super().__init__()

        # --- 1. Load Topology ---
        self.topology = nx.DiGraph()
        with open(topo_path) as f:
            topo_data = json.load(f)

        for node in topo_data["nodes"]:
            self.topology.add_node(node["id"])

        self.edges = []
        for link in topo_data["links"]:
            u, v = link["source"], link["target"]
            delay = link.get("delay_ms", link.get("delay", 5.0))  # ms
            cap = link.get("capacity_bps", link.get("capacity", CAPACITY_MBPS * 1e6))
            # Add bidirectional edges
            self.topology.add_edge(u, v, capacity=cap, delay=delay)
            self.topology.add_edge(v, u, capacity=cap, delay=delay)
            self.edges.append((u, v))
            self.edges.append((v, u))

        # Keep consistent edge ordering (42 directed links)
        self.edges = sorted(self.topology.edges())
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        self.num_edges = len(self.edges)

        # --- 2. Load Traffic Matrices ---
        # Shape: (8064, 14, 14)
        if os.path.exists(tm_path):
            self.tm_data = np.load(tm_path)
        else:
            print(f"Warning: TM data not found at {tm_path}. Using random mock data.")
            self.tm_data = np.random.rand(8064, 14, 14) * 2.0  # Mbps

        self.num_nodes = self.tm_data.shape[1]
        self.dataset_len = self.tm_data.shape[0]

        # Find top 20 OD pairs by average traffic
        avg_tm = np.mean(self.tm_data, axis=0)
        od_pairs = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    od_pairs.append((i, j, avg_tm[i, j]))

        od_pairs.sort(key=lambda x: x[2], reverse=True)
        self.managed_od_pairs = [(u, v) for u, v, _ in od_pairs[:20]]
        self.unmanaged_od_pairs = [(u, v) for u, v, _ in od_pairs[20:]]
        self.num_managed = len(self.managed_od_pairs)

        # --- 3. Pre-compute K=3 shortest paths ---
        self.K = 3
        self.disabled_links = set()
        self.scheduled_failures = []
        self._recompute_paths()

        # Calculate maximum possible demand for normalization
        self.d_max = np.max(self.tm_data) if np.max(self.tm_data) > 0 else 1.0

        # Calculate theoretical maximum delay for normalization
        self.max_delay_norm = 0.0
        for p in self.path_table.values():
            for path in p:
                d = 0.0
                for i in range(len(path) - 1):
                    e = (path[i], path[i + 1])
                    d += self.topology[e[0]][e[1]]["delay"] + MAX_QUEUE_DELAY_MS
                self.max_delay_norm = max(self.max_delay_norm, d)

        # --- 4. Define Spaces ---
        # State: 42 (util) + 42 (predicted util) + 20 (demands) = 104
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(104,), dtype=np.float32)
        # Action: 20 choices of 3 paths
        self.action_space = spaces.MultiDiscrete([self.K] * self.num_managed)

        # --- 5. State Variables ---
        self.time_index = 0
        self.step_count = 0
        self.link_utils = np.zeros(self.num_edges, dtype=np.float32)
        self.history_buffer = deque(maxlen=LOOKBACK_WINDOW)

        # Load FL model
        self.fl_model = DomainPredictor(fl_model_path) if fl_model_path else None

    def _recompute_paths(self):
        G_filtered = self.topology.copy()
        for u, v in self.disabled_links:
            if G_filtered.has_edge(u, v):
                G_filtered.remove_edge(u, v)

        self.path_table = {}
        for u, v in self.managed_od_pairs:
            try:
                paths = list(islice(nx.shortest_simple_paths(G_filtered, u, v), self.K))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths = [[]]

            while len(paths) > 0 and len(paths) < self.K:
                paths.append(paths[-1])
            self.path_table[(u, v)] = paths

        self.unmanaged_path_table = {}
        for u, v in self.unmanaged_od_pairs:
            try:
                paths = list(islice(nx.shortest_simple_paths(G_filtered, u, v), 1))
                self.unmanaged_path_table[(u, v)] = paths[0]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self.unmanaged_path_table[(u, v)] = []

    def disable_link(self, u: int, v: int):
        self.disabled_links.add((u, v))
        self.disabled_links.add((v, u))
        self._recompute_paths()

    def enable_link(self, u: int, v: int, capacity: float | None = None, delay: float | None = None):
        self.disabled_links.discard((u, v))
        self.disabled_links.discard((v, u))
        self._recompute_paths()

    def set_link_failure(self, u: int, v: int, failure_step: int):
        self.scheduled_failures.append((failure_step, u, v))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random starting point in the dataset (ensuring room for EPISODE_LENGTH)
        # Use only training split ideally, but for simplicity pick any valid offset
        max_start = self.dataset_len - EPISODE_LENGTH - 1
        self.time_index = self.np_random.integers(0, max_start)
        self.step_count = 0

        self.link_utils = np.zeros(self.num_edges, dtype=np.float32)
        for _ in range(LOOKBACK_WINDOW):
            self.history_buffer.append(np.zeros(self.num_edges, dtype=np.float32))

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        # 1. Current utilization
        u_t = np.clip(self.link_utils, 0.0, 1.0)

        # 2. Predicted utilization
        if self.fl_model is not None:
            hist_array = np.array(self.history_buffer, dtype=np.float32)  # (12, 42)
            u_pred = self.fl_model.predict(hist_array, self.num_edges)
        else:
            u_pred = np.zeros(self.num_edges, dtype=np.float32)

        u_pred = np.clip(u_pred, 0.0, 1.0)

        # 3. Normalized demands
        tm = self.tm_data[self.time_index]
        demands = np.array([tm[u, v] for u, v in self.managed_od_pairs], dtype=np.float32) / self.d_max
        demands = np.clip(demands, 0.0, 1.0)

        obs = np.concatenate([u_t, u_pred, demands])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one environment step.
        Spec: §6.3
        """
        for fail_step, u, v in self.scheduled_failures:
            if self.step_count == fail_step:
                self.disable_link(u, v)

        # a. Advance time index
        self.time_index += 1
        self.step_count += 1
        current_tm = self.tm_data[self.time_index]

        # b. Compute link loads
        link_loads = np.zeros(self.num_edges, dtype=np.float32)

        # Managed OD pairs

        managed_paths = []
        for i, (u, v) in enumerate(self.managed_od_pairs):
            path_idx = action[i]
            path = self.path_table[(u, v)][path_idx]
            managed_paths.append(path)
            demand = current_tm[u, v]
            for j in range(len(path) - 1):
                e = (path[j], path[j + 1])
                link_loads[self.edge_to_idx[e]] += demand

        # Unmanaged OD pairs
        for u, v in self.unmanaged_od_pairs:
            path = self.unmanaged_path_table[(u, v)]
            demand = current_tm[u, v]
            for j in range(len(path) - 1):
                e = (path[j], path[j + 1])
                link_loads[self.edge_to_idx[e]] += demand

        # c. Compute link utilizations using actual per-edge capacity
        capacities = np.array(
            [self.topology[e[0]][e[1]]["capacity"] if e not in self.disabled_links else 1.0 for e in self.edges],
            dtype=np.float32,
        )
        self.link_utils = link_loads / capacities
        for idx, e in enumerate(self.edges):
            if e in self.disabled_links:
                self.link_utils[idx] = 0.0
        self.link_utils = np.clip(self.link_utils, 0.0, MAX_UTIL_CLIP)

        # d. Update history buffer
        self.history_buffer.append(self.link_utils.copy())

        # e. Get observation (which runs the FL model internally)
        obs = self._get_observation()

        # f & g. Compute delay and loss metrics per link
        link_delays = np.zeros(self.num_edges)
        link_losses = np.zeros(self.num_edges)

        for idx, e in enumerate(self.edges):
            if e in self.disabled_links:
                link_delays[idx] = 0.0
                link_losses[idx] = 0.0
                continue

            u_e = self.link_utils[idx]
            prop_delay = self.topology[e[0]][e[1]]["delay"]
            cap_e = self.topology[e[0]][e[1]]["capacity"]

            # Queuing delay (M/M/1: d_queue = u_e / (C * (1 - u_e)), C in bps -> convert to ms)
            q_delay = (u_e / (cap_e * (1.0 - u_e))) * 1e3 if u_e < 0.95 else MAX_QUEUE_DELAY_MS
            link_delays[idx] = prop_delay + q_delay

            # Packet loss
            if u_e <= 0.90:
                l_e = 0.0
            elif u_e < 1.00:
                l_e = (u_e - 0.90) / 0.10
            else:
                l_e = 1.0
            link_losses[idx] = l_e

        # OD Pair metrics (managed only)
        total_delay = 0.0
        total_loss = 0.0

        for path in managed_paths:
            p_delay = 0.0
            p_success = 1.0
            for j in range(len(path) - 1):
                e = (path[j], path[j + 1])
                idx = self.edge_to_idx[e]
                p_delay += link_delays[idx]
                p_success *= 1.0 - link_losses[idx]

            total_delay += p_delay
            total_loss += 1.0 - p_success

        avg_delay = total_delay / self.num_managed
        avg_delay_norm = min(avg_delay / self.max_delay_norm, 1.0)
        avg_loss = total_loss / self.num_managed

        # h. Compute reward
        mlu = np.max(self.link_utils)
        reward = -(1.0 * mlu + 0.5 * avg_delay_norm + 2.0 * avg_loss)

        # i. Check done
        all_congested = np.all(self.link_utils > 0.95)
        done = bool(self.step_count >= EPISODE_LENGTH or all_congested)
        truncated = False

        info = {"mlu": float(mlu), "avg_delay": float(avg_delay), "avg_loss": float(avg_loss)}

        return obs, float(reward), done, truncated, info


if __name__ == "__main__":
    # Smoke test
    # Ensure dummy data exists
    os.makedirs("data/generated", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    topo = {
        "nodes": [{"id": i} for i in range(14)],
        "links": [{"source": 0, "target": 1}],  # Minimal dummy just to avoid crash, actual needs 14 nodes connected
    }
    # To run test properly, let's create a full mock topology matching the spec loosely
    edges = [
        (0, 1),
        (0, 2),
        (0, 7),
        (1, 2),
        (1, 3),
        (2, 5),
        (3, 4),
        (3, 8),
        (4, 5),
        (4, 6),
        (5, 10),
        (5, 13),
        (6, 7),
        (7, 9),
        (8, 9),
        (8, 11),
        (8, 12),
        (9, 10),
        (10, 11),
        (10, 12),
        (11, 13),
        (12, 13),
    ]
    mock_topo = {
        "nodes": [{"id": i} for i in range(14)],
        "links": [{"source": u, "target": v, "capacity": 10.0, "delay": 5.0} for u, v in edges],
    }

    with open("data/nsfnet_topology.json", "w") as f:
        json.dump(mock_topo, f)

    np.save("data/generated/traffic_matrices.npy", np.random.rand(1000, 14, 14) * 0.5)

    env = NSFNETRoutingEnv("data/nsfnet_topology.json", "data/generated/traffic_matrices.npy")
    obs, _ = env.reset()
    print("Obs shape:", obs.shape)

    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    print("Step 1 Reward:", reward)
    print("Step 1 Info:", info)
    print("Environment smoke test passed!")
