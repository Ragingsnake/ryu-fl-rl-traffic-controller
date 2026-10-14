"""
Traffic Orchestrator
Reads the traffic matrix and drives Iperf3 between Mininet hosts.
"""

import os
import time

import numpy as np


def main():
    print("Initializing Traffic Orchestrator...")
    tm_path = "../data/generated/traffic_matrices.npy"
    if not os.path.exists(tm_path):
        print("TM not found. Run generate_tm.py first.")
        return

    tms = np.load(tm_path)
    print(f"Loaded TM with {len(tms)} timesteps.")

    # Find top OD pairs for the demo
    avg_tm = np.mean(tms, axis=0)
    num_nodes = avg_tm.shape[0]

    od_pairs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                od_pairs.append((i, j, avg_tm[i, j]))

    od_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = od_pairs[:5]  # Top 5 for Iperf

    print("Starting traffic generation loop...")
    for step in range(10):  # Demo for 10 steps
        print(f"--- Step {step} (Epoch = 10s) ---")
        current_tm = tms[step]

        for u, v, _ in top_pairs:
            demand = current_tm[u, v]
            if demand > 0.1:
                # In Mininet we would execute:
                # h{u} iperf3 -c h{v} -u -b {demand}M -t 10
                print(f"Executing: h{u} -> h{v} | Demand: {demand:.2f} Mbps")

        time.sleep(10)


if __name__ == "__main__":
    main()
