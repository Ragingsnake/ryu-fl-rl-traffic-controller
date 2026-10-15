"""
Load and parse real-world Abilene NetFlow traffic matrices from Abilene_Evaluation.csv.
Internet2 Abilene dataset: 12 nodes, 132 OD pairs.
Converts raw NetFlow kbps traces into (N_timesteps, 12, 12) Mbps arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd


def load_abilene_traces(
    csv_path: str,
    output_path: str = "data/generated/abilene_traffic_matrices.npy",
    max_timesteps: int = 8064,
    scale_factor: float = 9.0,
) -> np.ndarray:
    """
    Parses Abilene CSV NetFlow evaluation data into 3D traffic matrices.

    Args:
        csv_path: Path to Abilene_Evaluation.csv
        output_path: Target .npy path
        max_timesteps: Maximum timesteps to extract (default 8064 = 4 weeks at 5-min intervals)
        scale_factor: Multiplier to scale historical NetFlow flows to 10Mbps link capacities (default 4.0)

    Returns:
        traffic_matrices: Array of shape (timesteps, 12, 12)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Abilene CSV not found at {csv_path}")

    print(f"Reading Abilene NetFlow data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_timesteps if max_timesteps > 0 else None)
    num_timesteps = len(df)
    print(f"Loaded {num_timesteps} snapshots from CSV.")

    # 12 nodes (0-indexed: 0..11, CSV columns are 1..12)
    num_nodes = 12
    tms = np.zeros((num_timesteps, num_nodes, num_nodes), dtype=np.float32)

    od_cols = [c for c in df.columns if c.startswith("OD_")]
    if len(od_cols) != 132:
        print(f"Warning: Expected 132 OD columns, found {len(od_cols)}")

    for col in od_cols:
        parts = col.replace("OD_", "").split("-")
        src = int(parts[0]) - 1
        dst = int(parts[1]) - 1

        values = df[col].to_numpy(dtype=np.float32) * scale_factor
        tms[:, src, dst] = values

    # Zero out diagonals
    for i in range(num_nodes):
        tms[:, i, i] = 0.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, tms)
    print(f"Saved Abilene Traffic Matrices to {output_path} with shape {tms.shape}")
    print(f"Demand statistics: Min={np.min(tms):.4f}, Mean={np.mean(tms):.4f}, Max={np.max(tms):.4f} Mbps")

    return tms


def main():
    parser = argparse.ArgumentParser(description="Parse Abilene NetFlow Traces")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/generate_nsfnets/Abilene_Evaluation.csv",
        help="Path to Abilene NetFlow CSV",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/generated/abilene_traffic_matrices.npy",
        help="Path to save traffic matrices",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=8064,
        help="Number of timesteps to parse (default 8064 = 4 weeks)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="Scaling multiplier for NetFlow demands to match 10Mbps link capacity (default 9.0)",
    )
    args = parser.parse_args()

    load_abilene_traces(args.csv_path, args.output_path, max_timesteps=args.timesteps, scale_factor=args.scale)


if __name__ == "__main__":
    main()
