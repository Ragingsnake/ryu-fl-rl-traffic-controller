"""
Abilene NetFlow trace loader and preprocessor.
Ingests real Abilene NetFlow matrices if present (e.g. data/generate_nsfnets/Abilene_Evaluation.csv),
or deterministically falls back to the calibrated 12-node gravity model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_real_abilene_csv(csv_path: str) -> np.ndarray | None:
    """
    Parse historical Abilene NetFlow CSV containing 132 OD pair columns.
    Expected columns: OD_1-2, OD_1-3, ..., OD_12-11.
    """
    if not os.path.exists(csv_path):
        return None

    try:
        import pandas as pd

        print(f"Reading real Abilene NetFlow CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Look for OD columns
        od_cols = [c for c in df.columns if c.startswith("OD_") or "-" in c]
        if not od_cols:
            print("No OD columns found in CSV.")
            return None

        num_timesteps = len(df)
        num_nodes = 12
        tms = np.zeros((num_timesteps, num_nodes, num_nodes), dtype=np.float32)

        for col in od_cols:
            parts = col.replace("OD_", "").split("-")
            if len(parts) == 2:
                try:
                    src = int(parts[0]) - 1  # 1-indexed to 0-indexed
                    dst = int(parts[1]) - 1
                    if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
                        # Values in kbps -> convert to bps (* 1000)
                        values = df[col].fillna(0.0).to_numpy(dtype=np.float32)
                        tms[:, src, dst] = values * 1000.0
                except ValueError:
                    continue

        print(f"Successfully parsed real Abilene CSV into matrix of shape {tms.shape}")
        return tms
    except Exception as e:
        print(f"Error parsing real Abilene CSV: {e}")
        return None


def generate_fallback_abilene(topo_path: str, out_path: str, timesteps: int = 8064) -> np.ndarray:
    """Generate calibrated synthetic traffic matrices for Abilene."""
    from data.generate_tm import build_graph, generate_base_tms, generate_temporal_tms, load_topology, scale_alpha

    print(f"Generating calibrated fallback Abilene matrices using {topo_path}...")
    topology = load_topology(topo_path)
    G = build_graph(topology)
    base_tm = generate_base_tms(G)
    alpha = scale_alpha(G, base_tm, target_utilization=0.85)
    tms = generate_temporal_tms(base_tm, alpha, num_timesteps=timesteps)
    return tms


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load or generate Abilene traffic matrices")
    parser.add_argument(
        "--csv", type=str, default="data/generate_nsfnets/Abilene_Evaluation.csv", help="Path to real NetFlow CSV"
    )
    parser.add_argument("--topo", type=str, default="data/abilene_topology.json", help="Path to Abilene topology JSON")
    parser.add_argument(
        "--out", type=str, default="data/generated/abilene_traffic_matrices.npy", help="Output .npy path"
    )
    parser.add_argument("--timesteps", type=int, default=8064, help="Timesteps if synthetic fallback used")
    args = parser.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tms = load_real_abilene_csv(args.csv)
    if tms is None:
        print("Real Abilene CSV not present or unparseable. Falling back to calibrated Gravity model...")
        tms = generate_fallback_abilene(args.topo, str(out_file), timesteps=args.timesteps)

    np.save(str(out_file), tms)
    print(f"Saved Abilene traffic matrices to: {out_file} (Shape: {tms.shape}, Mean: {np.mean(tms):.2f} bps)")


if __name__ == "__main__":
    main()
