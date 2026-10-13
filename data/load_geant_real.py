"""
Downloader and Ingestion parser for real-world GÉANT NetFlow traffic matrices from SNDlib / TOTEM.
Processes 22-node European academic traffic traces into (N_timesteps, 22, 22) numpy arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SNDLIB_GEANT_URL = "http://sndlib.zib.de/download/directed-geant-uhlig-15min-over-4months-ALL-native.tar.gz"


def download_geant_archive(target_dir: str = "data/raw_geant") -> str | None:
    """Attempts to download SNDlib GÉANT tar archive."""
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "geant_uhlig_traces.tar.gz")

    if os.path.exists(archive_path) and os.path.getsize(archive_path) > 1000:
        print(f"Existing GÉANT archive found at {archive_path}")
        return archive_path

    print(f"Attempting to download real GÉANT traces from {SNDLIB_GEANT_URL}...")
    try:
        req = urllib.request.Request(SNDLIB_GEANT_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=15) as response, open(archive_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Downloaded GÉANT archive ({len(data)} bytes) to {archive_path}")
        return archive_path
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"Warning: Could not automatically download SNDlib GÉANT archive ({e}).")
        print("Note: SNDlib server requires browser authentication or may be temporarily offline.")
        print("To use real traces: place the extracted SNDlib GÉANT matrix folder in data/raw_geant/.")
        return None


def parse_geant_matrices(
    raw_dir: str = "data/raw_geant",
    output_path: str = "data/generated/geant_real_traffic_matrices.npy",
    max_timesteps: int = 8064,
) -> np.ndarray | None:
    """
    Parses extracted SNDlib GÉANT files into 3D traffic matrices of shape (timesteps, 22, 22).
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"Raw directory {raw_dir} does not exist.")
        return None

    # Find matrix files
    matrix_files = sorted(list(raw_path.glob("*.txt")) + list(raw_path.glob("*.xml")))
    if not matrix_files:
        print(f"No matrix text or xml files found in {raw_dir}.")
        return None

    print(f"Found {len(matrix_files)} GÉANT snapshot files in {raw_dir}.")
    num_timesteps = min(len(matrix_files), max_timesteps) if max_timesteps > 0 else len(matrix_files)

    # 22 nodes
    tms = np.zeros((num_timesteps, 22, 22), dtype=np.float32)

    # SNDlib GÉANT node names mapping
    node_mapping = {
        "at": 0, "be": 1, "ch": 2, "cz": 3, "de": 4, "dk": 5, "es": 6, "fr": 7,
        "gr": 8, "hr": 9, "hu": 10, "ie": 11, "il": 12, "it": 13, "lu": 14, "nl": 15,
        "no": 16, "pl": 17, "pt": 18, "se": 19, "si": 20, "uk": 21
    }

    for t, file_path in enumerate(matrix_files[:num_timesteps]):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("demand_") or "(" in line:
                    parts = line.replace("(", "").replace(")", "").split()
                    if len(parts) >= 3:
                        src_str = parts[0].split(".")[0].lower()
                        dst_str = parts[1].split(".")[0].lower()
                        if src_str in node_mapping and dst_str in node_mapping:
                            src_idx = node_mapping[src_str]
                            dst_idx = node_mapping[dst_str]
                            try:
                                val = float(parts[2]) / 1000.0  # kbps to Mbps
                                tms[t, src_idx, dst_idx] = val
                            except ValueError:
                                pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, tms)
    print(f"Saved real GÉANT Traffic Matrices to {output_path} with shape {tms.shape}")
    return tms


def main():
    parser = argparse.ArgumentParser(description="Download and Parse Real GÉANT Traces")
    parser.add_argument("--raw_dir", type=str, default="data/raw_geant", help="Directory for raw GÉANT files")
    parser.add_argument("--out", type=str, default="data/generated/geant_real_traffic_matrices.npy", help="Output path")
    parser.add_argument("--timesteps", type=int, default=8064, help="Max timesteps to extract")
    args = parser.parse_args()

    # Attempt download
    download_geant_archive(args.raw_dir)
    res = parse_geant_matrices(args.raw_dir, args.out, max_timesteps=args.timesteps)
    if res is None:
        print("Real GÉANT trace files not present. Using synthetic calibrated GÉANT matrix generator.")
        synthetic_path = "data/generated/geant_traffic_matrices.npy"
        if not os.path.exists(synthetic_path):
            print("Generating synthetic GÉANT traffic matrices...")
            from data.generate_tm import build_graph, generate_base_tms, generate_temporal_tms, load_topology, scale_alpha
            topo = load_topology("data/geant_topology.json")
            G = build_graph(topo)
            base_tm = generate_base_tms(G)
            alpha = scale_alpha(G, base_tm, 0.85)
            tms = generate_temporal_tms(base_tm, alpha, 8064)
            np.save(synthetic_path, tms)
            print(f"Synthetic fallback matrices created at {synthetic_path}")


if __name__ == "__main__":
    main()
