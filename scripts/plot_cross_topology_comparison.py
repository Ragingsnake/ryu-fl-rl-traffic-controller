import sys
from pathlib import Path

# Ensure stdout handles unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "demo" / "cross_topology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load summary CSVs
nsfnet_csv = RESULTS_DIR / "summary_table.csv"
geant_csv = RESULTS_DIR / "summary_table_geant.csv"
abilene_csv = RESULTS_DIR / "summary_table_abilene.csv"

dfs = {}
if nsfnet_csv.exists():
    dfs["NSFNET (14 nodes, 44 links)"] = pd.read_csv(nsfnet_csv)
if geant_csv.exists():
    dfs["GÉANT (22 nodes, 72 links)"] = pd.read_csv(geant_csv)
if abilene_csv.exists():
    dfs["Abilene (12 nodes, 30 links)"] = pd.read_csv(abilene_csv)

scenarios = ["T1", "T2", "T3", "T4", "T5"]
scenario_labels = {
    "T1": "T1 (Normal)",
    "T2": "T2 (Spike)",
    "T3": "T3 (Gradual)",
    "T4": "T4 (Flash Crowd)",
    "T5": "T5 (Link Failure)",
}

metrics_to_plot = [
    ("MLU (Mean)", "Maximum Link Utilization (MLU)", "cross_topology_mlu_comparison.png", False),
    ("Delay (ms)", "End-to-End Delay (ms)", "cross_topology_delay_comparison.png", False),
    ("Loss Rate", "Packet Loss Ratio", "cross_topology_loss_comparison.png", True),
]

for metric_col, ylabel, filename, is_pct in metrics_to_plot:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    
    for ax_idx, (topo_name, df) in enumerate(dfs.items()):
        ax = axes[ax_idx]
        x = np.arange(len(scenarios))
        width = 0.35
        
        ospf_vals = []
        prop_vals = []
        for s in scenarios:
            sub_ospf = df[(df["Scenario"] == s) & (df["Baseline"] == "B1")]
            sub_prop = df[(df["Scenario"] == s) & (df["Baseline"] == "Proposed")]
            
            val_o = sub_ospf[metric_col].values[0] if len(sub_ospf) > 0 else 0.0
            val_p = sub_prop[metric_col].values[0] if len(sub_prop) > 0 else 0.0
            
            if is_pct:
                val_o *= 100.0
                val_p *= 100.0
                
            ospf_vals.append(val_o)
            prop_vals.append(val_p)
            
        ax.bar(x - width/2, ospf_vals, width, label="OSPF-SPF (B1)", color="#DD8452", alpha=0.85)
        ax.bar(x + width/2, prop_vals, width, label="Proposed (FL+RL)", color="#4C72B0", alpha=0.85)
        
        ax.set_title(topo_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Scenario", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel if not is_pct else f"{ylabel} (%)", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([scenario_labels[s] for s in scenarios], fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        
    plt.suptitle(f"Cross-Topology Performance Comparison: {ylabel}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_file = OUT_DIR / filename
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

print("Cross-topology comparative charts generated successfully!")
