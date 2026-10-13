import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

from rl.env import NetworkRoutingEnv, NSFNETRoutingEnv


def get_default_failure_link(env: NSFNETRoutingEnv) -> tuple[int, int]:
    """Finds a valid backbone link in the topology to disable for scenario T5."""
    candidates = [(8, 12), (4, 5), (0, 4), (1, 4), (0, 1), (0, 2)]
    for u, v in candidates:
        if env.topology.has_edge(u, v):
            return (u, v)
    edges = list(env.topology.edges())
    return edges[0] if edges else (0, 1)


def apply_scenario(env: NSFNETRoutingEnv, tm_data: np.ndarray, scenario: str, step: int) -> np.ndarray:
    """
    Apply scenario modifications to the environment or traffic matrix dynamically per step.
    T1: Normal Traffic - no changes.
    T2: Traffic Spike - at t=30, triple demand for top 3 OD pairs for 10 steps.
    T3: Gradual Increase - linear 50% increase over 100 steps.
    T4: Flash Crowd - at t=50, 5x demand on top 5 OD pairs for 5 steps.
    T5: Link Failure - at t=40, disable primary link.
    """
    current_tm = tm_data[step].copy()

    if scenario == "T1":
        pass

    elif scenario == "T2":
        if 30 <= step < 40:
            for k in range(min(3, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                current_tm[u, v] *= 3.0

    elif scenario == "T3":
        multiplier = 1.0 + (0.5 * (step / 100.0))
        current_tm *= multiplier

    elif scenario == "T4":
        if 50 <= step < 55:
            for k in range(min(5, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                current_tm[u, v] *= 5.0

    elif scenario == "T5" and step == 40:
        fail_u, fail_v = get_default_failure_link(env)
        if hasattr(env, "disable_link"):
            env.disable_link(fail_u, fail_v)
        elif env.topology.has_edge(fail_u, fail_v):
            env.topology[fail_u][fail_v]["capacity"] = 1e-9
            env.topology[fail_v][fail_u]["capacity"] = 1e-9

    return current_tm


def calculate_jains_fairness(utilizations: np.ndarray) -> float:
    """Calculate Jain's Fairness Index for link utilizations."""
    if np.sum(utilizations) == 0:
        return 1.0
    return float((np.sum(utilizations) ** 2) / (len(utilizations) * np.sum(utilizations**2)))


def evaluate_baseline(
    env: NSFNETRoutingEnv,
    model: Any | None,
    baseline_type: str,
    scenario: str,
    start_idx: int,
    num_steps: int = 100,
) -> dict[str, float]:
    """
    Evaluates a specific baseline and scenario combination over a single episode.
    """
    env.time_index = start_idx
    obs = env._get_observation()

    original_slice = env.tm_data[start_idx : start_idx + num_steps].copy()
    scenario_slice = original_slice.copy()

    if scenario == "T2":
        for t in range(30, min(40, num_steps)):
            for k in range(min(3, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                scenario_slice[t, u, v] *= 3.0
    elif scenario == "T3":
        for t in range(num_steps):
            scenario_slice[t] *= 1.0 + (0.5 * (t / float(num_steps)))
    elif scenario == "T4":
        for t in range(50, min(55, num_steps)):
            for k in range(min(5, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                scenario_slice[t, u, v] *= 5.0

    env.tm_data[start_idx : start_idx + num_steps] = scenario_slice

    metrics: dict[str, list[float]] = {
        "mlu": [],
        "avg_util": [],
        "jain": [],
        "delay": [],
        "loss": [],
        "throughput": [],
        "latency": [],
    }

    fail_u, fail_v = get_default_failure_link(env)
    original_capacities = {}
    if scenario == "T5" and env.topology.has_edge(fail_u, fail_v):
        original_capacities[(fail_u, fail_v)] = env.topology[fail_u][fail_v]["capacity"]
        original_capacities[(fail_v, fail_u)] = env.topology[fail_v][fail_u]["capacity"]

    for step in range(num_steps):
        # Apply link failure for T5 at step 40
        if scenario == "T5" and step == 40:
            if hasattr(env, "disable_link"):
                env.disable_link(fail_u, fail_v)
            elif env.topology.has_edge(fail_u, fail_v):
                env.topology[fail_u][fail_v]["capacity"] = 1e-9
                env.topology[fail_v][fail_u]["capacity"] = 1e-9

        # Determine Action and time it (Decision Latency)
        t_start = time.perf_counter()

        if baseline_type == "B1":
            # OSPF-SPF: All traffic on shortest path (action = all zeros)
            action = np.zeros(env.num_managed, dtype=np.int32)

        elif baseline_type == "B2":
            # ECMP: Random among equal-cost paths (proxy: random action)
            action = env.action_space.sample()

        elif baseline_type == "B3":
            # RL-Only (Reactive): Zero out predicted utils
            modified_obs = obs.copy()
            modified_obs[env.num_edges : 2 * env.num_edges] = 0.0
            if model:
                action, _ = model.predict(modified_obs, deterministic=True)
            else:
                action = np.zeros(env.num_managed, dtype=np.int32)

        elif baseline_type in ["B4", "Proposed"]:
            # B4 (Centralized proxy) and Proposed (FL+RL) use the full observation
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.num_managed, dtype=np.int32)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Step env
        obs, _reward, done, _trunc, info = env.step(action)

        # Calculate custom metrics
        link_utils = env.link_utils
        avg_util = float(np.mean(link_utils))
        jain = calculate_jains_fairness(link_utils)

        # Throughput calculation
        current_tm = env.tm_data[env.time_index - 1]
        total_demand = 0.0
        goodput = 0.0

        for u, v in env.managed_od_pairs:
            demand = current_tm[u, v]
            total_demand += demand
            goodput += demand * (1.0 - info["avg_loss"])

        throughput = (goodput / total_demand) * 100.0 if total_demand > 0 else 100.0

        # Record metrics
        metrics["mlu"].append(info["mlu"])
        metrics["avg_util"].append(avg_util)
        metrics["jain"].append(jain)
        metrics["delay"].append(info["avg_delay"])
        metrics["loss"].append(info["avg_loss"])
        metrics["throughput"].append(throughput)
        metrics["latency"].append(latency_ms)

        if done:
            break

    # Restore capacities if modified
    if scenario == "T5":
        if hasattr(env, "enable_link"):
            env.enable_link(fail_u, fail_v)
        for (u, v), cap in original_capacities.items():
            env.topology[u][v]["capacity"] = cap

    # Restore original traffic matrix slice
    env.tm_data[start_idx : start_idx + num_steps] = original_slice

    # Return averages for the episode
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def run_evaluations(args: argparse.Namespace):
    """Main evaluation orchestrator."""
    topo = args.topology.lower()
    os.makedirs(args.out_dir, exist_ok=True)

    if topo == "nsfnet":
        topo_path = "data/nsfnet_topology.json"
        tm_path = "data/generated/traffic_matrices.npy"
        domain_data_dir = "data/preprocessed"
        default_num_managed = 20
        default_fl = "models/fl_global_model.pt"
        default_rl = "models/ppo_agent.zip"
    elif topo == "geant":
        topo_path = "data/geant_topology.json"
        tm_path = "data/generated/geant_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_geant"
        default_num_managed = 30
        default_fl = "models/fl_global_model_geant_fedadam.pt"
        default_rl = "models/ppo_agent_geant.zip"
    elif topo == "abilene":
        topo_path = "data/abilene_topology.json"
        tm_path = "data/generated/abilene_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_abilene"
        default_num_managed = 15
        default_fl = "models/fl_global_model_abilene_fedadam.pt"
        default_rl = "models/ppo_agent_abilene.zip"
    else:
        raise ValueError(f"Unsupported topology: {topo}")

    num_managed = args.num_managed_od if args.num_managed_od is not None else default_num_managed
    fl_model_path = args.fl_model_path if args.fl_model_path is not None else default_fl
    rl_model_path = args.rl_model_path if args.rl_model_path is not None else default_rl

    print(f"============================================================")
    print(f"Evaluating Baselines on Topology: {topo.upper()}")
    print(f"  Topology file:    {topo_path}")
    print(f"  TM file:          {tm_path}")
    print(f"  FL Model:         {fl_model_path} (exists: {os.path.exists(fl_model_path)})")
    print(f"  RL Model:         {rl_model_path} (exists: {os.path.exists(rl_model_path)})")
    print(f"  Domain Data Dir:  {domain_data_dir}")
    print(f"  Managed OD pairs: {num_managed}")
    print(f"============================================================")

    if not os.path.exists(tm_path):
        print("Mocking TM data for evaluation test...")
        original_tm_data = np.random.rand(8064, 14, 14) * 0.5
    else:
        original_tm_data = np.load(tm_path)

    env = NetworkRoutingEnv(
        topo_path=topo_path,
        tm_path=tm_path,
        fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
        domain_data_dir=domain_data_dir,
        num_managed_od=num_managed,
    )

    model = None
    if PPO is not None and os.path.exists(rl_model_path):
        try:
            model = PPO.load(rl_model_path)
            print(f"Loaded RL model from {rl_model_path}")
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Failed to load RL model: {e}")
    else:
        print("Warning: RL model or stable-baselines3 not found. RL baselines will output zeros.")

    scenarios = args.scenarios
    baselines = args.baselines
    seeds = args.seeds

    results = []
    raw_data = {s: {b: [] for b in baselines} for s in scenarios}

    print("\nStarting Evaluation...")
    for scenario in scenarios:
        for baseline in baselines:
            print(f"Running Scenario: {scenario}, Baseline: {baseline}")
            seed_metrics = []
            for seed in seeds:
                np.random.seed(seed)
                env.reset(seed=seed)
                day_idx = (seed * 3) % 20 + 1
                start_idx = day_idx * 288
                res = evaluate_baseline(env, model, baseline, scenario, start_idx=start_idx, num_steps=100)
                seed_metrics.append(res)

            # Compute mean and std across seeds
            agg_res = {}
            for k in seed_metrics[0]:
                vals = [m[k] for m in seed_metrics]
                agg_res[f"{k}_mean"] = np.mean(vals)
                agg_res[f"{k}_std"] = np.std(vals)

            raw_data[scenario][baseline] = seed_metrics

            results.append(
                {
                    "Scenario": scenario,
                    "Baseline": baseline,
                    "MLU (Mean)": f"{agg_res['mlu_mean']:.3f} ± {agg_res['mlu_std']:.3f}",
                    "Avg Util (%)": f"{agg_res['avg_util_mean'] * 100:.2f} ± {agg_res['avg_util_std'] * 100:.2f}",
                    "Jain's": f"{agg_res['jain_mean']:.3f} ± {agg_res['jain_std']:.3f}",
                    "Delay (ms)": f"{agg_res['delay_mean']:.1f} ± {agg_res['delay_std']:.1f}",
                    "Loss Rate": f"{agg_res['loss_mean']:.4f} ± {agg_res['loss_std']:.4f}",
                    "Throughput (%)": f"{agg_res['throughput_mean']:.2f} ± {agg_res['throughput_std']:.2f}",
                    "Latency (ms)": f"{agg_res['latency_mean']:.2f} ± {agg_res['latency_std']:.2f}",
                }
            )

    # Statistical tests (Proposed vs Baselines)
    stats_results = []
    for scenario in scenarios:
        if "Proposed" not in baselines:
            continue

        prop_mlu = [m["mlu"] for m in raw_data[scenario]["Proposed"]]

        for baseline in baselines:
            if baseline == "Proposed":
                continue
            base_mlu = [m["mlu"] for m in raw_data[scenario][baseline]]

            try:
                # Two-sided wilcoxon signed-rank test
                _stat, p_val = wilcoxon(prop_mlu, base_mlu)
                sig = "*" if p_val < 0.05 else ""
            except (ValueError, TypeError, FloatingPointError, ZeroDivisionError):
                # If differences are exactly zero or too small a sample
                p_val = 1.0
                sig = ""

            stats_results.append(
                {"Scenario": scenario, "Comparison": f"Proposed vs {baseline}", "P-Value": p_val, "Significant": sig}
            )

    # Formatting outputs
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (Mean ± Std)")
    print("=" * 80)
    print(df.to_string(index=False))

    out_benchmark_csv = os.path.join(args.out_dir, f"benchmark_results_{topo}.csv" if topo != "nsfnet" else "benchmark_results.csv")
    df.to_csv(out_benchmark_csv, index=False)

    # Summary table just with means
    summary_data = []
    for r in results:
        summary_row = {"Scenario": r["Scenario"], "Baseline": r["Baseline"]}
        for k, v in r.items():
            if k not in ["Scenario", "Baseline"]:
                mean_val = float(v.split(" ± ")[0])
                summary_row[k] = mean_val
        summary_data.append(summary_row)

    df_summary = pd.DataFrame(summary_data)
    out_summary_csv = os.path.join(args.out_dir, f"summary_table_{topo}.csv" if topo != "nsfnet" else "summary_table.csv")
    df_summary.to_csv(out_summary_csv, index=False)

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test on MLU)")
    print("=" * 80)
    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        print(df_stats.to_string(index=False))
    else:
        print("Proposed baseline not evaluated. Skipping stats.")

    print("\nResults saved to:")
    print(f"  - {out_benchmark_csv}")
    print(f"  - {out_summary_csv}")

    # Generate charts
    chart_dir = os.path.join(args.out_dir, "demo" if topo == "nsfnet" else f"demo/{topo}")
    plot_evaluation_charts(df_summary, chart_dir)


def plot_evaluation_charts(df_summary: pd.DataFrame, output_dir: str = "results/demo"):
    """Generates comparison bar charts across scenarios and baselines."""
    os.makedirs(output_dir, exist_ok=True)
    scenarios = df_summary["Scenario"].unique()
    baselines = df_summary["Baseline"].unique()

    metrics = [
        ("MLU (Mean)", "Maximum Link Utilization (MLU)", "benchmark_mlu_comparison.png"),
        ("Delay (ms)", "End-to-End Delay (ms)", "benchmark_delay_comparison.png"),
        ("Loss Rate", "Packet Loss Ratio", "benchmark_loss_comparison.png"),
        ("Throughput (%)", "Throughput (%)", "benchmark_throughput_comparison.png"),
    ]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for col_name, ylabel, filename in metrics:
        if col_name not in df_summary.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(scenarios))
        width = 0.8 / len(baselines)

        for i, baseline in enumerate(baselines):
            sub = df_summary[df_summary["Baseline"] == baseline]
            # Match scenarios in order
            vals = [sub[sub["Scenario"] == s][col_name].values[0] if len(sub[sub["Scenario"] == s]) > 0 else 0 for s in scenarios]
            ax.bar(x + i * width, vals, width, label=baseline, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xlabel("Scenario", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(f"Benchmark Comparison: {ylabel}", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * (len(baselines) - 1) / 2)
        ax.set_xticklabels(scenarios, fontsize=10)
        ax.legend(title="Baseline", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"  - Chart saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FL+RL Traffic Engineering")
    parser.add_argument("--topology", type=str, default="nsfnet", choices=["nsfnet", "geant", "abilene"], help="Network topology")
    parser.add_argument("--rl_model_path", type=str, default=None, help="Path to trained RL model")
    parser.add_argument("--fl_model_path", type=str, default=None, help="Path to FL prediction model")
    parser.add_argument("--num_managed_od", type=int, default=None, help="Number of managed OD pairs")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of random seeds to evaluate"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["T1", "T2", "T3", "T4", "T5"],
        help="List of test scenarios (T1-T5)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["B1", "B2", "B3", "B4", "Proposed"],
        help="List of baselines to evaluate (B1, B2, B3, B4, Proposed)",
    )

    args = parser.parse_args()
    run_evaluations(args)
