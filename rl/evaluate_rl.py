import os
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy.stats import wilcoxon

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

from rl.env import NSFNETRoutingEnv

def apply_scenario(env: NSFNETRoutingEnv, tm_data: np.ndarray, scenario: str, step: int) -> np.ndarray:
    """
    Apply scenario modifications to the environment or traffic matrix dynamically per step.
    T1: Normal Traffic - no changes.
    T2: Traffic Spike - at t=30, triple demand for top 3 OD pairs for 10 steps.
    T3: Gradual Increase - linear 50% increase over 100 steps.
    T4: Flash Crowd - at t=50, 5x demand on top 5 OD pairs for 5 steps.
    T5: Link Failure - at t=40, disable link (4,5) (capacity -> near zero).
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
                
    elif scenario == "T5":
        # Link failure at t=40
        if step == 40:
            if hasattr(env, "disable_link"):
                env.disable_link(4, 5)
            elif env.topology.has_edge(4, 5):
                env.topology[4][5]["capacity"] = 1e-9
                env.topology[5][4]["capacity"] = 1e-9

    return current_tm


def calculate_jains_fairness(utilizations: np.ndarray) -> float:
    """Calculate Jain's Fairness Index for link utilizations."""
    if np.sum(utilizations) == 0:
        return 1.0
    return float((np.sum(utilizations) ** 2) / (len(utilizations) * np.sum(utilizations ** 2)))

def evaluate_baseline(env: NSFNETRoutingEnv, model: Optional[Any], baseline_type: str, 
                      scenario: str, original_tm_data: np.ndarray, num_steps: int = 100) -> Dict[str, float]:
    """
    Evaluates a specific baseline and scenario combination over a single episode.
    """
    obs, _ = env.reset()
    
    metrics: Dict[str, List[float]] = {
        "mlu": [],
        "avg_util": [],
        "jain": [],
        "delay": [],
        "loss": [],
        "throughput": [],
        "latency": []
    }
    
    # Store original capacities to restore after T5 link failure
    original_capacities = {}
    if scenario == "T5":
        if env.topology.has_edge(4, 5):
            original_capacities[(4, 5)] = env.topology[4][5]["capacity"]
            original_capacities[(5, 4)] = env.topology[5][4]["capacity"]
    
    for step in range(num_steps):
        # Apply scenario modifications for this step and update env's dataset
        env.tm_data[env.time_index] = apply_scenario(env, original_tm_data, scenario, step)
        
        # Determine Action and time it (Decision Latency)
        t_start = time.perf_counter()
        
        if baseline_type == "B1":
            # OSPF-SPF: All traffic on shortest path (action = all zeros)
            action = np.zeros(env.num_managed, dtype=np.int32)
            
        elif baseline_type == "B2":
            # ECMP: Random among equal-cost paths (proxy: random action)
            action = env.action_space.sample()
            
        elif baseline_type == "B3":
            # RL-Only (Reactive): Zero out predicted utils (indices 42 to 83)
            modified_obs = obs.copy()
            modified_obs[42:84] = 0.0
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
        obs, reward, done, trunc, info = env.step(action)
        
        # Calculate custom metrics
        link_utils = env.link_utils
        avg_util = float(np.mean(link_utils))
        jain = calculate_jains_fairness(link_utils)
        
        # Throughput calculation
        current_tm = env.tm_data[env.time_index - 1] # previous is the current one we acted on
        total_demand = 0.0
        goodput = 0.0
        
        for u, v in env.managed_od_pairs:
            demand = current_tm[u, v]
            total_demand += demand
            # Approx goodput via path average loss
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
            env.enable_link(4, 5)
        for (u, v), cap in original_capacities.items():
            env.topology[u][v]["capacity"] = cap
            
    # Return averages for the episode
    return {k: float(np.mean(v)) for k, v in metrics.items()}

def run_evaluations(args: argparse.Namespace):
    """Main evaluation orchestrator."""
    topo_path = "data/nsfnet_topology.json"
    tm_path = "data/generated/traffic_matrices.npy"
    fl_model_path = "models/fl_global_model.pt"
    rl_model_path = "models/ppo_agent.zip"
    
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists(tm_path):
        print("Mocking TM data for evaluation test...")
        original_tm_data = np.random.rand(8064, 14, 14) * 0.5
    else:
        original_tm_data = np.load(tm_path)
        
    env = NSFNETRoutingEnv(topo_path=topo_path, tm_path=tm_path, fl_model_path=fl_model_path)
    
    model = None
    if PPO is not None and os.path.exists(rl_model_path):
        try:
            model = PPO.load(rl_model_path)
            print(f"Loaded RL model from {rl_model_path}")
        except Exception as e:
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
                
                res = evaluate_baseline(env, model, baseline, scenario, original_tm_data, num_steps=100)
                seed_metrics.append(res)
                
            # Compute mean and std across seeds
            agg_res = {}
            for k in seed_metrics[0].keys():
                vals = [m[k] for m in seed_metrics]
                agg_res[f"{k}_mean"] = np.mean(vals)
                agg_res[f"{k}_std"] = np.std(vals)
                
            raw_data[scenario][baseline] = seed_metrics
            
            results.append({
                "Scenario": scenario,
                "Baseline": baseline,
                "MLU (Mean)": f"{agg_res['mlu_mean']:.3f} ± {agg_res['mlu_std']:.3f}",
                "Avg Util (%)": f"{agg_res['avg_util_mean']*100:.2f} ± {agg_res['avg_util_std']*100:.2f}",
                "Jain's": f"{agg_res['jain_mean']:.3f} ± {agg_res['jain_std']:.3f}",
                "Delay (ms)": f"{agg_res['delay_mean']:.1f} ± {agg_res['delay_std']:.1f}",
                "Loss Rate": f"{agg_res['loss_mean']:.4f} ± {agg_res['loss_std']:.4f}",
                "Throughput (%)": f"{agg_res['throughput_mean']:.2f} ± {agg_res['throughput_std']:.2f}",
                "Latency (ms)": f"{agg_res['latency_mean']:.2f} ± {agg_res['latency_std']:.2f}",
            })
            
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
                stat, p_val = wilcoxon(prop_mlu, base_mlu)
                sig = "*" if p_val < 0.05 else ""
            except Exception:
                # If differences are exactly zero or too small a sample
                p_val = 1.0
                sig = ""
                
            stats_results.append({
                "Scenario": scenario,
                "Comparison": f"Proposed vs {baseline}",
                "P-Value": p_val,
                "Significant": sig
            })
            
    # Formatting outputs
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("EVALUATION RESULTS (Mean ± Std)")
    print("="*80)
    print(df.to_string(index=False))
    
    df.to_csv("results/benchmark_results.csv", index=False)
    
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
    df_summary.to_csv("results/summary_table.csv", index=False)
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test on MLU)")
    print("="*80)
    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        print(df_stats.to_string(index=False))
    else:
        print("Proposed baseline not evaluated. Skipping stats.")
        
    print("\nResults saved to:")
    print("  - results/benchmark_results.csv")
    print("  - results/summary_table.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FL+RL Traffic Engineering")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="List of random seeds to evaluate")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["T1", "T2", "T3", "T4", "T5"],
                        help="List of test scenarios (T1-T5)")
    parser.add_argument("--baselines", type=str, nargs="+", default=["B1", "B2", "B3", "B4", "Proposed"],
                        help="List of baselines to evaluate (B1, B2, B3, B4, Proposed)")
    
    args = parser.parse_args()
    run_evaluations(args)
