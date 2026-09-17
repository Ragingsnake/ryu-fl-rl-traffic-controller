"""
Demo script for FL+RL Traffic Engineering prototype.
Compares Trained FL+RL Agent against Untrained Baselines (OSPF Shortest-Path & Random Agent)
across ALL test scenarios (T1 Normal, T2 Spike, T3 Gradual, T4 Flash Crowd, T5 Link Failure)
over multiple random seeds/tries with statistical reporting (mean ± std).

Produces rich multi-metric comparative plots in results/demo/:
    - overview_all_scenarios_mlu.png: 5-panel MLU timeline
    - overview_all_scenarios_delay.png: 5-panel End-to-End Delay timeline
    - overview_all_scenarios_loss.png: 5-panel Packet Loss timeline
    - overview_all_scenarios_reward.png: 5-panel Step Reward timeline
    - scenario_t4_flash_crowd.png: 4-subplot deep dive into T4 (MLU, Loss, Delay, Reward)
    - scenario_t2_traffic_spike.png: 4-subplot deep dive into T2 (MLU, Loss, Delay, Reward)
    - summary_metrics_barchart.png: Grouped bar charts with error bars across all scenarios
"""

import argparse
import os
import sys
from pathlib import Path

# Add prototype root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def run_episode(env, policy_fn, start_idx: int, seed: int, scenario: str = "T1", num_steps: int = 100):
    """Run an episode using the given policy function under a specific scenario and seed."""
    env.reset(seed=seed)
    env.time_index = start_idx
    obs = env._get_observation()

    # Create scenario-specific copy of the TM slice
    original_slice = env.tm_data[start_idx : start_idx + num_steps].copy()
    scenario_slice = original_slice.copy()

    if scenario == "T2":
        for t in range(30, 40):
            for k in range(min(3, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                scenario_slice[t, u, v] *= 3.0
    elif scenario == "T3":
        for t in range(num_steps):
            scenario_slice[t] *= 1.0 + (0.5 * (t / float(num_steps)))
    elif scenario == "T4":
        for t in range(50, 55):
            for k in range(min(5, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                scenario_slice[t, u, v] *= 5.0

    env.tm_data[start_idx : start_idx + num_steps] = scenario_slice

    rewards = []
    mlus = []
    mean_utils = []
    avg_delays = []
    avg_losses = []
    churns = []

    for step in range(num_steps):
        # Apply link failure for T5 at step 40 (fail primary backbone link 8-12)
        if scenario == "T5" and step == 40:
            env.disable_link(8, 12)

        action = policy_fn(obs, env)
        obs, reward, done, _truncated, info = env.step(action)

        rewards.append(reward)
        mlus.append(info["mlu"])
        mean_utils.append(float(np.mean(env.link_utils)))
        avg_delays.append(info["avg_delay"])
        avg_losses.append(info["avg_loss"] * 100.0)  # Convert to percentage
        churns.append(info.get("churn", 0.0) * 100.0)  # Convert to percentage

        if done:
            break

    # Cleanup link failure if modified
    if scenario == "T5":
        env.enable_link(8, 12)

    # Restore original traffic matrix
    env.tm_data[start_idx : start_idx + num_steps] = original_slice

    return {
        "timesteps": np.arange(len(rewards)),
        "rewards": np.array(rewards),
        "mlus": np.array(mlus),
        "mean_utils": np.array(mean_utils),
        "avg_delays": np.array(avg_delays),
        "avg_losses": np.array(avg_losses),
        "churns": np.array(churns),
    }


def run_demo():
    parser = argparse.ArgumentParser(description="Multi-Scenario & Multi-Seed FL+RL Evaluation Demo")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds for testing")
    parser.add_argument("--steps", type=int, default=100, help="Timesteps per episode")
    parser.add_argument("--out_dir", type=str, default="results/demo", help="Directory to save generated plots")
    parser.add_argument("--no_random", action="store_true", help="Skip the Untrained (Random Policy) baseline")
    args = parser.parse_args()

    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    topo_path = "data/nsfnet_topology.json"
    tm_path = "data/generated/traffic_matrices.npy"
    fl_model_path = "models/fl_global_model.pt"
    rl_model_path = "models/ppo_agent.zip"

    if not (os.path.exists(topo_path) and os.path.exists(tm_path)):
        print("Data files not found. Please run: python scripts/run_full_pipeline.py --quick")
        return

    from rl.env import NSFNETRoutingEnv

    env = NSFNETRoutingEnv(
        topo_path=topo_path,
        tm_path=tm_path,
        fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
    )

    if not os.path.exists(rl_model_path):
        raise FileNotFoundError(
            f"Trained RL agent model file not found at '{rl_model_path}'. "
            "Please ensure models/ppo_agent.zip exists in the repository before running the evaluation demo."
        )

    try:
        from stable_baselines3 import PPO

        trained_agent = PPO.load(rl_model_path)
        print(f"Loaded trained RL agent from {rl_model_path}.")
    except (RuntimeError, ValueError, OSError) as e:
        raise RuntimeError(f"Failed to load RL agent from {rl_model_path}: {e}") from e

    # Policies
    def policy_ospf(_obs, env):
        return np.zeros(env.num_managed, dtype=np.int32)

    def policy_random(_obs, env):
        return env.action_space.sample()

    def policy_trained(obs, _env):
        action, _ = trained_agent.predict(obs, deterministic=True)
        return action

    policies = {
        "OSPF": policy_ospf,
    }
    if not args.no_random:
        policies["Random"] = policy_random
    policies["Trained"] = policy_trained

    active_policy_names = list(policies.keys())

    scenarios = ["T1", "T2", "T3", "T4", "T5"]
    scenario_titles = {
        "T1": "T1: Normal Traffic",
        "T2": "T2: Traffic Spike (3x at t=30..40)",
        "T3": "T3: Gradual Load (+50% over 100 steps)",
        "T4": "T4: Flash Crowd (5x at t=50..55)",
        "T5": "T5: Link Failure (Backbone Link 8-12 down at t=40)",
    }

    print(f"\nRunning benchmark across {len(scenarios)} scenarios and {len(args.seeds)} random seeds: {args.seeds}...")

    # Data structure: all_runs[scenario][policy] = list of episode result dicts (one per seed)
    all_runs = {sc: {p: [] for p in policies} for sc in scenarios}

    for s_idx, seed in enumerate(args.seeds):
        # Pick consistent starting index for this seed:
        # Start at the beginning of daytime (t=0) on Day (seed % 20 + 1)
        # Curves ramp from base traffic (1.0x) -> daily peak (1.30x at step 72) -> starts dying down (1.25x at step 100)
        day_idx = (seed * 3) % 20 + 1
        start_idx = day_idx * 288 + 0

        for sc in scenarios:
            for p_name, p_fn in policies.items():
                run_data = run_episode(env, p_fn, start_idx, seed=seed, scenario=sc, num_steps=args.steps)
                all_runs[sc][p_name].append(run_data)

    # --- Print Statistical Summary Table (Mean ± Std) ---
    print("\n" + "=" * 122)
    print(
        f"{'SCENARIO':<6} | {'POLICY':<20} | {'MEAN MLU':<14} | {'MAX MLU':<14} | {'DELAY (ms)':<14} | {'LOSS (%)':<14} | {'CHURN (%)':<14} | {'REWARD':<14}"
    )
    print("=" * 122)

    summary_stats = {sc: {p: {} for p in policies} for sc in scenarios}

    for sc in scenarios:
        for p_name in active_policy_names:
            seed_runs = all_runs[sc][p_name]

            # Aggregate across seeds
            mean_mlus = [np.mean(r["mlus"]) for r in seed_runs]
            max_mlus = [np.max(r["mlus"]) for r in seed_runs]
            delays = [np.mean(r["avg_delays"]) for r in seed_runs]
            losses = [np.mean(r["avg_losses"]) for r in seed_runs]
            churns = [np.mean(r["churns"]) for r in seed_runs]
            rewards = [np.mean(r["rewards"]) for r in seed_runs]

            summary_stats[sc][p_name] = {
                "mlu_mean": np.mean(mean_mlus),
                "mlu_std": np.std(mean_mlus),
                "max_mlu_mean": np.mean(max_mlus),
                "max_mlu_std": np.std(max_mlus),
                "delay_mean": np.mean(delays),
                "delay_std": np.std(delays),
                "loss_mean": np.mean(losses),
                "loss_std": np.std(losses),
                "churn_mean": np.mean(churns),
                "churn_std": np.std(churns),
                "reward_mean": np.mean(rewards),
                "reward_std": np.std(rewards),
            }

            label = (
                "OSPF (Shortest Path)"
                if p_name == "OSPF"
                else ("Untrained (Random)" if p_name == "Random" else "Trained FL+RL Agent")
            )

            st = summary_stats[sc][p_name]
            print(
                f"{sc:<6} | {label:<20} | "
                f"{st['mlu_mean']:.3f} ± {st['mlu_std']:.3f} | "
                f"{st['max_mlu_mean']:.3f} ± {st['max_mlu_std']:.3f} | "
                f"{st['delay_mean']:5.2f} ± {st['delay_std']:4.2f} | "
                f"{st['loss_mean']:5.2f} ± {st['loss_std']:4.2f} | "
                f"{st['churn_mean']:5.2f} ± {st['churn_std']:4.2f} | "
                f"{st['reward_mean']:6.2f} ± {st['reward_std']:4.2f}"
            )
        print("-" * 122)

    # --- Plot Generation ---
    print(f"\nGenerating rich comparative plots in {results_dir.resolve()}...")

    colors = {"OSPF": "#1f77b4", "Random": "#ff7f0e", "Trained": "#2ca02c"}
    styles = {"OSPF": "-", "Random": "--", "Trained": "-"}
    widths = {"OSPF": 1.8, "Random": 1.2, "Trained": 2.2}
    alphas = {"OSPF": 0.9, "Random": 0.6, "Trained": 1.0}

    # 1. Five-Panel Timelines for All Metrics (Averaged across seeds with std shading)
    metric_configs = [
        ("mlus", "Maximum Link Utilization (MLU)", "overview_all_scenarios_mlu.png", True),
        ("avg_delays", "End-to-End Delay (ms)", "overview_all_scenarios_delay.png", False),
        ("avg_losses", "Packet Loss Ratio (%)", "overview_all_scenarios_loss.png", False),
        ("rewards", "Per-Step Reward", "overview_all_scenarios_reward.png", False),
    ]

    for metric_key, ylabel, fname, has_threshold in metric_configs:
        _fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        for idx, sc in enumerate(scenarios):
            ax = axes[idx]
            timesteps = all_runs[sc]["OSPF"][0]["timesteps"]

            for p_name in active_policy_names:
                # Stack all seeds for this metric
                matrix = np.array([r[metric_key] for r in all_runs[sc][p_name]])  # (n_seeds, num_steps)
                mean_curve = np.mean(matrix, axis=0)
                std_curve = np.std(matrix, axis=0)

                label_p = (
                    "OSPF-SPF" if p_name == "OSPF" else ("Random Policy" if p_name == "Random" else "Trained FL+RL")
                )
                ax.plot(
                    timesteps,
                    mean_curve,
                    label=label_p,
                    color=colors[p_name],
                    linestyle=styles[p_name],
                    linewidth=widths[p_name],
                    alpha=alphas[p_name],
                )
                ax.fill_between(
                    timesteps,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=colors[p_name],
                    alpha=0.15,
                )

            if has_threshold:
                ax.axhline(
                    0.90, color="red", linestyle=":", alpha=0.7, label="Congestion Threshold (0.90)" if idx == 0 else ""
                )

            ax.set_title(scenario_titles[sc], fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks([0, 24, 48, 72, 96, 100])
            ax.set_xticklabels(["08:00", "10:00", "12:00", "14:00 (Peak)", "16:00", "16:20"], fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.9)

        axes[-1].set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-minute intervals)", fontsize=11)
        plt.tight_layout()
        plt.savefig(results_dir / fname, dpi=200)
        plt.close()

    # 2. Comprehensive 4-Subplot Deep Dive into Scenario T4 (Flash Crowd)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t4_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T4"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t4_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T4"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = "OSPF-SPF" if p_name == "OSPF" else ("Random Policy" if p_name == "Random" else "Trained FL+RL")
            ax.plot(
                timesteps, mean_c, label=p_lbl, color=colors[p_name], linewidth=widths[p_name], linestyle=styles[p_name]
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors[p_name], alpha=0.15)

        ax.axvspan(50, 55, color="purple", alpha=0.18, label="Flash Crowd Surge (5x)")
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T4 Flash Crowd: {m_label}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 24, 48, 72, 96, 100])
        ax.set_xticklabels(["08:00", "10:00", "12:00", "14:00 (Peak)", "16:00", "16:20"], fontsize=8)
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T4 (Flash Crowd Surge) — Comprehensive Multi-Metric Comparison", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t4_flash_crowd.png", dpi=200)
    plt.close()

    # 3. Comprehensive 4-Subplot Deep Dive into Scenario T2 (Traffic Spike)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t2_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T2"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t2_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T2"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = "OSPF-SPF" if p_name == "OSPF" else ("Random Policy" if p_name == "Random" else "Trained FL+RL")
            ax.plot(
                timesteps, mean_c, label=p_lbl, color=colors[p_name], linewidth=widths[p_name], linestyle=styles[p_name]
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors[p_name], alpha=0.15)

        ax.axvspan(30, 40, color="purple", alpha=0.18, label="Spike Surge (3x)")
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T2 Traffic Spike: {m_label}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 24, 48, 72, 96, 100])
        ax.set_xticklabels(["08:00", "10:00", "12:00", "14:00 (Peak)", "16:00", "16:20"], fontsize=8)
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T2 (Traffic Spike 3x) — Comprehensive Multi-Metric Comparison", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t2_traffic_spike.png", dpi=200)
    plt.close()

    # 4. Comprehensive 4-Subplot Deep Dive into Scenario T5 (Link Failure)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t5_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T5"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t5_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T5"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = "OSPF-SPF" if p_name == "OSPF" else ("Random Policy" if p_name == "Random" else "Trained FL+RL")
            ax.plot(
                timesteps, mean_c, label=p_lbl, color=colors[p_name], linewidth=widths[p_name], linestyle=styles[p_name]
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors[p_name], alpha=0.15)

        ax.axvline(40, color="darkred", linestyle="--", linewidth=1.8, label="Link (8, 12) Fails at t=40")
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T5 Link Failure: {m_label}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 24, 48, 72, 96, 100])
        ax.set_xticklabels(["08:00", "10:00", "12:00", "14:00 (Peak)", "16:00", "16:20"], fontsize=8)
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T5 (Backbone Link Failure) — Comprehensive Multi-Metric Comparison", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t5_link_failure.png", dpi=200)
    plt.close()

    # 5. Summary Grouped Bar Chart Across Scenarios (with Error Bars)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_metrics = [
        ("mlu_mean", "mlu_std", "Mean MLU (Lower is better)", axes[0, 0]),
        ("delay_mean", "delay_std", "Mean Delay (ms) (Lower is better)", axes[0, 1]),
        ("loss_mean", "loss_std", "Packet Loss (%) (Lower is better)", axes[1, 0]),
        ("reward_mean", "reward_std", "Mean Reward (Higher is better)", axes[1, 1]),
    ]

    x = np.arange(len(scenarios))
    num_p = len(active_policy_names)
    bar_width = 0.35 if num_p == 2 else 0.26

    for m_mean, m_std, title, ax in bar_metrics:
        for i, p_name in enumerate(active_policy_names):
            vals = [summary_stats[sc][p_name][m_mean] for sc in scenarios]
            errs = [summary_stats[sc][p_name][m_std] for sc in scenarios]

            p_lbl = "OSPF-SPF" if p_name == "OSPF" else ("Random Policy" if p_name == "Random" else "Trained FL+RL")
            offset = (i - (num_p - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                vals,
                bar_width,
                yerr=errs,
                capsize=4,
                label=p_lbl,
                color=colors[p_name],
                alpha=0.85,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.legend(loc="upper right" if "Reward" not in title else "lower right", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Overall Performance Benchmark Across All Scenarios (5-Seed Mean ± Std)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "summary_metrics_barchart.png", dpi=200)
    plt.close()

    print("All multi-metric and multi-seed comparative plots successfully generated and saved!")


if __name__ == "__main__":
    run_demo()
