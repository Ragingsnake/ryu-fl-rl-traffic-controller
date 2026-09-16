"""
Demo script for FL+RL Traffic Engineering prototype.
Runs a single episode using Scenario T2 (Traffic Spike) and visualizes results.

Usage:
    python scripts/demo.py

Produces plots in results/demo/:
    - utilization_time.png: Mean link utilization over time
    - reward_curve.png: Per-step reward
    - mlu_time.png: Maximum Link Utilization over time
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for servers/CI
import matplotlib.pyplot as plt


def run_demo():
    """Run demo with actual env + model if available, else mock data."""
    results_dir = Path("results/demo")
    results_dir.mkdir(parents=True, exist_ok=True)

    topo_path = "data/nsfnet_topology.json"
    tm_path = "data/generated/traffic_matrices.npy"
    fl_model_path = "models/fl_global_model.pt"
    rl_model_path = "models/ppo_agent.zip"

    use_real = os.path.exists(topo_path) and os.path.exists(tm_path)

    if use_real:
        print("Using REAL environment and data...")
        # Add prototype root to path for imports
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from rl.env import NSFNETRoutingEnv

        env = NSFNETRoutingEnv(
            topo_path=topo_path,
            tm_path=tm_path,
            fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
        )

        # Try loading trained RL agent
        agent = None
        if os.path.exists(rl_model_path):
            try:
                from stable_baselines3 import PPO

                agent = PPO.load(rl_model_path)
                print("Loaded trained RL agent.")
            except (RuntimeError, ValueError, OSError) as e:
                print(f"Could not load RL agent: {e}. Using OSPF baseline.")

        # --- Apply T2 scenario: 3x demand on top 3 OD pairs at t=30 for 10 steps ---
        obs, _ = env.reset(seed=42)
        start_idx = env.time_index

        # Modify TM data for T2 in-place (we'll restore after)
        original_slice = env.tm_data[start_idx : start_idx + 100].copy()
        for t in range(30, 40):
            abs_t = start_idx + t
            if abs_t < len(env.tm_data):
                for k in range(min(3, len(env.managed_od_pairs))):
                    u, v = env.managed_od_pairs[k]
                    env.tm_data[abs_t, u, v] *= 3.0

        # Run episode
        rewards = []
        mlus = []
        mean_utils = []
        avg_delays = []
        avg_losses = []

        for step in range(100):
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                # OSPF baseline: always pick path 0 (shortest)
                action = np.zeros(env.num_managed, dtype=np.int32)

            obs, reward, done, _truncated, info = env.step(action)
            rewards.append(reward)
            mlus.append(info["mlu"])
            mean_utils.append(np.mean(env.link_utils))
            avg_delays.append(info["avg_delay"])
            avg_losses.append(info["avg_loss"])

            if done:
                break

        # Restore original TM data
        env.tm_data[start_idx : start_idx + 100] = original_slice

        timesteps = np.arange(len(rewards))
        mean_utilization = np.array(mean_utils)
        mlu = np.array(mlus)
        rewards = np.array(rewards)

        agent_label = "PPO Agent" if agent else "OSPF Baseline"
    else:
        print("Data not found — using MOCK demo data...")
        agent_label = "Mock"
        timesteps = np.arange(100)
        mean_utilization = 0.4 + 0.3 * np.sin(timesteps / 10.0)
        mean_utilization[30:40] += 0.25
        mean_utilization = np.clip(mean_utilization, 0, 1)
        rewards = -mean_utilization + 0.5
        mlu = mean_utilization + 0.15 + np.random.rand(100) * 0.05
        mlu = np.clip(mlu, 0, 1)

    # --- Plots ---
    print("Generating plots...")

    # 1. Link Utilization over time
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, mean_utilization, label=f"Mean Link Util ({agent_label})", color="blue")
    plt.axvspan(30, 40, color="red", alpha=0.2, label="Traffic Spike (T2)")
    plt.title("Network Utilization over Time (Scenario T2)")
    plt.xlabel("Timestep")
    plt.ylabel("Utilization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "utilization_time.png", dpi=150)
    plt.close()

    # 2. Reward Curve
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, rewards, label="Reward", color="green")
    plt.title("RL Agent Reward per Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "reward_curve.png", dpi=150)
    plt.close()

    # 3. MLU
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, mlu, label="MLU", color="red")
    plt.axhline(0.9, color="black", linestyle="--", label="Congestion Threshold")
    plt.axvspan(30, 40, color="orange", alpha=0.15, label="Spike Window")
    plt.title("Maximum Link Utilization (MLU)")
    plt.xlabel("Timestep")
    plt.ylabel("MLU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "mlu_time.png", dpi=150)
    plt.close()

    print(f"\nDemo complete! Plots saved to {results_dir}/")
    print(f"\nSummary Metrics ({agent_label}):")
    print(f"  Average MLU:          {np.mean(mlu):.3f}")
    print(f"  Max MLU (spike):      {np.max(mlu[30 : min(40, len(mlu))]):.3f}")
    print(f"  Average Reward:       {np.mean(rewards):.3f}")
    print(f"  Mean Utilization:     {np.mean(mean_utilization):.3f}")


if __name__ == "__main__":
    run_demo()
