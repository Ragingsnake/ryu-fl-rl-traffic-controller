from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.callbacks import MetricsCallback
from rl.env import NetworkRoutingEnv


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Traffic Engineering")
    parser.add_argument(
        "--topology", type=str, default="nsfnet", choices=["nsfnet", "geant", "abilene"], help="Network topology"
    )
    parser.add_argument("--topo_path", type=str, default=None, help="Custom path to topology JSON")
    parser.add_argument("--tm_path", type=str, default=None, help="Custom path to traffic matrices .npy")
    parser.add_argument("--fl_model_path", type=str, default=None, help="Path to FL global model")
    parser.add_argument("--num_managed_od", type=int, default=None, help="Number of managed OD pairs")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    parser.add_argument("--prior_bias", type=float, default=0.0, help="Initial bias for Path 0 (0.0=neutral)")
    parser.add_argument("--surge_prob", type=float, default=0.25, help="Probability of traffic surge during training")
    parser.add_argument("--link_fail_prob", type=float, default=0.20, help="Probability of link failure during training")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="Directory for logs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Resolve paths based on topology
    if args.topology == "nsfnet":
        topo_path = "data/nsfnet_topology.json" if args.topo_path is None else args.topo_path
        tm_path = "data/generated/traffic_matrices.npy" if args.tm_path is None else args.tm_path
        default_fl = "models/fl_global_model.pt"
    elif args.topology == "geant":
        topo_path = "data/geant_topology.json" if args.topo_path is None else args.topo_path
        tm_path = "data/generated/geant_traffic_matrices.npy" if args.tm_path is None else args.tm_path
        default_fl = (
            "models/fl_geant_fedadam.pt"
            if os.path.exists("models/fl_geant_fedadam.pt")
            else "models/fl_global_model.pt"
        )
    else:  # abilene
        topo_path = "data/abilene_topology.json" if args.topo_path is None else args.topo_path
        tm_path = "data/generated/abilene_traffic_matrices.npy" if args.tm_path is None else args.tm_path
        default_fl = (
            "models/fl_abilene_fedadam.pt"
            if os.path.exists("models/fl_abilene_fedadam.pt")
            else "models/fl_global_model.pt"
        )

    fl_model_path = (
        args.fl_model_path if args.fl_model_path is not None else (default_fl if os.path.exists(default_fl) else None)
    )

    print(f"Training PPO Agent for Topology: {args.topology}")
    print(f"Topology: {topo_path} | TM: {tm_path} | FL Model: {fl_model_path}")

    def make_env():
        return NetworkRoutingEnv(
            topo_path=topo_path,
            tm_path=tm_path,
            fl_model_path=fl_model_path,
            num_managed_od=args.num_managed_od,
            surge_prob=args.surge_prob,
            link_fail_prob=args.link_fail_prob,
        )

    # Use DummyVecEnv or SubprocVecEnv
    try:
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    except (RuntimeError, ValueError, OSError, TimeoutError) as e:
        print(f"SubprocVecEnv failed: {e}. Falling back to DummyVecEnv.")
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=DummyVecEnv)

    callback = MetricsCallback(log_dir=args.log_dir, verbose=1)

    policy_kwargs = {"net_arch": {"pi": [256, 128], "vf": [256, 128]}}

    try:
        import tensorboard  # noqa: F401

        tb_log = args.log_dir
    except ImportError:
        tb_log = None

    # Hyperparameters tuned for multi-path exploration and proactive offloading
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log,
    )

    import torch

    # Inspect environment's num_managed
    sample_env = make_env()
    num_managed = sample_env.num_managed

    if args.prior_bias > 0.0:
        with torch.no_grad():
            bias = model.policy.action_net.bias
            for i in range(num_managed):
                bias[3 * i] = args.prior_bias  # Path 0 prior
                bias[3 * i + 1] = 0.0
                bias[3 * i + 2] = 0.0
        print(f"Initialized policy head with prior bias {args.prior_bias} towards Path 0 for all {num_managed} managed OD pairs.")
    else:
        print(f"Neutral initialization (prior_bias=0.0) across all {num_managed} managed OD pairs.")

    try:
        import tqdm  # noqa: F401

        use_pb = True
    except ImportError:
        use_pb = False

    print(f"Starting training for {args.timesteps} timesteps on {args.topology}...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=use_pb)

    if args.topology == "nsfnet":
        save_path = os.path.join(args.model_dir, "ppo_agent.zip")
    else:
        save_path = os.path.join(args.model_dir, f"ppo_agent_{args.topology}.zip")

    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
