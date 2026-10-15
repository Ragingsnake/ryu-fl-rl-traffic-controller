from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.callbacks import MetricsCallback
from rl.env import NetworkRoutingEnv, NSFNETRoutingEnv


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Traffic Engineering")
    parser.add_argument("--topology", type=str, default="nsfnet", choices=["nsfnet", "geant", "abilene"], help="Network topology")
    parser.add_argument("--timesteps", type=int, default=25000, help="Total timesteps to train")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num_managed_od", type=int, default=None, help="Number of managed OD pairs")
    parser.add_argument("--surge_prob", type=float, default=None, help="Probability of traffic surge during training (None = topology default)")
    parser.add_argument("--ent_coef", type=float, default=None, help="Entropy exploration coefficient (None = topology default)")
    parser.add_argument("--prior_bias", type=float, default=None, help="Prior bias towards Path 0 (None = topology default)")
    parser.add_argument("--fl_model_path", type=str, default=None, help="Path to FL predictor model")
    parser.add_argument("--model_save_path", type=str, default=None, help="Path to save trained RL model")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="Directory for logs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    topo = args.topology.lower()
    if topo == "nsfnet":
        topo_path = "data/nsfnet_topology.json"
        tm_path = "data/generated/traffic_matrices.npy"
        domain_data_dir = "data/preprocessed"
        default_num_managed = 20
        default_fl = "models/fl_global_model.pt"
        default_rl = os.path.join(args.model_dir, "ppo_agent.zip")
    elif topo == "geant":
        topo_path = "data/geant_topology.json"
        tm_path = "data/generated/geant_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_geant"
        default_num_managed = 30
        default_fl = "models/fl_global_model_geant_fedadam.pt"
        default_rl = os.path.join(args.model_dir, "ppo_agent_geant.zip")
    elif topo == "abilene":
        topo_path = "data/abilene_topology.json"
        tm_path = "data/generated/abilene_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_abilene"
        default_num_managed = 15
        default_fl = "models/fl_global_model_abilene_fedadam.pt"
        default_rl = os.path.join(args.model_dir, "ppo_agent_abilene.zip")
    else:
        raise ValueError(f"Unsupported topology: {topo}")

    num_managed = args.num_managed_od if args.num_managed_od is not None else default_num_managed
    fl_model_path = args.fl_model_path if args.fl_model_path is not None else default_fl
    save_path = args.model_save_path if args.model_save_path is not None else default_rl

    # Calibrated 25k convergence hyperparameters:
    # 1. Gentle prior bias 0.5 (~45% P0, ~27.5% P1, ~27.5% P2) encourages Path 0 without locking logits,
    #    preventing policy freeze and enabling smooth ascending convergence from -115 to -65.
    # 2. Balanced entropy coefficient (0.002) supports exploration of candidate paths during surges.
    # 3. Controlled surge probability (0.25) provides frequent congestion events to learn rerouting.
    default_ent_coef = 0.005
    default_prior_bias = 0.2
    default_surge_prob = 0.30

    ent_coef = args.ent_coef if args.ent_coef is not None else default_ent_coef
    prior_bias = args.prior_bias if args.prior_bias is not None else default_prior_bias
    surge_prob = args.surge_prob if args.surge_prob is not None else default_surge_prob

    print(f"============================================================")
    print(f"Training PPO on Topology: {topo.upper()}")
    print(f"  Topology file:    {topo_path}")
    print(f"  TM file:          {tm_path}")
    print(f"  FL Model:         {fl_model_path} (exists: {os.path.exists(fl_model_path)})")
    print(f"  Domain Data Dir:  {domain_data_dir}")
    print(f"  Managed OD pairs: {num_managed}")
    print(f"  Total Timesteps:  {args.timesteps}")
    print(f"  Exploration Ent:  {ent_coef}")
    print(f"  Prior Bias (P0):  {prior_bias}")
    print(f"  Surge Prob:       {surge_prob}")
    print(f"  Output Model:     {save_path}")
    print(f"============================================================")

    def make_env():
        return NetworkRoutingEnv(
            topo_path=topo_path,
            tm_path=tm_path,
            fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
            surge_prob=surge_prob,
            domain_data_dir=domain_data_dir,
            num_managed_od=num_managed,
        )

    # Use DummyVecEnv for stability across different OS/environments, or SubprocVecEnv for speed
    try:
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    except (RuntimeError, ValueError, OSError, TimeoutError) as e:
        print(f"SubprocVecEnv failed: {e}. Falling back to DummyVecEnv.")
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=DummyVecEnv)

    default_log_dir = os.path.join(args.log_dir, topo)
    default_demo_dir = os.path.join("results/demo", topo)
    callback = MetricsCallback(
        log_dir=default_log_dir,
        output_dir=default_demo_dir,
        title_suffix=topo.upper(),
        verbose=1,
    )

    policy_kwargs = {"net_arch": {"pi": [256, 128], "vf": [256, 128]}}

    try:
        import tensorboard  # noqa: F401

        tb_log = args.log_dir
    except ImportError:
        tb_log = None

    # Hyperparameters tuned for coordination, rapid updates, and low entropy
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=linear_schedule(4e-4, 1e-4),
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log,
    )

    import torch

    # Warm start: initialize policy head bias to favor Path 0 (OSPF baseline prior)
    with torch.no_grad():
        bias = model.policy.action_net.bias
        for i in range(num_managed):
            bias[3 * i] = float(prior_bias)  # Path 0 (Shortest Path / OSPF prior)
            bias[3 * i + 1] = 0.0  # Path 1
            bias[3 * i + 2] = 0.0  # Path 2
    print(f"Initialized policy head with prior bias {prior_bias:.2f} towards Path 0 for {num_managed} managed OD pairs.")

    try:
        import tqdm  # noqa: F401

        use_pb = True
    except ImportError:
        use_pb = False

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=use_pb)

    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
