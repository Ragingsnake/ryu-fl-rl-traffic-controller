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
from rl.env import NSFNETRoutingEnv


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Traffic Engineering")
    parser.add_argument("--timesteps", type=int, default=2000000, help="Total timesteps to train")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="Directory for logs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    topo_path = "data/nsfnet_topology.json"
    tm_path = "data/generated/traffic_matrices.npy"
    fl_model_path = "models/fl_global_model.pt"

    def make_env():
        return NSFNETRoutingEnv(
            topo_path=topo_path,
            tm_path=tm_path,
            fl_model_path=fl_model_path,
            surge_prob=0.15,
        )

    # Use DummyVecEnv for stability across different OS/environments, or SubprocVecEnv for speed
    # Falling back to DummyVecEnv for broader compatibility if Subproc fails
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

    # Hyperparameters tuned for coordination, rapid updates, and low entropy
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
        ent_coef=0.0005,
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
        num_managed = 20
        for i in range(num_managed):
            bias[3 * i] = 2.0  # Path 0 (Shortest Path / OSPF prior)
            bias[3 * i + 1] = 0.0  # Path 1
            bias[3 * i + 2] = 0.0  # Path 2
    print("Initialized policy head with prior bias towards Path 0 (OSPF baseline).")

    try:
        import tqdm  # noqa: F401

        use_pb = True
    except ImportError:
        use_pb = False

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=use_pb)

    save_path = os.path.join(args.model_dir, "ppo_agent.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
