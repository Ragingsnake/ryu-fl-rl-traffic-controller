import argparse
import os
from collections.abc import Callable

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
        return NSFNETRoutingEnv(topo_path=topo_path, tm_path=tm_path, fl_model_path=fl_model_path)

    # Use DummyVecEnv for stability across different OS/environments, or SubprocVecEnv for speed
    # Falling back to DummyVecEnv for broader compatibility if Subproc fails
    try:
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    except (RuntimeError, ValueError, OSError, TimeoutError) as e:
        print(f"SubprocVecEnv failed: {e}. Falling back to DummyVecEnv.")
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=DummyVecEnv)

    callback = MetricsCallback(log_dir=args.log_dir, verbose=1)

    policy_kwargs = {"net_arch": {"pi": [256, 128], "vf": [256, 128]}}

    # Hyperparameters per §5.6
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    save_path = os.path.join(args.model_dir, "ppo_agent.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
