"""
Script for Few-Shot Fine-Tuning of Pretrained RecurrentPPO Agent on Target Topologies.

Takes the GÉANT-trained RecurrentPPO model and fine-tunes it on NSFNET and/or Abilene
using a topology padding wrapper with proper learning rate unfreezing and exploration entropy.

CRITICAL SAFETY:
Guaranteed NEVER to overwrite any existing base models (ppo_agent.zip, ppo_agent_geant.zip, ppo_agent_abilene.zip).
Output models are saved explicitly as:
    models/ppo_agent_geant_finetuned_nsfnet.zip
    models/ppo_agent_geant_finetuned_abilene.zip
"""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure prototype root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from rl.env import NetworkRoutingEnv


def linear_schedule(initial_value: float, final_value: float = 3e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class TopologyPaddingWrapper(gym.Wrapper):
    """
    Adapts a target environment (e.g., NSFNET: 44 links, 20 flows) to match
    the exact observation and action space of the source GÉANT model (72 links, 30 flows).
    """

    def __init__(self, env, src_edges: int = 72, src_managed: int = 30):
        super().__init__(env)
        self.src_edges = src_edges
        self.src_managed = src_managed
        self.src_dim = src_edges * 2 + src_managed

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.src_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * src_managed)

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        n_e = self.env.num_edges
        n_m = self.env.num_managed
        padded = np.zeros(self.src_dim, dtype=np.float32)
        # 1. Current link utilizations
        padded[: min(n_e, self.src_edges)] = obs[: min(n_e, self.src_edges)]
        # 2. Predicted link utilizations
        padded[self.src_edges : self.src_edges + min(n_e, self.src_edges)] = obs[n_e : n_e + min(n_e, self.src_edges)]
        # 3. Demands
        padded[self.src_edges * 2 : self.src_edges * 2 + min(n_m, self.src_managed)] = obs[n_e * 2 : n_e * 2 + min(n_m, self.src_managed)]
        return padded

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info

    def step(self, action):
        target_action = np.array(action[: self.env.num_managed], dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(target_action)
        return self._pad_obs(obs), reward, terminated, truncated, info


class TransferEvalCallback(BaseCallback):
    """Evaluates fine-tuned transfer model on target topology every check_freq steps."""

    def __init__(self, target_topology: str, check_freq: int = 2500, verbose: int = 1):
        super().__init__(verbose)
        self.target_topology = target_topology
        self.check_freq = check_freq
        self.last_eval_step = 0
        self.best_score = 999.0
        self.best_model_path = f"models/ppo_agent_geant_finetuned_{target_topology}_best.zip"

        if target_topology == "nsfnet":
            self.topo_path = "data/nsfnet_topology.json"
            self.tm_path = "data/generated/traffic_matrices.npy"
            self.domain_data_dir = "data/preprocessed"
            self.fl_model_path = "models/fl_global_model.pt"
            self.num_managed = 20
        else:
            self.topo_path = "data/abilene_topology.json"
            self.tm_path = "data/generated/abilene_traffic_matrices.npy"
            self.domain_data_dir = "data/preprocessed_abilene"
            self.fl_model_path = (
                "models/fl_global_model_abilene_fedadam.pt"
                if os.path.exists("models/fl_global_model_abilene_fedadam.pt")
                else "models/fl_global_model.pt"
            )
            self.num_managed = 15

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.check_freq:
            return True
        self.last_eval_step = self.num_timesteps

        env = NetworkRoutingEnv(
            topo_path=self.topo_path,
            tm_path=self.tm_path,
            fl_model_path=self.fl_model_path if os.path.exists(self.fl_model_path) else None,
            domain_data_dir=self.domain_data_dir,
            num_managed_od=self.num_managed,
        )

        src_dim = 72 * 2 + 30
        n_e = env.num_edges
        n_m = env.num_managed

        def policy_fn(obs, lstm_state, ep_start):
            padded = np.zeros(src_dim, dtype=np.float32)
            padded[: min(n_e, 72)] = obs[: min(n_e, 72)]
            padded[72 : 72 + min(n_e, 72)] = obs[n_e : n_e + min(n_e, 72)]
            padded[72 * 2 : 72 * 2 + min(n_m, 30)] = obs[n_e * 2 : n_e * 2 + min(n_m, 30)]
            act, next_state = self.model.predict(
                padded, state=lstm_state, episode_start=np.array([ep_start]), deterministic=True
            )
            return act[:n_m], next_state

        seeds = [42, 43, 44]
        t1_mlus = []
        t3_losses = []
        t5_losses = []

        for s in seeds:
            start_idx = min(((s * 3) % 20 + 1) * 288, env.dataset_len - 100 - 1)

            # T1: Normal
            env.reset(seed=s)
            env.time_index = start_idx
            env.step_count = 0
            state = None
            ep_start = True
            for _ in range(100):
                obs = env._get_observation()
                act, state = policy_fn(obs, state, ep_start)
                ep_start = False
                obs, _, _, _, info = env.step(act)
                t1_mlus.append(info["mlu"])

            # T3: Gradual Load
            env.reset(seed=s)
            env.time_index = start_idx
            env.step_count = 0
            orig_slice = env.tm_data[start_idx : start_idx + 100].copy()
            for t in range(100):
                env.tm_data[start_idx + t] *= 1.0 + (0.5 * (t / 100.0))
            state = None
            ep_start = True
            for _ in range(100):
                obs = env._get_observation()
                act, state = policy_fn(obs, state, ep_start)
                ep_start = False
                obs, _, _, _, info = env.step(act)
                t3_losses.append(info["avg_loss"] * 100.0)
            env.tm_data[start_idx : start_idx + 100] = orig_slice

            # T5: Link Failure
            fail_link = (8, 12) if self.target_topology == "nsfnet" else (1, 11)
            env.reset(seed=s)
            env.time_index = start_idx
            env.step_count = 0
            state = None
            ep_start = True
            for step in range(100):
                if step == 40:
                    env.disable_link(fail_link[0], fail_link[1])
                obs = env._get_observation()
                act, state = policy_fn(obs, state, ep_start)
                ep_start = False
                obs, _, _, _, info = env.step(act)
                t5_losses.append(info["avg_loss"] * 100.0)
            env.enable_link(fail_link[0], fail_link[1])

        mean_t1_mlu = np.mean(t1_mlus)
        mean_t3_loss = np.mean(t3_losses)
        mean_t5_loss = np.mean(t5_losses)
        score = mean_t1_mlu + (mean_t3_loss * 0.1) + (mean_t5_loss * 0.1)

        print(f"\n--- [Fine-tune Eval @ step {self.num_timesteps}] ---")
        print(f"  T1 MLU: {mean_t1_mlu:.3f} | T3 Loss: {mean_t3_loss:.2f}% | T5 Loss: {mean_t5_loss:.2f}%")

        if score < self.best_score:
            self.best_score = score
            self.model.save(self.best_model_path)
            print(f"  >>> New best checkpoint saved to {self.best_model_path} (Score={score:.3f}) <<<")
        print("-" * 55)

        return True


PROTECTED_BASE_MODELS = {
    "models/ppo_agent.zip",
    "models/ppo_agent_geant.zip",
    "models/ppo_agent_geant_backup.zip",
    "models/ppo_agent_abilene.zip",
    "models/fl_global_model.pt",
    "models/fl_global_model_geant_fedadam.pt",
    "models/fl_global_model_abilene_fedadam.pt",
}


def finetune_for_topology(target_topology: str, timesteps: int = 25000, n_envs: int = 4):
    target_topology = target_topology.lower()
    source_model_path = "models/ppo_agent_geant.zip"

    if not os.path.exists(source_model_path):
        raise FileNotFoundError(f"Source model not found at {source_model_path}")

    save_path = f"models/ppo_agent_geant_finetuned_{target_topology}.zip"

    norm_save = str(Path(save_path).as_posix())
    if norm_save in PROTECTED_BASE_MODELS:
        raise ValueError(f"FATAL: Attempting to overwrite protected base model {norm_save}!")

    if target_topology == "nsfnet":
        topo_path = "data/nsfnet_topology.json"
        tm_path = "data/generated/traffic_matrices.npy"
        domain_data_dir = "data/preprocessed"
        fl_model_path = "models/fl_global_model.pt"
        num_managed = 20
    elif target_topology == "abilene":
        topo_path = "data/abilene_topology.json"
        tm_path = "data/generated/abilene_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_abilene"
        fl_model_path = (
            "models/fl_global_model_abilene_fedadam.pt"
            if os.path.exists("models/fl_global_model_abilene_fedadam.pt")
            else "models/fl_global_model.pt"
        )
        num_managed = 15
    else:
        raise ValueError(f"Unsupported target topology: {target_topology}")

    print("\n" + "=" * 80)
    print(f"FEW-SHOT FINE-TUNING: GÉANT -> {target_topology.upper()}")
    print(f"  Source Model:      {source_model_path} (Read-Only)")
    print(f"  Target Save Path:  {save_path} (NEW SEPARATE FILE)")
    print(f"  Target Topology:   {topo_path}")
    print(f"  Fine-tune Steps:   {timesteps}")
    print(f"  Parallel Envs:     {n_envs}")
    print(f"  Surge Prob:        0.25")
    print(f"  Failure Prob:      0.20")
    print("=" * 80)

    def make_env():
        raw_env = NetworkRoutingEnv(
            topo_path=topo_path,
            tm_path=tm_path,
            fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
            surge_prob=0.25,
            failure_prob=0.20,
            domain_data_dir=domain_data_dir,
            num_managed_od=num_managed,
        )
        return TopologyPaddingWrapper(raw_env, src_edges=72, src_managed=30)

    try:
        vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    except Exception as e:
        print(f"SubprocVecEnv fallback to DummyVecEnv: {e}")
        vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=DummyVecEnv)

    print(f"[{target_topology.upper()}] Loading source GÉANT model...")
    model = RecurrentPPO.load(source_model_path, env=vec_env)

    # Re-initialize learning rate schedule and parameter groups for smooth fine-tuning
    initial_lr = 1.0e-4
    model.lr_schedule = linear_schedule(initial_lr, 1.5e-5)
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = initial_lr
    model.ent_coef = 0.003

    eval_callback = TransferEvalCallback(target_topology=target_topology, check_freq=2500)

    print(f"[{target_topology.upper()}] Starting fine-tuning for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)

    best_checkpoint = f"models/ppo_agent_geant_finetuned_{target_topology}_best.zip"
    if os.path.exists(best_checkpoint):
        print(f"[{target_topology.upper()}] Loading best evaluated checkpoint...")
        best_model = RecurrentPPO.load(best_checkpoint)
        best_model.save(save_path)
    else:
        model.save(save_path)

    print(f"[{target_topology.upper()}] Fine-tuned model successfully saved to: {save_path}")
    print("Original models remained 100% untouched and safe.")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Fine-Tuning of GÉANT model on other topologies")
    parser.add_argument(
        "--target",
        type=str,
        default="nsfnet",
        choices=["nsfnet", "abilene", "all"],
        help="Target topology to fine-tune on",
    )
    parser.add_argument("--timesteps", type=int, default=60000, help="Timesteps for fine-tuning (default: 60000)")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    args = parser.parse_args()

    targets = ["nsfnet", "abilene"] if args.target == "all" else [args.target]
    for target in targets:
        finetune_for_topology(target, timesteps=args.timesteps, n_envs=args.n_envs)

    print("\nAll fine-tuning runs completed successfully!")


if __name__ == "__main__":
    main()
