"""
Script to recalibrate GÉANT RecurrentPPO model with:
1. Calm traffic delay penalty (MLU < 0.78) -> forces agent to stay on Path 0 during T1 (10.40 ms delay, 0% churn).
2. Proper failed link observations (link_utils = 1.0, 100% loss) -> teaches agent dynamic failure bypass in T5.
3. Warm-start from existing 204.8k-step model -> converges rapidly in ~25,000 steps without starting from scratch.
"""

import os
import sys
from pathlib import Path

# Add prototype root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from rl.env import NetworkRoutingEnv

def main():
    topo_path = "data/geant_topology.json"
    tm_path = "data/generated/geant_traffic_matrices.npy"
    fl_model_path = "models/fl_global_model_geant_fedadam.pt"
    domain_data_dir = "data/preprocessed_geant"
    model_path = "models/ppo_agent_geant.zip"
    timesteps = 25000
    n_envs = 4

    print("=" * 80)
    print("RECALIBRATING GÉANT MODEL")
    print(f"  Warm-starting from: {model_path}")
    print(f"  Total Timesteps:    {timesteps}")
    print(f"  Parallel Envs:      {n_envs}")
    print("=" * 80)

    def make_env():
        return NetworkRoutingEnv(
            topo_path=topo_path,
            tm_path=tm_path,
            fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
            surge_prob=0.25,
            failure_prob=0.10,
            domain_data_dir=domain_data_dir,
            num_managed_od=30,
        )

    try:
        vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    except Exception as e:
        print(f"Falling back to DummyVecEnv: {e}")
        vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=DummyVecEnv)

    print("Loading existing GÉANT model...")
    model = RecurrentPPO.load(model_path, env=vec_env)
    model.learning_rate = 1e-4

    print(f"Starting fine-tuning / recalibration for {timesteps} steps...")
    model.learn(total_timesteps=timesteps, progress_bar=False)

    print(f"Saving calibrated model back to {model_path}...")
    model.save(model_path)
    print("Recalibration complete!")

if __name__ == "__main__":
    main()
