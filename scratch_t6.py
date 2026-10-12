import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import PPO
from rl.env import NSFNETRoutingEnv

def run_t6_debug():
    env = NSFNETRoutingEnv(
        topo_path="data/nsfnet_topology.json",
        tm_path="data/generated/traffic_matrices.npy",
        fl_model_path="models/fl_global_model.pt"
    )
    agent = PPO.load("models/ppo_agent.zip")
    
    for seed in [42, 43, 44, 45, 46]:
        day_idx = (seed * 3) % 20 + 1
        start_idx = day_idx * 288 + 0
        num_steps = 100
        
        env.reset(seed=seed)
        env.time_index = start_idx
        obs = env._get_observation()
        
        original_slice = env.tm_data[start_idx : start_idx + num_steps].copy()
        scenario_slice = original_slice.copy()
        
        for t in range(0, 40):
            scenario_slice[t] *= 2.0
        for t in range(60, num_steps):
            scenario_slice[t] *= 0.3
            
        env.tm_data[start_idx : start_idx + num_steps] = scenario_slice
        
        print(f"--- Seed {seed} ---")
        for step in range(45):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _truncated, info = env.step(action)
            if step >= 30 and info.get('churn', 0) > 0:
                print(f"Step {step:4d} | Churn: {info.get('churn', 0)*100:7.1f} | ActionSum: {np.sum(action)}")

if __name__ == '__main__':
    run_t6_debug()
