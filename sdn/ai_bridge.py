from __future__ import annotations

"""
AI Bridge
Connects the Ryu SDN controller to the FL+RL models.
Reads stats from Ryu, queries RL model, (mock) installs flow mods.
"""

import json
import os
import sys
import time

from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.env import NSFNETRoutingEnv


def load_models():
    ppo_path = "../models/ppo_agent.zip"
    if not os.path.exists(ppo_path):
        print(f"Error: RL model not found at {ppo_path}")
        return None, None

    agent = PPO.load(
        ppo_path, custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    )

    # We will instantiate the env to reuse its FL and routing logic
    # Use random TM since we just need the environment's internal logic,
    # not the actual simulated traffic step.
    env = NSFNETRoutingEnv(
        topo_path="../data/nsfnet_topology.json",
        tm_path="../data/generated/traffic_matrices.npy",
        fl_model_path="../models/fl_global_model.pt",
    )
    env.reset()
    return agent, env


def main():
    agent, env = load_models()
    if agent is None:
        return

    print("AI Bridge initialized. Waiting for SDN stats...")

    while True:
        time.sleep(3)  # Polling check

        stats_file = "/tmp/sdn_stats.json"
        if not os.path.exists(stats_file):
            print("Waiting for Ryu stats...")
            continue

        try:
            with open(stats_file) as f:
                stats = json.load(f)
        except Exception as e:
            print(f"Error reading stats: {e}")
            continue

        # In a real setup, we would map dpid + port_no to our edge indices
        # Here we mock the utilization for demonstration of the loop
        # and instead just show that the loop is running and querying the model.

        # Read mock or actual util
        # env.link_utils = ... (calculate from stats)

        obs = env._get_observation()
        action, _states = agent.predict(obs, deterministic=True)

        print(f"[AI Bridge] Calculated new routing action: {action}")
        # Next step would be sending these actions to Ryu via REST
        # (For Phase 2 path 1 demo, printing the action is sufficient to show closed loop)


if __name__ == "__main__":
    main()
