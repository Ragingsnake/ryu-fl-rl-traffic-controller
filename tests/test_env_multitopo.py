import os
from pathlib import Path
import numpy as np
import pytest

from rl.env import NetworkRoutingEnv

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("topo,num_edges,num_managed,expected_obs_dim", [
    ("nsfnet", 44, 20, 44 * 2 + 20),
    ("geant", 72, 30, 72 * 2 + 30),
    ("abilene", 30, 15, 30 * 2 + 15),
])
def test_environment_initialization_and_spaces(topo, num_edges, num_managed, expected_obs_dim):
    if topo == "nsfnet":
        topo_path = str(BASE_DIR / "data" / "nsfnet_topology.json")
        tm_path = str(BASE_DIR / "data" / "generated" / "traffic_matrices.npy")
        domain_dir = str(BASE_DIR / "data" / "preprocessed")
    elif topo == "geant":
        topo_path = str(BASE_DIR / "data" / "geant_topology.json")
        tm_path = str(BASE_DIR / "data" / "generated" / "geant_traffic_matrices.npy")
        domain_dir = str(BASE_DIR / "data" / "preprocessed_geant")
    elif topo == "abilene":
        topo_path = str(BASE_DIR / "data" / "abilene_topology.json")
        tm_path = str(BASE_DIR / "data" / "generated" / "abilene_traffic_matrices.npy")
        domain_dir = str(BASE_DIR / "data" / "preprocessed_abilene")

    env = NetworkRoutingEnv(
        topo_path=topo_path,
        tm_path=tm_path,
        num_managed_od=num_managed,
        domain_data_dir=domain_dir,
    )

    assert env.num_edges == num_edges, f"Expected {num_edges} edges, got {env.num_edges}"
    assert env.num_managed == num_managed, f"Expected {num_managed} managed OD pairs, got {env.num_managed}"
    assert env.observation_space.shape == (expected_obs_dim,), f"Expected obs shape ({expected_obs_dim},), got {env.observation_space.shape}"
    assert len(env.action_space.nvec) == num_managed, f"Expected action space len {num_managed}, got {len(env.action_space.nvec)}"

    obs, info = env.reset(seed=42)
    assert obs.shape == (expected_obs_dim,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Normalized observation values must be in [0, 1]"

    # Execute 5 sample steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, trunc, step_info = env.step(action)
        assert not np.isnan(reward), "Reward must not be NaN"
        assert obs.shape == (expected_obs_dim,)
        assert step_info["mlu"] >= 0.0


def test_link_failure_mechanism():
    topo_path = str(BASE_DIR / "data" / "geant_topology.json")
    tm_path = str(BASE_DIR / "data" / "generated" / "geant_traffic_matrices.npy")
    env = NetworkRoutingEnv(
        topo_path=topo_path,
        tm_path=tm_path,
        num_managed_od=30,
    )

    env.reset(seed=10)
    # Target link
    fail_link = (0, 4)
    assert fail_link in env.edges or (4, 0) in env.edges

    env.disable_link(fail_link[0], fail_link[1])
    assert fail_link in env.disabled_links
    assert (fail_link[1], fail_link[0]) in env.disabled_links

    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    assert not np.isnan(reward)

    env.enable_link(fail_link[0], fail_link[1])
    assert fail_link not in env.disabled_links
