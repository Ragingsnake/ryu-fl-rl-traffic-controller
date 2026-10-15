from pathlib import Path

import numpy as np
import pytest
from sb3_contrib import RecurrentPPO

from rl.env import NetworkRoutingEnv
from scripts.demo import make_transfer_policy

BASE_DIR = Path(__file__).resolve().parent.parent


def test_recurrent_ppo_instantiation_and_step():
    topo_path = BASE_DIR / "data" / "nsfnet_topology.json"
    tm_path = BASE_DIR / "data" / "generated" / "traffic_matrices.npy"

    if not topo_path.exists() or not tm_path.exists():
        pytest.skip("Topology or TM data not found for RecurrentPPO test.")

    env = NetworkRoutingEnv(
        topo_path=str(topo_path),
        tm_path=str(tm_path),
        fl_model_path=None,
        surge_prob=0.0,
        num_managed_od=5,
    )

    recurrent_policy_kwargs = {
        "lstm_hidden_size": 32,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        policy_kwargs=recurrent_policy_kwargs,
        verbose=0,
    )

    # Test predict with LSTM state tracking
    obs, _ = env.reset(seed=42)
    lstm_state = None
    ep_start = np.array([True])

    action, lstm_state = model.predict(obs, state=lstm_state, episode_start=ep_start, deterministic=True)
    assert action.shape == (5,)
    assert lstm_state is not None

    # Step in environment
    next_obs, reward, done, _, info = env.step(action)
    assert not np.isnan(reward)
    assert next_obs.shape == obs.shape

    # Second step with preserved state
    ep_start = np.array([False])
    action2, lstm_state = model.predict(next_obs, state=lstm_state, episode_start=ep_start, deterministic=True)
    assert action2.shape == (5,)


def test_make_transfer_policy_recurrent():
    topo_path = BASE_DIR / "data" / "abilene_topology.json"
    tm_path = BASE_DIR / "data" / "generated" / "abilene_traffic_matrices.npy"

    if not topo_path.exists() or not tm_path.exists():
        pytest.skip("Abilene topology or TM data not found.")

    env = NetworkRoutingEnv(
        topo_path=str(topo_path),
        tm_path=str(tm_path),
        fl_model_path=None,
        surge_prob=0.0,
        num_managed_od=15,
    )

    # Create dummy source agent with recurrent policy
    dummy_env = NetworkRoutingEnv(
        topo_path=str(BASE_DIR / "data" / "nsfnet_topology.json"),
        tm_path=str(BASE_DIR / "data" / "generated" / "traffic_matrices.npy"),
        fl_model_path=None,
        surge_prob=0.0,
        num_managed_od=20,
    )
    src_agent = RecurrentPPO(
        "MlpLstmPolicy",
        dummy_env,
        n_steps=16,
        batch_size=16,
        policy_kwargs={"lstm_hidden_size": 32},
        verbose=0,
    )

    transfer_fn = make_transfer_policy(
        src_agent,
        src_edges=dummy_env.num_edges,
        src_managed=dummy_env.num_managed,
        with_ospf_fallback=True,
    )

    obs, _ = env.reset(seed=42)
    action = transfer_fn(obs, env)
    assert len(action) == env.num_managed
    assert np.all(action >= 0) and np.all(action < 3)
