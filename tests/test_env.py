import numpy as np


def test_env_reset(dummy_env):
    obs, _info = dummy_env.reset()
    assert obs.shape == dummy_env.observation_space.shape, (
        f"Expected observation shape {dummy_env.observation_space.shape}, got {obs.shape}"
    )
    assert np.all(obs >= 0) and np.all(obs <= 1), "Observation values must be in [0, 1]"


def test_env_step(dummy_env):
    dummy_env.reset()
    # 20 OD pairs, 3 paths each
    action = np.random.randint(0, 3, size=(20,))

    obs, reward, terminated, truncated, _info = dummy_env.step(action)
    assert obs.shape == dummy_env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(reward, float), "Reward must be a float"
    assert isinstance(terminated, bool), "terminated must be boolean"
    assert isinstance(truncated, bool), "truncated must be boolean"


def test_env_episode_length(dummy_env):
    dummy_env.reset()
    steps = 0
    terminated, truncated = False, False

    while not (terminated or truncated) and steps < 150:
        action = np.random.randint(0, 3, size=(20,))
        _obs, _reward, terminated, truncated, _info = dummy_env.step(action)
        steps += 1

    assert steps <= 100, "Episode should terminate after 100 steps"
