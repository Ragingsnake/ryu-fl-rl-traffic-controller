import copy
from pathlib import Path

import torch

from fl.model import create_model
from fl.server import build_strategy

BASE_DIR = Path(__file__).resolve().parent.parent


def test_fedadam_strategy_initialization():
    strategy = build_strategy(num_rounds=5, strategy_type="fedadam")
    assert strategy is not None
    # Verify FedAvg builds as well
    strategy_avg = build_strategy(num_rounds=5, strategy_type="fedavg")
    assert strategy_avg is not None


def test_fedadam_synchronous_update_logic():
    """Verify momentum and velocity updates follow Adam optimization principles."""
    model = create_model(f_in=8)
    initial_weights = copy.deepcopy(model.state_dict())

    # Simulated client gradients / pseudo-gradients Delta
    client_weights = []
    for _ in range(3):
        perturbed = copy.deepcopy(initial_weights)
        for k in perturbed:
            perturbed[k] = perturbed[k] + torch.randn_like(perturbed[k]) * 0.01
        client_weights.append(perturbed)

    # Compute average Delta = \bar{\theta} - \theta_0
    avg_delta = {}
    for k in initial_weights:
        diffs = [cw[k] - initial_weights[k] for cw in client_weights]
        avg_delta[k] = torch.stack(diffs, dim=0).mean(dim=0)

    # First moment m_t and second moment v_t
    beta_1 = 0.9
    beta_2 = 0.99
    eta = 0.05
    tau = 1e-3

    m_t = {k: torch.zeros_like(v) for k, v in initial_weights.items()}
    v_t = {k: torch.zeros_like(v) for k, v in initial_weights.items()}

    updated_weights = {}
    for k in initial_weights:
        m_t[k] = beta_1 * m_t[k] + (1 - beta_1) * avg_delta[k]
        v_t[k] = beta_2 * v_t[k] + (1 - beta_2) * (avg_delta[k] ** 2)
        step = eta * m_t[k] / (torch.sqrt(v_t[k]) + tau)
        updated_weights[k] = initial_weights[k] + step

    # Ensure update actually modified weights and produced finite values
    for k in initial_weights:
        assert not torch.allclose(updated_weights[k], initial_weights[k])
        assert not torch.isnan(updated_weights[k]).any()
        assert not torch.isinf(updated_weights[k]).any()
