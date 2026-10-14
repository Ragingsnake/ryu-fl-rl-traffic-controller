import os
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from fl.model import create_model


def save_aggregated_model(
    aggregated_parameters: fl.common.Parameters,
    model_save_path: str,
    f_in: int,
    server_round: int,
) -> None:
    """Helper to save Flower Parameters to PyTorch state dict or raw numpy."""
    aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    try:
        model = create_model(f_in=f_in)
        params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        torch.save(model.state_dict(), model_save_path)
        print(f"Round {server_round}: Global model saved to {model_save_path}")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(
            f"Round {server_round}: Could not save PyTorch model due to architecture mismatch. Saving raw weights instead. Error: {e}"
        )
        np.save(model_save_path.replace(".pt", "_weights.npy"), np.array(aggregated_ndarrays, dtype=object))


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy to save the global model after training."""

    def __init__(self, model_save_path: str, f_in: int = 7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.f_in = f_in

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[fl.common.Parameters | None, dict[str, fl.common.Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            save_aggregated_model(aggregated_parameters, self.model_save_path, self.f_in, server_round)
        return aggregated_parameters, aggregated_metrics


class SaveModelFedAdamStrategy(fl.server.strategy.FedAdam):
    """Custom FedAdam strategy with server-side momentum and adaptive learning rate."""

    def __init__(self, model_save_path: str, f_in: int = 7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.f_in = f_in

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[fl.common.Parameters | None, dict[str, fl.common.Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            save_aggregated_model(aggregated_parameters, self.model_save_path, self.f_in, server_round)
        return aggregated_parameters, aggregated_metrics


def build_strategy(
    strategy_name: str = "fedadam",
    num_rounds: int = 50,
    num_clients: int = 3,
    model_save_path: str = "models/fl_global_model.pt",
    f_in: int = 24,
) -> fl.server.strategy.Strategy:
    """Builds the FedAvg or FedAdam strategy for FL training."""

    def fit_config(server_round: int):
        return {"local_epochs": 5, "lr": 1e-3, "round": server_round}

    common_kwargs = {
        "model_save_path": model_save_path,
        "f_in": f_in,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": num_clients,
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "on_fit_config_fn": fit_config,
    }

    if strategy_name.lower() == "fedavg":
        return SaveModelStrategy(**common_kwargs)

    # Initialize parameters for FedAdam
    init_model = create_model(f_in=f_in)
    init_ndarrays = [val.cpu().numpy() for val in init_model.state_dict().values()]
    init_parameters = fl.common.ndarrays_to_parameters(init_ndarrays)

    return SaveModelFedAdamStrategy(
        initial_parameters=init_parameters,
        eta=1e-3,
        eta_l=1e-3,
        beta_1=0.9,
        beta_2=0.99,
        tau=1e-5,
        **common_kwargs,
    )
