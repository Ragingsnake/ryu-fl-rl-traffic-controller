from __future__ import annotations

import os
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from fl.model import create_model


def get_initial_parameters(f_in: int = 24) -> fl.common.Parameters:
    """Instantiate a base model and extract initial parameters."""
    model = create_model(f_in=f_in)
    ndarrays = [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(ndarrays)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy to save the global model after training.
    """

    def __init__(self, model_save_path: str, f_in: int = 24, *args, **kwargs):
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
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

            try:
                model = create_model(f_in=self.f_in)
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                torch.save(model.state_dict(), self.model_save_path)
                print(f"Round {server_round}: Global model saved to {self.model_save_path}")
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                print(f"Round {server_round}: Saving raw weights due to: {e}")
                np.save(
                    self.model_save_path.replace(".pt", "_weights.npy"), np.array(aggregated_ndarrays, dtype=object)
                )

        return aggregated_parameters, aggregated_metrics


class SaveModelFedAdamStrategy(fl.server.strategy.FedAdam):
    """
    Custom FedAdam strategy with adaptive server momentum to combat client drift in non-IID traffic.
    """

    def __init__(self, model_save_path: str, f_in: int = 24, *args, **kwargs):
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
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

            try:
                model = create_model(f_in=self.f_in)
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                torch.save(model.state_dict(), self.model_save_path)
                print(f"Round {server_round}: Global FedAdam model saved to {self.model_save_path}")
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                print(f"Round {server_round}: Saving raw weights due to: {e}")
                np.save(
                    self.model_save_path.replace(".pt", "_weights.npy"), np.array(aggregated_ndarrays, dtype=object)
                )

        return aggregated_parameters, aggregated_metrics


def build_strategy(
    num_rounds: int,
    num_clients: int = 3,
    model_save_path: str = "models/fl_global_model.pt",
    f_in: int = 24,
    strategy_type: str = "fedadam",
) -> fl.server.strategy.Strategy:
    """
    Builds the FedAvg or FedAdam strategy for FL training.
    """

    def fit_config(server_round: int):
        return {"local_epochs": 5, "lr": 1e-3, "round": server_round}

    if strategy_type.lower() == "fedadam":
        return SaveModelFedAdamStrategy(
            model_save_path=model_save_path,
            f_in=f_in,
            initial_parameters=get_initial_parameters(f_in=f_in),
            eta=1e-3,
            eta_l=1e-3,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-5,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            on_fit_config_fn=fit_config,
        )

    return SaveModelStrategy(
        model_save_path=model_save_path,
        f_in=f_in,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
    )
