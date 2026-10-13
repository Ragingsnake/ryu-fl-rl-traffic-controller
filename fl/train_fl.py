from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import flwr as fl
import numpy as np
import torch

from fl.client import TrafficClient, load_data
from fl.model import create_model
from fl.server import build_strategy


def main():
    parser = argparse.ArgumentParser(description="Train Federated Learning Model for Traffic Prediction")
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds to train")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedadam",
        choices=["fedavg", "fedadam"],
        help="Federated optimization strategy (fedavg or fedadam)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="nsfnet",
        choices=["nsfnet", "geant", "abilene"],
        help="Target topology",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Directory containing preprocessed domain data"
    )
    parser.add_argument(
        "--model_save_path", type=str, default=None, help="Path to save the global model"
    )

    args = parser.parse_args()

    # Determine default paths
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        if args.topology == "nsfnet":
            data_dir = "data/preprocessed"
        else:
            data_dir = f"data/preprocessed_{args.topology}"

    if args.model_save_path is not None:
        model_save_path = args.model_save_path
    else:
        if args.topology == "nsfnet":
            model_save_path = f"models/fl_global_model_{args.strategy}.pt" if args.strategy != "fedavg" else "models/fl_global_model.pt"
        else:
            model_save_path = f"models/fl_global_model_{args.topology}_{args.strategy}.pt"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"Starting FL Training with strategy={args.strategy.upper()} on topology={args.topology.upper()}...")
    print(f"  Data dir:        {data_dir}")
    print(f"  Model save path: {model_save_path}")

    # Determine MAX_F_IN dynamically across domain data
    max_f = 7
    for did in [0, 1, 2]:
        p = os.path.join(data_dir, f"domain_{did}_train_X.npy")
        if os.path.exists(p):
            max_f = max(max_f, np.load(p, mmap_mode="r").shape[2])
    MAX_F_IN = max_f
    print(f"  Detected Max Domain Input Dimension (MAX_F_IN): {MAX_F_IN}")

    def client_fn(cid: str) -> fl.client.Client:
        domain_id = int(cid)
        train_loader, val_loader, f_in = load_data(
            domain_id, data_dir=data_dir, batch_size=32, target_f_in=MAX_F_IN
        )
        model = create_model(f_in=f_in)
        return TrafficClient(
            domain_id=domain_id, model=model, train_loader=train_loader, val_loader=val_loader
        ).to_client()

    strategy = build_strategy(
        num_rounds=args.rounds,
        num_clients=3,
        model_save_path=model_save_path,
        f_in=MAX_F_IN,
        strategy_type=args.strategy,
    )

    try:
        import ray  # noqa: F401

        print(f"Starting Flower simulation for {args.rounds} rounds...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )
    except (ImportError, RuntimeError, OSError) as e:
        print(f"Flower simulation engine unavailable ({e}). Using direct synchronous {args.strategy.upper()} runner...")
        run_local_fl(
            rounds=args.rounds,
            num_clients=3,
            data_dir=data_dir,
            model_save_path=model_save_path,
            max_f_in=MAX_F_IN,
            strategy_type=args.strategy,
        )

    print(f"FL Training ({args.strategy.upper()} on {args.topology}) complete!")


def run_local_fl(
    rounds: int,
    num_clients: int,
    data_dir: str,
    model_save_path: str,
    max_f_in: int,
    strategy_type: str = "fedadam",
) -> None:
    """
    Direct synchronous federated optimization loop supporting both FedAvg and FedAdam.
    Guarantees reliable execution in CI and local test environments.
    """
    print(f"Running synchronous {strategy_type.upper()} for {rounds} rounds across {num_clients} clients...")
    global_model = create_model(f_in=max_f_in)
    global_params = [val.detach().cpu().numpy() for val in global_model.state_dict().values()]

    # FedAdam hyperparameters
    eta = 1e-3
    beta_1 = 0.9
    beta_2 = 0.99
    tau = 1e-5

    # Momentum and variance accumulators
    m_moments = [np.zeros_like(p) for p in global_params]
    v_moments = [np.zeros_like(p) for p in global_params]

    clients = []
    for cid in range(num_clients):
        train_loader, val_loader, f_in = load_data(cid, data_dir=data_dir, batch_size=32, target_f_in=max_f_in)
        model = create_model(f_in=f_in)
        client = TrafficClient(domain_id=cid, model=model, train_loader=train_loader, val_loader=val_loader)
        clients.append(client)

    for r in range(1, rounds + 1):
        client_weights = []
        total_samples = 0
        for client in clients:
            weights, num_samples, _ = client.fit(global_params, {"local_epochs": 5, "lr": 1e-3, "round": r})
            client_weights.append((weights, num_samples))
            total_samples += num_samples

        if total_samples > 0:
            if strategy_type.lower() == "fedadam":
                # 1. Compute pseudo-gradients: Delta = sum_k (n_k / n) * (w_k - w_t)
                delta_params = []
                for idx in range(len(global_params)):
                    weighted_update = sum(
                        (w[idx] - global_params[idx]) * (n / total_samples) for w, n in client_weights
                    )
                    delta_params.append(weighted_update)

                # 2. Update server momentum and velocity
                new_params = []
                for idx in range(len(global_params)):
                    delta = delta_params[idx]
                    m_moments[idx] = beta_1 * m_moments[idx] + (1.0 - beta_1) * delta
                    v_moments[idx] = beta_2 * v_moments[idx] + (1.0 - beta_2) * (delta**2)
                    # 3. Apply server Adam step: w_{t+1} = w_t + eta * m / (sqrt(v) + tau)
                    step = eta * m_moments[idx] / (np.sqrt(v_moments[idx]) + tau)
                    new_params.append(global_params[idx] + step)

                global_params = new_params
            else:
                # Standard FedAvg: weighted average of client weights
                new_params = []
                for idx in range(len(global_params)):
                    weighted = sum(w[idx] * (n / total_samples) for w, n in client_weights)
                    new_params.append(weighted)
                global_params = new_params

        # Evaluate validation loss
        eval_losses = []
        for client in clients:
            loss, _, _ = client.evaluate(global_params, {})
            eval_losses.append(loss)
        avg_loss = float(np.mean(eval_losses))
        print(f"Round {r}/{rounds} [{strategy_type.upper()}] - Mean Validation Loss: {avg_loss:.6f}", flush=True)

    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    state_dict = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(global_model.state_dict().keys(), global_params, strict=True)
    )
    global_model.load_state_dict(state_dict)
    torch.save(global_model.state_dict(), model_save_path)
    print(f"FL Global model saved to {model_save_path}", flush=True)


if __name__ == "__main__":
    main()
