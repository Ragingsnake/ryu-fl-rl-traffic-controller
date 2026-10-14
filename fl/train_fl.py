import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
import numpy as np
import torch

from fl.client import TrafficClient, load_data
from fl.model import create_model
from fl.server import build_strategy


def run_local_fedavg(
    rounds: int,
    num_clients: int,
    data_dir: str,
    model_save_path: str,
    max_f_in: int,
) -> None:
    """
    Direct synchronous FedAvg loop using TrafficClient without requiring Ray/Flower background daemons.
    Guarantees reliable execution in CI, Windows, and containerized environments.
    """
    print(f"Running synchronous FedAvg for {rounds} rounds across {num_clients} clients...")
    global_model = create_model(f_in=max_f_in)
    global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]

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
            new_params = []
            for idx in range(len(global_params)):
                weighted = sum(w[idx] * (n / total_samples) for w, n in client_weights)
                new_params.append(weighted)
            global_params = new_params

        # Evaluate progress
        eval_losses = []
        for client in clients:
            loss, _, _ = client.evaluate(global_params, {})
            eval_losses.append(loss)
        avg_loss = float(np.mean(eval_losses))
        print(f"Round {r}/{rounds} - Mean Validation Loss: {avg_loss:.6f}")

    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    state_dict = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(global_model.state_dict().keys(), global_params, strict=True)
    )
    global_model.load_state_dict(state_dict)
    torch.save(global_model.state_dict(), model_save_path)
    print(f"FL Global model saved to {model_save_path}")


def run_local_fedadam(
    rounds: int,
    num_clients: int,
    data_dir: str,
    model_save_path: str,
    max_f_in: int,
    eta: float = 1e-3,
    beta_1: float = 0.9,
    beta_2: float = 0.99,
    tau: float = 1e-5,
) -> None:
    """
    Direct synchronous FedAdam adaptive server optimizer (Reddi et al., ICLR 2021).
    Maintains server-side first (m) and second (v) moment estimates.
    """
    print(
        f"Running synchronous FedAdam (eta={eta}, beta1={beta_1}, beta2={beta_2}, tau={tau}) for {rounds} rounds across {num_clients} clients..."
    )
    global_model = create_model(f_in=max_f_in)
    global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]

    # Initialize first and second moments on server
    m = [np.zeros_like(p, dtype=np.float32) for p in global_params]
    v = [np.zeros_like(p, dtype=np.float32) for p in global_params]

    clients = []
    for cid in range(num_clients):
        train_loader, val_loader, f_in = load_data(cid, data_dir=data_dir, batch_size=32, target_f_in=max_f_in)
        model = create_model(f_in=f_in)
        client = TrafficClient(domain_id=cid, model=model, train_loader=train_loader, val_loader=val_loader)
        clients.append(client)

    for r in range(1, rounds + 1):
        client_deltas = []
        total_samples = 0
        for client in clients:
            weights, num_samples, _ = client.fit(global_params, {"local_epochs": 5, "lr": 1e-3, "round": r})
            # Client delta = weights - global_params
            deltas = [w - g for w, g in zip(weights, global_params)]
            client_deltas.append((deltas, num_samples))
            total_samples += num_samples

        if total_samples > 0:
            # Weighted average pseudo-gradient: Delta_t
            delta_avg = []
            for idx in range(len(global_params)):
                weighted_delta = sum(d[idx] * (n / total_samples) for d, n in client_deltas)
                delta_avg.append(weighted_delta)

            # Update moments and server parameters
            new_params = []
            for idx in range(len(global_params)):
                # m_t = beta1 * m_{t-1} + (1 - beta1) * Delta_t
                m[idx] = beta_1 * m[idx] + (1.0 - beta_1) * delta_avg[idx]
                # v_t = beta2 * v_{t-1} + (1 - beta2) * (Delta_t)^2
                v[idx] = beta_2 * v[idx] + (1.0 - beta_2) * (delta_avg[idx] ** 2)
                # w_{t+1} = w_t + eta * m_t / (sqrt(v_t) + tau)
                update_step = eta * (m[idx] / (np.sqrt(v[idx]) + tau))
                new_params.append(global_params[idx] + update_step)

            global_params = new_params

        # Evaluate progress
        eval_losses = []
        for client in clients:
            loss, _, _ = client.evaluate(global_params, {})
            eval_losses.append(loss)
        avg_loss = float(np.mean(eval_losses))
        print(f"Round {r}/{rounds} - Mean Validation Loss: {avg_loss:.6f}")

    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    state_dict = OrderedDict(
        (k, torch.tensor(val)) for k, val in zip(global_model.state_dict().keys(), global_params, strict=True)
    )
    global_model.load_state_dict(state_dict)
    torch.save(global_model.state_dict(), model_save_path)
    print(f"FedAdam Global model saved to {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Federated Learning Model for Traffic Prediction")
    parser.add_argument(
        "--topology", type=str, default="nsfnet", choices=["nsfnet", "geant", "abilene"], help="Topology name"
    )
    parser.add_argument(
        "--strategy", type=str, default="fedadam", choices=["fedavg", "fedadam"], help="FL strategy (fedavg or fedadam)"
    )
    parser.add_argument("--rounds", type=int, default=50, help="Number of FL rounds to train")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing preprocessed data")
    parser.add_argument("--model_save_path", type=str, default=None, help="Path to save the global model")

    args = parser.parse_args()

    # Determine default data_dir and model_save_path
    if args.data_dir is None:
        if args.topology == "nsfnet":
            data_dir = "data/preprocessed"
        else:
            data_dir = f"data/preprocessed_{args.topology}"
    else:
        data_dir = args.data_dir

    if args.model_save_path is None:
        if args.topology == "nsfnet" and args.strategy == "fedavg":
            model_save_path = "models/fl_global_model.pt"
        elif args.topology == "nsfnet":
            model_save_path = f"models/fl_global_model_{args.strategy}.pt"
        else:
            model_save_path = f"models/fl_{args.topology}_{args.strategy}.pt"
    else:
        model_save_path = args.model_save_path

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Determine MAX_F_IN dynamically across domain data
    max_f = 24
    for did in [0, 1, 2]:
        p = os.path.join(data_dir, f"domain_{did}_train_X.npy")
        if os.path.exists(p):
            max_f = max(max_f, np.load(p, mmap_mode="r").shape[2])
    MAX_F_IN = max_f
    print(
        f"Topology: {args.topology} | Strategy: {args.strategy} | MAX_F_IN: {MAX_F_IN} | Data: {data_dir} | Output: {model_save_path}"
    )

    def client_fn(cid: str) -> fl.client.Client:
        domain_id = int(cid)
        train_loader, val_loader, f_in = load_data(domain_id, data_dir=data_dir, batch_size=32, target_f_in=MAX_F_IN)
        model = create_model(f_in=f_in)
        return TrafficClient(
            domain_id=domain_id, model=model, train_loader=train_loader, val_loader=val_loader
        ).to_client()

    strategy = build_strategy(
        strategy_name=args.strategy,
        num_rounds=args.rounds,
        num_clients=3,
        model_save_path=model_save_path,
        f_in=MAX_F_IN,
    )

    try:
        # Check if ray / flower simulation engine is available
        import ray  # noqa: F401

        print(f"Starting Flower simulation for {args.rounds} rounds using {args.strategy}...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )
    except (ImportError, RuntimeError, OSError) as e:
        print(f"Flower simulation engine unavailable ({e}). Using direct synchronous {args.strategy} runner...")
        if args.strategy.lower() == "fedadam":
            run_local_fedadam(
                rounds=args.rounds,
                num_clients=3,
                data_dir=data_dir,
                model_save_path=model_save_path,
                max_f_in=MAX_F_IN,
            )
        else:
            run_local_fedavg(
                rounds=args.rounds,
                num_clients=3,
                data_dir=data_dir,
                model_save_path=model_save_path,
                max_f_in=MAX_F_IN,
            )

    print("FL Training complete!")


if __name__ == "__main__":
    main()
