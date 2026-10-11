import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
import numpy as np

from fl.client import TrafficClient, load_data
from fl.model import create_model
from fl.server import build_strategy


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 3

# NSFNET:
# 14 nodes -> 14 * 13 = 182 directed OD pairs
F_IN = 182

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 1e-3


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train Federated Learning Model for NSFNET Traffic Prediction"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of FL rounds to train"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/preprocessed",
        help="Directory containing preprocessed NSFNET data"
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/fl_global_model.pt",
        help="Path to save the global model"
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.model_save_path),
        exist_ok=True
    )

    print("=" * 60)
    print("FEDERATED LEARNING - NSFNET TRAFFIC PREDICTION")
    print("=" * 60)

    print(f"Number of FL clients : {NUM_CLIENTS}")
    print(f"Number of OD pairs   : {F_IN}")
    print(f"FL rounds            : {args.rounds}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Local epochs         : {LOCAL_EPOCHS}")
    print(f"Learning rate        : {LEARNING_RATE}")

    # ========================================================
    # CHECK DATA
    # ========================================================

    print("\nChecking client datasets...")

    for domain_id in range(1, NUM_CLIENTS + 1):

        train_path = os.path.join(
            args.data_dir,
            f"domain_{domain_id}_train_X.npy"
        )

        val_path = os.path.join(
            args.data_dir,
            f"domain_{domain_id}_val_X.npy"
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Training data not found:\n{train_path}"
            )

        if not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Validation data not found:\n{val_path}"
            )

        train_x = np.load(
            train_path,
            mmap_mode="r"
        )

        val_x = np.load(
            val_path,
            mmap_mode="r"
        )

        print(
            f"Client {domain_id}: "
            f"train={train_x.shape}, "
            f"val={val_x.shape}"
        )

        if train_x.shape[-1] != F_IN:
            raise ValueError(
                f"Client {domain_id} has "
                f"{train_x.shape[-1]} features. "
                f"Expected {F_IN}."
            )

    print("\nDataset check completed.")

    # ========================================================
    # FLOWER CLIENT FUNCTION
    # ========================================================

    def client_fn(cid: str) -> fl.client.Client:

        # Client IDs are 1, 2, 3
        domain_id = int(cid) + 1

        train_loader, val_loader, f_in = load_data(
            domain_id,
            data_dir=args.data_dir,
            batch_size=BATCH_SIZE,
            target_f_in=F_IN
        )

        if f_in != F_IN:
            raise ValueError(
                f"Client {domain_id}: "
                f"f_in={f_in}, expected {F_IN}"
            )

        model = create_model(
            f_in=F_IN
        )

        return TrafficClient(
            domain_id=domain_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        ).to_client()

    # ========================================================
    # FEDERATED STRATEGY
    # ========================================================

    strategy = build_strategy(
        num_rounds=args.rounds,
        num_clients=NUM_CLIENTS,
        model_save_path=args.model_save_path,
        f_in=F_IN
    )

    # ========================================================
    # FLOWER SIMULATION
    # ========================================================

    try:

        import ray  # noqa: F401

        print(
            f"\nStarting Flower simulation "
            f"for {args.rounds} rounds..."
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(
                num_rounds=args.rounds
            ),
            strategy=strategy,
            client_resources={
                "num_cpus": 1,
                "num_gpus": 0
            }
        )

    except (ImportError, RuntimeError, OSError) as e:

        print(
            "\nFlower/Ray simulation unavailable."
        )

        print(
            f"Reason: {e}"
        )

        print(
            "\nTraining using synchronous fallback..."
        )

        run_local_fedavg(
            rounds=args.rounds,
            num_clients=NUM_CLIENTS,
            data_dir=args.data_dir,
            model_save_path=args.model_save_path,
            f_in=F_IN
        )

    print("\nFL Training complete!")


# ============================================================
# SYNCHRONOUS FALLBACK
# ============================================================

def run_local_fedavg(
    rounds: int,
    num_clients: int,
    data_dir: str,
    model_save_path: str,
    f_in: int,
) -> None:

    """
    Direct synchronous FL fallback.

    NOTE:
    This fallback uses FedAvg.
    For the final KLTN experiment, the preferred
    implementation should use the configured FedAdam
    strategy in fl.server.
    """

    from collections import OrderedDict

    import torch

    print(
        f"\nRunning synchronous FedAvg "
        f"for {rounds} rounds "
        f"across {num_clients} clients..."
    )

    # --------------------------------------------------------
    # Global model
    # --------------------------------------------------------

    global_model = create_model(
        f_in=f_in
    )

    global_params = [
        val.cpu().numpy()
        for val in global_model.state_dict().values()
    ]

    # --------------------------------------------------------
    # Create clients
    # --------------------------------------------------------

    clients = []

    for domain_id in range(
        1,
        num_clients + 1
    ):

        train_loader, val_loader, client_f_in = load_data(
            domain_id,
            data_dir=data_dir,
            batch_size=BATCH_SIZE,
            target_f_in=f_in
        )

        if client_f_in != f_in:
            raise ValueError(
                f"Client {domain_id}: "
                f"f_in={client_f_in}, "
                f"expected {f_in}"
            )

        model = create_model(
            f_in=f_in
        )

        client = TrafficClient(
            domain_id=domain_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )

        clients.append(client)

    # --------------------------------------------------------
    # Federated training
    # --------------------------------------------------------

    for r in range(
        1,
        rounds + 1
    ):

        client_weights = []

        total_samples = 0

        # ----------------------------------------------------
        # Local training
        # ----------------------------------------------------

        for client in clients:

            weights, num_samples, _ = client.fit(
                global_params,
                {
                    "local_epochs": LOCAL_EPOCHS,
                    "lr": LEARNING_RATE,
                    "round": r
                }
            )

            client_weights.append(
                (
                    weights,
                    num_samples
                )
            )

            total_samples += num_samples

        # ----------------------------------------------------
        # FedAvg aggregation
        # ----------------------------------------------------

        if total_samples > 0:

            new_params = []

            for idx in range(
                len(global_params)
            ):

                weighted = sum(
                    w[idx]
                    * (n / total_samples)
                    for w, n in client_weights
                )

                new_params.append(
                    weighted
                )

            global_params = new_params

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        eval_losses = []

        for client in clients:

            loss, _, _ = client.evaluate(
                global_params,
                {}
            )

            eval_losses.append(
                loss
            )

        avg_loss = float(
            np.mean(eval_losses)
        )

        print(
            f"Round {r}/{rounds} "
            f"- Mean Validation Loss: "
            f"{avg_loss:.6f}"
        )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(model_save_path),
        exist_ok=True
    )

    state_dict = OrderedDict(
        (
            k,
            torch.tensor(v)
        )
        for k, v in zip(
            global_model.state_dict().keys(),
            global_params,
            strict=True
        )
    )

    global_model.load_state_dict(
        state_dict
    )

    torch.save(
        global_model.state_dict(),
        model_save_path
    )

    print(
        f"\nFL Global model saved to:\n"
        f"{model_save_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()