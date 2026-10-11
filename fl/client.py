import os
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class TrafficClient(fl.client.NumPyClient):
    """
    Flower Client for federated learning of traffic prediction.
    """

    def __init__(self, domain_id: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        self.domain_id = domain_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]):
        """Set the local model parameters to the given values."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Train the local model for a few epochs."""
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 5)
        lr = config.get("lr", 1e-3)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for X_batch, Y_batch in self.train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the local model on the validation set."""
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in self.val_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, Y_batch)
                total_loss += loss.item() * X_batch.size(0)

        if len(self.val_loader.dataset) == 0:
            return 0.0, 0, {"val_loss": 0.0}

        avg_loss = total_loss / len(self.val_loader.dataset)
        return float(avg_loss), len(self.val_loader.dataset), {"val_loss": float(avg_loss)}


# def load_data(
#     domain_id: int, data_dir: str = "data/preprocessed", batch_size: int = 32, target_f_in: int | None = None
# ) -> tuple[DataLoader, DataLoader, int]:
#     """
#     Load data for a specific domain ID and return DataLoaders and F_in.
#     Assumes data is in {data_dir}/domain_{id}_*.npy.
#     If target_f_in is provided and the local f_in < target_f_in, zero-pad features.
#     """
#     x_train_path = os.path.join(data_dir, f"domain_{domain_id}_train_X.npy")
#     y_train_path = os.path.join(data_dir, f"domain_{domain_id}_train_Y.npy")
#     x_val_path = os.path.join(data_dir, f"domain_{domain_id}_val_X.npy")
#     y_val_path = os.path.join(data_dir, f"domain_{domain_id}_val_Y.npy")

#     try:
#         X_train = np.load(x_train_path)
#         Y_train = np.load(y_train_path)
#         X_val = np.load(x_val_path)
#         Y_val = np.load(y_val_path)
#     except FileNotFoundError:
#         print(f"Warning: Data for domain {domain_id} not found at {data_dir}. Using dummy data for testing.")
#         # Default fallback feature dimension
#         fallback_dim = target_f_in if target_f_in is not None else 24
#         X_train = np.random.rand(100, 12, fallback_dim).astype(np.float32)
#         Y_train = np.random.rand(100, 3, fallback_dim).astype(np.float32)
#         X_val = np.random.rand(20, 12, fallback_dim).astype(np.float32)
#         Y_val = np.random.rand(20, 3, fallback_dim).astype(np.float32)

#     F_in = X_train.shape[2]

#     if target_f_in is not None and F_in < target_f_in:
#         pad_width = target_f_in - F_in
#         X_train = np.pad(X_train, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
#         Y_train = np.pad(Y_train, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
#         X_val = np.pad(X_val, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
#         Y_val = np.pad(Y_val, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
#         F_in = target_f_in

#     train_dataset = TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
#     )
#     val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, F_in

def load_data(
    domain_id: int,
    data_dir: str = "data/preprocessed",
    batch_size: int = 32,
    input_len: int = 12,
    pred_len: int = 3,
    num_clients: int = 3,
):
    """
    Load preprocessed NSFNET traffic data for one FL client.

    Dataset:
        NSFNET with 182 directed OD pairs.

    Input:
        X = [input_len, 182]

    Target:
        Y = [pred_len, 182]

    Clients:
        domain_1
        domain_2
        domain_3
    """

    # ========================================================
    # 1. Check client ID
    # ========================================================

    if domain_id < 0 or domain_id >= num_clients:
        raise ValueError(
            f"domain_id must be in 0..{num_clients - 1}, "
            f"but received {domain_id}"
        )

        # ========================================================
    # 2. File paths
    # ========================================================

    file_domain_id = domain_id + 1

    x_train_path = os.path.join(
        data_dir, f"domain_{file_domain_id}_train_X.npy"
    )
    y_train_path = os.path.join(
        data_dir, f"domain_{file_domain_id}_train_Y.npy"
    )
    x_val_path = os.path.join(
        data_dir, f"domain_{file_domain_id}_val_X.npy"
    )
    y_val_path = os.path.join(
        data_dir, f"domain_{file_domain_id}_val_Y.npy"
    )

    # ========================================================
    # 3. Check files
    # ========================================================

    required_files = [
        x_train_path,
        y_train_path,
        x_val_path,
        y_val_path,
    ]

    for path in required_files:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy file:\n{path}\n\n"
                f"Hãy chạy preprocess_nsfnet.py trước."
            )

    # ========================================================
    # 4. Load data
    # ========================================================

    X_train = np.load(x_train_path)
    Y_train = np.load(y_train_path)

    X_val = np.load(x_val_path)
    Y_val = np.load(y_val_path)

    # ========================================================
    # 5. Validate shapes
    # ========================================================

    expected_features = 182  # Number of directed OD pairs in NSFNET

    if X_train.ndim != 3:
        raise ValueError(
            f"X_train phải có 3 chiều "
            f"(N, input_len, 182), "
            f"nhưng nhận được {X_train.shape}"
        )

    if Y_train.ndim != 3:
        raise ValueError(
            f"Y_train phải có 3 chiều "
            f"(N, pred_len, 182), "
            f"nhưng nhận được {Y_train.shape}"
        )

    if X_train.shape[-1] != expected_features:
        raise ValueError(
            f"X_train phải có {expected_features} OD pairs, "
            f"nhưng nhận được {X_train.shape[-1]}"
        )

    if Y_train.shape[-1] != expected_features:
        raise ValueError(
            f"Y_train phải có {expected_features} OD pairs, "
            f"nhưng nhận được {Y_train.shape[-1]}"
        )

    if X_val.shape[-1] != expected_features:
        raise ValueError(
            f"X_val phải có {expected_features} OD pairs, "
            f"nhưng nhận được {X_val.shape[-1]}"
        )

    if Y_val.shape[-1] != expected_features:
        raise ValueError(
            f"Y_val phải có {expected_features} OD pairs, "
            f"nhưng nhận được {Y_val.shape[-1]}"
        )

    # ========================================================
    # 6. Convert to TensorDataset
    # ========================================================

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            Y_train,
            dtype=torch.float32
        )
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32
        ),
        torch.tensor(
            Y_val,
            dtype=torch.float32
        )
    )

    # ========================================================
    # 7. DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ========================================================
    # 8. Feature dimension
    # ========================================================

    F_in = expected_features

    print(
        f"Client {domain_id}: "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"X={X_train.shape}, "
        f"Y={Y_train.shape}, "
        f"F_in={F_in}"
    )

    return train_loader, val_loader, F_in


# def main():
#     import argparse

#     from model import create_model

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--domain_id", type=int, required=True)
#     parser.add_argument(
#         "--server_address",
#         type=str,
#         default="127.0.0.1:8080",
#     )
#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         default="data/preprocessed",
#     )
#     args = parser.parse_args()

#     print(f"[Client {args.domain_id}] Loading data...", flush=True)

#     train_loader, val_loader, f_in = load_data(
#         domain_id=args.domain_id,
#         data_dir=args.data_dir,
#         batch_size=32,
#         input_len=12,
#         pred_len=3,
#         num_clients=3,
#     )

#     print(f"[Client {args.domain_id}] Creating model...", flush=True)

#     model = create_model(f_in=f_in)

#     client = TrafficClient(
#         domain_id=args.domain_id,
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#     )

#     print(
#         f"[Client {args.domain_id}] Connecting to "
#         f"{args.server_address}...",
#         flush=True,
#     )

#     fl.client.start_client(
#         server_address=args.server_address,
#         client=client.to_client(),
#     )


# if __name__ == "__main__":
#     main()
