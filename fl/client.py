import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

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
        
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the local model parameters to the given values."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
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

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
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

def load_data(domain_id: int, data_dir: str = "data/preprocessed", batch_size: int = 32, target_f_in: Optional[int] = None) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load data for a specific domain ID and return DataLoaders and F_in.
    Assumes data is in {data_dir}/domain_{id}_*.npy.
    If target_f_in is provided and the local f_in < target_f_in, zero-pad features.
    """
    x_train_path = os.path.join(data_dir, f"domain_{domain_id}_train_X.npy")
    y_train_path = os.path.join(data_dir, f"domain_{domain_id}_train_Y.npy")
    x_val_path = os.path.join(data_dir, f"domain_{domain_id}_val_X.npy")
    y_val_path = os.path.join(data_dir, f"domain_{domain_id}_val_Y.npy")
    
    try:
        X_train = np.load(x_train_path)
        Y_train = np.load(y_train_path)
        X_val = np.load(x_val_path)
        Y_val = np.load(y_val_path)
    except FileNotFoundError:
        print(f"Warning: Data for domain {domain_id} not found at {data_dir}. Using dummy data for testing.")
        # Default fallback feature dimension
        fallback_dim = target_f_in if target_f_in is not None else 24
        X_train = np.random.rand(100, 12, fallback_dim).astype(np.float32)
        Y_train = np.random.rand(100, 3, fallback_dim).astype(np.float32)
        X_val = np.random.rand(20, 12, fallback_dim).astype(np.float32)
        Y_val = np.random.rand(20, 3, fallback_dim).astype(np.float32)
        
    F_in = X_train.shape[2]
    
    if target_f_in is not None and F_in < target_f_in:
        pad_width = target_f_in - F_in
        X_train = np.pad(X_train, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        Y_train = np.pad(Y_train, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        X_val = np.pad(X_val, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        Y_val = np.pad(Y_val, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        F_in = target_f_in
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, F_in
