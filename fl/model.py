import torch
from torch import nn


class TrafficLSTM(nn.Module):
    """
    LSTM model for predicting future link utilization.

    Architecture:
    - 2 LSTM layers, hidden_size=64, batch_first=True
    - Dropout 0.2
    - Fully connected layer outputting (H * F_in)
    - Reshape to (H, F_in)
    """

    def __init__(self, f_in: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, h: int = 3):
        super().__init__()
        self.f_in = f_in
        self.h = h
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=f_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, h * f_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TrafficLSTM.
        Args:
            x: Tensor of shape (batch_size, T, F_in)
        Returns:
            Tensor of shape (batch_size, H, F_in)
        """
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_out)  # (batch_size, H * F_in)
        out = out.view(-1, self.h, self.f_in)  # (batch_size, H, F_in)
        return out


def create_model(f_in: int) -> nn.Module:
    """
    Helper function to create the TrafficLSTM model for a given input dimension.

    Args:
        f_in: Number of input features (links).

    Returns:
        Instantiated TrafficLSTM model.
    """
    return TrafficLSTM(f_in=f_in)


def predict(model: nn.Module, history_buffer: torch.Tensor) -> torch.Tensor:
    """
    Utility function for inference on a given history buffer.

    Args:
        model: Trained TrafficLSTM model.
        history_buffer: Tensor of shape (1, T, F_in) or (T, F_in) containing past utilization.

    Returns:
        Predicted utilization of shape (H, F_in) or (1, H, F_in).
    """
    model.eval()
    with torch.no_grad():
        if history_buffer.dim() == 2:
            # Add batch dimension if missing
            history_buffer = history_buffer.unsqueeze(0)
            return model(history_buffer).squeeze(0)
        return model(history_buffer)
