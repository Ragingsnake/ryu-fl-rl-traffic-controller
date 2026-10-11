
import torch
from torch import nn


class TrafficLSTM(nn.Module):
    """
    LSTM model for predicting future link utilization.

    Input:
        (batch_size, 12, 44)

    Output:
        (batch_size, 3, 44)

    Architecture:
        - 2 LSTM layers
        - hidden_size = 64
        - Dropout = 0.2
        - Fully connected layer
        - Predict 3 future time steps for 44 directed links
    """

    def __init__(
        self,
        f_in: int = 44,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        h: int = 3,
    ):
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

        self.fc = nn.Linear(
            hidden_size,
            h * f_in,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, T, F_in)

        Returns:
            (batch_size, H, F_in)
        """
        lstm_out, _ = self.lstm(x)

        # Output at the final input time step
        last_out = lstm_out[:, -1, :]

        # Predict H * F_in values
        out = self.fc(last_out)

        # Reshape to (batch_size, H, F_in)
        out = out.reshape(-1, self.h, self.f_in)

        return out


def create_model(f_in: int = 44) -> nn.Module:
    """
    Create the TrafficLSTM model.

    Args:
        f_in: Number of link-utilization features.
              For the current NSFNET preprocessing, f_in = 44.

    Returns:
        TrafficLSTM model.
    """
    return TrafficLSTM(f_in=f_in)


def predict(
    model: nn.Module,
    history_buffer: torch.Tensor,
) -> torch.Tensor:
    """
    Predict future link utilization.

    Args:
        history_buffer:
            (T, F_in) or (batch_size, T, F_in)

    Returns:
        (H, F_in) if input is 2D;
        (batch_size, H, F_in) if input is 3D.
    """
    model.eval()

    with torch.no_grad():
        if history_buffer.dim() == 2:
            history_buffer = history_buffer.unsqueeze(0)
            return model(history_buffer).squeeze(0)

        return model(history_buffer)
