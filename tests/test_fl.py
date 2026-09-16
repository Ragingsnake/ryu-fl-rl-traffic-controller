"""
Tests for the Federated Learning model (LSTM traffic predictor).
"""

import torch

from fl.model import create_model, predict


def test_fl_model_forward_pass():
    """Test that the LSTM model produces correct output shape."""
    F_in = 7  # Typical domain link count
    model = create_model(f_in=F_in)

    batch_size = 32
    T = 12  # Lookback window
    x = torch.rand(batch_size, T, F_in)

    out = model(x)
    assert out.shape == (batch_size, 3, F_in), f"Expected output shape {(batch_size, 3, F_in)}, got {out.shape}"


def test_fl_model_parameter_count():
    """Test that model parameter count is approximately ~37k (CPU-friendly)."""
    F_in = 7
    model = create_model(f_in=F_in)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 30000 < total_params < 60000, f"Expected params around 37k-55k, got {total_params}"


def test_fl_model_predict_utility():
    """Test the predict() utility function works for single and batch inputs."""
    F_in = 7
    model = create_model(f_in=F_in)

    # Single input (no batch dim)
    history = torch.rand(12, F_in)
    pred = predict(model, history)
    assert pred.shape == (3, F_in), f"Single predict shape: {pred.shape}"

    # Batched input
    history_batch = torch.rand(4, 12, F_in)
    pred_batch = predict(model, history_batch)
    assert pred_batch.shape == (4, 3, F_in), f"Batch predict shape: {pred_batch.shape}"


def test_fl_model_different_domains():
    """Test that model works for different domain input sizes."""
    for f_in in [5, 6, 7, 8]:  # Different domain link counts
        model = create_model(f_in=f_in)
        x = torch.rand(8, 12, f_in)
        out = model(x)
        assert out.shape == (8, 3, f_in), f"F_in={f_in}: got shape {out.shape}"
