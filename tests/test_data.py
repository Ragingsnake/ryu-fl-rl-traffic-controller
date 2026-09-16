"""
Tests for the data pipeline: TM generation and preprocessing.
"""

import pytest
import numpy as np


def test_generate_base_tm():
    """Test that the gravity model generates a valid base TM."""
    from data.generate_tm import load_topology, build_graph, generate_base_tms
    
    topology = load_topology("data/nsfnet_topology.json")
    G = build_graph(topology)
    base_tm = generate_base_tms(G)
    
    assert base_tm.shape == (14, 14), f"Expected (14, 14), got {base_tm.shape}"
    assert np.all(base_tm >= 0), "Base TM values must be non-negative"
    # Diagonal should be zero (no self-traffic)
    assert np.all(np.diag(base_tm) == 0), "Diagonal entries must be zero"


def test_generate_temporal_tms():
    """Test temporal TM generation produces correct shape and non-negative values."""
    from data.generate_tm import load_topology, build_graph, generate_base_tms, generate_temporal_tms
    
    topology = load_topology("data/nsfnet_topology.json")
    G = build_graph(topology)
    base_tm = generate_base_tms(G)
    
    # Generate a small batch for speed
    tms = generate_temporal_tms(base_tm, alpha=1.0, num_timesteps=100)
    
    assert tms.shape == (100, 14, 14), f"Expected (100, 14, 14), got {tms.shape}"
    assert np.all(tms >= 0), "Traffic matrix values must be non-negative"


def test_create_windows():
    """Test sliding window creation for time series."""
    from data.preprocess import create_windows
    
    data = np.random.rand(100, 42).astype(np.float32)
    X, Y = create_windows(data, T=12, H=3)
    
    expected_samples = 100 - 12 - 3 + 1  # = 86
    assert X.shape == (expected_samples, 12, 42), f"X shape mismatch: {X.shape}"
    assert Y.shape == (expected_samples, 3, 42), f"Y shape mismatch: {Y.shape}"
    assert len(X) == len(Y), "X and Y must have same number of samples"
