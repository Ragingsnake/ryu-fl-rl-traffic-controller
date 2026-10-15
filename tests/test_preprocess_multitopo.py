from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "topo,domain_dir,expected_max_f_in",
    [
        ("nsfnet", "preprocessed", 24),
        ("geant", "preprocessed_geant", 38),
        ("abilene", "preprocessed_abilene", 16),
    ],
)
def test_preprocessed_tensors_invariants(topo, domain_dir, expected_max_f_in):
    data_dir = BASE_DIR / "data" / domain_dir
    assert data_dir.exists(), f"Preprocessed directory {data_dir} must exist"

    for d in [0, 1, 2]:
        train_x = np.load(data_dir / f"domain_{d}_train_X.npy")
        train_y = np.load(data_dir / f"domain_{d}_train_Y.npy")
        val_x = np.load(data_dir / f"domain_{d}_val_X.npy")
        val_y = np.load(data_dir / f"domain_{d}_val_Y.npy")
        test_x = np.load(data_dir / f"domain_{d}_test_X.npy")
        test_y = np.load(data_dir / f"domain_{d}_test_Y.npy")

        # Zero NaNs
        assert not np.isnan(train_x).any(), f"NaNs detected in {topo} domain {d} train_X"
        assert not np.isnan(train_y).any(), f"NaNs detected in {topo} domain {d} train_Y"

        # Temporal lookback window T=12, prediction horizon H=3
        assert train_x.shape[1] == 12, f"Lookback window must be 12, got {train_x.shape[1]}"
        assert train_y.shape[1] == 3, f"Prediction horizon must be 3, got {train_y.shape[1]}"

        # Feature dimension must be <= expected_max_f_in
        f_in = train_x.shape[2]
        assert f_in <= expected_max_f_in, f"f_in {f_in} exceeds max {expected_max_f_in} for {topo}"
        assert train_y.shape[2] == f_in

        # Utilization bounds
        assert np.all(train_x >= 0.0), "Link utilization must be non-negative"
