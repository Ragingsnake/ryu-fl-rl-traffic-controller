"""
Shared pytest fixtures for the FL+RL Traffic Engineering prototype tests.
Provides topology data, small test TMs, and env instances for testing.
"""

import json

import numpy as np
import pytest


# --- Topology fixture ---
@pytest.fixture
def dummy_topology_data():
    """Full NSFNET topology matching the actual nsfnet_topology.json structure."""
    return {
        "nodes": [
            {"id": 0, "label": "WA", "city": "Seattle", "domain": "A", "population_weight": 4.0},
            {"id": 1, "label": "CA1", "city": "Palo Alto", "domain": "A", "population_weight": 7.0},
            {"id": 2, "label": "CA2", "city": "San Diego", "domain": "A", "population_weight": 3.3},
            {"id": 3, "label": "NE", "city": "Lincoln", "domain": "B", "population_weight": 1.3},
            {"id": 4, "label": "IL", "city": "Champaign", "domain": "B", "population_weight": 3.0},
            {"id": 5, "label": "MI", "city": "Ann Arbor", "domain": "B", "population_weight": 4.3},
            {"id": 6, "label": "GA", "city": "Atlanta", "domain": "C", "population_weight": 6.1},
            {"id": 7, "label": "TX", "city": "Houston", "domain": "A", "population_weight": 7.1},
            {"id": 8, "label": "UT", "city": "Salt Lake City", "domain": "A", "population_weight": 1.2},
            {"id": 9, "label": "CO", "city": "Boulder", "domain": "B", "population_weight": 2.9},
            {"id": 10, "label": "PA", "city": "Pittsburgh", "domain": "C", "population_weight": 2.3},
            {"id": 11, "label": "NY", "city": "Ithaca", "domain": "C", "population_weight": 1.0},
            {"id": 12, "label": "NJ", "city": "Princeton", "domain": "C", "population_weight": 20.0},
            {"id": 13, "label": "DC", "city": "College Park", "domain": "C", "population_weight": 6.0},
        ],
        "links": [
            {"source": 0, "target": 1, "capacity_bps": 10000000, "delay_ms": 4},
            {"source": 0, "target": 2, "capacity_bps": 10000000, "delay_ms": 6},
            {"source": 0, "target": 7, "capacity_bps": 10000000, "delay_ms": 10},
            {"source": 1, "target": 2, "capacity_bps": 10000000, "delay_ms": 3},
            {"source": 1, "target": 3, "capacity_bps": 10000000, "delay_ms": 8},
            {"source": 2, "target": 5, "capacity_bps": 10000000, "delay_ms": 10},
            {"source": 3, "target": 4, "capacity_bps": 10000000, "delay_ms": 3},
            {"source": 3, "target": 8, "capacity_bps": 10000000, "delay_ms": 4},
            {"source": 4, "target": 5, "capacity_bps": 10000000, "delay_ms": 2},
            {"source": 4, "target": 6, "capacity_bps": 10000000, "delay_ms": 3},
            {"source": 5, "target": 10, "capacity_bps": 10000000, "delay_ms": 2},
            {"source": 5, "target": 13, "capacity_bps": 10000000, "delay_ms": 3},
            {"source": 6, "target": 7, "capacity_bps": 10000000, "delay_ms": 4},
            {"source": 7, "target": 9, "capacity_bps": 10000000, "delay_ms": 5},
            {"source": 8, "target": 9, "capacity_bps": 10000000, "delay_ms": 2},
            {"source": 8, "target": 11, "capacity_bps": 10000000, "delay_ms": 9},
            {"source": 8, "target": 12, "capacity_bps": 10000000, "delay_ms": 9},
            {"source": 9, "target": 10, "capacity_bps": 10000000, "delay_ms": 7},
            {"source": 10, "target": 11, "capacity_bps": 10000000, "delay_ms": 2},
            {"source": 10, "target": 12, "capacity_bps": 10000000, "delay_ms": 2},
            {"source": 11, "target": 13, "capacity_bps": 10000000, "delay_ms": 3},
            {"source": 12, "target": 13, "capacity_bps": 10000000, "delay_ms": 2},
        ],
    }


@pytest.fixture
def small_tm_data():
    """Small traffic matrix dataset for quick tests (200 timesteps, 14 nodes)."""
    np.random.seed(123)
    return np.random.rand(200, 14, 14).astype(np.float32) * 0.5


@pytest.fixture
def tmp_test_files(tmp_path, dummy_topology_data, small_tm_data):
    """
    Create temporary topology JSON and TM .npy files for env testing.
    Returns (topo_path, tm_path) as strings.
    """
    topo_path = tmp_path / "nsfnet_topology.json"
    tm_path = tmp_path / "traffic_matrices.npy"

    with open(topo_path, "w") as f:
        json.dump(dummy_topology_data, f)

    np.save(str(tm_path), small_tm_data)

    return str(topo_path), str(tm_path)


@pytest.fixture
def dummy_env(tmp_test_files):
    """Create a real NSFNETRoutingEnv instance using temporary test data."""
    topo_path, tm_path = tmp_test_files
    from rl.env import NSFNETRoutingEnv

    return NSFNETRoutingEnv(topo_path=topo_path, tm_path=tm_path)
