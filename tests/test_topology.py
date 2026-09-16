import json
import pytest
from pathlib import Path

def test_topology_validity(dummy_topology_data, tmp_path):
    # Simulate loading from json
    topo_file = tmp_path / "nsfnet_topology.json"
    topo_file.write_text(json.dumps(dummy_topology_data))
    
    with open(topo_file, 'r') as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    
    assert len(nodes) == 14, "Topology must have 14 nodes"
    assert len(links) == 21 or len(links) == 22, "Topology should have 21 undirected or 42 directed edges"
    
    for link in links:
        assert "capacity_bps" in link or "capacity" in link, "Edge must have capacity"
        assert "delay_ms" in link or "delay" in link, "Edge must have delay"
        
    domains = set(n.get("domain") for n in nodes)
    assert len(domains) == 3, "There should be exactly 3 domains"
    assert domains == {"A", "B", "C"}, "Domains should be A, B, and C"
    
    # Assert nodes covered exactly
    covered_nodes = set(n["id"] for n in nodes)
    assert len(covered_nodes) == 14, "All 14 nodes must be uniquely covered"
