"""
Unit tests for multi-topology specifications: NSFNET, GEANT, and Abilene.
Verifies node counts, directed edge counts, connectivity, and domain partitioning.
"""

import json

import networkx as nx


def load_and_build(topo_path: str) -> tuple[dict, nx.DiGraph]:
    with open(topo_path) as f:
        data = json.load(f)
    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(n["id"], **n)
    for link in data["links"]:
        G.add_edge(link["source"], link["target"], capacity=link["capacity_bps"], delay=link["delay_ms"])
        G.add_edge(link["target"], link["source"], capacity=link["capacity_bps"], delay=link["delay_ms"])
    return data, G


def test_nsfnet_topology():
    data, G = load_and_build("data/nsfnet_topology.json")
    assert len(data["nodes"]) == 14
    assert len(data["links"]) in [21, 22]
    assert G.number_of_nodes() == 14
    assert G.number_of_edges() in [42, 44]
    assert nx.is_strongly_connected(G)
    domains = {n["domain"] for n in data["nodes"]}
    assert len(domains) == 3


def test_geant_topology():
    data, G = load_and_build("data/geant_topology.json")
    assert len(data["nodes"]) == 22, "GEANT must have exactly 22 nodes"
    assert len(data["links"]) == 36, "GEANT must have 36 bidirectional links"
    assert G.number_of_nodes() == 22
    assert G.number_of_edges() == 72, "GEANT must have 72 directed links"
    assert nx.is_strongly_connected(G), "GEANT graph must be strongly connected"

    domain_counts = {}
    for n in data["nodes"]:
        d = n["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    assert len(domain_counts) == 3, "GEANT must have 3 domains"
    assert domain_counts["A"] == 7
    assert domain_counts["B"] == 8
    assert domain_counts["C"] == 7


def test_abilene_topology():
    data, G = load_and_build("data/abilene_topology.json")
    assert len(data["nodes"]) == 12, "Abilene must have exactly 12 nodes"
    assert len(data["links"]) == 15, "Abilene must have 15 bidirectional links"
    assert G.number_of_nodes() == 12
    assert G.number_of_edges() == 30, "Abilene must have 30 directed links"
    assert nx.is_strongly_connected(G), "Abilene graph must be strongly connected"

    domains = {n["domain"] for n in data["nodes"]}
    assert len(domains) == 3, "Abilene must have 3 domains"
