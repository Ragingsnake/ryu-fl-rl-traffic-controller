import json
from pathlib import Path

import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent


def test_nsfnet_topology_invariants():
    topo_path = BASE_DIR / "data" / "nsfnet_topology.json"
    assert topo_path.exists(), "nsfnet_topology.json must exist"
    with open(topo_path) as f:
        data = json.load(f)

    assert len(data["nodes"]) == 14
    G = nx.Graph()
    for link in data["links"]:
        G.add_edge(link["source"], link["target"])
    assert nx.is_connected(G), "NSFNET graph must be connected"
    assert G.number_of_edges() == 21 or G.number_of_edges() == 22


def test_geant_topology_invariants():
    topo_path = BASE_DIR / "data" / "geant_topology.json"
    assert topo_path.exists(), "geant_topology.json must exist"
    with open(topo_path) as f:
        data = json.load(f)

    # |V| = 22
    assert len(data["nodes"]) == 22
    # 3 domains covering all nodes
    domains = {"A": set(), "B": set(), "C": set()}
    for node in data["nodes"]:
        domains[str(node["domain"])].add(node["id"])
    assert sum(len(s) for s in domains.values()) == 22

    # Graph connectivity
    G = nx.DiGraph()
    for link in data["links"]:
        u, v = link["source"], link["target"]
        assert link.get("capacity_bps", 0) > 0, "Capacity must be positive"
        assert link.get("delay_ms", 0) > 0, "Delay must be positive"
        G.add_edge(u, v)
        G.add_edge(v, u)

    assert nx.is_strongly_connected(G), "GEANT bidirectional graph must be strongly connected"
    # 36 undirected links -> 72 directed links
    assert G.number_of_edges() == 72


def test_abilene_topology_invariants():
    topo_path = BASE_DIR / "data" / "abilene_topology.json"
    assert topo_path.exists(), "abilene_topology.json must exist"
    with open(topo_path) as f:
        data = json.load(f)

    # |V| = 12
    assert len(data["nodes"]) == 12
    labels = {n.get("label", n.get("name")) for n in data["nodes"]}
    assert len(labels) == 12

    # 3 domains covering all 12 nodes (4 + 4 + 4)
    domains = {"A": set(), "B": set(), "C": set()}
    for node in data["nodes"]:
        domains[str(node["domain"])].add(node["id"])
    assert sorted([len(s) for s in domains.values()]) == [4, 4, 4]

    # Graph connectivity
    G = nx.DiGraph()
    for link in data["links"]:
        u, v = link["source"], link["target"]
        assert link.get("capacity_bps", 0) > 0, "Capacity must be positive"
        assert link.get("delay_ms", 0) > 0, "Delay must be positive"
        G.add_edge(u, v)
        G.add_edge(v, u)

    assert nx.is_strongly_connected(G), "Abilene bidirectional graph must be strongly connected"
    # 15 undirected links -> 30 directed links
    assert G.number_of_edges() == 30
