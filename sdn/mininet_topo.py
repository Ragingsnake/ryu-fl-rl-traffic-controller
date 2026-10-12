"""
Mininet Topology for NSFNET (14 nodes).
Matches capacity and delays from data/nsfnet_topology.json.
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from functools import partial
import json
import os

class NSFNETTopo(Topo):
    def build(self, topo_path="../data/nsfnet_topology.json"):
        with open(topo_path) as f:
            topo_data = json.load(f)

        nodes = {}
        for node in topo_data["nodes"]:
            # Add switch
            sw = self.addSwitch('s{}'.format(node["id"]))
            # Add one host per switch for traffic generation
            host = self.addHost('h{}'.format(node["id"]), ip='10.0.0.{}/24'.format(node["id"] + 1))
            self.addLink(host, sw, bw=1000) # Fast link to host
            nodes[node["id"]] = sw

        for link in topo_data["links"]:
            u = link["source"]
            v = link["target"]
            # capacity in bps -> Mbps
            bw_mbps = link["capacity_bps"] / 1e6
            delay_ms = link["delay_ms"]
            self.addLink(nodes[u], nodes[v], bw=bw_mbps, delay='{}ms'.format(delay_ms))

def run():
    topo_path = os.path.join(os.path.dirname(__file__), "../data/nsfnet_topology.json")
    topo = NSFNETTopo(topo_path=topo_path)
    Switch = partial(OVSSwitch, datapath='user')
    net = Mininet(
        topo=topo,
        controller=RemoteController('c0', ip='127.0.0.1', port=6653),
        switch=Switch,
        link=TCLink
    )
    
    net.start()
    info("*** Started Mininet with NSFNET Topology\n")
    
    info("*** Testing baseline L2 connectivity with pingall...\n")
    net.pingAll()
    
    # Optional: run some simple tests or drop into CLI
    if os.environ.get("HEADLESS"):
        info("*** Headless mode active: keeping network alive...\n")
        import time
        while True:
            time.sleep(1)
    else:
        CLI(net)
        net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
