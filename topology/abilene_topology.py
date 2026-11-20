#!/usr/bin/env python3
"""
11-router Abilene Backbone topology for Mininet + Ryu/OpenFlow 1.3.

Topology:
- 11 OVS logical routers: r0-r10
- 14 bidirectional inter-router links
- 11 end hosts: h0-h10 (each attached to its corresponding router)

End hosts:
- h{i}: 10.0.{i}.2/24, gateway 10.0.{i}.1

Ryu Controller:
- Default IP: 127.0.0.1
- Default port: 6633
"""

import argparse

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info


# ============================================================
# 1. ABILENE ROUTERS (11 NODES: r0 -> r10)
# ============================================================

ROUTERS = {
    0: "Seattle",
    1: "Sunnyvale",
    2: "Los Angeles",
    3: "Denver",
    4: "Kansas City",
    5: "Houston",
    6: "Chicago",
    7: "Indianapolis",
    8: "Atlanta",
    9: "New York",
    10: "Washington",
}


# ============================================================
# 2. ABILENE PHYSICAL LINKS (14 LINKS)
# ============================================================

LINKS = [
    (0, 1),   # Seattle - Sunnyvale (55)
    (0, 3),   # Seattle - Denver (2095 / 2100)
    (1, 2),   # Sunnyvale - Los Angeles (366)
    (1, 3),   # Sunnyvale - Denver (1295)
    (2, 5),   # Los Angeles - Houston (1893 / 1898)
    (3, 4),   # Denver - Kansas City (639 / 644)
    (4, 5),   # Kansas City - Houston (902)
    (4, 7),   # Kansas City - Indianapolis (548)
    (5, 8),   # Houston - Atlanta (1176 / 1175)
    (6, 7),   # Chicago - Indianapolis (260)
    (6, 9),   # Chicago - New York (700)
    (7, 8),   # Indianapolis - Atlanta (587)
    (8, 10),  # Atlanta - Washington (846)
    (9, 10),  # New York - Washington (253)
]


# ============================================================
# 3. END HOST CONFIGURATION
# ============================================================

HOSTS = {
    f"h{i}": {
        "ip": f"10.0.{i}.2/24",
        "mac": f"00:00:00:00:00:{i + 1:02x}",  # 00:00:00:00:00:01 -> 00:00:00:00:00:0b
        "gateway": f"10.0.{i}.1",
        "router": i,
    }
    for i in range(len(ROUTERS))
}


# ============================================================
# 4. ABILENE TOPOLOGY
# ============================================================

class AbileneTopo(Topo):
    """
    11 logical routers implemented by Open vSwitch datapaths.

    Inter-router links are created before host links so that
    router interconnection ports remain consistently numbered.
    """

    def build(self):
        routers = {}

        # Create 11 OVS routers (dpid: 1 -> 11)
        for router_id in ROUTERS:
            routers[router_id] = self.addSwitch(
                f"r{router_id}",
                dpid=f"{router_id + 1:016x}",
                protocols="OpenFlow13",
                failMode="secure",
            )

        # Create 14 bidirectional inter-router links
        for src, dst in LINKS:
            self.addLink(
                routers[src],
                routers[dst],
            )

        # Create and connect all 11 end hosts
        for host_name, config in HOSTS.items():
            host = self.addHost(
                host_name,
                ip=config["ip"],
                mac=config["mac"],
            )
            self.addLink(
                host,
                routers[config["router"]],
            )


# ============================================================
# 5. START MININET
# ============================================================

def run(controller_ip="127.0.0.1", controller_port=6633):
    topo = AbileneTopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        autoSetMacs=False,
        autoStaticArp=False,
        build=True,
    )

    # Add Ryu Remote Controller
    net.addController(
        "c0",
        controller=RemoteController,
        ip=controller_ip,
        port=int(controller_port),
    )

    info("\n*** Starting Abilene topology\n")
    info(f"*** Routers: {len(ROUTERS)}\n")
    info(f"*** Inter-router links: {len(LINKS)}\n")
    info(f"*** End hosts: {len(HOSTS)}\n")

    # Start switches, hosts and controller connections
    net.start()

    # Configure default gateways on all 11 end hosts
    info("*** Configuring host default gateways\n")
    for host_name, config in HOSTS.items():
        net.get(host_name).cmd(f"ip route replace default via {config['gateway']}")

    info("*** Default gateways configured for all hosts (h0 - h10)\n")
    info(f"*** Ryu Controller: {controller_ip}:{controller_port}\n")

    info("\n*** Abilene topology started successfully\n")
    info("*** Available commands:\n")
    info("*** nodes | net | links | dump\n")
    info("*** h0 ip addr | h10 ip addr\n")
    info("*** h0 ip route | h10 ip route\n")

    # Enter Mininet CLI
    CLI(net)

    # Stop topology after exiting CLI
    info("*** Stopping Abilene topology\n")
    net.stop()


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="11-router Abilene Backbone topology."
    )

    parser.add_argument(
        "--controller",
        default="127.0.0.1",
        help="Ryu Controller IP address",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6633,
        help="Ryu OpenFlow port",
    )

    args = parser.parse_args()

    setLogLevel("info")

    run(
        controller_ip=args.controller,
        controller_port=args.port,
    )
