#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSFNET Mininet topology for the SDN/FL/RL KLTN prototype.

Topology: 14 OpenFlow 1.3 switches, 14 hosts, and 27 inter-switch links.
Controller: Ryu at 127.0.0.1:6653.
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info


class NSFNETTopo(Topo):
    """14-switch NSFNET-inspired topology with one host per switch."""

    def build(self):
        info('*** Creating 14 core switches...\n')
        switches = {
            i: self.addSwitch(f's{i}', protocols='OpenFlow13')
            for i in range(1, 15)
        }

        info('*** Creating 14 hosts and host-access links...\n')
        for i in range(1, 15):
            host = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            self.addLink(
                host,
                switches[i],
                cls=TCLink,
                bw=100,
                delay='1ms',
                loss=0,
            )

        # This list contains 27 links. Keep the controller topology discovery
        # enabled so it learns the actual switch ports from LLDP.
        nsfnet_links = [
            # West region
            (1, 2), (1, 4), (1, 3), (2, 3), (2, 5), (3, 5), (3, 4), (3, 8),
            # Central region
            (4, 7), (4, 10), (5, 6), (5, 7), (5, 9), (6, 9), (7, 8),
            (7, 10), (8, 9),
            # East region
            (9, 10), (9, 12), (9, 11), (10, 11), (10, 12), (10, 13),
            (11, 12), (11, 14), (12, 13), (13, 14),
        ]

        info('*** Creating inter-switch links...\n')
        for src, dst in nsfnet_links:
            self.addLink(
                switches[src],
                switches[dst],
                cls=TCLink,
                bw=100,
                delay='5ms',
                loss=0,
            )

        info(f'*** Created {len(switches)} switches, 14 hosts, '
             f'{len(nsfnet_links)} core links.\n')


def run():
    topo = NSFNETTopo()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=False,
        build=False,
    )
    net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6653,
    )

    try:
        info('*** Starting NSFNET Mininet network...\n')
        net.build()
        net.start()
        info('\n*** NSFNET network started.\n')
        info('*** Switches: 14 | Hosts: 14 | Core links: 27 | OpenFlow: 1.3\n')
        info('*** Wait for Ryu log "Topology refreshed ... links=27" before pingall.\n')
        info('*** Useful commands: nodes, links, net, pingall, h1 ping -c 3 10.0.0.2\n')
        CLI(net)
    finally:
        info('*** Stopping NSFNET network...\n')
        net.stop()
        info('*** Network stopped.\n')


if __name__ == '__main__':
    setLogLevel('info')
    run()


# Mininet topology discovery is dynamic; no static routes are configured here.
topos = {'nsfnet': (lambda: NSFNETTopo())}
