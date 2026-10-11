#!/usr/bin/env python3

"""
NSFNET Topology for Mininet
===========================

KLTN:
Traffic Prediction and Proactive Routing for Internet Networks
Based on SDN, Federated Learning, and Reinforcement Learning

Topology:
    - 14 core switches (s1 - s14)
    - 14 hosts (h1 - h14)
    - 22 core links
    - 1 host connected to each core switch
    - OpenFlow 1.3
    - Remote Ryu Controller
    - Open vSwitch
    - TCLink for bandwidth / delay / loss configuration

Controller:
    IP   : 127.0.0.1
    Port : 6653
"""

# ==============================================================
# 1. Import libraries
# ==============================================================

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.clean import cleanup


# ==============================================================
# 2. NSFNET Topology
# ==============================================================

class NSFNETTopo(Topo):
    """
    NSFNET topology.

    14 core switches:
        s1 - s14

    14 hosts:
        h1 - h14

    22 core links.

    Each host is connected to its corresponding core switch.
    """

    def build(self):

        # ======================================================
        # 2.1 Create 14 core switches
        # ======================================================

        info('*** Creating NSFNET core switches...\n')

        switches = {}

        for i in range(1, 15):

            switches[i] = self.addSwitch(
                f's{i}',
                protocols='OpenFlow13'
            )

        info('*** Created 14 core switches.\n')


        # ======================================================
        # 2.2 Create one host for each core switch
        # ======================================================

        info('*** Creating NSFNET hosts...\n')

        for i in range(1, 15):

            host = self.addHost(
            f'h{i}',
            ip=f'10.0.0.{i}/24'
        )

            self.addLink(
                host,
                switches[i],

                # Host-to-switch bandwidth
                bw=1000,

                # Host-to-switch delay
                delay='1ms',

                # No packet loss
                loss=0
            )

        info('*** Created 14 hosts.\n')


        # ======================================================
        # 2.3 NSFNET core topology
        # ======================================================
        #
        # 14 nodes
        # 22 core links
        #
        # Each tuple represents:
        #
        #       (source switch, destination switch)
        #
        # ======================================================

        nsfnet_links = [

            # --------------------------------------------------
            # North / Upper part
            # --------------------------------------------------

            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 6),

            # --------------------------------------------------
            # Central part
            # --------------------------------------------------

            (4, 5),
            (4, 11),
            (5, 6),
            (5, 7),
            (5, 9),
            (6, 9),

            # --------------------------------------------------
            # Lower / Middle part
            # --------------------------------------------------

            (7, 8),
            (7, 10),
            (8, 9),
            (8, 10),
            (9, 10),

            # --------------------------------------------------
            # East / Lower part
            # --------------------------------------------------

            (9, 12),
            (10, 13),
            (11, 12),
            (11, 14),
            (12, 13),
            (13, 14)
        ]


        # ======================================================
        # 2.4 Create core links
        # ======================================================

        info('*** Creating NSFNET core links...\n')

        for src, dst in nsfnet_links:

            self.addLink(
                switches[src],
                switches[dst],

                # ------------------------------------------------
                # Core link bandwidth
                # ------------------------------------------------
                bw=100,

                # ------------------------------------------------
                # Propagation delay
                # ------------------------------------------------
                delay='5ms',

                # ------------------------------------------------
                # Packet loss
                # ------------------------------------------------
                loss=0
            )

        info(
            f'*** Created {len(nsfnet_links)} core links.\n'
        )


# ==============================================================
# 3. Create and start Mininet
# ==============================================================

def run():

    # ----------------------------------------------------------
    # Display information
    # ----------------------------------------------------------

    info('\n')
    info('==================================================\n')
    info('              NSFNET MININET TESTBED             \n')
    info('==================================================\n')
    info('Topology      : NSFNET\n')
    info('Core switches : 14\n')
    info('Hosts         : 14\n')
    info('Core links    : 22\n')
    info('OpenFlow      : 1.3\n')
    info('Switch        : Open vSwitch\n')
    info('Controller    : Ryu\n')
    info('Controller IP : 127.0.0.1\n')
    info('Controller    : 6653\n')
    info('==================================================\n')
    info('\n')


    # ----------------------------------------------------------
    # Create topology
    # ----------------------------------------------------------

    info('*** Creating NSFNET topology...\n')

    topo = NSFNETTopo()


    # ----------------------------------------------------------
    # Create Mininet network
    # ----------------------------------------------------------

    net = Mininet(
        topo=topo,

        # Controller will be added manually
        controller=None,

        # Open vSwitch
        switch=OVSSwitch,

        # Traffic Control links
        link=TCLink,

        # Automatically assign MAC addresses
        autoSetMacs=True
    )


    # ==========================================================
    # 4. Add Ryu Remote Controller
    # ==========================================================

    info('*** Adding Ryu controller...\n')

    net.addController(
        'c0',

        controller=RemoteController,

        # Ryu controller IP
        ip='127.0.0.1',

        # Ryu OpenFlow port
        port=6653
    )


    # ==========================================================
    # 5. Start network
    # ==========================================================

    try:

        info('\n')
        info('*** Starting NSFNET network...\n')

        net.start()

        info('\n')
        info('==================================================\n')
        info('        NSFNET NETWORK STARTED SUCCESSFULLY       \n')
        info('==================================================\n')

        info('Core switches : 14\n')
        info('Hosts         : 14\n')
        info('Core links    : 22\n')
        info('Controller    : 127.0.0.1:6653\n')
        info('OpenFlow      : 1.3\n')

        info('==================================================\n')
        info('\n')


        # ======================================================
        # 6. Basic test instructions
        # ======================================================

        info('*** Mininet CLI is ready.\n')
        info('\n')

        info('Useful commands:\n')
        info('  nodes                 -> List all nodes\n')
        info('  net                   -> Display network links\n')
        info('  links                -> Display all links\n')
        info('  dump                 -> Display node information\n')
        info('  pingall              -> Test connectivity\n')
        info('  h1 ping h14          -> Test h1 -> h14\n')
        info('  h1 ping h10          -> Test h1 -> h10\n')
        info('  s1 ovs-ofctl -O OpenFlow13 dump-flows s1\n')
        info('  exit                 -> Stop Mininet\n')
        info('\n')


        # ======================================================
        # 7. Start Mininet CLI
        # ======================================================

        CLI(net)


    except Exception as e:

        info('\n')
        info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        info('ERROR: Failed to start NSFNET network\n')
        info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')

        info(f'{e}\n')


    finally:

        # ======================================================
        # 8. Stop network
        # ======================================================

        info('\n')
        info('*** Stopping NSFNET network...\n')

        net.stop()

        info('*** NSFNET network stopped.\n')
        info('\n')


# ==============================================================
# 9. Main entry point
# ==============================================================

if __name__ == '__main__':

    # Set Mininet log level
    setLogLevel('info')

    # Run NSFNET
    run()
