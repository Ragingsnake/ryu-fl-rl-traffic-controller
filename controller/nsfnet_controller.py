
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NSFNET SDN Controller: topology discovery + SPF routing."""

from collections import defaultdict

import networkx as nx

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    DEAD_DISPATCHER,
    set_ev_cls,
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, arp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link


class NSFNETController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NSFNETController, self).__init__(*args, **kwargs)

        self.net = nx.Graph()
        self.datapaths = {}
        self.host_location = {}
        self.link_ports = {}
        self.tree_ports = defaultdict(set)
        self.switch_ports = defaultdict(set)

        self.logger.info("NSFNET SPF controller initialized")

    @set_ev_cls(
        ofp_event.EventOFPStateChange,
        [MAIN_DISPATCHER, DEAD_DISPATCHER]
    )
    def state_change_handler(self, ev):
        dp = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.logger.info("Switch connected: dpid=%s", dp.id)

        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)
            self.host_location = {
                mac: loc
                for mac, loc in self.host_location.items()
                if loc[0] != dp.id
            }

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER
    )
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofproto = dp.ofproto

        # Gửi gói chưa khớp flow lên controller.
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(
            dp, priority=0, match=match,
            actions=actions, idle_timeout=0
        )

        self.logger.info("Table-miss installed: dpid=%s", dp.id)

    def add_flow(
        self, dp, priority, match, actions, idle_timeout=60
    ):
        parser = dp.ofproto_parser

        instructions = [
            parser.OFPInstructionActions(
                dp.ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        dp.send_msg(
            parser.OFPFlowMod(
                datapath=dp,
                priority=priority,
                match=match,
                instructions=instructions,
                idle_timeout=idle_timeout
            )
        )

    @set_ev_cls([
        event.EventSwitchEnter,
        event.EventSwitchLeave,
        event.EventLinkAdd,
        event.EventLinkDelete,
    ])
    def topology_change_handler(self, ev):
        self.refresh_topology()

    def refresh_topology(self):
        """Discover links and rebuild the SPF graph."""
        try:
            switches = get_switch(self, None)
            links = get_link(self, None)
        except Exception as exc:
            self.logger.warning(
                "Topology discovery error: %s", exc
            )
            return

        graph = nx.Graph()
        link_ports = {}
        switch_ports = defaultdict(set)

        for sw in switches:
            dpid = sw.dp.id
            graph.add_node(dpid)

            for port in sw.ports:
                if port.port_no < sw.dp.ofproto.OFPP_MAX:
                    switch_ports[dpid].add(port.port_no)

        for link in links:
            src = link.src.dpid
            dst = link.dst.dpid

            if src == dst:
                continue

            # SPF metric: each discovered link has cost 1.
            # This selects a minimum-hop path.
            graph.add_edge(src, dst, cost=1)

            link_ports[(src, dst)] = link.src.port_no
            link_ports[(dst, src)] = link.dst.port_no

        self.net = graph
        self.link_ports = link_ports
        self.switch_ports = switch_ports

        # Build a spanning tree for loop-free broadcast flooding.
        self.tree_ports = defaultdict(set)

        for component in nx.connected_components(graph):
            tree = nx.minimum_spanning_tree(
                graph.subgraph(component),
                weight="cost"
            )

            for a, b in tree.edges():
                port_ab = link_ports.get((a, b))
                port_ba = link_ports.get((b, a))

                if port_ab is not None:
                    self.tree_ports[a].add(port_ab)

                if port_ba is not None:
                    self.tree_ports[b].add(port_ba)

        self.logger.info(
            "Topology refreshed: switches=%d links=%d",
            graph.number_of_nodes(),
            graph.number_of_edges()
        )

        if graph.number_of_nodes() != 14:
            self.logger.warning(
                "Expected 14 switches, discovered %d",
                graph.number_of_nodes()
            )

        if graph.number_of_edges() != 27:
            self.logger.warning(
                "Expected 27 inter-switch links, discovered %d",
                graph.number_of_edges()
            )

    def flood_ports(self, dpid, in_port):
        """Return host ports and spanning-tree ports."""
        inter_switch_ports = {
            port
            for (src, _dst), port in self.link_ports.items()
            if src == dpid
        }

        ports = set(self.tree_ports.get(dpid, set()))

        for port in self.switch_ports.get(dpid, set()):
            if port not in inter_switch_ports:
                ports.add(port)

        ports.discard(in_port)
        return sorted(ports)

    def send_packet(self, dp, in_port, out_ports, data):
        """Send a packet to selected output ports."""
        if not out_ports:
            return

        parser = dp.ofproto_parser

        actions = [
            parser.OFPActionOutput(port)
            for port in out_ports
        ]

        dp.send_msg(
            parser.OFPPacketOut(
                datapath=dp,
                buffer_id=dp.ofproto.OFP_NO_BUFFER,
                in_port=in_port,
                actions=actions,
                data=data
            )
        )

    @set_ev_cls(
        ofp_event.EventOFPPacketIn,
        MAIN_DISPATCHER
    )
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        dpid = dp.id
        in_port = msg.match.get("in_port")

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None:
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        src_mac = eth.src
        dst_mac = eth.dst
        arp_pkt = pkt.get_protocol(arp.arp)

        inter_switch_ports = {
            port
            for (src, _dst), port in self.link_ports.items()
            if src == dpid
        }

        # Learn MAC addresses from host-facing ports only.
        if in_port not in inter_switch_ports:
            self.host_location[src_mac] = (dpid, in_port)

        self.logger.info(
            "Packet-In: dpid=%s in_port=%s src=%s dst=%s arp=%s",
            dpid, in_port, src_mac, dst_mac,
            arp_pkt is not None
        )

        # ARP and broadcast/multicast packets.
        first_octet = int(dst_mac.split(":")[0], 16)
        is_multicast = bool(first_octet & 1)

        if arp_pkt is not None or is_multicast:
            out_ports = self.flood_ports(dpid, in_port)

            self.logger.info(
                "Flood: dpid=%s output_ports=%s",
                dpid, out_ports
            )

            if not out_ports:
                self.logger.warning(
                    "No flood ports: dpid=%s", dpid
                )
                return

            self.send_packet(
                dp, in_port, out_ports, msg.data
            )
            return

        # Unknown destination: flood until the destination is learned.
        if dst_mac not in self.host_location:
            out_ports = self.flood_ports(dpid, in_port)

            self.logger.info(
                "Unknown destination %s; ports=%s",
                dst_mac, out_ports
            )

            self.send_packet(
                dp, in_port, out_ports, msg.data
            )
            return

        dst_dpid, dst_port = self.host_location[dst_mac]

        # Calculate an SPF path through the discovered topology.
        if dst_dpid == dpid:
            out_port = dst_port
            path = [dpid]
        else:
            try:
                path = nx.shortest_path(
                    self.net,
                    source=dpid,
                    target=dst_dpid,
                    weight="cost",
                    method="dijkstra"
                )

                out_port = self.link_ports[
                    (path[0], path[1])
                ]

            except (
                nx.NetworkXNoPath,
                nx.NodeNotFound,
                KeyError
            ):
                self.logger.warning(
                    "No SPF path: %s -> %s",
                    dpid, dst_dpid
                )
                return

        self.logger.info(
            "SPF path: %s -> %s, path=%s, output_port=%s",
            dpid, dst_dpid, path, out_port
        )

        parser = dp.ofproto_parser
        actions = [parser.OFPActionOutput(out_port)]

        match = parser.OFPMatch(
            in_port=in_port,
            eth_src=src_mac,
            eth_dst=dst_mac
        )

        self.add_flow(
            dp,
            priority=10,
            match=match,
            actions=actions
        )

        data = (
            None
            if msg.buffer_id != dp.ofproto.OFP_NO_BUFFER
            else msg.data
        )

        dp.send_msg(
            parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=data
            )
        )
