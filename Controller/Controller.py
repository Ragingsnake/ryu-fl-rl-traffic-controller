#!/usr/bin/env python3
"""
Universal SDN L3 Router & Topology-Aware Controller (OpenFlow 1.3)
- Automatically discovers any arbitrary topology via LLDP (requires --observe-links flag)
- Dynamically learns MAC / IP / attachment points of all end hosts
- Packet buffering for unknown destinations (prevents 1st-packet loss during ARP discovery)
- Proxy ARP: Directly answers ARP requests (prevents broadcast storms in loopy topologies)
- ICMP Responder: Answers ping (ICMP Echo Request) to any router gateway IP (10.0.x.1)
- Shortest Path Routing (Dijkstra) between any pair of hosts across different subnets
"""

import networkx as nx
from ryu.lib import hub
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4, icmp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link


class UniversalRouterController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(UniversalRouterController, self).__init__(*args, **kwargs)
        # Directed graph storing network topology (Nodes = DPID, Edges = Links with output ports)
        self.net = nx.DiGraph()

        self.monitor_thread = hub.spawn(self._topology_sync_loop)

        # Connected switches dictionary: {dpid: datapath}
        self.datapaths = {}

        # Dynamically learned host table: {ip_addr: {'mac': mac, 'dpid': dpid, 'port': in_port}}
        self.hosts = {}

        # Buffer for IPv4 packets waiting for destination host ARP discovery: {dst_ip: [(src_ip, msg), ...]}
        self.packet_buffer = {}

        # Virtual MAC address representing all router gateways
        self.ROUTER_MAC = "00:00:00:00:fe:fe"

    def _topology_sync_loop(self):
        while True:
            hub.sleep(2)
            try:
                self.update_topology(None)
            except Exception:
                pass

    # ============================================================
    # 1. SWITCH INITIALIZATION & TABLE-MISS FLOW
    # ============================================================
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[datapath.id] = datapath

        # Install table-miss flow entry to forward unmatched packets to the controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

        self.logger.info(f"[Switch Connected] r{datapath.id - 1} (DPID: {datapath.id})")

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout
        )
        datapath.send_msg(mod)

    # ============================================================
    # 2. DYNAMIC TOPOLOGY DISCOVERY (LLDP)
    # ============================================================
    @set_ev_cls([event.EventSwitchEnter, event.EventSwitchLeave,
                 event.EventLinkAdd, event.EventLinkDelete])
    def update_topology(self, ev):
        switches = get_switch(self, None)
        links = get_link(self, None)

        old_nodes = self.net.number_of_nodes()
        old_edges = self.net.number_of_edges()

        self.net.clear()
        for sw in switches:
            self.net.add_node(sw.dp.id)

        for link in links:
            src = link.src.dpid
            dst = link.dst.dpid
            # Store output port and default link metric (weight=1) for routing algorithms
            self.net.add_edge(src, dst, port=link.src.port_no, weight=1)

        new_nodes = self.net.number_of_nodes()
        new_edges = self.net.number_of_edges()

        if new_nodes != old_nodes or new_edges != old_edges:
            self.logger.info(
                f"[Topology Updated] Nodes: {new_nodes} | Directed Links: {new_edges}"
            )

    # ============================================================
    # 3. PACKET-IN HANDLER (HOST LEARNING, PROXY ARP, ICMP & L3 ROUTING)
    # ============================================================
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # Ignore LLDP packets (used internally for topology discovery) and IPv6
        if eth.ethertype in (ether_types.ETH_TYPE_LLDP, ether_types.ETH_TYPE_IPV6):
            return

        # Check whether the ingress port is an inter-router link
        is_inter_router_port = any(
            data['port'] == in_port for _, _, data in self.net.out_edges(dpid, data=True)
        )

        # --- HANDLE ARP PACKETS (HOST LEARNING & PROXY ARP) ---
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            src_ip = arp_pkt.src_ip
            src_mac = arp_pkt.src_mac
            dst_ip = arp_pkt.dst_ip

            # Learn host location only if packet arrives from an edge (host-facing) port
            if not is_inter_router_port and src_ip not in self.hosts:
                self.hosts[src_ip] = {'mac': src_mac, 'dpid': dpid, 'port': in_port}
                self.logger.info(f"[Host Learned] IP: {src_ip} | MAC: {src_mac} | Router: r{dpid-1} (port {in_port})")

                # Flush any buffered packets waiting for this newly discovered host
                if src_ip in self.packet_buffer:
                    for buf_src_ip, buf_msg in self.packet_buffer.pop(src_ip):
                        self.install_shortest_path(buf_src_ip, src_ip, buf_msg)

            # If host sends an ARP Request -> controller replies directly on behalf of the gateway/target
            if arp_pkt.opcode == arp.ARP_REQUEST:
                target_mac = self.hosts[dst_ip]['mac'] if dst_ip in self.hosts else self.ROUTER_MAC
                self.send_arp_reply(datapath, in_port, target_mac, dst_ip, src_mac, src_ip)
            return

        # --- HANDLE IPv4 PACKETS (ICMP GATEWAY PING & L3 ROUTING) ---
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst

            # Learn source host if the first packet is IPv4
            if not is_inter_router_port and src_ip not in self.hosts:
                self.hosts[src_ip] = {'mac': eth.src, 'dpid': dpid, 'port': in_port}
                self.logger.info(f"[Host Learned] IP: {src_ip} | MAC: {eth.src} | Router: r{dpid-1} (port {in_port})")

            # Reply to ICMP Echo Request (ping) destined to any Router Gateway IP (10.0.x.1)
            icmp_pkt = pkt.get_protocol(icmp.icmp)
            if icmp_pkt and icmp_pkt.type == icmp.ICMP_ECHO_REQUEST and dst_ip.endswith(".1"):
                try:
                    target_router_id = int(dst_ip.split('.')[2])
                    if (target_router_id + 1) in self.datapaths:
                        self.send_icmp_reply(datapath, in_port, eth.src, src_ip, dst_ip, ip_pkt, icmp_pkt)
                except ValueError:
                    pass
                return

            # If destination host is unknown, buffer packet and send ARP probes out edge ports
            if dst_ip not in self.hosts:
                self.packet_buffer.setdefault(dst_ip, []).append((src_ip, msg))
                self.discover_host(dst_ip)
                return

            # Compute shortest path and install flow rules along the entire route
            self.install_shortest_path(src_ip, dst_ip, msg)

    # ============================================================
    # 4. ARP REPLY, ICMP REPLY & UNKNOWN HOST DISCOVERY
    # ============================================================
    def send_arp_reply(self, datapath, out_port, src_mac, src_ip, dst_mac, dst_ip):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_ARP,
            dst=dst_mac,
            src=src_mac
        ))
        pkt.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=src_mac,
            src_ip=src_ip,
            dst_mac=dst_mac,
            dst_ip=dst_ip
        ))
        pkt.serialize()

        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=pkt.data
        )
        datapath.send_msg(out)

    def send_icmp_reply(self, datapath, out_port, dst_mac, dst_ip, src_ip, ip_pkt, icmp_pkt):
        """Send ICMP Echo Reply when a host pings a router gateway IP (10.0.x.1)"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_IP,
            dst=dst_mac,
            src=self.ROUTER_MAC
        ))
        pkt.add_protocol(ipv4.ipv4(
            dst=dst_ip,
            src=src_ip,
            proto=ip_pkt.proto
        ))
        pkt.add_protocol(icmp.icmp(
            type_=icmp.ICMP_ECHO_REPLY,
            code=icmp.ICMP_ECHO_REPLY_CODE,
            csum=0,
            data=icmp_pkt.data
        ))
        pkt.serialize()

        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=pkt.data
        )
        datapath.send_msg(out)

    def discover_host(self, target_ip):
        """Broadcast ARP requests only on edge (non-inter-router) ports to locate target host"""
        parts = target_ip.split('.')
        if len(parts) != 4:
            return
        gw_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.1"

        for dpid, datapath in self.datapaths.items():
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            inter_ports = {data['port'] for _, _, data in self.net.out_edges(dpid, data=True)}

            for port_no in datapath.ports:
                if port_no <= ofproto.OFPP_MAX and port_no not in inter_ports:
                    pkt = packet.Packet()
                    pkt.add_protocol(ethernet.ethernet(
                        ethertype=ether_types.ETH_TYPE_ARP,
                        dst="ff:ff:ff:ff:ff:ff",
                        src=self.ROUTER_MAC
                    ))
                    pkt.add_protocol(arp.arp(
                        opcode=arp.ARP_REQUEST,
                        src_mac=self.ROUTER_MAC,
                        src_ip=gw_ip,
                        dst_mac="00:00:00:00:00:00",
                        dst_ip=target_ip
                    ))
                    pkt.serialize()
                    actions = [parser.OFPActionOutput(port_no)]
                    datapath.send_msg(parser.OFPPacketOut(
                        datapath=datapath,
                        buffer_id=ofproto.OFP_NO_BUFFER,
                        in_port=ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=pkt.data
                    ))

    # ============================================================
    # 5. ROUTING ALGORITHM (DIJKSTRA / CUSTOMIZABLE)
    # ============================================================
    def install_shortest_path(self, src_ip, dst_ip, msg):
        src_info = self.hosts[src_ip]
        dst_info = self.hosts[dst_ip]

        src_dpid = src_info['dpid']
        dst_dpid = dst_info['dpid']

        try:
            # Compute shortest path using edge 'weight' (can be swapped with Bellman-Ford, A*, etc.)
            path = nx.shortest_path(self.net, source=src_dpid, target=dst_dpid, weight='weight')
        except nx.NetworkXNoPath:
            self.logger.warning(f"[No Path] No route found from r{src_dpid-1} to r{dst_dpid-1}")
            return

        self.logger.info(f"[Route Installed] {src_ip} -> {dst_ip} | Path: {[f'r{n-1}' for n in path]}")

        # Install flow rules on each router along the computed path
        for i in range(len(path)):
            curr_dpid = path[i]
            dp = self.datapaths[curr_dpid]
            parser = dp.ofproto_parser

            if curr_dpid == dst_dpid:
                # Egress router: rewrite Ethernet MACs for destination host and forward to host port
                out_port = dst_info['port']
                actions = [
                    parser.OFPActionSetField(eth_src=self.ROUTER_MAC),
                    parser.OFPActionSetField(eth_dst=dst_info['mac']),
                    parser.OFPActionOutput(out_port)
                ]
            else:
                # Transit router: forward packet to the next hop router along the path
                next_dpid = path[i + 1]
                out_port = self.net[curr_dpid][next_dpid]['port']
                actions = [parser.OFPActionOutput(out_port)]

            match = parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_dst=dst_ip
            )
            self.add_flow(dp, priority=10, match=match, actions=actions, idle_timeout=60)

        # Immediately forward the current packet out of the ingress router
        first_dp = self.datapaths[src_dpid]
        parser = first_dp.ofproto_parser
        ofproto = first_dp.ofproto

        if src_dpid == dst_dpid:
            first_actions = [
                parser.OFPActionSetField(eth_src=self.ROUTER_MAC),
                parser.OFPActionSetField(eth_dst=dst_info['mac']),
                parser.OFPActionOutput(dst_info['port'])
            ]
        else:
            next_dpid = path[1]
            out_port = self.net[src_dpid][next_dpid]['port']
            first_actions = [parser.OFPActionOutput(out_port)]

        out = parser.OFPPacketOut(
            datapath=first_dp,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=msg.match['in_port'],
            actions=first_actions,
            data=msg.data
        )
        first_dp.send_msg(out)
