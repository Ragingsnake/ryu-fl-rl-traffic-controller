#!/usr/bin/env python3
import networkx as nx
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4, icmp
from ryu.topology import event
from ryu.topology.api import get_link


# Cấu trúc liên kết chuẩn của GEANT (23 node) để bù đắp ngay lập tức khi LLDP bị rớt gói trên máy ảo
GEANT_LINKS = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 3),
    (2, 3), (2, 9), (3, 4), (3, 8), (4, 5),
    (5, 6), (5, 8), (6, 7), (7, 10), (8, 9),
    (8, 10), (9, 11), (9, 12), (10, 14), (11, 12),
    (11, 13), (12, 13), (12, 14), (12, 15), (13, 15),
    (14, 16), (15, 16), (15, 17), (16, 18), (16, 20),
    (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
]


class UniversalRouterController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(UniversalRouterController, self).__init__(*args, **kwargs)
        self.net = nx.DiGraph()
        self.datapaths = {}
        self.hosts = {}
        self.packet_buffer = {}
        self.ROUTER_MAC = "00:00:00:00:fe:fe"
        self.monitor_thread = hub.spawn(self._topology_sync_loop)

    def _topology_sync_loop(self):
        while True:
            hub.sleep(2)
            try:
                self._rebuild_graph()
            except Exception:
                pass

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
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

    @set_ev_cls([event.EventSwitchEnter, event.EventSwitchLeave,
                 event.EventLinkAdd, event.EventLinkDelete])
    def update_topology(self, ev):
        self._rebuild_graph()

    def _rebuild_graph(self):
        old_nodes = self.net.number_of_nodes()
        old_edges = self.net.number_of_edges()

        new_net = nx.DiGraph()
        for dpid in self.datapaths:
            new_net.add_node(dpid)

        # 1. Lấy các cạnh từ LLDP
        for link in get_link(self, None):
            new_net.add_edge(link.src.dpid, link.dst.dpid, port=link.src.port_no, weight=1)

        # 2. Nếu là mạng GEANT (23 router) và LLDP chưa quét đủ 70 cạnh, tự động ánh xạ cổng chính xác
        # Trong file topology hiện tại của bạn: port 1 nối tới host h{i}, các port 2, 3, 4... nối theo thứ tự GEANT_LINKS
        if len(self.datapaths) == 23 and new_net.number_of_edges() < 70:
            port_counter = {i: 2 for i in range(23)}  # port 1 là host, inter-router bắt đầu từ port 2
            for u, v in GEANT_LINKS:
                u_dpid, v_dpid = u + 1, v + 1
                u_port = port_counter[u]
                v_port = port_counter[v]
                port_counter[u] += 1
                port_counter[v] += 1
                if not new_net.has_edge(u_dpid, v_dpid):
                    new_net.add_edge(u_dpid, v_dpid, port=u_port, weight=1)
                if not new_net.has_edge(v_dpid, u_dpid):
                    new_net.add_edge(v_dpid, u_dpid, port=v_port, weight=1)

        self.net = new_net
        new_nodes = self.net.number_of_nodes()
        new_edges = self.net.number_of_edges()

        if new_nodes != old_nodes or new_edges != old_edges:
            self.logger.info(f"[Topology Updated] Nodes: {new_nodes} | Directed Links: {new_edges}")

    def _is_valid_local_host(self, dpid, in_port, src_ip):
        if in_port == ofproto_v1_3.OFPP_LOCAL:
            return False
        is_inter_router = any(
            data['port'] == in_port for _, _, data in self.net.out_edges(dpid, data=True)
        )
        if is_inter_router:
            return False
        parts = src_ip.split('.')
        if len(parts) == 4 and parts[0] == '10' and parts[1] == '0':
            return int(parts[2]) == (dpid - 1) and parts[3] != '1'
        return True

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth.ethertype in (ether_types.ETH_TYPE_LLDP, ether_types.ETH_TYPE_IPV6):
            return

        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            src_ip = arp_pkt.src_ip
            src_mac = arp_pkt.src_mac
            dst_ip = arp_pkt.dst_ip

            if self._is_valid_local_host(dpid, in_port, src_ip) and src_ip not in self.hosts:
                self.hosts[src_ip] = {'mac': src_mac, 'dpid': dpid, 'port': in_port}
                self.logger.info(f"[Host Learned] IP: {src_ip} | MAC: {src_mac} | Router: r{dpid-1} (port {in_port})")
                if src_ip in self.packet_buffer:
                    for buf_src_ip, buf_msg in self.packet_buffer.pop(src_ip):
                        self.install_shortest_path(buf_src_ip, src_ip, buf_msg)

            if arp_pkt.opcode == arp.ARP_REQUEST:
                target_mac = self.hosts[dst_ip]['mac'] if dst_ip in self.hosts else self.ROUTER_MAC
                self.send_arp_reply(datapath, in_port, target_mac, dst_ip, src_mac, src_ip)
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst

            if self._is_valid_local_host(dpid, in_port, src_ip) and src_ip not in self.hosts:
                self.hosts[src_ip] = {'mac': eth.src, 'dpid': dpid, 'port': in_port}
                self.logger.info(f"[Host Learned] IP: {src_ip} | MAC: {eth.src} | Router: r{dpid-1} (port {in_port})")

            icmp_pkt = pkt.get_protocol(icmp.icmp)
            if icmp_pkt and icmp_pkt.type == icmp.ICMP_ECHO_REQUEST and dst_ip.endswith(".1"):
                try:
                    target_router_id = int(dst_ip.split('.')[2])
                    if (target_router_id + 1) in self.datapaths:
                        self.send_icmp_reply(datapath, in_port, eth.src, src_ip, dst_ip, ip_pkt, icmp_pkt)
                except ValueError:
                    pass
                return

            if dst_ip not in self.hosts:
                buf = self.packet_buffer.setdefault(dst_ip, [])
                if len(buf) < 5:
                    buf.append((src_ip, msg))
                self.discover_host(dst_ip)
                return

            self.install_shortest_path(src_ip, dst_ip, msg)

    def send_arp_reply(self, datapath, out_port, src_mac, src_ip, dst_mac, dst_ip):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_ARP, dst=dst_mac, src=src_mac))
        pkt.add_protocol(arp.arp(opcode=arp.ARP_REPLY, src_mac=src_mac, src_ip=src_ip, dst_mac=dst_mac, dst_ip=dst_ip))
        pkt.serialize()
        actions = [parser.OFPActionOutput(out_port)]
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=ofproto.OFPP_CONTROLLER,
            actions=actions, data=pkt.data
        ))

    def send_icmp_reply(self, datapath, out_port, dst_mac, dst_ip, src_ip, ip_pkt, icmp_pkt):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IP, dst=dst_mac, src=self.ROUTER_MAC))
        pkt.add_protocol(ipv4.ipv4(dst=dst_ip, src=src_ip, proto=ip_pkt.proto))
        pkt.add_protocol(icmp.icmp(type_=icmp.ICMP_ECHO_REPLY, code=icmp.ICMP_ECHO_REPLY_CODE, csum=0, data=icmp_pkt.data))
        pkt.serialize()
        actions = [parser.OFPActionOutput(out_port)]
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=ofproto.OFPP_CONTROLLER,
            actions=actions, data=pkt.data
        ))

    def discover_host(self, target_ip):
        parts = target_ip.split('.')
        if len(parts) != 4:
            return
        gw_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.1"
        target_dpids = []
        try:
            candidate_dpid = int(parts[2]) + 1
            if candidate_dpid in self.datapaths:
                target_dpids = [candidate_dpid]
        except ValueError:
            pass
        if not target_dpids:
            target_dpids = list(self.datapaths.keys())

        for dpid in target_dpids:
            datapath = self.datapaths[dpid]
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            inter_ports = {data['port'] for _, _, data in self.net.out_edges(dpid, data=True)}
            for port_no in datapath.ports:
                if port_no <= ofproto.OFPP_MAX and port_no not in inter_ports:
                    pkt = packet.Packet()
                    pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_ARP, dst="ff:ff:ff:ff:ff:ff", src=self.ROUTER_MAC))
                    pkt.add_protocol(arp.arp(opcode=arp.ARP_REQUEST, src_mac=self.ROUTER_MAC, src_ip=gw_ip, dst_mac="00:00:00:00:00:00", dst_ip=target_ip))
                    pkt.serialize()
                    actions = [parser.OFPActionOutput(port_no)]
                    datapath.send_msg(parser.OFPPacketOut(
                        datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=ofproto.OFPP_CONTROLLER,
                        actions=actions, data=pkt.data
                    ))

    def install_shortest_path(self, src_ip, dst_ip, msg):
        if src_ip not in self.hosts or dst_ip not in self.hosts:
            return
        src_info = self.hosts[src_ip]
        dst_info = self.hosts[dst_ip]
        src_dpid = src_info['dpid']
        dst_dpid = dst_info['dpid']

        try:
            path = nx.shortest_path(self.net, source=src_dpid, target=dst_dpid, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.logger.warning(f"[No Path] No route from r{src_dpid-1} to r{dst_dpid-1}")
            return

        self.logger.info(f"[Route Installed] {src_ip} -> {dst_ip} | Path: {[f'r{n-1}' for n in path]}")

        for i in range(len(path)):
            curr_dpid = path[i]
            dp = self.datapaths[curr_dpid]
            parser = dp.ofproto_parser
            if curr_dpid == dst_dpid:
                out_port = dst_info['port']
                actions = [
                    parser.OFPActionSetField(eth_src=self.ROUTER_MAC),
                    parser.OFPActionSetField(eth_dst=dst_info['mac']),
                    parser.OFPActionOutput(out_port)
                ]
            else:
                next_dpid = path[i + 1]
                out_port = self.net[curr_dpid][next_dpid]['port']
                actions = [parser.OFPActionOutput(out_port)]
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_dst=dst_ip)
            self.add_flow(dp, priority=10, match=match, actions=actions, idle_timeout=60)

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
        first_dp.send_msg(parser.OFPPacketOut(
            datapath=first_dp, buffer_id=ofproto.OFP_NO_BUFFER, in_port=msg.match['in_port'],
            actions=first_actions, data=msg.data
        ))
