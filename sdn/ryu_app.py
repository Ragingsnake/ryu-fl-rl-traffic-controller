"""
Ryu OpenFlow 1.3 Controller
- Inherits from simple_switch_13 for basic L2 connectivity
- Polls port statistics every 5 seconds
- Writes stats to a JSON file for the AI Bridge
"""

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
import time
import json
import os

class SDNController(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SDNController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.stats = {} 
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5) # Poll every 5 seconds

    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.stats.setdefault(dpid, {})
        for stat in body:
            # We track tx_bytes, rx_bytes for utilization calculation
            # Port no > 1000 usually means LOCAL or internal, ignore if needed
            self.stats[dpid][stat.port_no] = {
                'tx_bytes': stat.tx_bytes,
                'rx_bytes': stat.rx_bytes,
                'timestamp': time.time()
            }
        
        # Write to JSON for AI Bridge
        with open('/tmp/sdn_stats.json', 'w') as f:
            json.dump(self.stats, f)
