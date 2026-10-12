#!/bin/bash
# Start Open vSwitch service
service openvswitch-switch start || true
if ! pidof ovs-vswitchd >/dev/null 2>&1; then
    ovs-vswitchd --pidfile --detach --log-file
fi

# Start bash or run command
exec "$@"
