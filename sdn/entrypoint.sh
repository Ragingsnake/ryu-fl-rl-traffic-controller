#!/bin/bash
# Start Open vSwitch service
service openvswitch-switch start

# Start bash or run command
exec "$@"
