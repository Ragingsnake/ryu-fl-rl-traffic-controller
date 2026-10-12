#!/bin/bash
export PYTHONUNBUFFERED=1

# Patch eventlet for Python 3.10 staticmethod compatibility
python3 -c "p = '/usr/local/lib/python3.10/dist-packages/eventlet/greenio/py3.py'; c = open(p).read().replace('_original_pyio.open', 'getattr(_original_pyio.open, \'__func__\', _original_pyio.open)'); open(p, 'w').write(c)" 2>/dev/null || true

if ! pidof ovs-vswitchd >/dev/null 2>&1; then
    ovs-vswitchd --pidfile --detach --log-file
fi

echo "[1/4] Starting Ryu Controller..."
ryu-manager --ofp-tcp-listen-port 6653 sdn/ryu_app.py > /tmp/ryu.log 2>&1 &
sleep 3

echo "[2/4] Starting Mininet (NSFNET Topology)..."
HEADLESS=1 python3 sdn/mininet_topo.py > /tmp/mininet.log 2>&1 &

echo "  -> Waiting for Mininet topology & pingall to complete..."
for attempt in {1..30}; do
    if grep -q "Headless mode active" /tmp/mininet.log 2>/dev/null; then
        echo "  -> Mininet topology & connectivity ready!"
        break
    fi
    sleep 2
done

echo "[3/4] Generating Traffic Matrices & Starting Orchestrator..."
python3 data/generate_tm.py > /tmp/generate_tm.log 2>&1

cd sdn
python3 traffic_orchestrator.py > /tmp/traffic.log 2>&1 &
sleep 2

echo "[4/4] Starting AI Bridge & Monitoring Closed Loop..."
python3 ai_bridge.py > /tmp/ai.log 2>&1 &

for i in {1..6}; do
    echo "  -> Monitoring loop epoch $i/6..."
    sleep 3
done

echo ""
echo "==========================================="
echo "               TEST RESULTS                "
echo "==========================================="

echo -e "\n--- Ryu Controller Log (tail) ---"
cat /tmp/ryu.log | tail -n 10

echo -e "\n--- Mininet Log (tail) ---"
cat /tmp/mininet.log | tail -n 10

echo -e "\n--- Traffic Orchestrator Log (tail) ---"
cat /tmp/traffic.log | tail -n 10

echo -e "\n--- AI Bridge Log (tail) ---"
cat /tmp/ai.log | tail -n 15

# Cleanup
echo -e "\nCleaning up..."
pkill -9 -f ryu-manager 2>/dev/null || true
pkill -9 -f mininet_topo.py 2>/dev/null || true
pkill -9 -f traffic_orchestrator.py 2>/dev/null || true
pkill -9 -f ai_bridge.py 2>/dev/null || true
echo "Test complete!"
