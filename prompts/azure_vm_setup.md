# Azure Linux VM Provisioning, Setup, and Execution Guide
## FL + RL Proactive Traffic Controller for Multi-Topology SDN Networks

> **Document Type:** Production Deployment & Operation Prompt  
> **Target OS:** Ubuntu 22.04 LTS (x86_64)  
> **Recommended Azure VM Size:** `Standard_D8s_v5` (8 vCPUs, 32 GiB RAM, Premium SSD)

---

## 1. Connection Configurations & Host Pointer

Fill in the connection details below once your Azure VM is deployed:

```bash
# ==============================================================================
# AZURE VM CONNECTION POINTERS
# ==============================================================================
export AZURE_VM_IP="<YOUR_AZURE_VM_PUBLIC_IP>"          # e.g., "20.219.145.88"
export AZURE_SSH_KEY="<PATH_TO_YOUR_SSH_PRIVATE_KEY>"   # e.g., "~/.ssh/azure_vm_key.pem" or "C:/Users/.../azure.pem"
export AZURE_USER="azureuser"                           # Default Azure Ubuntu user
export APP_PORT="8501"                                  # Web UI Application Port
# ==============================================================================
```

### Quick SSH One-Liner
```bash
ssh -i "$AZURE_SSH_KEY" "$AZURE_USER@$AZURE_VM_IP"
```

If you are using Windows PowerShell:
```powershell
$AZURE_VM_IP = "<YOUR_AZURE_VM_PUBLIC_IP>"
$AZURE_SSH_KEY = "<PATH_TO_YOUR_SSH_PRIVATE_KEY>"
ssh -i $AZURE_SSH_KEY azureuser@$AZURE_VM_IP
```

---

## 2. Azure Network Security Group (NSG) Rules
Ensure the following inbound ports are open in the Azure NSG:

| Priority | Name | Port | Protocol | Source | Purpose |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | SSH | 22 | TCP | Any / My IP | Remote Shell Access |
| 110 | WebApp_Demo | 8501 | TCP | Any / My IP | Streamlit Web Demo UI |
| 120 | Flower_FL | 8080 | TCP | VirtualNetwork | gRPC Federated Learning Aggregation |
| 130 | Ryu_REST | 8080 / 6653 | TCP | Localhost / Docker | OpenFlow 1.3 & REST Interface |

---

## 3. Remote Server Initialization Script (Run on Azure VM)

Once SSH'd into the Azure VM, run these commands to configure the kernel environment, Open vSwitch, Docker, and Python:

```bash
#!/bin/bash
set -e

echo "=== 1. Updating System Packages ==="
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    tmux \
    htop \
    iperf \
    iperf3 \
    openvswitch-switch \
    openvswitch-testcontroller \
    mininet \
    python3 \
    python3-pip \
    python3-venv \
    net-tools \
    iproute2 \
    ca-certificates

echo "=== 2. Setting Up Docker & Docker Compose ==="
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER

echo "=== 3. Starting Open vSwitch Kernel Service ==="
sudo systemctl enable --now openvswitch-switch
sudo ovs-vsctl show

echo "=== 4. Cloning Codebase ==="
cd /home/$USER
if [ ! -d "ryu-fl-rl-traffic-controller" ]; then
    git clone https://github.com/Ragingsnake/ryu-fl-rl-traffic-controller.git
fi
cd ryu-fl-rl-traffic-controller
git checkout main
git pull origin main

echo "=== 5. Setting up Virtualenv ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit plotly

echo "=== Environment Provisioned Successfully! ==="
```

---

## 4. Multi-Topology Training Commands (Inside `tmux`)

Always launch long-running training inside `tmux` so that disconnecting from SSH does not terminate your job:

```bash
tmux new -s train_multitopo
source venv/bin/activate
```

### 4.1 NSFNET (Baseline Reference)
```bash
# 1. Generate & Preprocess
python -m data.generate_tm --topo data/nsfnet_topology.json --timesteps 8064
python -m data.preprocess --topology nsfnet

# 2. Train FL (FedAdam, 20 rounds)
python -m fl.train_fl --topology nsfnet --strategy fedadam --rounds 20

# 3. Train RL Agent (PPO, 200,000 steps across 8 CPU cores)
python -m rl.train_rl --topology nsfnet --timesteps 200000 --n_envs 8
```

### 4.2 GÉANT (Large-Scale European Research Mesh)
```bash
# 1. Generate & Preprocess
python -m data.generate_tm --topo data/geant_topology.json --timesteps 8064
python -m data.preprocess --topology geant

# 2. Train FL (FedAdam, 20 rounds)
python -m fl.train_fl --topology geant --strategy fedadam --rounds 20

# 3. Train RL Agent (PPO, 200,000 steps)
python -m rl.train_rl --topology geant --timesteps 200000 --n_envs 8
```

### 4.3 Abilene (US Research Backbone) & Cross-Topology Transfer
```bash
# 1. Ingest/Generate & Preprocess
python -m data.load_abilene --timesteps 8064
python -m data.preprocess --topology abilene

# 2. Train FL (FedAdam, 20 rounds)
python -m fl.train_fl --topology abilene --strategy fedadam --rounds 20

# 3. Train Native RL Agent
python -m rl.train_rl --topology abilene --timesteps 200000 --n_envs 8

# 4. Or Run Cross-Topology Transfer (GEANT policy -> Abilene)
python -m scripts.demo --topology abilene --cross_eval
```

---

## 5. Hosting the Web App on Azure for Live Demonstrations

Run the web UI inside Docker or directly via Streamlit:

### Method A: Docker (Recommended for Isolated Live Demos)
```bash
cd /home/$USER/ryu-fl-rl-traffic-controller
docker compose -f app/docker-compose.yml up -d --build
```
Access the dashboard in your web browser at:
`http://<YOUR_AZURE_VM_PUBLIC_IP>:8501`

### Method B: Streamlit Direct
```bash
cd /home/$USER/ryu-fl-rl-traffic-controller
source venv/bin/activate
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

---

## 6. Downloading Benchmark Results & Model Weights to Local Machine

From your local machine terminal:
```bash
# Download all demo plots and benchmark charts
scp -i "$AZURE_SSH_KEY" -r "$AZURE_USER@$AZURE_VM_IP:/home/azureuser/ryu-fl-rl-traffic-controller/results/demo" ./local_results/

# Download trained models
scp -i "$AZURE_SSH_KEY" -r "$AZURE_USER@$AZURE_VM_IP:/home/azureuser/ryu-fl-rl-traffic-controller/models" ./local_models/
```
