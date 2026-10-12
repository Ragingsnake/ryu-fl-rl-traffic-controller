# Current System Architecture & Implementation State (Branch: main)

**Date**: October 12, 2026  
**Repository**: `prototype` (`https://github.com/Ragingsnake/ryu-fl-rl-traffic-controller.git`)  
**Active Working Branch**: `main`  
**Status**: Fully integrated closed-loop SDN + FL + RL prototype (End-to-End verified)

---

## 1. System Overview & Core Pipeline

The system is a proactive Traffic Engineering framework for SDN-controlled ISP backbones that combines:
1. **Federated Learning (FL)**: Decentralized LSTM models forecasting link utilization across administrative network domains.
2. **Deep Reinforcement Learning (DRL)**: A PPO agent dynamically routing top traffic flows across candidate paths to prevent congestion before it occurs.
3. **Software-Defined Networking (SDN)**: A Mininet emulation using Open vSwitch (OVS) and a Ryu OpenFlow 1.3 controller, linked via an asynchronous AI Bridge.

```
       [Historical Telemetry (12 steps)]
                      │
                      ▼
            ┌───────────────────┐
            │  FL Model (LSTM)  │ ──► [Predicted Link Utils (3 steps ahead)]
            └───────────────────┘                      │
                      ▲                                │
                      │                                ▼
┌─────────────────────────────────┐         ┌─────────────────────────┐
│     Current Network State       │ ──────► │      RL Agent (PPO)     │
│  (Current Utils + Flow Demands) │         └─────────────────────────┘
└─────────────────────────────────┘                      │
                                                         ▼
                                            [Path Selection (K=3)]
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │ Ryu Controller (OF 1.3) │
                                            └─────────────────────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │ Mininet Switch Datapath │
                                            └─────────────────────────┘
```

---

## 2. Network Topology: NSFNET

- **Nodes ($|V| = 14$)**: Metropolitan backbone nodes.
- **Edges ($|E| = 42$ directed links, 21 bidirectional links)**:
  - Bandwidth: 10 Mbps per link.
  - Delay: 2.0 ms to 10.0 ms propagation delay.
- **FL Domain Partitioning (0-Indexed)**:
  - **Domain 0 (West)**: Nodes 0, 1, 2, 7, 8 (24 directed links monitored)
  - **Domain 1 (Central)**: Nodes 3, 4, 5, 9 (18 directed links monitored)
  - **Domain 2 (East)**: Nodes 6, 10, 11, 12, 13 (20 directed links monitored)
- **Managed OD Pairs**: Top 20 origin-destination pairs managed dynamically via PPO ($K = 3$ precomputed candidate paths per pair via Yen's KSP). The remaining 162 OD pairs follow static OSPF shortest-path routing.

---

## 3. Data Generation & Preprocessing

- **Generator (`data/generate_tm.py`)**:
  - Metropolitan Gravity Model for spatial traffic distribution.
  - Modulated by daily ($T_{day}=288$) and weekly ($T_{week}=2016$) sine waves + Gaussian noise over 8,064 timesteps (4 weeks at 5-minute sampling intervals).
  - Stored at `data/generated/traffic_matrices.npy`.
- **Preprocessor (`data/preprocess.py`)**:
  - Simulates baseline OSPF routing to compute link utilization time-series for all 42 directed links.
  - Partitions into Domains 0, 1, and 2.
  - Creates sliding window samples: Lookback window $W_{in} = 12$ steps (1 hr), Forecast horizon $H = 3$ steps (15 min).
  - Pads link dimensions to `MAX_F_IN = 24` to ensure unified model tensor dimensions across domains during FedAvg.
  - Data stored in `data/preprocessed/` with splits: `domain_{id}_train_X.npy`, `val_X.npy`, `test_X.npy`.

---

## 4. Federated Learning Subsystem (`fl/`)

- **Model Architecture (`fl/model.py`)**:
  - `TrafficLSTM`: 2-layer LSTM, hidden dimension = 64, dropout = 0.2, linear projection to $H \times F_{in}$.
  - Target features: **Link Utilization** (normalized $[0.0, 1.0]$).
  - Dynamically configurable input dimension (padded to 24 for cross-domain FedAvg compatibility).
- **Federated Training (`fl/train_fl.py`)**:
  - Flower (`flwr`) framework using `FedAvg`.
  - Includes a direct synchronous local FedAvg fallback (`run_local_fedavg`) to ensure reliable training in headless, containerized, or non-daemon environments.
  - Saved model artifact: `models/fl_global_model.pt`.
- **Evaluation (`fl/evaluate_fl.py`)**:
  - Computes MAE, RMSE, MAPE, and $R^2$ across all test domain datasets.

---

## 5. Reinforcement Learning Subsystem (`rl/`)

- **Gym Environment (`rl/env.py`)**:
  - `NSFNETRoutingEnv`: Vectorized network simulator modeling M/M/1 queuing delays, packet loss, and link saturation.
  - **Observation Space (108 dimensions)**:
    - 44 actual link utilizations (padded/aligned).
    - 44 predicted future link utilizations from `DomainPredictor`.
    - 20 current demands of the managed OD pairs.
  - **Action Space**: MultiDiscrete array of shape `(20,)` with values in `{0, 1, 2}` representing path selection for the top 20 OD pairs.
  - **Reward Function**: Penalizes maximum link utilization (MLU), queueing delay, packet loss, and route churn (frequent switching).
  - **Dynamic Link Failure Support**: `disable_link()` and `set_link_failure()` methods for injecting edge drops during simulation without breaking observation space shape.
- **Training (`rl/train_rl.py`)**:
  - Stable-Baselines3 PPO with linear learning rate decay ($3 \times 10^{-4} \to 1 \times 10^{-5}$).
  - Saved model artifact: `models/ppo_agent.zip`.
- **Evaluation (`rl/evaluate_rl.py` & `scripts/demo.py`)**:
  - Scenarios tested:
    - **T1**: Normal Traffic
    - **T2**: Traffic Spike (+300% on top 3 OD pairs)
    - **T3**: Gradual Load (+50% ramp)
    - **T4**: Flash Crowd (5x surge on top 5 OD pairs)
    - **T5**: Link Failure (Link 4-5 down at $t=40$)
    - **T6**: Irregular Day (2.0x morning surge, 0.3x afternoon quiet period)
  - Statistical testing: Wilcoxon signed-rank tests against OSPF and Random baselines over 5 random seeds.
  - Outputs multi-panel visual timeline dashboards to `results/demo/`.

---

## 6. SDN Emulation & AI Closed Loop (`sdn/`)

- **Containerized Environment (`Dockerfile`, `entrypoint.sh`)**:
  - Ubuntu 22.04 base running Python 3.10, Mininet, Open vSwitch, and Ryu.
  - Patched `eventlet` compatibility for Python 3.10.
  - Runs OVS with **userspace datapath** (`OVSSwitch(datapath='user')`) to run seamlessly inside WSL2 and Docker containers without requiring Linux host openvswitch kernel modules.
- **Ryu Controller (`sdn/ryu_app.py`)**:
  - OpenFlow 1.3 controller monitoring ports and dumping live link byte counters to `sdn_stats.json`.
  - Listens for REST route instructions from the AI bridge.
- **AI Bridge (`sdn/ai_bridge.py`)**:
  - Reads `sdn_stats.json`, feeds telemetry into `models/ppo_agent.zip` (querying `models/fl_global_model.pt`), and computes real-time path updates.
  - Translates selected paths into OpenFlow 1.3 flow-mod rules and installs them into the switch fabric.
- **Traffic Orchestrator (`sdn/traffic_orchestrator.py`)**:
  - Generates real background TCP/UDP traffic flows between Mininet hosts using `iperf`.
- **Automated Validation (`test_sdn_headless.sh`)**:
  - Verifies the full closed-loop emulation headlessly in Docker without UI hangs.

---

## 7. Modified & Untracked Files on `main` to be Committed

- `Dockerfile`: Added userspace OVS dependencies, proxy args, and environment configs.
- `fl/model.py`: Import cleanups.
- `models/fl_global_model.pt` & `models/ppo_agent.zip`: Trained production weights.
- `rl/env.py`: Refined observation alignment and link failure support.
- `rl/train_rl.py`: Linear schedule calibration.
- `scripts/demo.py`: Added Scenario T6 (Irregular Day) and dedicated deep-dive plots.
- `sdn/ai_bridge.py`: Real-time telemetry ingestion and error handling.
- `sdn/entrypoint.sh`: Headless service orchestration.
- `sdn/mininet_topo.py`: Configured `OVSSwitch(datapath='user')` and port mappings.
- `sdn/ryu_app.py`: Telemetry reporting and flow rule installation.
- `test_sdn_headless.sh`: Automated end-to-end SDN verification script.
- `scratch_t6.py`: T6 anomaly diagnosis script.
