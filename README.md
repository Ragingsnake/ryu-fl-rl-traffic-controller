# FL+RL Proactive Traffic Engineering for SDN

A proactive traffic engineering system utilizing a Federated-Learning LSTM model to predict future traffic and a Reinforcement Learning (PPO) agent to pre-compute routing paths before congestion occurs.

## Prerequisites
- Python 3.10+
- (Optional) Mininet + Ryu for Phase 2 validation

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/MacOS
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

You can run the entire pipeline (generate data, train FL, train RL, benchmark) automatically using the provided orchestrator script:

```bash
python scripts/run_full_pipeline.py --quick
```

## Full Pipeline Commands

If you prefer to run steps individually:

1. **Generate Data:**
```bash
python data/generate_tm.py
python data/preprocess.py
```

2. **Train FL Model:**
```bash
python fl/train_fl.py
```

3. **Train RL Agent:**
```bash
python rl/train_rl.py
```

4. **Evaluate/Benchmark:**
```bash
python rl/evaluate_rl.py
```

5. **Demo Visualization:**
```bash
python scripts/demo.py
```

## Project Structure

- `data/`: Traffic matrix generation and preprocessing
- `fl/`: Federated Learning (LSTM) models and training logic
- `rl/`: Reinforcement Learning (PPO) environment and agent
- `sdn/`: SDN Controller (Ryu) and Network Emulation (Mininet)
- `tests/`: PyTest unit and integration tests
- `scripts/`: Orchestration and demo scripts

## CI/CD
This project uses GitHub Actions for continuous integration, automatically running code linting (`ruff`) and unit tests (`pytest`) upon pushes and PRs to `main`.
