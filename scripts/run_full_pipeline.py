import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_step(command: list[str], step_name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"Starting {step_name}...")
    print(f"Command: {' '.join(command)}")
    print(f"{'=' * 50}\n", flush=True)

    start_time = time.time()
    root_dir = str(Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        subprocess.run(command, check=True, env=env, cwd=root_dir)
    except subprocess.CalledProcessError as e:
        print(f"\nError: {step_name} failed with exit code {e.returncode}.", flush=True)
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\n{step_name} completed in {duration:.2f} seconds.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-End FL+RL Traffic Engineering Pipeline")
    parser.add_argument(
        "--topology", type=str, default="nsfnet", choices=["nsfnet", "geant", "abilene"], help="Topology name"
    )
    parser.add_argument(
        "--strategy", type=str, default="fedadam", choices=["fedavg", "fedadam"], help="FL aggregation strategy"
    )
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (2 FL rounds, 1000 RL steps)")
    args = parser.parse_args()

    # Base commands
    if args.topology == "nsfnet":
        topo_json = "data/nsfnet_topology.json"
    elif args.topology == "geant":
        topo_json = "data/geant_topology.json"
    else:
        topo_json = "data/abilene_topology.json"

    cmd_generate = [sys.executable, "-m", "data.generate_tm", "--topo", topo_json]
    cmd_preprocess = [sys.executable, "-m", "data.preprocess", "--topology", args.topology]
    cmd_fl = [sys.executable, "-m", "fl.train_fl", "--topology", args.topology, "--strategy", args.strategy]
    cmd_eval_fl = [sys.executable, "-m", "fl.evaluate_fl", "--topology", args.topology]
    cmd_rl = [sys.executable, "-m", "rl.train_rl", "--topology", args.topology]
    cmd_benchmark = [sys.executable, "-m", "scripts.demo", "--topology", args.topology]

    if args.quick:
        print(f"Running in QUICK mode for {args.topology} with {args.strategy}...")
        cmd_generate.extend(["--timesteps", "200"])
        cmd_fl.extend(["--rounds", "2"])
        cmd_rl.extend(["--timesteps", "1000"])
        cmd_benchmark.extend(["--seeds", "42", "--steps", "20"])
    else:
        print(f"Running full pipeline for {args.topology} with {args.strategy}...")

    # Validate we're in the right directory
    if not Path("data").exists() or not Path("fl").exists():
        print("Error: Please run this script from the project root directory (prototype/).")
        print("Example: python scripts/run_full_pipeline.py")
        sys.exit(1)

    total_start_time = time.time()

    # Execute steps
    run_step(cmd_generate, "Step 1: Generate Traffic Matrices")
    run_step(cmd_preprocess, "Step 2: Preprocess Data")
    run_step(cmd_fl, "Step 3: Train FL Model")
    run_step(cmd_eval_fl, "Step 3.5: Evaluate FL Predictions")
    run_step(cmd_rl, "Step 4: Train RL Agent")
    run_step(cmd_benchmark, "Step 5: Run Benchmarks")

    total_duration = time.time() - total_start_time
    print(f"\n{'=' * 50}")
    print(f"PIPELINE COMPLETED in {total_duration:.2f} seconds.")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
