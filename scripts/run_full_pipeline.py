import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(command: list[str], step_name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"Starting {step_name}...")
    print(f"Command: {' '.join(command)}")
    print(f"{'=' * 50}\n")

    start_time = time.time()
    root_dir = str(Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(command, check=True, env=env, cwd=root_dir)
    except subprocess.CalledProcessError as e:
        print(f"\nError: {step_name} failed with exit code {e.returncode}.")
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\n{step_name} completed in {duration:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="End-to-End FL+RL Traffic Engineering Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (2 FL rounds, 1000 RL steps)")
    args = parser.parse_args()

    # Base commands
    cmd_generate = [sys.executable, "data/generate_tm.py"]
    cmd_preprocess = [sys.executable, "data/preprocess.py"]
    cmd_fl = [sys.executable, "fl/train_fl.py"]
    cmd_eval_fl = [sys.executable, "fl/evaluate_fl.py"]
    cmd_rl = [sys.executable, "rl/train_rl.py"]
    cmd_benchmark = [sys.executable, "rl/evaluate_rl.py"]

    if args.quick:
        print("Running in QUICK mode...")
        cmd_fl.extend(["--rounds", "2"])
        cmd_rl.extend(["--timesteps", "1000"])
        cmd_benchmark.extend(["--seeds", "42", "--scenarios", "T1", "T2", "--baselines", "B1", "Proposed"])
    else:
        print("Running full pipeline...")

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
