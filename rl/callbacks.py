import csv
import os

from stable_baselines3.common.callbacks import BaseCallback


def plot_training_curves(csv_path: str, output_dir: str = "results/demo") -> None:
    """Standalone utility to plot publication-quality training curves from CSV."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        print(f"Warning: {csv_path} has fewer than 2 data points.")
        return

    os.makedirs(output_dir, exist_ok=True)
    window = min(50, max(5, len(df) // 20))
    df["reward_smooth"] = df["reward"].rolling(window=window, min_periods=1).mean()
    df["mlu_smooth"] = df["mlu"].rolling(window=window, min_periods=1).mean()
    df["delay_smooth"] = df["avg_delay"].rolling(window=window, min_periods=1).mean()
    df["loss_smooth"] = df["avg_loss"].rolling(window=window, min_periods=1).mean()

    # 1. Reward Curve
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df["step"], df["reward"], color="#90caf9", alpha=0.35, label="Raw Episode Reward")
    ax.plot(df["step"], df["reward_smooth"], color="#1565c0", linewidth=2.2, label=f"Moving Average (window={window})")
    ax.set_title("PPO Agent Training Convergence: Episode Reward over Time", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel("Cumulative Episode Reward", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    reward_path = os.path.join(output_dir, "rl_training_reward_curve.png")
    plt.savefig(reward_path)
    plt.close()

    # 2. Comprehensive 4-Panel Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    axes[0, 0].plot(df["step"], df["reward"], color="#90caf9", alpha=0.3)
    axes[0, 0].plot(df["step"], df["reward_smooth"], color="#1976d2", linewidth=2)
    axes[0, 0].set_title("(a) Episode Cumulative Reward", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Training Timesteps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    axes[0, 1].plot(df["step"], df["mlu"], color="#ffab91", alpha=0.3)
    axes[0, 1].plot(df["step"], df["mlu_smooth"], color="#d84315", linewidth=2)
    axes[0, 1].axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Capacity Limit (1.0)")
    axes[0, 1].set_title("(b) Maximum Link Utilization (MLU)", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Training Timesteps")
    axes[0, 1].set_ylabel("MLU")
    axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].plot(df["step"], df["avg_delay"], color="#ce93d8", alpha=0.3)
    axes[1, 0].plot(df["step"], df["delay_smooth"], color="#6a1b9a", linewidth=2)
    axes[1, 0].set_title("(c) Average End-to-End Delay (ms)", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Training Timesteps")
    axes[1, 0].set_ylabel("Delay (ms)")
    axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    axes[1, 1].plot(df["step"], df["avg_loss"], color="#ef9a9a", alpha=0.3)
    axes[1, 1].plot(df["step"], df["loss_smooth"], color="#c62828", linewidth=2)
    axes[1, 1].set_title("(d) Average Packet Loss Ratio", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Training Timesteps")
    axes[1, 1].set_ylabel("Packet Loss")
    axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.suptitle("Deep Reinforcement Learning (PPO) Training Dynamics Across Timesteps", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    diag_path = os.path.join(output_dir, "rl_training_curves.png")
    plt.savefig(diag_path)
    plt.close()
    print(f"Training curves saved to {reward_path} and {diag_path}")


class MetricsCallback(BaseCallback):
    """
    Custom SB3 Callback for logging and saving metrics to CSV.
    Logs MLU, avg_delay, avg_loss, and reward per episode.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Initialize or truncate CSV only when training begins."""
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "episode", "reward", "mlu", "avg_delay", "avg_loss"])

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    if "episode" in info:
                        self.episode_count += 1
                        ep_reward = info["episode"]["r"]
                        mlu = info.get("mlu", 0.0)
                        avg_delay = info.get("avg_delay", 0.0)
                        avg_loss = info.get("avg_loss", 0.0)

                        with open(self.csv_path, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [self.num_timesteps, self.episode_count, ep_reward, mlu, avg_delay, avg_loss]
                            )

                        if self.verbose > 0 and self.episode_count % 10 == 0:
                            print(
                                f"Episode {self.episode_count} | Step {self.num_timesteps} | Reward: {ep_reward:.2f} | MLU: {mlu:.3f}"
                            )

        return True

    def _on_training_end(self) -> None:
        """Automatically plot training curves when training completes."""
        try:
            demo_dir = os.path.join(os.path.dirname(self.log_dir), "demo")
            plot_training_curves(self.csv_path, demo_dir)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to generate training plots: {e}")


if __name__ == "__main__":
    cb = MetricsCallback(log_dir="./test_logs", verbose=1)
    print("MetricsCallback initialized successfully.")

