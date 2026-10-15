import csv
import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


def plot_training_curves(csv_path: str, output_dir: str = "results/demo", title_suffix: str = "") -> None:
    """Standalone utility to plot publication-quality, smooth training curves from CSV."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        print(f"Warning: {csv_path} has fewer than 2 data points.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Aggregate parallel environments by step (eliminates vertical step-ladder lines and square waves)
    df_step = df.groupby("step", as_index=False).agg({
        "reward": ["mean", "std"],
        "mlu": ["mean", "std"],
        "avg_delay": ["mean", "std"],
        "avg_loss": ["mean", "std"],
    })
    df_step.columns = [
        "step",
        "reward", "reward_std",
        "mlu", "mlu_std",
        "avg_delay", "delay_std",
        "avg_loss", "loss_std",
    ]
    df_step["reward_std"] = df_step["reward_std"].fillna(0.0)
    df_step["mlu_std"] = df_step["mlu_std"].fillna(0.0)
    df_step["delay_std"] = df_step["delay_std"].fillna(0.0)
    df_step["loss_std"] = df_step["loss_std"].fillna(0.0)

    # 2. Compute smooth Exponential Moving Average (EMA) across distinct timesteps
    span = max(3, min(20, len(df_step) // 4))
    df_step["reward_smooth"] = df_step["reward"].ewm(span=span, adjust=False).mean()
    df_step["mlu_smooth"] = df_step["mlu"].ewm(span=span, adjust=False).mean()
    df_step["delay_smooth"] = df_step["avg_delay"].ewm(span=span, adjust=False).mean()
    df_step["loss_smooth"] = df_step["avg_loss"].ewm(span=span, adjust=False).mean()

    # Helper function for publication-grade continuous spline interpolation
    def spline_smooth(x, y, num_points=300):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 4:
            return x, y
        try:
            from scipy.interpolate import make_interp_spline
            spl = make_interp_spline(x, y, k=min(3, len(x) - 1))
            x_new = np.linspace(x.min(), x.max(), num_points)
            y_new = spl(x_new)
            return x_new, y_new
        except Exception:
            return x, y

    x_raw = df_step["step"].to_numpy()
    x_spline, reward_trend = spline_smooth(x_raw, df_step["reward_smooth"].to_numpy())
    _, reward_lo = spline_smooth(x_raw, (df_step["reward"] - 0.75 * df_step["reward_std"]).to_numpy())
    _, reward_hi = spline_smooth(x_raw, (df_step["reward"] + 0.75 * df_step["reward_std"]).to_numpy())

    # 3. Single Reward Convergence Curve
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Shaded confidence band (±0.75 std across parallel envs)
    ax.fill_between(
        x_spline,
        reward_lo,
        reward_hi,
        color="#90caf9",
        alpha=0.25,
        label="Cross-Env Variance (±0.75σ)",
    )
    ax.scatter(df_step["step"], df_step["reward"], color="#64b5f6", alpha=0.5, s=18, label="Step Evaluation Points")
    ax.plot(x_spline, reward_trend, color="#1565c0", linewidth=2.6, label="Convergence Trend (Spline EMA)")

    heading = f"PPO Training Convergence ({title_suffix}): Episode Reward over Timesteps" if title_suffix else "PPO Agent Training Convergence: Episode Reward over Time"
    ax.set_title(heading, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Timesteps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Episode Reward", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9)
    plt.tight_layout()
    reward_path = os.path.join(output_dir, "rl_training_reward_curve.png")
    plt.savefig(reward_path)
    plt.close()

    # 4. Comprehensive 4-Panel Diagnostics (Smooth Splines)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)

    # (a) Reward
    axes[0, 0].fill_between(x_spline, reward_lo, reward_hi, color="#90caf9", alpha=0.25)
    axes[0, 0].scatter(df_step["step"], df_step["reward"], color="#90caf9", alpha=0.5, s=14)
    axes[0, 0].plot(x_spline, reward_trend, color="#1976d2", linewidth=2.4, label="Spline EMA")
    axes[0, 0].set_title("(a) Episode Cumulative Reward", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Training Timesteps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="lower right", framealpha=0.9, fontsize=9)

    # (b) MLU
    _, mlu_trend = spline_smooth(x_raw, df_step["mlu_smooth"].to_numpy())
    _, mlu_lo = spline_smooth(x_raw, np.maximum(0, df_step["mlu"] - 0.75 * df_step["mlu_std"]).to_numpy())
    _, mlu_hi = spline_smooth(x_raw, (df_step["mlu"] + 0.75 * df_step["mlu_std"]).to_numpy())
    axes[0, 1].fill_between(x_spline, mlu_lo, mlu_hi, color="#ffab91", alpha=0.25)
    axes[0, 1].scatter(df_step["step"], df_step["mlu"], color="#ffab91", alpha=0.5, s=14)
    axes[0, 1].plot(x_spline, mlu_trend, color="#d84315", linewidth=2.4, label="Spline EMA")
    axes[0, 1].axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Capacity Limit (1.0)")
    axes[0, 1].set_title("(b) Maximum Link Utilization (MLU)", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Training Timesteps")
    axes[0, 1].set_ylabel("MLU")
    axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right", framealpha=0.9, fontsize=9)

    # (c) Delay
    _, delay_trend = spline_smooth(x_raw, df_step["delay_smooth"].to_numpy())
    _, delay_lo = spline_smooth(x_raw, np.maximum(0, df_step["avg_delay"] - 0.75 * df_step["delay_std"]).to_numpy())
    _, delay_hi = spline_smooth(x_raw, (df_step["avg_delay"] + 0.75 * df_step["delay_std"]).to_numpy())
    axes[1, 0].fill_between(x_spline, delay_lo, delay_hi, color="#ce93d8", alpha=0.25)
    axes[1, 0].scatter(df_step["step"], df_step["avg_delay"], color="#ce93d8", alpha=0.5, s=14)
    axes[1, 0].plot(x_spline, delay_trend, color="#6a1b9a", linewidth=2.4, label="Spline EMA")
    axes[1, 0].set_title("(c) Average End-to-End Delay (ms)", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Training Timesteps")
    axes[1, 0].set_ylabel("Delay (ms)")
    axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", framealpha=0.9, fontsize=9)

    # (d) Packet Loss
    _, loss_trend = spline_smooth(x_raw, df_step["loss_smooth"].to_numpy())
    _, loss_lo = spline_smooth(x_raw, np.maximum(0, df_step["avg_loss"] - 0.75 * df_step["loss_std"]).to_numpy())
    _, loss_hi = spline_smooth(x_raw, (df_step["avg_loss"] + 0.75 * df_step["loss_std"]).to_numpy())
    axes[1, 1].fill_between(x_spline, loss_lo, loss_hi, color="#ef9a9a", alpha=0.25)
    axes[1, 1].scatter(df_step["step"], df_step["avg_loss"], color="#ef9a9a", alpha=0.5, s=14)
    axes[1, 1].plot(x_spline, loss_trend, color="#c62828", linewidth=2.4, label="Spline EMA")
    axes[1, 1].set_title("(d) Average Packet Loss Ratio", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Training Timesteps")
    axes[1, 1].set_ylabel("Packet Loss")
    axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.suptitle(
        f"Deep Reinforcement Learning (PPO) Training Dynamics ({title_suffix})",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
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

    def __init__(self, log_dir: str, output_dir: str = None, title_suffix: str = "", verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(self.log_dir), "demo")
        self.title_suffix = title_suffix
        self.csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
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
            plot_training_curves(self.csv_path, self.output_dir, self.title_suffix)
            if self.title_suffix == "NSFNET" and self.output_dir != "results/demo":
                plot_training_curves(self.csv_path, "results/demo", self.title_suffix)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to generate training plots: {e}")


if __name__ == "__main__":
    cb = MetricsCallback(log_dir="./test_logs", verbose=1)
    print("MetricsCallback initialized successfully.")

