import csv
import os

from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    Custom SB3 Callback for logging and saving metrics to CSV.
    Logs MLU, avg_delay, avg_loss, and reward per episode.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(self.log_dir, "training_metrics.csv")

        # Create dir if not exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize CSV
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "episode", "reward", "mlu", "avg_delay", "avg_loss"])

        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode is done in any environment
        # self.locals contains locals() from the step function in stable baselines
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    if "episode" in info:
                        self.episode_count += 1
                        ep_reward = info["episode"]["r"]
                        # Metrics should be in info dictionary by env.step()
                        mlu = info.get("mlu", 0.0)
                        avg_delay = info.get("avg_delay", 0.0)
                        avg_loss = info.get("avg_loss", 0.0)

                        # Save to CSV
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


if __name__ == "__main__":
    # Smoke test for callback
    cb = MetricsCallback(log_dir="./test_logs", verbose=1)
    print("MetricsCallback initialized successfully.")
