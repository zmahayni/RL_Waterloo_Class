"""Plot learning curves for NFSP baseline experiments."""

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=200):
    """Compute moving average with given window size."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_learning_curve(run_dir, window=200):
    """
    Plot learning curve from training and evaluation logs.

    Args:
        run_dir: Path to run directory containing train_log.csv and eval_log.csv
        window: Window size for moving average
    """
    run_dir = Path(run_dir)
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"

    # Read training log
    train_episodes = []
    train_rewards = []

    with open(train_log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_episodes.append(int(row["episode"]))
            train_rewards.append(float(row["episode_reward"]))

    # Read evaluation log
    eval_episodes = []
    eval_rewards = []

    if eval_log_path.exists():
        with open(eval_log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_episodes.append(int(row["episode"]))
                eval_rewards.append(float(row["mean_reward"]))

    # Compute moving average for training
    train_ma = moving_average(train_rewards, window)
    train_ma_episodes = (
        train_episodes[window - 1 :]
        if len(train_episodes) >= window
        else train_episodes
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training moving average
    ax.plot(
        train_ma_episodes,
        train_ma,
        label=f"Training (MA {window})",
        linewidth=2,
        alpha=0.8,
    )

    # Plot evaluation points
    if eval_episodes:
        ax.scatter(
            eval_episodes,
            eval_rewards,
            label="Evaluation",
            color="red",
            s=50,
            zorder=5,
            alpha=0.7,
        )
        ax.plot(eval_episodes, eval_rewards, color="red", linewidth=1.5, alpha=0.5)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(
        "Leduc Hold'em: NFSP Baseline Learning Curve", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = run_dir / "learning_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Learning curve saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves for NFSP baseline experiments"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to run directory"
    )
    parser.add_argument(
        "--window", type=int, default=200, help="Moving average window size"
    )

    args = parser.parse_args()

    plot_learning_curve(args.run_dir, args.window)


if __name__ == "__main__":
    main()
