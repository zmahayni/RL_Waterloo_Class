"""Plot learning curves for MiniGrid FourRooms baseline experiments."""

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
    """Plot learning curve from training and evaluation logs."""
    run_dir = Path(run_dir)
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Read training log
    train_episodes = []
    train_rewards = []
    train_success = []

    with open(train_log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_episodes.append(int(row["episode"]))
            train_rewards.append(float(row["episode_reward"]))
            train_success.append(float(row["success"]))

    # Read evaluation log
    eval_episodes = []
    eval_rewards = []
    eval_success_rates = []

    if eval_log_path.exists():
        with open(eval_log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_episodes.append(int(row["episode"]))
                eval_rewards.append(float(row["mean_reward"]))
                eval_success_rates.append(float(row["success_rate"]))

    # Plot 1: Reward curve
    train_rewards_ma = moving_average(train_rewards, window)
    train_episodes_ma = train_episodes[len(train_episodes) - len(train_rewards_ma) :]

    ax1.plot(
        train_episodes_ma,
        train_rewards_ma,
        label=f"Training (MA {window})",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )

    if eval_episodes:
        ax1.scatter(
            eval_episodes,
            eval_rewards,
            label="Evaluation",
            color="red",
            s=50,
            zorder=5,
            alpha=0.7,
        )
        ax1.plot(eval_episodes, eval_rewards, color="red", linewidth=1.5, alpha=0.5)

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.set_title(
        "MiniGrid FourRooms: Reward Learning Curve", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success curve
    train_success_ma = moving_average(train_success, window)
    train_episodes_ma_success = train_episodes[
        len(train_episodes) - len(train_success_ma) :
    ]

    ax2.plot(
        train_episodes_ma_success,
        train_success_ma,
        label=f"Training Success (MA {window})",
        color="green",
        linewidth=2,
        alpha=0.8,
    )

    if eval_episodes:
        ax2.scatter(
            eval_episodes,
            eval_success_rates,
            label="Evaluation Success Rate",
            color="red",
            s=50,
            zorder=5,
            alpha=0.7,
        )
        ax2.plot(
            eval_episodes, eval_success_rates, color="red", linewidth=1.5, alpha=0.5
        )

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Success Rate", fontsize=12)
    ax2.set_title(
        "MiniGrid FourRooms: Success Rate Curve", fontsize=14, fontweight="bold"
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Save plot
    output_path = run_dir / "learning_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Learning curve saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves for MiniGrid FourRooms baseline experiments"
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
