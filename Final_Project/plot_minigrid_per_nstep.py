"""Plot learning curves for PER+N-step DQN on MiniGrid FourRooms (stacked observations)."""

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
    """Plot learning curve for PER+N-step DQN."""
    run_dir = Path(run_dir)
    train_log_path = run_dir / "train_log.csv"

    # Read training log
    train_frames = []
    train_rewards = []

    with open(train_log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_frames.append(int(row["frames"]))
            train_rewards.append(float(row["episode_reward"]))

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training curve (moving average)
    train_rewards_ma = moving_average(train_rewards, window)
    train_frames_ma = train_frames[len(train_frames) - len(train_rewards_ma) :]

    ax.plot(
        train_frames_ma,
        train_rewards_ma,
        label=f"Training (MA {window})",
        color="steelblue",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("MiniGrid FourRooms: PER+N-step DQN", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = run_dir / "learning_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Learning curve saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curve for PER+N-step DQN on MiniGrid FourRooms"
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
