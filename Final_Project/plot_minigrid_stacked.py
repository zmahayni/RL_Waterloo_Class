"""Plot learning curves for MiniGrid FourRooms with stacked observations (frame-based)."""

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


def plot_learning_curves(run_dir, window=200):
    """Plot reward and success learning curves from training and evaluation logs."""
    run_dir = Path(run_dir)
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"

    # Read training log
    train_episodes = []
    train_frames = []
    train_rewards = []
    train_success = []

    with open(train_log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_episodes.append(int(row["episode"]))
            train_frames.append(int(row["frames"]))
            train_rewards.append(float(row["episode_reward"]))
            train_success.append(float(row["success"]))

    # Read evaluation log (separate greedy and exploration)
    eval_frames_greedy = []
    eval_rewards_greedy = []
    eval_success_greedy = []

    eval_frames_explore = []
    eval_rewards_explore = []
    eval_success_explore = []

    if eval_log_path.exists():
        with open(eval_log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames = int(row["frames"])
                epsilon = float(row["eval_epsilon"])
                reward = float(row["mean_reward"])
                success = float(row["success_rate"])

                if epsilon == 0.0:  # Greedy
                    eval_frames_greedy.append(frames)
                    eval_rewards_greedy.append(reward)
                    eval_success_greedy.append(success)
                elif epsilon == 0.1:  # Exploration
                    eval_frames_explore.append(frames)
                    eval_rewards_explore.append(reward)
                    eval_success_explore.append(success)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Reward curve
    train_rewards_ma = moving_average(train_rewards, window)
    # Use frames for x-axis (align with moving average)
    train_frames_ma = train_frames[len(train_frames) - len(train_rewards_ma) :]

    ax1.plot(
        train_frames_ma,
        train_rewards_ma,
        label=f"Training (MA {window})",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )

    if eval_frames_greedy:
        ax1.scatter(
            eval_frames_greedy,
            eval_rewards_greedy,
            label="Eval (greedy, ε=0.0)",
            color="red",
            s=50,
            zorder=5,
            alpha=0.7,
        )
        ax1.plot(
            eval_frames_greedy,
            eval_rewards_greedy,
            color="red",
            linewidth=1.5,
            alpha=0.5,
        )

    if eval_frames_explore:
        ax1.scatter(
            eval_frames_explore,
            eval_rewards_explore,
            label="Eval (explore, ε=0.1)",
            color="orange",
            s=50,
            zorder=5,
            alpha=0.7,
            marker="x",
        )
        ax1.plot(
            eval_frames_explore,
            eval_rewards_explore,
            color="orange",
            linewidth=1.5,
            alpha=0.5,
            linestyle="--",
        )

    ax1.set_xlabel("Frames", fontsize=12)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.set_title(
        "MiniGrid FourRooms (Stacked): Reward Curve", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success curve
    train_success_ma = moving_average(train_success, window)
    train_frames_ma_success = train_frames[len(train_frames) - len(train_success_ma) :]

    ax2.plot(
        train_frames_ma_success,
        train_success_ma,
        label=f"Training Success (MA {window})",
        color="green",
        linewidth=2,
        alpha=0.8,
    )

    if eval_frames_greedy:
        ax2.scatter(
            eval_frames_greedy,
            eval_success_greedy,
            label="Eval Success (greedy, ε=0.0)",
            color="red",
            s=50,
            zorder=5,
            alpha=0.7,
        )
        ax2.plot(
            eval_frames_greedy,
            eval_success_greedy,
            color="red",
            linewidth=1.5,
            alpha=0.5,
        )

    if eval_frames_explore:
        ax2.scatter(
            eval_frames_explore,
            eval_success_explore,
            label="Eval Success (explore, ε=0.1)",
            color="orange",
            s=50,
            zorder=5,
            alpha=0.7,
            marker="x",
        )
        ax2.plot(
            eval_frames_explore,
            eval_success_explore,
            color="orange",
            linewidth=1.5,
            alpha=0.5,
            linestyle="--",
        )

    ax2.set_xlabel("Frames", fontsize=12)
    ax2.set_ylabel("Success Rate", fontsize=12)
    ax2.set_title(
        "MiniGrid FourRooms (Stacked): Success Rate Curve",
        fontsize=14,
        fontweight="bold",
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
        description="Plot learning curves for MiniGrid FourRooms with stacked observations"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to run directory"
    )
    parser.add_argument(
        "--window", type=int, default=200, help="Moving average window size"
    )

    args = parser.parse_args()

    plot_learning_curves(args.run_dir, args.window)


if __name__ == "__main__":
    main()
