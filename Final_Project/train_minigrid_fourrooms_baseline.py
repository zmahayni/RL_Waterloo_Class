"""Training script for baseline DQN on MiniGrid FourRooms."""

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from minigrid_env import make_env, get_env_info
from dqn_agent_minigrid import DQNAgentMiniGrid
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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


def evaluate_agent(agent, num_episodes=200, seed=None):
    """
    Evaluate agent performance.

    Args:
        agent: DQN agent to evaluate
        num_episodes: Number of evaluation episodes
        seed: Random seed for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    if seed is not None:
        set_seed(seed)

    env = make_env(seed=seed)

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_len = 0
        done = False

        while not done:
            # Greedy action selection (epsilon=0)
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_len += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

        # Check if goal was reached (success)
        # MiniGrid gives reward > 0 when goal is reached
        if episode_reward > 0:
            successes += 1

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}")

    env.close()

    results = {
        "mean_reward": np.mean(episode_rewards),
        "success_rate": successes / num_episodes,
        "avg_episode_len": np.mean(episode_lengths),
        "num_episodes": num_episodes,
    }

    return results


def train_dqn(args):
    """Train DQN agent on MiniGrid FourRooms."""
    # Set seed
    set_seed(args.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = Path(f"runs/minigrid/fourrooms/baseline/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Create environment and get dimensions
    state_size, action_size = get_env_info()
    print("\nEnvironment info:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")

    # Initialize agent
    agent = DQNAgentMiniGrid(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_decay=1.0,  # Manual decay
        epsilon_min=args.epsilon_end,
        hidden_size=args.hidden_size,
        buffer_capacity=100000,
    )

    # Epsilon schedule
    epsilon_decay_episodes = int(args.num_episodes * 0.6)
    epsilon_step = (args.epsilon_start - args.epsilon_end) / epsilon_decay_episodes

    # Initialize CSV logs
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"

    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "episode_reward", "episode_len", "epsilon", "loss", "success"]
        )

    with open(eval_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "mean_reward", "success_rate", "avg_episode_len"])

    # Training loop
    print(f"\nStarting training for {args.num_episodes} episodes...")
    print("=" * 80)

    env = make_env(seed=args.seed)

    for episode in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_len = 0
        losses = []
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs, training=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)

            # Train agent
            loss = agent.train(batch_size=args.batch_size)
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            episode_reward += reward
            episode_len += 1

        # Determine success (goal reached)
        success = 1 if episode_reward > 0 else 0

        # Decay epsilon
        if episode < epsilon_decay_episodes:
            agent.epsilon = max(
                args.epsilon_end, args.epsilon_start - epsilon_step * episode
            )

        # Update target network
        if (episode + 1) % args.target_update_freq == 0:
            agent.update_target_network()

        # Log training metrics
        avg_loss = np.mean(losses) if losses else 0.0
        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode + 1,
                    episode_reward,
                    episode_len,
                    agent.epsilon,
                    avg_loss,
                    success,
                ]
            )

        # Print progress
        if (episode + 1) % 1000 == 0:
            print(
                f"Episode {episode + 1}/{args.num_episodes} | Reward: {episode_reward:.2f} | Success: {success} | Epsilon: {agent.epsilon:.3f}"
            )

        # Periodic evaluation
        if (episode + 1) % args.eval_every == 0:
            print(
                f"\nEpisode {episode + 1}/{args.num_episodes} - Running evaluation..."
            )
            eval_results = evaluate_agent(
                agent, num_episodes=args.num_eval_episodes, seed=args.seed + 999
            )

            print(f"  Mean reward: {eval_results['mean_reward']:.4f}")
            print(f"  Success rate: {eval_results['success_rate']:.3f}")
            print(f"  Avg episode length: {eval_results['avg_episode_len']:.1f}")

            # Log evaluation metrics
            with open(eval_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode + 1,
                        eval_results["mean_reward"],
                        eval_results["success_rate"],
                        eval_results["avg_episode_len"],
                    ]
                )

        # Save checkpoint
        if (episode + 1) % args.checkpoint_every == 0:
            checkpoint_path = run_dir / f"checkpoint_ep{episode + 1}"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint at episode {episode + 1}")

    env.close()

    # Save final model
    final_path = run_dir / "final_model"
    agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {run_dir}")

    # Generate learning curve
    plot_learning_curve(run_dir, window=args.plot_window)

    # Final evaluation
    print("\nRunning final evaluation...")
    final_eval = evaluate_agent(
        agent, num_episodes=args.num_eval_episodes, seed=args.seed + 999
    )

    print("\n" + "=" * 80)
    print("Training Summary:")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Final eval mean reward: {final_eval['mean_reward']:.4f}")
    print(f"Final eval success rate: {final_eval['success_rate']:.3f}")
    print(f"Final eval avg episode length: {final_eval['avg_episode_len']:.1f}")
    print("=" * 80)

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline DQN on MiniGrid FourRooms"
    )

    # Training params
    parser.add_argument(
        "--num_episodes", type=int, default=50000, help="Number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=1000,
        help="Target network update frequency",
    )

    # Epsilon schedule
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="Starting epsilon"
    )
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")

    # Evaluation
    parser.add_argument(
        "--eval_every", type=int, default=2000, help="Evaluate every N episodes"
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=200,
        help="Number of episodes for evaluation",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Save checkpoint every N episodes",
    )

    # Misc
    parser.add_argument("--run_name", type=str, default="", help="Name for this run")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--plot_window", type=int, default=200, help="Moving average window for plots"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dqn(args)


if __name__ == "__main__":
    main()
