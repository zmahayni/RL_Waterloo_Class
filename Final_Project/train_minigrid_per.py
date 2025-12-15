"""Training script for DQN with PER on MiniGrid FourRooms (frame-based).

Supports:
- PER-only (1-step + PER)
- PER + N-step (N-step + PER)
"""

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from minigrid_stacked_obs_wrapper import make_env, get_env_info
from dqn_agent_minigrid_per import DQNAgentPER
from dqn_agent_minigrid_per_nstep import DQNAgentPERNStep


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(agent, num_episodes=200, seed=None, eval_epsilon=0.0):
    """
    Evaluate agent performance.

    Args:
        agent: DQN agent to evaluate
        num_episodes: Number of evaluation episodes
        seed: Random seed for evaluation
        eval_epsilon: Epsilon for evaluation (0.0 = greedy, 0.1 = mild exploration)

    Returns:
        Dictionary with evaluation metrics
    """
    if seed is not None:
        set_seed(seed)

    env = make_env(seed=seed)

    # Save original epsilon and set eval epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = eval_epsilon

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_len = 0
        done = False

        while not done:
            # Use eval_epsilon for action selection
            action = agent.select_action(
                obs, training=True
            )  # training=True to use epsilon
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_len += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

        # Check if goal was reached (success)
        if episode_reward > 0:
            successes += 1

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}")

    env.close()

    # Restore original epsilon
    agent.epsilon = original_epsilon

    results = {
        "mean_reward": np.mean(episode_rewards),
        "success_rate": successes / num_episodes,
        "avg_episode_len": np.mean(episode_lengths),
        "num_episodes": num_episodes,
        "eval_epsilon": eval_epsilon,
    }

    return results


def train_dqn(args):
    """Train DQN agent with PER on MiniGrid FourRooms (frame-based)."""
    # Set seed
    set_seed(args.seed)

    # Determine run directory based on configuration
    if args.use_per and args.n_step > 1:
        base_dir = "runs/minigrid/fourrooms/per_nstep"
    elif args.use_per:
        base_dir = "runs/minigrid/fourrooms/per"
    else:
        raise ValueError("This script requires --use_per flag")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = Path(f"{base_dir}/{run_name}")
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
    print(f"  PER enabled: {args.use_per}")
    print(f"  N-step: {args.n_step}")

    # Initialize agent
    if args.n_step > 1:
        # PER + N-step
        agent = DQNAgentPERNStep(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            n_step=args.n_step,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            hidden_size=args.hidden_size,
            buffer_capacity=args.buffer_capacity,
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=args.total_frames,
            per_eps=args.per_eps,
            device=args.device,
        )
    else:
        # PER only (1-step)
        agent = DQNAgentPER(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            hidden_size=args.hidden_size,
            buffer_capacity=args.buffer_capacity,
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=args.total_frames,
            per_eps=args.per_eps,
            device=args.device,
        )

    # Epsilon schedule (frame-based)
    epsilon_decay_frames = int(args.total_frames * 0.6)

    # Initialize CSV logs
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"

    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "frames",
                "episode_reward",
                "episode_len",
                "epsilon",
                "loss",
                "success",
            ]
        )

    with open(eval_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frames", "eval_epsilon", "mean_reward", "success_rate", "avg_episode_len"]
        )

    # Training loop
    print(f"\nStarting training for {args.total_frames} frames...")
    print("=" * 80)

    env = make_env(seed=args.seed)

    total_frames = 0
    episode = 0
    next_eval_frames = args.eval_every_frames
    next_checkpoint_frames = args.checkpoint_every_frames

    while total_frames < args.total_frames:
        episode += 1
        obs, info = env.reset()
        episode_reward = 0
        episode_len = 0
        done = False
        losses = []

        while not done and total_frames < args.total_frames:
            # Select action
            action = agent.select_action(obs, training=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)

            # Train
            loss = agent.train_step(args.batch_size)
            if loss > 0:
                losses.append(loss)

            # Update state
            obs = next_obs
            episode_reward += reward
            episode_len += 1
            total_frames += 1

            # Update epsilon
            agent.update_epsilon(total_frames, epsilon_decay_frames)

            # Update target network
            if total_frames % args.target_update_freq == 0:
                agent.update_target_network()

            # Periodic evaluation
            if total_frames >= next_eval_frames:
                print(
                    f"\nFrames {total_frames}/{args.total_frames} - Running evaluation..."
                )

                # Get PER stats
                per_stats = agent.get_per_stats()
                print("  PER stats:")
                print(
                    f"    Priority: min={per_stats['priority_min']:.6f}, mean={per_stats['priority_mean']:.6f}, max={per_stats['priority_max']:.6f}"
                )
                print(f"    Beta: {per_stats['beta']:.4f}")
                print(f"    Buffer size: {per_stats['buffer_size']}")

                # Evaluate with greedy policy (epsilon=0.0)
                print("  [Greedy policy, epsilon=0.0]")
                eval_greedy = evaluate_agent(
                    agent,
                    num_episodes=args.num_eval_episodes,
                    seed=args.seed + 999,
                    eval_epsilon=0.0,
                )
                print(f"    Mean reward: {eval_greedy['mean_reward']:.4f}")
                print(f"    Success rate: {eval_greedy['success_rate']:.3f}")
                print(f"    Avg episode length: {eval_greedy['avg_episode_len']:.1f}")

                # Log greedy evaluation
                with open(eval_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            total_frames,
                            0.0,
                            eval_greedy["mean_reward"],
                            eval_greedy["success_rate"],
                            eval_greedy["avg_episode_len"],
                        ]
                    )

                # Evaluate with mild exploration (epsilon=0.1)
                print("  [Mild exploration, epsilon=0.1]")
                eval_explore = evaluate_agent(
                    agent,
                    num_episodes=args.num_eval_episodes,
                    seed=args.seed + 999,
                    eval_epsilon=0.1,
                )
                print(f"    Mean reward: {eval_explore['mean_reward']:.4f}")
                print(f"    Success rate: {eval_explore['success_rate']:.3f}")
                print(f"    Avg episode length: {eval_explore['avg_episode_len']:.1f}")

                # Log exploration evaluation
                with open(eval_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            total_frames,
                            0.1,
                            eval_explore["mean_reward"],
                            eval_explore["success_rate"],
                            eval_explore["avg_episode_len"],
                        ]
                    )

                next_eval_frames += args.eval_every_frames

            # Save checkpoint
            if total_frames >= next_checkpoint_frames:
                checkpoint_path = run_dir / f"checkpoint_frames{total_frames}"
                agent.save(str(checkpoint_path))
                print(f"Saved checkpoint at {total_frames} frames")
                next_checkpoint_frames += args.checkpoint_every_frames

        # Determine success (goal reached)
        success = 1 if episode_reward > 0 else 0

        # Log training metrics
        avg_loss = np.mean(losses) if losses else 0.0
        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    total_frames,
                    episode_reward,
                    episode_len,
                    agent.epsilon,
                    avg_loss,
                    success,
                ]
            )

        # Print progress
        if episode % 100 == 0:
            print(
                f"Episode {episode} | Frames {total_frames}/{args.total_frames} | Reward: {episode_reward:.2f} | Success: {success} | Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    # Save final model
    final_path = run_dir / "final_model"
    agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {run_dir}")

    # Final evaluation
    print("\nRunning final evaluation...")
    print("  [Greedy policy, epsilon=0.0]")
    final_eval_greedy = evaluate_agent(
        agent,
        num_episodes=args.num_eval_episodes,
        seed=args.seed + 999,
        eval_epsilon=0.0,
    )
    print("  [Mild exploration, epsilon=0.1]")
    final_eval_explore = evaluate_agent(
        agent,
        num_episodes=args.num_eval_episodes,
        seed=args.seed + 999,
        eval_epsilon=0.1,
    )

    print("\n" + "=" * 80)
    print("Training Summary:")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Total frames: {total_frames}")
    print(f"Total episodes: {episode}")
    print("\nFinal eval (greedy, eps=0.0):")
    print(f"  Mean reward: {final_eval_greedy['mean_reward']:.4f}")
    print(f"  Success rate: {final_eval_greedy['success_rate']:.3f}")
    print(f"  Avg episode length: {final_eval_greedy['avg_episode_len']:.1f}")
    print("\nFinal eval (explore, eps=0.1):")
    print(f"  Mean reward: {final_eval_explore['mean_reward']:.4f}")
    print(f"  Success rate: {final_eval_explore['success_rate']:.3f}")
    print(f"  Avg episode length: {final_eval_explore['avg_episode_len']:.1f}")
    print("=" * 80)

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN with PER on MiniGrid FourRooms"
    )

    # Training params (frame-based)
    parser.add_argument(
        "--total_frames",
        type=int,
        default=300000,
        help="Total number of frames to train",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden layer size"
    )
    parser.add_argument(
        "--buffer_capacity", type=int, default=100000, help="Replay buffer capacity"
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=1000,
        help="Target network update frequency (frames)",
    )

    # N-step parameter
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="N-step for N-step returns (default: 1 for PER-only)",
    )

    # PER parameters
    parser.add_argument(
        "--use_per",
        action="store_true",
        help="Use Prioritized Experience Replay",
    )
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=0.6,
        help="PER alpha (priority exponent)",
    )
    parser.add_argument(
        "--per_beta_start",
        type=float,
        default=0.4,
        help="PER beta start (IS correction)",
    )
    parser.add_argument(
        "--per_eps",
        type=float,
        default=1e-6,
        help="PER epsilon (small constant for priority floor)",
    )

    # Epsilon schedule
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="Starting epsilon"
    )
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")

    # Evaluation and checkpointing
    parser.add_argument(
        "--eval_every_frames",
        type=int,
        default=20000,
        help="Evaluate every N frames",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=200,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--checkpoint_every_frames",
        type=int,
        default=50000,
        help="Save checkpoint every N frames",
    )

    # Run config
    parser.add_argument("--run_name", type=str, default="", help="Run name for logging")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )

    args = parser.parse_args()

    if not args.use_per:
        raise ValueError(
            "This script requires --use_per flag. Use train_minigrid_stacked.py for baseline DQN."
        )

    train_dqn(args)


if __name__ == "__main__":
    main()
