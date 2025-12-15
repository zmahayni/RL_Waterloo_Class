"""Training script for baseline DQN on Leduc Hold'em."""

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from pettingzoo.classic import leduc_holdem_v4
from dqn_agent import DQNAgent
from rule_based_opponent import RuleBasedOpponent


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(
    env,
    dqn_agent,
    rule_based,
    num_episodes,
    dqn_agent_name="player_0",
    seed_offset=10000,
):
    """
    Evaluate DQN agent against rule-based opponent.

    Returns:
        dict with mean_reward, winrate, lossrate, drawrate, avg_episode_len
    """
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0

    for ep in range(num_episodes):
        env.reset(seed=seed_offset + ep)
        episode_reward = 0.0
        episode_len = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if agent == dqn_agent_name:
                episode_reward += reward

            episode_len += 1

            if termination or truncation:
                action = None
            else:
                obs = observation["observation"]
                legal_actions = observation["action_mask"]

                if agent == dqn_agent_name:
                    action = dqn_agent.select_action(obs, legal_actions, training=False)
                else:
                    action = rule_based.get_action(obs, legal_actions)

            env.step(action)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

        # Determine outcome
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "mean_reward": np.mean(episode_rewards),
        "winrate": wins / num_episodes,
        "lossrate": losses / num_episodes,
        "drawrate": draws / num_episodes,
        "avg_episode_len": np.mean(episode_lengths),
    }


def train_baseline_dqn(args):
    """Train baseline DQN on Leduc Hold'em."""
    # Set seed
    set_seed(args.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = Path(f"runs/leduc/baseline/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Initialize environment
    env = leduc_holdem_v4.env(render_mode=None)
    env.reset(seed=args.seed)

    state_size = 36
    action_size = 4
    dqn_agent_name = "player_0"

    # Initialize DQN agent
    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_decay=1.0,  # We'll manually decay
        epsilon_min=args.epsilon_end,
        hidden_size=args.hidden_size,
    )

    # Initialize rule-based opponent
    rule_based = RuleBasedOpponent()

    # Epsilon schedule
    epsilon_decay_episodes = int(args.num_episodes * 0.6)
    epsilon_delta = (args.epsilon_start - args.epsilon_end) / epsilon_decay_episodes

    # Training logs
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"

    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "episode_reward", "episode_len", "epsilon", "loss"])

    with open(eval_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "mean_reward", "winrate", "avg_episode_len"])

    # Training loop
    print(f"\nStarting training for {args.num_episodes} episodes...")
    print("=" * 80)

    for episode in range(args.num_episodes):
        env.reset(seed=args.seed + episode)

        episode_reward = 0.0
        episode_len = 0
        prev_state = None
        prev_action = None
        losses = []

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if agent == dqn_agent_name:
                episode_reward += reward

            episode_len += 1

            if termination or truncation:
                if agent == dqn_agent_name and prev_state is not None:
                    dqn_agent.store_transition(
                        prev_state,
                        prev_action,
                        reward,
                        observation["observation"],
                        True,
                    )
                action = None
            else:
                obs = observation["observation"]
                legal_actions = observation["action_mask"]

                if agent == dqn_agent_name:
                    if prev_state is not None:
                        dqn_agent.store_transition(
                            prev_state, prev_action, reward, obs, False
                        )

                    action = dqn_agent.select_action(obs, legal_actions, training=True)
                    prev_state = obs
                    prev_action = action
                else:
                    action = rule_based.get_action(obs, legal_actions)

            env.step(action)

            # Train DQN
            if agent == dqn_agent_name:
                loss = dqn_agent.train(batch_size=args.batch_size)
                if loss is not None:
                    losses.append(loss)

        # Decay epsilon
        if episode < epsilon_decay_episodes:
            dqn_agent.epsilon = max(args.epsilon_end, dqn_agent.epsilon - epsilon_delta)

        # Log training episode
        avg_loss = np.mean(losses) if losses else None
        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [episode, episode_reward, episode_len, dqn_agent.epsilon, avg_loss]
            )

        # Periodic evaluation
        if (episode + 1) % args.eval_every == 0:
            print(
                f"\nEpisode {episode + 1}/{args.num_episodes} - Running evaluation..."
            )
            eval_results = evaluate_agent(
                env,
                dqn_agent,
                rule_based,
                args.num_eval_episodes,
                dqn_agent_name,
                seed_offset=args.seed + 100000 + episode,
            )

            print(f"  Mean reward: {eval_results['mean_reward']:.4f}")
            print(f"  Winrate: {eval_results['winrate']:.3f}")
            print(f"  Avg episode length: {eval_results['avg_episode_len']:.1f}")

            with open(eval_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode + 1,
                        eval_results["mean_reward"],
                        eval_results["winrate"],
                        eval_results["avg_episode_len"],
                    ]
                )

        # Save checkpoint
        if (episode + 1) % args.checkpoint_every == 0:
            checkpoint_path = run_dir / f"q_network_ep{episode + 1}.pt"
            dqn_agent.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

        # Progress update
        if (episode + 1) % 1000 == 0:
            print(
                f"Episode {episode + 1}/{args.num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Epsilon: {dqn_agent.epsilon:.3f}"
            )

    # Save final model
    final_path = run_dir / "q_network_final.pt"
    dqn_agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved: {final_path}")

    env.close()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train baseline DQN on Leduc Hold'em")

    # Training params
    parser.add_argument(
        "--num_episodes", type=int, default=50000, help="Number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
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
        default=1000,
        help="Number of evaluation episodes",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10000,
        help="Save checkpoint every N episodes",
    )

    # Run management
    parser.add_argument("--run_name", type=str, default=None, help="Name for this run")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )

    args = parser.parse_args()

    run_dir = train_baseline_dqn(args)
    print(f"\nRun artifacts saved to: {run_dir}")
    print(f"To plot results, run: python plot_leduc.py --run_dir {run_dir}")


if __name__ == "__main__":
    main()
