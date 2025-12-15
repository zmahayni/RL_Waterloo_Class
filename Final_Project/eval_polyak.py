"""Standalone evaluation script for Leduc Hold'em DQN agent with Polyak updates."""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from pettingzoo.classic import leduc_holdem_v4
from dqn_agent_polyak import DQNAgentPolyak
from rule_based_opponent import RuleBasedOpponent


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(
    checkpoint_path, num_episodes=1000, seed=42, dqn_agent_name="player_0"
):
    """
    Evaluate DQN agent from checkpoint against rule-based opponent.

    Args:
        checkpoint_path: Path to saved model checkpoint
        num_episodes: Number of evaluation episodes
        seed: Random seed
        dqn_agent_name: Name of DQN agent in environment

    Returns:
        dict with evaluation metrics
    """
    set_seed(seed)

    # Initialize environment
    env = leduc_holdem_v4.env(render_mode=None)
    env.reset(seed=seed)

    state_size = 36
    action_size = 4

    # Initialize and load DQN agent
    dqn_agent = DQNAgentPolyak(
        state_size=state_size,
        action_size=action_size,
        epsilon=0.0,  # Greedy evaluation
    )
    dqn_agent.load(checkpoint_path)

    # Initialize rule-based opponent
    rule_based = RuleBasedOpponent()

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0

    print(f"Evaluating for {num_episodes} episodes...")

    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
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

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}")

    env.close()

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "winrate": wins / num_episodes,
        "lossrate": losses / num_episodes,
        "drawrate": draws / num_episodes,
        "avg_episode_len": np.mean(episode_lengths),
        "num_episodes": num_episodes,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DQN agent with Polyak updates on Leduc Hold'em"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_log", action="store_true", help="Save evaluation log to CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save eval log"
    )

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    results = evaluate_agent(args.checkpoint, args.num_eval_episodes, args.seed)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Episodes: {results['num_episodes']}")
    print(f"Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Winrate: {results['winrate']:.3f}")
    print(f"Lossrate: {results['lossrate']:.3f}")
    print(f"Drawrate: {results['drawrate']:.3f}")
    print(f"Avg episode length: {results['avg_episode_len']:.1f}")
    print("=" * 60)

    # Save log if requested
    if args.save_log:
        output_dir = (
            Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "eval_log.csv"

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "mean_reward", "winrate", "avg_episode_len"])
            writer.writerow(
                [
                    results["num_episodes"],
                    results["mean_reward"],
                    results["winrate"],
                    results["avg_episode_len"],
                ]
            )

        print(f"\nEvaluation log saved to: {log_path}")


if __name__ == "__main__":
    main()
