"""Standalone evaluation script for N-step DQN on MiniGrid FourRooms with stacked observations."""

import argparse
import numpy as np
import torch
from minigrid_stacked_obs_wrapper import make_env, get_env_info
from dqn_agent_minigrid_nstep import DQNAgentNStep


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(checkpoint_path, num_episodes=200, seed=None):
    """
    Evaluate N-step DQN agent on MiniGrid FourRooms.

    Args:
        checkpoint_path: Path to model checkpoint (without .pt extension)
        num_episodes: Number of evaluation episodes
        seed: Random seed for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(seed)

    # Get environment info
    state_size, action_size = get_env_info()

    # Initialize agent
    agent = DQNAgentNStep(
        state_size=state_size,
        action_size=action_size,
        n_step=3,  # Default, doesn't matter for evaluation
        epsilon_start=0.0,  # Greedy policy for evaluation
        epsilon_end=0.0,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    # Create environment
    env = make_env(seed=seed)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    successes = 0

    print(f"Evaluating for {num_episodes} episodes...")
    print("Using greedy policy (epsilon=0)")

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_len = 0
        done = False

        while not done:
            # Greedy action selection
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_len += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

        # Check if goal was reached
        if episode_reward > 0:
            successes += 1

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}")

    env.close()

    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = successes / num_episodes
    avg_episode_len = np.mean(episode_lengths)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results (N-step DQN):")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Avg episode length: {avg_episode_len:.1f}")
    print("=" * 60)

    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "avg_episode_len": avg_episode_len,
        "num_episodes": num_episodes,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate N-step DQN on MiniGrid FourRooms with stacked observations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (without .pt extension)",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=200,
        help="Number of episodes for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    evaluate_agent(args.checkpoint, args.num_eval_episodes, args.seed)


if __name__ == "__main__":
    main()
