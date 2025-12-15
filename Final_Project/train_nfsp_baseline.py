"""Training script for NFSP with baseline DQN on Leduc Hold'em (self-play)."""

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from pettingzoo.classic import leduc_holdem_v4
from nfsp_agent_baseline import NFSPAgentBaseline
from rule_based_opponent import RuleBasedOpponent
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


def evaluate_agent(
    env, agent, rule_based, num_episodes, agent_name="player_0", seed_offset=10000
):
    """
    Evaluate NFSP agent (using average policy) against rule-based opponent.

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

        for agent_iter in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if agent_iter == agent_name:
                episode_reward += reward

            episode_len += 1

            if termination or truncation:
                action = None
            else:
                obs = observation["observation"]
                legal_actions = observation["action_mask"]

                if agent_iter == agent_name:
                    # Use average policy greedily for evaluation
                    action = agent.select_action(obs, legal_actions, training=False)
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


def train_nfsp(args):
    """Train NFSP agents on Leduc Hold'em (self-play)."""
    # Set seed
    set_seed(args.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = Path(f"runs/leduc/nfsp_baseline/{run_name}")
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

    # Initialize two NFSP agents for self-play
    agent_names = ["player_0", "player_1"]
    agents = {
        "player_0": NFSPAgentBaseline(
            state_size=state_size,
            action_size=action_size,
            rl_lr=args.rl_lr,
            sl_lr=args.sl_lr,
            gamma=args.gamma,
            epsilon=args.epsilon_start,
            epsilon_decay=1.0,  # Manual decay
            epsilon_min=args.epsilon_end,
            hidden_size=args.hidden_size,
            eta=args.eta,
            rl_buffer_size=args.rl_buffer_size,
            sl_buffer_size=args.sl_buffer_size,
        ),
        "player_1": NFSPAgentBaseline(
            state_size=state_size,
            action_size=action_size,
            rl_lr=args.rl_lr,
            sl_lr=args.sl_lr,
            gamma=args.gamma,
            epsilon=args.epsilon_start,
            epsilon_decay=1.0,  # Manual decay
            epsilon_min=args.epsilon_end,
            hidden_size=args.hidden_size,
            eta=args.eta,
            rl_buffer_size=args.rl_buffer_size,
            sl_buffer_size=args.sl_buffer_size,
        ),
    }

    # Initialize rule-based opponent for evaluation
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
    print(f"\nStarting NFSP self-play training for {args.num_episodes} episodes...")
    print(f"Eta (RL mode prob): {args.eta}")
    print("=" * 80)

    env_step = 0

    for episode in range(args.num_episodes):
        env.reset(seed=args.seed + episode)

        # Each agent samples mode for this episode
        for agent_name in agent_names:
            agents[agent_name].begin_episode()

        episode_reward_p0 = 0.0
        episode_len = 0

        # Track previous states/actions/legal_actions for each agent
        prev_states = {name: None for name in agent_names}
        prev_actions = {name: None for name in agent_names}
        prev_legal_actions = {name: None for name in agent_names}

        rl_losses = []
        sl_losses = []

        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Track player_0 reward for logging
            if agent_name == "player_0":
                episode_reward_p0 += reward

            episode_len += 1
            env_step += 1

            if termination or truncation:
                # Store terminal transition
                if prev_states[agent_name] is not None:
                    agents[agent_name].store_transition(
                        prev_states[agent_name],
                        prev_actions[agent_name],
                        reward,
                        observation["observation"],
                        True,
                        prev_legal_actions[agent_name],  # current legal actions
                        observation["action_mask"],  # next legal actions (terminal)
                    )
                action = None
            else:
                obs = observation["observation"]
                legal_actions = observation["action_mask"]

                # Store transition if not first step
                if prev_states[agent_name] is not None:
                    agents[agent_name].store_transition(
                        prev_states[agent_name],
                        prev_actions[agent_name],
                        reward,
                        obs,
                        False,
                        prev_legal_actions[agent_name],  # current legal actions
                        legal_actions,  # next legal actions
                    )

                # Select action
                action = agents[agent_name].select_action(
                    obs, legal_actions, training=True
                )
                prev_states[agent_name] = obs
                prev_actions[agent_name] = action
                prev_legal_actions[agent_name] = legal_actions

            env.step(action)

            # Train RL for current agent
            if agent_name in agents:
                rl_loss = agents[agent_name].train_rl(batch_size=args.rl_batch_size)
                if rl_loss is not None:
                    rl_losses.append(rl_loss)

                # Train SL periodically
                if env_step % args.sl_update_every == 0:
                    for _ in range(args.sl_updates_per_step):
                        sl_loss = agents[agent_name].train_sl(
                            batch_size=args.sl_batch_size
                        )
                        if sl_loss is not None:
                            sl_losses.append(sl_loss)

        # Decay epsilon for both agents
        if episode < epsilon_decay_episodes:
            for agent_name in agent_names:
                agents[agent_name].epsilon = max(
                    args.epsilon_end, agents[agent_name].epsilon - epsilon_delta
                )

        # Log training episode (player_0 perspective)
        avg_loss = np.mean(rl_losses + sl_losses) if (rl_losses or sl_losses) else None
        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    episode_reward_p0,
                    episode_len,
                    agents["player_0"].epsilon,
                    avg_loss,
                ]
            )

        # Periodic evaluation (player_0 vs rule-based)
        if (episode + 1) % args.eval_every == 0:
            print(
                f"\nEpisode {episode + 1}/{args.num_episodes} - Running evaluation..."
            )
            eval_results = evaluate_agent(
                env,
                agents["player_0"],
                rule_based,
                args.num_eval_episodes,
                "player_0",
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
            for agent_name in agent_names:
                checkpoint_path = run_dir / f"{agent_name}_ep{episode + 1}"
                agents[agent_name].save(str(checkpoint_path))
            print(f"Saved checkpoint at episode {episode + 1}")

        # Progress update
        if (episode + 1) % 1000 == 0:
            print(
                f"Episode {episode + 1}/{args.num_episodes} | "
                f"P0 Reward: {episode_reward_p0:.2f} | "
                f"Epsilon: {agents['player_0'].epsilon:.3f}"
            )

    # Save final models
    for agent_name in agent_names:
        final_path = run_dir / f"{agent_name}_final"
        agents[agent_name].save(str(final_path))
    print(f"\nTraining complete! Final models saved to: {run_dir}")

    # Generate plot
    plot_learning_curve(run_dir)

    # Final evaluation
    print("\nRunning final evaluation (player_0 vs rule-based)...")
    final_eval = evaluate_agent(
        env,
        agents["player_0"],
        rule_based,
        args.num_eval_episodes,
        "player_0",
        seed_offset=args.seed + 200000,
    )

    env.close()

    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary:")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Final eval mean reward: {final_eval['mean_reward']:.4f}")
    print(f"Final eval winrate: {final_eval['winrate']:.3f}")
    print(f"Final eval lossrate: {final_eval['lossrate']:.3f}")
    print(f"Final eval drawrate: {final_eval['drawrate']:.3f}")
    print("=" * 80)

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train NFSP with baseline DQN on Leduc Hold'em"
    )

    # Training params
    parser.add_argument(
        "--num_episodes", type=int, default=50000, help="Number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--rl_batch_size", type=int, default=32, help="Batch size for RL training"
    )
    parser.add_argument(
        "--sl_batch_size", type=int, default=128, help="Batch size for SL training"
    )
    parser.add_argument(
        "--rl_lr", type=float, default=1e-3, help="Learning rate for RL"
    )
    parser.add_argument(
        "--sl_lr", type=float, default=1e-3, help="Learning rate for SL"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
    )

    # Epsilon schedule
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="Starting epsilon"
    )
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")

    # NFSP params
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Anticipatory parameter (prob of RL mode)",
    )
    parser.add_argument(
        "--rl_buffer_size", type=int, default=10000, help="RL replay buffer size"
    )
    parser.add_argument(
        "--sl_buffer_size", type=int, default=100000, help="SL reservoir buffer size"
    )
    parser.add_argument(
        "--sl_update_every",
        type=int,
        default=64,
        help="SL update frequency (env steps)",
    )
    parser.add_argument(
        "--sl_updates_per_step",
        type=int,
        default=1,
        help="Number of SL updates per trigger",
    )

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

    train_nfsp(args)


if __name__ == "__main__":
    main()
