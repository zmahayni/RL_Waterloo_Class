"""
DQN Agent with Prioritized Experience Replay (PER) + N-step returns for MiniGrid.

Combines PER with N-step TD targets for improved learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from prioritized_replay_buffer import PrioritizedReplayBuffer


class QNetwork(nn.Module):
    """Q-Network for MiniGrid with stacked observations (510 dims)."""

    def __init__(self, state_size, action_size, hidden_size=512):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NStepBuffer:
    """N-step transition buffer for computing N-step returns."""

    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        """Add transition to N-step buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def get_n_step_transition(self):
        """
        Compute N-step return and return transition.

        Returns:
            (state, action, n_step_return, n_step_next_state, n_step_done)
            or None if buffer not full enough
        """
        if len(self.buffer) == 0:
            return None

        # Get first transition
        state, action, _, _, _ = self.buffer[0]

        # Compute N-step return
        n_step_return = 0.0
        gamma_power = 1.0

        for i, (_, _, reward, next_state, done) in enumerate(self.buffer):
            n_step_return += gamma_power * reward
            gamma_power *= self.gamma

            # If episode ends, stop accumulating
            if done:
                return (state, action, n_step_return, next_state, done)

        # If we reach here, episode didn't end within N steps
        # Use the last next_state and done flag
        _, _, _, n_step_next_state, n_step_done = self.buffer[-1]
        return (state, action, n_step_return, n_step_next_state, n_step_done)

    def is_full(self):
        """Check if buffer has N transitions."""
        return len(self.buffer) == self.n_step

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class DQNAgentPERNStep:
    """DQN Agent with PER + N-step returns."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-4,
        gamma=0.99,
        n_step=3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        hidden_size=512,
        buffer_capacity=100000,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=300000,
        per_eps=1e-6,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.n_step = n_step
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.device = device

        # Q-networks
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
            eps=per_eps,
        )

        # N-step buffer
        self.n_step_buffer = NStepBuffer(n_step, gamma)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in N-step buffer.
        When buffer is full or episode ends, push N-step transition to PER buffer.
        """
        # Add to N-step buffer
        self.n_step_buffer.push(state, action, reward, next_state, done)

        # If buffer is full or episode ended, compute N-step return and store
        if self.n_step_buffer.is_full() or done:
            n_step_transition = self.n_step_buffer.get_n_step_transition()
            if n_step_transition is not None:
                self.replay_buffer.push(*n_step_transition)

            # If episode ended, flush remaining transitions
            if done:
                # Process remaining transitions in buffer
                while len(self.n_step_buffer) > 1:
                    # Remove first transition
                    self.n_step_buffer.buffer.popleft()
                    # Get N-step transition from remaining
                    n_step_transition = self.n_step_buffer.get_n_step_transition()
                    if n_step_transition is not None:
                        self.replay_buffer.push(*n_step_transition)
                # Clear buffer for next episode
                self.n_step_buffer.clear()

    def train_step(self, batch_size):
        """Perform one training step with PER + N-step returns."""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch with priorities
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights,
            indices,
        ) = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        )

        # Target Q values with N-step returns
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # Target: R_n + (gamma^n) * max_a' Q_target(s_{t+n}, a') * (1 - done)
            target_q_values = rewards + (self.gamma**self.n_step) * next_q_values * (
                1 - dones
            )

        # TD errors
        td_errors = target_q_values - current_q_values

        # Compute priorities for update (|TD_error| + eps)
        priorities = td_errors.abs().detach().cpu().numpy() + self.replay_buffer.eps

        # Weighted loss with Huber/SmoothL1
        loss = (
            weights
            * nn.functional.smooth_l1_loss(
                current_q_values, target_q_values, reduction="none"
            )
        ).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, priorities)

        return loss.item()

    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self, frame, total_decay_frames):
        """Update epsilon based on frame count (linear decay)."""
        if frame < total_decay_frames:
            self.epsilon = self.epsilon_start - (
                self.epsilon_start - self.epsilon_end
            ) * (frame / total_decay_frames)
        else:
            self.epsilon = self.epsilon_end

    def get_per_stats(self):
        """Get PER statistics for debugging."""
        priority_stats = self.replay_buffer.get_priority_stats()
        return {
            "priority_min": priority_stats["min"],
            "priority_mean": priority_stats["mean"],
            "priority_max": priority_stats["max"],
            "beta": self.replay_buffer._get_beta(),
            "buffer_size": len(self.replay_buffer),
        }

    def save(self, path):
        """Save model checkpoint."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            f"{path}.pt",
        )

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
