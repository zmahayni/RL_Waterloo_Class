"""
N-step DQN Agent for MiniGrid FourRooms with Stacked Observations.

Implements N-step returns for improved credit assignment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


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


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


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


class DQNAgentNStep:
    """N-step DQN Agent for MiniGrid."""

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

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

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
        When buffer is full or episode ends, push N-step transition to replay buffer.
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
        """Perform one training step with N-step returns."""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values with N-step returns
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # Target: R_n + (gamma^n) * max_a' Q_target(s_{t+n}, a') * (1 - done)
            target_q_values = rewards + (self.gamma**self.n_step) * next_q_values * (
                1 - dones
            )

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
