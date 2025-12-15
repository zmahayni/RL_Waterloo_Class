"""DQN agent with Polyak (soft) target network updates."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """Neural network for Q-value estimation."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgentPolyak:
    """Deep Q-Network agent with Polyak (soft) target updates."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_size=128,
        target_update_mode="hard",
        tau=0.995,
        target_update_every=1000,
    ):
        """
        Initialize DQN agent with Polyak target updates.

        Args:
            state_size: Size of the observation space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            hidden_size: Hidden layer size for Q-network
            target_update_mode: 'hard' or 'polyak'
            tau: Polyak averaging coefficient (only used if mode='polyak')
            target_update_every: Hard update frequency (only used if mode='hard')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_mode = target_update_mode
        self.tau = tau
        self.target_update_every = target_update_every

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-networks
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Training step counter
        self.train_step = 0

    def select_action(self, state, legal_actions, training=True):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current observation
            legal_actions: Mask of legal actions
            training: Whether in training mode

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Explore: select random legal action
            legal_action_indices = np.where(legal_actions)[0]
            return np.random.choice(legal_action_indices)
        else:
            # Exploit: select best legal action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)

                # Mask illegal actions with very negative values
                q_values_masked = q_values.clone()
                q_values_masked[0, ~legal_actions.astype(bool)] = -1e9

                action = q_values_masked.argmax(1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def _polyak_update(self):
        """Perform Polyak (soft) target network update."""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * target_param.data + (1.0 - self.tau) * online_param.data
            )

    def _hard_update(self):
        """Perform hard (periodic copy) target network update."""
        if self.train_step % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, batch_size=32):
        """
        Train the Q-network using a batch from the replay buffer.

        Args:
            batch_size: Size of the training batch

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q-values for current states
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_step += 1

        # Update target network based on mode
        if self.target_update_mode == "polyak":
            self._polyak_update()
        else:  # hard
            self._hard_update()

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save agent weights."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self.train_step,
                "target_update_mode": self.target_update_mode,
                "tau": self.tau,
            },
            filepath,
        )

    def load(self, filepath):
        """Load agent weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_step = checkpoint["train_step"]
        if "target_update_mode" in checkpoint:
            self.target_update_mode = checkpoint["target_update_mode"]
        if "tau" in checkpoint:
            self.tau = checkpoint["tau"]
