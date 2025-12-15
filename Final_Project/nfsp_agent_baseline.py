"""NFSP agent with baseline DQN (no dueling, no polyak) and average policy."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """Standard Q-network for baseline DQN."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class AveragePolicyNetwork(nn.Module):
    """Average policy network for supervised learning."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(AveragePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for RL transitions."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, next_legal_actions):
        self.buffer.append(
            (state, action, reward, next_state, done, next_legal_actions)
        )

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_legal_actions = zip(
            *transitions
        )

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        next_legal_actions = torch.BoolTensor(np.array(next_legal_actions))

        return states, actions, rewards, next_states, dones, next_legal_actions

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer:
    """Reservoir sampling buffer for supervised learning."""

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.num_added = 0

    def add(self, state, action, legal_actions):
        """Add transition using reservoir sampling."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, legal_actions))
        else:
            # Reservoir sampling
            idx = random.randint(0, self.num_added)
            if idx < self.capacity:
                self.buffer[idx] = (state, action, legal_actions)
        self.num_added += 1

    def sample(self, batch_size):
        """Sample batch for supervised learning."""
        transitions = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, legal_actions = zip(*transitions)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        legal_actions = torch.BoolTensor(np.array(legal_actions))

        return states, actions, legal_actions

    def __len__(self):
        return len(self.buffer)


class NFSPAgentBaseline:
    """NFSP agent with baseline DQN and average policy."""

    def __init__(
        self,
        state_size,
        action_size,
        rl_lr=1e-3,
        sl_lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_size=128,
        eta=0.1,
        rl_buffer_size=10000,
        sl_buffer_size=100000,
        target_update_freq=1000,
    ):
        """
        Initialize NFSP agent.

        Args:
            state_size: Size of observation space
            action_size: Number of possible actions
            rl_lr: Learning rate for RL (DQN)
            sl_lr: Learning rate for SL (average policy)
            gamma: Discount factor
            epsilon: Initial exploration rate for RL
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            hidden_size: Hidden layer size
            eta: Anticipatory parameter (prob of using RL policy)
            rl_buffer_size: RL replay buffer capacity
            sl_buffer_size: SL reservoir buffer capacity
            target_update_freq: Frequency of hard target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eta = eta
        self.target_update_freq = target_update_freq

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RL: Q-networks (baseline DQN)
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # SL: Average policy network
        self.avg_policy = AveragePolicyNetwork(state_size, action_size, hidden_size).to(
            self.device
        )

        # Optimizers
        self.rl_optimizer = optim.Adam(self.q_network.parameters(), lr=rl_lr)
        self.sl_optimizer = optim.Adam(self.avg_policy.parameters(), lr=sl_lr)

        # Buffers
        self.rl_buffer = ReplayBuffer(capacity=rl_buffer_size)
        self.sl_buffer = ReservoirBuffer(capacity=sl_buffer_size)

        # Training counters
        self.rl_train_step = 0
        self.sl_train_step = 0

        # Episode mode tracking
        self.current_mode = None  # 'rl' or 'avg'

    def begin_episode(self):
        """Sample mode for the episode based on eta."""
        self.current_mode = "rl" if random.random() < self.eta else "avg"
        return self.current_mode

    def select_action(self, state, legal_actions, training=True):
        """
        Select action based on current mode.

        Args:
            state: Current observation
            legal_actions: Boolean mask of legal actions
            training: Whether in training mode

        Returns:
            Selected action index
        """
        if not training:
            # During evaluation, use average policy greedily
            return self._select_avg_action(state, legal_actions, greedy=True)

        # Training: use current episode mode
        if self.current_mode == "rl":
            return self._select_rl_action(state, legal_actions)
        else:
            return self._select_avg_action(state, legal_actions, greedy=False)

    def _select_rl_action(self, state, legal_actions):
        """Select action using RL policy (epsilon-greedy)."""
        if random.random() < self.epsilon:
            # Explore: random legal action
            legal_action_indices = np.where(legal_actions)[0]
            return np.random.choice(legal_action_indices)
        else:
            # Exploit: best legal Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                q_values_masked = q_values.clone()
                q_values_masked[0, ~legal_actions.astype(bool)] = -1e9
                action = q_values_masked.argmax(1).item()
            return action

    def _select_avg_action(self, state, legal_actions, greedy=False):
        """Select action using average policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.avg_policy(state_tensor)

            # Mask illegal actions
            logits_masked = logits.clone()
            logits_masked[0, ~legal_actions.astype(bool)] = -1e9

            if greedy:
                action = logits_masked.argmax(1).item()
            else:
                # Sample from softmax distribution
                probs = torch.softmax(logits_masked, dim=1)
                action = torch.multinomial(probs, 1).item()

        return action

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        current_legal_actions,
        next_legal_actions,
    ):
        """Store transition in appropriate buffers based on current mode.

        Args:
            current_legal_actions: Legal action mask for current state (when action was taken)
            next_legal_actions: Legal action mask for next_state (for DQN target computation)
        """
        if self.current_mode == "rl":
            # Store in both RL buffer and SL reservoir
            self.rl_buffer.add(
                state, action, reward, next_state, done, next_legal_actions
            )
            # SL buffer stores current state's action with its legal actions
            self.sl_buffer.add(state, action, current_legal_actions)

    def train_rl(self, batch_size=32):
        """Train RL policy (DQN)."""
        if len(self.rl_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones, next_legal_actions = (
            self.rl_buffer.sample(batch_size)
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_legal_actions = next_legal_actions.to(self.device)

        # Compute Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values with legal action masking
        with torch.no_grad():
            q_next = self.target_network(next_states)  # (B, A)

            # Mask illegal actions with very negative values
            q_next_masked = q_next.clone()
            q_next_masked[~next_legal_actions] = -1e9

            next_q_values = q_next_masked.max(dim=1).values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Backpropagation
        self.rl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.rl_optimizer.step()

        self.rl_train_step += 1

        # Hard target update
        if self.rl_train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train_sl(self, batch_size=128):
        """Train average policy (supervised learning)."""
        if len(self.sl_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, legal_actions = self.sl_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        legal_actions = legal_actions.to(self.device)

        # Forward pass
        logits = self.avg_policy(states)

        # Mask illegal actions
        logits_masked = logits.clone()
        logits_masked[~legal_actions] = -1e9

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits_masked, actions)

        # Backpropagation
        self.sl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.avg_policy.parameters(), 1.0)
        self.sl_optimizer.step()

        self.sl_train_step += 1

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath_prefix):
        """Save agent networks."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "avg_policy": self.avg_policy.state_dict(),
                "rl_optimizer": self.rl_optimizer.state_dict(),
                "sl_optimizer": self.sl_optimizer.state_dict(),
                "epsilon": self.epsilon,
                "rl_train_step": self.rl_train_step,
                "sl_train_step": self.sl_train_step,
            },
            f"{filepath_prefix}.pt",
        )

    def load(self, filepath_prefix):
        """Load agent networks."""
        checkpoint = torch.load(f"{filepath_prefix}.pt", map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.avg_policy.load_state_dict(checkpoint["avg_policy"])
        self.rl_optimizer.load_state_dict(checkpoint["rl_optimizer"])
        self.sl_optimizer.load_state_dict(checkpoint["sl_optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.rl_train_step = checkpoint["rl_train_step"]
        self.sl_train_step = checkpoint["sl_train_step"]
