"""
Prioritized Experience Replay (PER) implementation with SumTree.

Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
"""

import numpy as np
import random


class SumTree:
    """
    SumTree data structure for efficient proportional sampling.

    Binary tree where:
    - Leaf nodes store priorities
    - Parent nodes store sum of children
    - Root stores total priority sum
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Store transitions
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Retrieve leaf index for given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1

        # If leaf node, return index
        if left >= len(self.tree):
            return idx

        # Traverse left or right based on cumulative sum
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority sum (root value)."""
        return self.tree[0]

    def add(self, priority, data):
        """Add new transition with given priority."""
        idx = self.write_idx + self.capacity - 1  # Leaf index in tree

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """Update priority at given tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """
        Get transition for cumulative sum s.

        Returns:
            (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    Samples transitions proportional to TD error priority.
    Uses importance sampling weights to correct bias.
    """

    def __init__(
        self,
        capacity,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=300000,
        eps=1e-6,
    ):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta_start: Initial importance sampling correction
            beta_frames: Frames to anneal beta from beta_start to 1.0
            eps: Small constant to prevent zero priority
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.max_priority = 1.0  # Initial max priority
        self.frame = 0

    def _get_beta(self):
        """Compute current beta (anneals from beta_start to 1.0)."""
        progress = min(self.frame / self.beta_frames, 1.0)
        return self.beta_start + (1.0 - self.beta_start) * progress

    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        transition = (state, action, reward, next_state, done)
        priority = self.max_priority**self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        """
        Sample batch with priorities.

        Returns:
            states, actions, rewards, next_states, dones, weights, indices
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        beta = self._get_beta()

        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Get transition
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()

        # IS weights: (N * P(i))^(-beta)
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        # Normalize by max weight for stability
        weights = weights / weights.max()

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        self.frame += 1

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights, dtype=np.float32),
            np.array(indices),
        )

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.

        Args:
            indices: Tree indices from sample()
            priorities: New priorities (already computed as |TD_error| + eps)
        """
        for idx, priority in zip(indices, priorities):
            # Apply alpha exponent
            priority_alpha = float(priority) ** self.alpha
            self.tree.update(idx, priority_alpha)
            self.max_priority = max(self.max_priority, priority)

    def get_avg_priority(self):
        """Get average priority for debugging."""
        if self.tree.n_entries == 0:
            return 0.0
        return self.tree.total() / self.tree.n_entries

    def get_priority_stats(self):
        """Get priority statistics for debugging."""
        if self.tree.n_entries == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}

        # Get all leaf priorities
        priorities = []
        for i in range(self.tree.n_entries):
            idx = i + self.capacity - 1
            priorities.append(self.tree.tree[idx])

        priorities = np.array(priorities)
        return {
            "min": float(np.min(priorities)),
            "mean": float(np.mean(priorities)),
            "max": float(np.max(priorities)),
        }

    def __len__(self):
        return self.tree.n_entries
