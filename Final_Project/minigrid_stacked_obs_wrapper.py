"""
Custom observation wrapper for MiniGrid FourRooms.

Produces 510-dim vector observation:
- 7x7 egocentric view encoded to 7x7x2 (98 features per frame)
- Frame stacking: last 5 frames (490 features)
- Direction history: last 5 directions one-hot encoded (20 features)
- Total: 490 + 20 = 510 features
"""

import gymnasium as gym
import numpy as np
from collections import deque
from minigrid.core.constants import OBJECT_TO_IDX
from gymnasium.core import ActionWrapper


class ActionRestrictionWrapper(ActionWrapper):
    """
    Wrapper that restricts action space to only left, right, and forward.

    MiniGrid action space:
    - 0: left
    - 1: right
    - 2: forward
    - 3: pickup
    - 4: drop
    - 5: toggle
    - 6: done

    This wrapper maps agent actions {0, 1, 2} to env actions {0, 1, 2}.
    """

    def __init__(self, env):
        super().__init__(env)
        # Restrict to 3 actions: left (0), right (1), forward (2)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        """Map agent action to environment action."""
        # Agent actions {0, 1, 2} map directly to env actions {0, 1, 2}
        # (left, right, forward)
        return action


class StackedEgoObsWrapper(gym.ObservationWrapper):
    """
    Wrapper that produces 510-dim vector observation from MiniGrid.

    Observation pipeline:
    1. Extract 7x7 egocentric view (obs["image"])
    2. Encode each cell to 2 channels: [wall_bit, goal_bit]
       - empty: [0, 0]
       - wall: [1, 0]
       - goal: [0, 1]
       - other: [0, 0]
    3. Flatten to 98 features (7*7*2)
    4. Stack last 5 frames (FIFO) → 490 features
    5. Append last 5 directions (one-hot, 4 dims each) → 20 features
    6. Total: 510 features
    """

    def __init__(self, env, stack_size=5):
        super().__init__(env)

        self.stack_size = stack_size

        # Get object IDs from MiniGrid constants
        self.wall_id = OBJECT_TO_IDX["wall"]
        self.empty_id = OBJECT_TO_IDX["empty"]
        self.goal_id = OBJECT_TO_IDX["goal"]

        # Frame stack (FIFO queue of encoded frames)
        # Each frame is 98 features (7x7x2 flattened)
        self.frame_stack = deque(maxlen=stack_size)

        # Direction history (FIFO queue of directions)
        # Each direction is 0-3 (will be one-hot encoded)
        self.direction_history = deque(maxlen=stack_size)

        # Define observation space: 510-dim vector
        # 490 (stacked frames) + 20 (direction history)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(510,), dtype=np.float32
        )

    def _encode_frame(self, image):
        """
        Encode 7x7x3 image to 7x7x2 binary encoding.

        Args:
            image: numpy array of shape (7, 7, 3)
                   image[:, :, 0] contains object IDs

        Returns:
            Flattened numpy array of shape (98,) with values in {0, 1}
        """
        # Extract object IDs (first channel)
        objects = image[:, :, 0]  # Shape: (7, 7)

        # Create 2-channel encoding
        encoded = np.zeros((7, 7, 2), dtype=np.float32)

        # Channel 0: wall bit
        encoded[:, :, 0] = (objects == self.wall_id).astype(np.float32)

        # Channel 1: goal bit
        encoded[:, :, 1] = (objects == self.goal_id).astype(np.float32)

        # Flatten to 98 features
        return encoded.flatten()

    def _encode_direction(self, direction):
        """
        One-hot encode direction (0-3) to 4-dim vector.

        Args:
            direction: int in range [0, 3]

        Returns:
            One-hot encoded numpy array of shape (4,)
        """
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[direction] = 1.0
        return one_hot

    def _get_stacked_obs(self):
        """
        Construct 510-dim observation from frame stack and direction history.

        Returns:
            numpy array of shape (510,)
        """
        # Stack frames: 5 frames × 98 features = 490 features
        stacked_frames = np.concatenate(list(self.frame_stack))

        # Encode direction history: 5 directions × 4 dims = 20 features
        direction_features = np.concatenate(
            [self._encode_direction(d) for d in self.direction_history]
        )

        # Concatenate: 490 + 20 = 510
        obs = np.concatenate([stacked_frames, direction_features])

        return obs.astype(np.float32)

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack and direction history."""
        obs, info = self.env.reset(**kwargs)

        # Extract image and direction from observation dict
        image = obs["image"]  # Shape: (7, 7, 3)
        direction = obs["direction"]  # int in [0, 3]

        # Encode initial frame
        encoded_frame = self._encode_frame(image)

        # Fill frame stack with initial frame (repeated 5 times)
        self.frame_stack.clear()
        for _ in range(self.stack_size):
            self.frame_stack.append(encoded_frame)

        # Fill direction history with initial direction (repeated 5 times)
        self.direction_history.clear()
        for _ in range(self.stack_size):
            self.direction_history.append(direction)

        # Return 510-dim observation
        return self._get_stacked_obs(), info

    def observation(self, obs):
        """
        Process observation dict to 510-dim vector.
        Called automatically by step().

        Args:
            obs: observation dict from environment

        Returns:
            510-dim numpy array
        """
        # Extract image and direction
        image = obs["image"]
        direction = obs["direction"]

        # Encode frame and add to stack (FIFO)
        encoded_frame = self._encode_frame(image)
        self.frame_stack.append(encoded_frame)

        # Add direction to history (FIFO)
        self.direction_history.append(direction)

        # Return stacked observation
        return self._get_stacked_obs()


def make_env(seed=None):
    """
    Create MiniGrid FourRooms environment with stacked egocentric observations.

    Args:
        seed: Random seed for environment

    Returns:
        Wrapped gymnasium environment with 510-dim vector observations and 3 actions
    """
    # Create base environment
    env = gym.make("MiniGrid-FourRooms-v0")

    # Restrict action space to left, right, forward (3 actions)
    env = ActionRestrictionWrapper(env)

    # Wrap with stacked observation wrapper
    env = StackedEgoObsWrapper(env, stack_size=5)

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    return env


def get_env_info():
    """Get environment observation and action space information."""
    env = make_env()
    obs, _ = env.reset()

    state_size = obs.shape[0]
    action_size = env.action_space.n

    env.close()

    return state_size, action_size


if __name__ == "__main__":
    # Test environment setup
    print("Testing MiniGrid FourRooms with stacked egocentric observations...")

    env = make_env(seed=42)
    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation min: {obs.min()}, max: {obs.max()}")
    print(f"Action space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")

    # Verify observation size
    assert obs.shape == (510,), f"Expected (510,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    print("\n✓ Observation shape and dtype correct!")

    # Test a few steps
    print("\nTesting environment steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {i + 1}: action={action}, reward={reward}, obs_shape={obs.shape}, terminated={terminated}"
        )

        assert obs.shape == (510,), f"Step {i + 1}: Expected (510,), got {obs.shape}"

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            assert obs.shape == (510,), f"After reset: Expected (510,), got {obs.shape}"

    env.close()
    print("\n✓ Environment test complete!")
    print(f"\nFinal observation vector size: {obs.shape[0]} features")
    print("  - Stacked frames: 5 × 98 = 490 features")
    print("  - Direction history: 5 × 4 = 20 features")
    print("  - Total: 510 features")
    print(f"\nAction space: {env.action_space.n} actions (left, right, forward)")
