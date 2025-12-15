"""MiniGrid environment setup with vector observation wrappers."""

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
import numpy as np


class FlattenObservation(gym.ObservationWrapper):
    """Flatten the observation to a 1D vector."""

    def __init__(self, env):
        super().__init__(env)
        # Get the original observation space shape
        obs_shape = env.observation_space.shape
        # Flatten to 1D
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(np.prod(obs_shape),), dtype=np.uint8
        )

    def observation(self, obs):
        """Flatten the observation."""
        return obs.flatten()


def make_env(seed=None):
    """
    Create MiniGrid FourRooms environment with vector observations.

    Args:
        seed: Random seed for environment

    Returns:
        Wrapped gymnasium environment with flat vector observations
    """
    # Create base environment
    env = gym.make("MiniGrid-FourRooms-v0")

    # Wrap to get full observability (agent sees entire grid)
    env = FullyObsWrapper(env)

    # Flatten the observation to 1D vector
    env = FlatObsWrapper(env)

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
    print("Testing MiniGrid FourRooms environment setup...")

    env = make_env(seed=42)
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")

    # Test a few steps
    print("\nTesting environment steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i + 1}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

    env.close()
    print("\nEnvironment test complete!")
