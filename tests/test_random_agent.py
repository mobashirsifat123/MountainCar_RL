"""Tests for the no-learning random-policy baseline."""

import gymnasium as gym
import numpy as np

from mountaincar_rl.agents.random_agent import RandomAgent


def test_random_agent_emits_valid_discrete_actions() -> None:
    """Every selected action must belong to MountainCar's discrete action set."""
    action_space = gym.spaces.Discrete(3)
    agent = RandomAgent(action_space=action_space, seed=31)
    observation = np.array([-0.5, 0.0], dtype=np.float32)

    actions = [agent.select_action(observation, explore=True) for _ in range(50)]

    assert all(action_space.contains(action) for action in actions)


def test_random_agent_sequence_is_reproducible_for_equal_seeds() -> None:
    """Two fresh agents with equal seeds should generate equal action sequences."""
    first = RandomAgent(action_space=gym.spaces.Discrete(3), seed=4)
    second = RandomAgent(action_space=gym.spaces.Discrete(3), seed=4)
    observation = np.array([-0.5, 0.0], dtype=np.float32)

    first_actions = [first.select_action(observation, explore=True) for _ in range(32)]
    second_actions = [
        second.select_action(observation, explore=True) for _ in range(32)
    ]

    assert first_actions == second_actions
