"""Evaluation-quality episode rollout shared by training and evaluation CLIs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np

from mountaincar_rl.agents.base import Agent


EPISODE_FIELDNAMES = [
    "condition",
    "split",
    "training_seed",
    "episode",
    "environment_seed",
    "steps",
    "episode_return",
    "terminated",
    "truncated",
    "environment_completion",
    "assignment_success",
]


def training_episode_seeds(training_seed: int, episodes: int) -> list[int]:
    """Generate a transparent training-only seed sequence.

    Training episode seeds begin at 100,000 and occupy a separate million-value
    block per training seed. Validation (1000-series) and held-out test
    (2000-series) seeds are therefore structurally disjoint.
    """

    if training_seed < 0:
        raise ValueError("training_seed must be nonnegative.")
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    first = 100_000 + training_seed * 1_000_000
    last = first + episodes - 1
    if last > np.iinfo(np.uint32).max:
        raise ValueError("Requested training episode seeds exceed uint32 range.")
    return list(range(first, last + 1))


def rollout_episodes(
    *,
    env: gym.Env[Any, int],
    agent: Agent,
    condition: str,
    split: str,
    training_seed: int,
    episode_seeds: Iterable[int],
    record_trajectories: bool,
    explore: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run complete episodes and return episode-level and transition records.

    Rollouts stop on either Gymnasium termination or truncation. Completion is
    inferred from the car reaching the unwrapped environment's goal position,
    whereas assignment success additionally requires at most 100 actual steps.
    """

    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    goal_position = float(env.unwrapped.goal_position)

    for episode_index, environment_seed in enumerate(episode_seeds):
        observation, _ = env.reset(seed=int(environment_seed))
        env.action_space.seed(int(environment_seed))
        episode_return = 0.0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(
                agent.select_action(np.asarray(observation), explore=explore)
            )
            next_observation, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            episode_return += float(reward)

            if record_trajectories:
                transitions.append(
                    {
                        "condition": condition,
                        "split": split,
                        "training_seed": training_seed,
                        "episode": episode_index,
                        "environment_seed": int(environment_seed),
                        "step": step_count,
                        "position": float(observation[0]),
                        "velocity": float(observation[1]),
                        "action": action,
                        "reward": float(reward),
                        "next_position": float(next_observation[0]),
                        "next_velocity": float(next_observation[1]),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
            observation = next_observation

        environment_completion = bool(float(observation[0]) >= goal_position)
        episodes.append(
            {
                "condition": condition,
                "split": split,
                "training_seed": training_seed,
                "episode": episode_index,
                "environment_seed": int(environment_seed),
                "steps": step_count,
                "episode_return": episode_return,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "environment_completion": environment_completion,
                "assignment_success": bool(
                    environment_completion and step_count <= 100
                ),
            }
        )

    if not episodes:
        raise ValueError("episode_seeds must contain at least one seed.")
    return episodes, transitions
