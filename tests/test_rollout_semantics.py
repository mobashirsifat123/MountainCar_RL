"""Contract tests for evaluation stopping and the two success definitions."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from mountaincar_rl.agents.random_agent import RandomAgent
from mountaincar_rl.analysis.statistics import summarize_episodes
from mountaincar_rl.experiments.rollout import rollout_episodes


class ScriptedGoalEnv(gym.Env[np.ndarray, int]):
    """Minimal environment that reaches the MountainCar goal on a fixed step."""

    metadata: dict[str, Any] = {}

    def __init__(self, *, goal_step: int | None, horizon: int = 200) -> None:
        self.goal_position = 0.5
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.2, -0.07], dtype=np.float32),
            high=np.array([0.6, 0.07], dtype=np.float32),
            dtype=np.float32,
        )
        self._goal_step = goal_step
        self._horizon = horizon
        self._step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.array([-0.5, 0.0], dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self._step += 1
        reached_goal = self._goal_step is not None and self._step >= self._goal_step
        position = self.goal_position if reached_goal else -0.5
        observation = np.array([position, 0.0], dtype=np.float32)
        return (
            observation,
            -1.0,
            bool(reached_goal),
            bool(self._step >= self._horizon),
            {},
        )


def _single_episode(*, goal_step: int | None, horizon: int = 200) -> dict[str, Any]:
    env = ScriptedGoalEnv(goal_step=goal_step, horizon=horizon)
    agent = RandomAgent(env.action_space, seed=7)
    try:
        episodes, _ = rollout_episodes(
            env=env,
            agent=agent,
            condition="scripted",
            split="validation",
            training_seed=0,
            episode_seeds=[1000],
            record_trajectories=False,
            explore=False,
        )
    finally:
        env.close()
    return episodes[0]


def test_completion_after_100_steps_is_not_assignment_success() -> None:
    """A 150-step flag reach must retain completion but fail the assignment."""

    episode = _single_episode(goal_step=150)

    assert episode["steps"] == 150
    assert episode["environment_completion"] is True
    assert episode["assignment_success"] is False


def test_completion_at_100_steps_meets_assignment_target() -> None:
    """The assignment threshold is inclusive at exactly 100 actual steps."""

    episode = _single_episode(goal_step=100)

    assert episode["steps"] == 100
    assert episode["environment_completion"] is True
    assert episode["assignment_success"] is True


def test_goal_is_preserved_when_termination_and_truncation_coincide() -> None:
    """A final-horizon goal cannot be erased by a simultaneous truncation flag."""

    episode = _single_episode(goal_step=200, horizon=200)

    assert episode["terminated"] is True
    assert episode["truncated"] is True
    assert episode["environment_completion"] is True
    assert episode["assignment_success"] is False


def test_no_completion_uses_null_conditional_step_summaries() -> None:
    """Do not fabricate successful-step statistics when no goal was reached."""

    episode = _single_episode(goal_step=None, horizon=2)
    summary = summarize_episodes([episode])

    assert episode["terminated"] is False
    assert episode["truncated"] is True
    assert summary["environment_completion_rate"] == 0.0
    assert summary["assignment_success_rate"] == 0.0
    assert summary["mean_steps_among_completions"] is None
    assert summary["median_steps_among_completions"] is None

