"""Contract tests for the shared mutation-free evaluation protocol."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.experiments.evaluation import (
    EvaluationEpisode,
    evaluate_agent,
    summarize_evaluation_episodes,
)


class ScriptedEvaluationEnv(gym.Env[np.ndarray, int]):
    """Seeded MountainCar-shaped environment with controlled boundary flags."""

    metadata: dict[str, Any] = {}

    def __init__(self, *, goal_step: int | None, horizon: int) -> None:
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
        self._initial_position = -0.5
        self.close_calls = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._step = 0
        self._initial_position = float(self.np_random.uniform(-0.6, -0.4))
        return np.array([self._initial_position, 0.0], dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self._step += 1
        reached_goal = self._goal_step is not None and self._step >= self._goal_step
        position = (
            self.goal_position
            if reached_goal
            else self._initial_position + 0.0001 * self._step * (action - 1)
        )
        velocity = 0.01 * (action - 1)
        observation = np.array([position, velocity], dtype=np.float32)
        reward = -1.0 - 0.25 * action
        return (
            observation,
            reward,
            bool(reached_goal),
            bool(self._step >= self._horizon),
            {"shaped_reward": 10_000.0},
        )

    def close(self) -> None:
        self.close_calls += 1


class ScriptedFactory:
    """Capture factory arguments and environments for ownership assertions."""

    def __init__(self, *, goal_step: int | None, horizon: int | None = None) -> None:
        self.goal_step = goal_step
        self.horizon = horizon
        self.calls: list[tuple[int, int]] = []
        self.environments: list[ScriptedEvaluationEnv] = []

    def __call__(
        self, *, seed: int, max_episode_steps: int
    ) -> ScriptedEvaluationEnv:
        self.calls.append((seed, max_episode_steps))
        environment = ScriptedEvaluationEnv(
            goal_step=self.goal_step,
            horizon=self.horizon or max_episode_steps,
        )
        self.environments.append(environment)
        return environment


class SpyAgent(Agent):
    """Record API calls while returning a fixed valid action."""

    def __init__(self, action: int = 2) -> None:
        self.action = action
        self.explore_arguments: list[bool] = []
        self.observe_calls = 0
        self.end_episode_calls = 0

    def select_action(
        self, observation: np.ndarray, *, explore: bool = True
    ) -> int:
        assert observation.shape == (2,)
        self.explore_arguments.append(explore)
        return self.action

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        del observation, action, reward, next_observation, terminated, truncated, info
        self.observe_calls += 1

    def end_episode(self) -> None:
        self.end_episode_calls += 1


class StatefulValueAgent(Agent):
    """Greedy policy whose learner state can be compared before and after eval."""

    def __init__(self) -> None:
        self.environment_steps = 37
        self.optimizer_steps = 11
        self.replay = [("sentinel", 5)]
        self.rng = np.random.default_rng(8128)
        self.observe_calls = 0
        self.end_episode_calls = 0

    def select_action(
        self, observation: np.ndarray, *, explore: bool = True
    ) -> int:
        if explore:
            raise AssertionError("evaluation requested exploratory behavior")
        # No counters or RNG are touched on the greedy path.
        return int(observation[1] >= 0.0) + 1

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        del observation, action, reward, next_observation, terminated, truncated, info
        self.observe_calls += 1
        self.environment_steps += 1
        self.optimizer_steps += 1
        self.replay.append(("transition", self.observe_calls))
        self.rng.random()

    def end_episode(self) -> None:
        self.end_episode_calls += 1


def _evaluate(
    *,
    goal_step: int | None,
    horizon: int | None = None,
    agent: Agent | None = None,
    episode_seeds: list[int] | None = None,
    capture_trajectories: bool = True,
) -> tuple[list[EvaluationEpisode], ScriptedFactory, Agent]:
    factory = ScriptedFactory(goal_step=goal_step, horizon=horizon)
    selected_agent = agent or SpyAgent()
    episodes = evaluate_agent(
        selected_agent,
        episode_seeds=episode_seeds or [1000],
        training_seed=0,
        condition="scripted",
        max_episode_steps=200,
        env_factory=factory,
        capture_trajectories=capture_trajectories,
    )
    return episodes, factory, selected_agent


def test_evaluation_returns_complete_trajectories_and_original_rewards() -> None:
    episodes, factory, agent = _evaluate(goal_step=3)
    episode = episodes[0]

    assert factory.calls == [(1000, 200)]
    assert factory.environments[0].close_calls == 1
    assert isinstance(agent, SpyAgent)
    assert agent.explore_arguments == [False, False, False]
    assert agent.observe_calls == 0
    assert agent.end_episode_calls == 0

    assert episode.steps == 3
    assert episode.action_trajectory == (2, 2, 2)
    assert episode.original_reward_trajectory == (-1.5, -1.5, -1.5)
    assert episode.original_return == -4.5
    assert episode.shaped_return is None
    assert len(episode.state_trajectory) == 4
    assert len(episode.position_trajectory) == 4
    assert len(episode.velocity_trajectory) == 4
    assert episode.position_trajectory == tuple(
        state[0] for state in episode.state_trajectory
    )
    assert episode.velocity_trajectory == tuple(
        state[1] for state in episode.state_trajectory
    )
    assert episode.state_trajectory[-1][0] == 0.5
    assert episode.terminated is True
    assert episode.truncated is False
    assert episode.environment_completion is True
    assert episode.assignment_success is True


def test_episode_and_transition_record_conversion_preserves_both_flags() -> None:
    episode = _evaluate(goal_step=3, horizon=3)[0][0]

    record = episode.to_episode_record(
        condition="dqn_replay", split="validation", training_seed=4, episode_index=7
    )
    transitions = episode.to_transition_records(
        condition="dqn_replay", split="validation", training_seed=4, episode_index=7
    )

    assert record["episode_return"] == record["original_return"] == -4.5
    assert record["shaped_return"] is None
    assert record["terminated"] is True
    assert record["truncated"] is True
    assert len(transitions) == 3
    assert transitions[0]["terminated"] is False
    assert transitions[0]["truncated"] is False
    assert transitions[-1]["terminated"] is True
    assert transitions[-1]["truncated"] is True
    assert transitions[-1]["reward"] == transitions[-1]["original_reward"] == -1.5


@pytest.mark.parametrize(
    ("goal_step", "expected_assignment_success"),
    [(100, True), (150, False)],
)
def test_assignment_threshold_is_inclusive_without_relabeling_late_completion(
    goal_step: int, expected_assignment_success: bool
) -> None:
    episode = _evaluate(goal_step=goal_step)[0][0]

    assert episode.steps == goal_step
    assert episode.environment_completion is True
    assert episode.assignment_success is expected_assignment_success


def test_pure_time_limit_truncation_is_not_completion() -> None:
    episode = _evaluate(goal_step=None, horizon=4)[0][0]

    assert episode.steps == 4
    assert episode.terminated is False
    assert episode.truncated is True
    assert episode.environment_completion is False
    assert episode.assignment_success is False


def test_goal_is_detected_when_termination_and_truncation_are_simultaneous() -> None:
    episode = _evaluate(goal_step=4, horizon=4)[0][0]

    assert episode.steps == 4
    assert episode.terminated is True
    assert episode.truncated is True
    assert episode.environment_completion is True
    assert episode.assignment_success is True


def test_fixed_episode_seeds_reproduce_complete_evaluation_results() -> None:
    agent = StatefulValueAgent()
    first = _evaluate(
        goal_step=None,
        horizon=5,
        agent=agent,
        episode_seeds=[1000, 1001, 1002],
    )[0]
    second = _evaluate(
        goal_step=None,
        horizon=5,
        agent=agent,
        episode_seeds=[1000, 1001, 1002],
    )[0]

    assert first == second
    assert [episode.environment_seed for episode in first] == [1000, 1001, 1002]


def test_evaluation_does_not_mutate_learning_replay_optimizer_counter_or_rng_state() -> None:
    agent = StatefulValueAgent()
    before = {
        "environment_steps": agent.environment_steps,
        "optimizer_steps": agent.optimizer_steps,
        "replay": copy.deepcopy(agent.replay),
        "rng": copy.deepcopy(agent.rng.bit_generator.state),
    }

    _evaluate(goal_step=None, horizon=5, agent=agent, episode_seeds=[1000, 1001])

    assert agent.environment_steps == before["environment_steps"]
    assert agent.optimizer_steps == before["optimizer_steps"]
    assert agent.replay == before["replay"]
    assert agent.rng.bit_generator.state == before["rng"]
    assert agent.observe_calls == 0
    assert agent.end_episode_calls == 0


def test_trajectory_capture_can_be_disabled_without_changing_metrics() -> None:
    captured = _evaluate(goal_step=3, capture_trajectories=True)[0][0]
    omitted = _evaluate(goal_step=3, capture_trajectories=False)[0][0]

    assert omitted.trajectories_captured is False
    assert omitted.state_trajectory == ()
    assert omitted.action_trajectory == ()
    assert omitted.position_trajectory == ()
    assert omitted.velocity_trajectory == ()
    assert omitted.original_reward_trajectory == ()
    assert omitted.steps == captured.steps
    assert omitted.original_return == captured.original_return
    assert omitted.environment_completion == captured.environment_completion
    assert omitted.assignment_success == captured.assignment_success


def test_evaluation_summary_uses_original_returns_and_null_success_steps() -> None:
    episodes = _evaluate(
        goal_step=None, horizon=3, episode_seeds=[1000, 1001]
    )[0]

    summary = summarize_evaluation_episodes(episodes)

    assert summary == {
        "episodes": 2,
        "environment_completion_rate": 0.0,
        "assignment_success_rate": 0.0,
        "mean_steps_among_completions": None,
        "median_steps_among_completions": None,
        "mean_episode_return": -4.5,
    }
