"""Mutation-free, original-objective evaluation shared by every agent type.

This module is intentionally separate from training rollouts. Evaluation calls
only ``agent.select_action(..., explore=False)``: it never invokes learning or
episode hooks, applies reward shaping, or advances a training schedule itself.
The policy implementation remains responsible for making its greedy action
path side-effect free.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.analysis.statistics import summarize_episodes
from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.environments.observation import validate_observation
from mountaincar_rl.utils.seeding import validate_seed


State = tuple[float, float]


class EnvironmentFactory(Protocol):
    """Callable contract used to create an evaluation-owned environment."""

    def __call__(
        self, *, seed: int, max_episode_steps: int
    ) -> gym.Env[Any, Any]: ...


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    """Immutable result of one greedy, fixed-seed evaluation episode.

    State, position, and velocity trajectories include the reset observation
    followed by every post-transition observation, so their length is
    ``steps + 1``. Action and original-reward trajectories contain one entry per
    environment step. All trajectory tuples are empty only when the caller
    explicitly disables capture.
    """

    environment_seed: int
    steps: int
    original_return: float
    environment_completion: bool
    assignment_success: bool
    terminated: bool
    truncated: bool
    state_trajectory: tuple[State, ...]
    action_trajectory: tuple[int, ...]
    position_trajectory: tuple[float, ...]
    velocity_trajectory: tuple[float, ...]
    original_reward_trajectory: tuple[float, ...]
    shaped_return: float | None = None

    def __post_init__(self) -> None:
        validate_seed(self.environment_seed)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("steps must be an integer")
        if self.steps < 1:
            raise ValueError("an evaluated episode must contain at least one step")
        if not math.isfinite(self.original_return):
            raise ValueError("original_return must be finite")
        if self.shaped_return is not None and not math.isfinite(self.shaped_return):
            raise ValueError("shaped_return must be finite when provided")
        if not self.terminated and not self.truncated:
            raise ValueError("an evaluated episode must end by termination or truncation")
        if self.assignment_success and not self.environment_completion:
            raise ValueError("assignment success requires environment completion")

        if self.state_trajectory:
            if len(self.state_trajectory) != self.steps + 1:
                raise ValueError("state_trajectory length must equal steps + 1")
            if len(self.action_trajectory) != self.steps:
                raise ValueError("action_trajectory length must equal steps")
            if len(self.original_reward_trajectory) != self.steps:
                raise ValueError("original_reward_trajectory length must equal steps")
            if len(self.position_trajectory) != self.steps + 1:
                raise ValueError("position_trajectory length must equal steps + 1")
            if len(self.velocity_trajectory) != self.steps + 1:
                raise ValueError("velocity_trajectory length must equal steps + 1")
            for state, position, velocity in zip(
                self.state_trajectory,
                self.position_trajectory,
                self.velocity_trajectory,
                strict=True,
            ):
                if len(state) != 2 or not all(math.isfinite(value) for value in state):
                    raise ValueError("state_trajectory must contain finite 2-D states")
                if state != (position, velocity):
                    raise ValueError(
                        "position and velocity trajectories must match state_trajectory"
                    )
            reward_sum = math.fsum(self.original_reward_trajectory)
            if not math.isclose(
                reward_sum, self.original_return, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise ValueError(
                    "original_return must equal the sum of original rewards"
                )
        elif any(
            (
                self.action_trajectory,
                self.position_trajectory,
                self.velocity_trajectory,
                self.original_reward_trajectory,
            )
        ):
            raise ValueError("trajectory fields must be captured together")

    @property
    def trajectories_captured(self) -> bool:
        """Whether full state/action/reward trajectories are retained."""

        return bool(self.state_trajectory)

    def to_episode_record(
        self,
        *,
        condition: str,
        split: str,
        training_seed: int,
        episode_index: int,
    ) -> dict[str, Any]:
        """Convert to the structured episode schema used by result writers."""

        _validate_record_context(
            condition=condition,
            split=split,
            training_seed=training_seed,
            episode_index=episode_index,
        )
        return {
            "condition": condition,
            "split": split,
            "training_seed": int(training_seed),
            "episode": int(episode_index),
            "environment_seed": self.environment_seed,
            "steps": self.steps,
            # Keep the phase-1 field for existing statistical/reporting code,
            # while making the original-objective meaning explicit.
            "episode_return": self.original_return,
            "original_return": self.original_return,
            "shaped_return": self.shaped_return,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "environment_completion": self.environment_completion,
            "assignment_success": self.assignment_success,
            "trajectories_captured": self.trajectories_captured,
        }

    def to_transition_records(
        self,
        *,
        condition: str,
        split: str,
        training_seed: int,
        episode_index: int,
    ) -> list[dict[str, Any]]:
        """Flatten captured trajectories into one original-reward row per step."""

        _validate_record_context(
            condition=condition,
            split=split,
            training_seed=training_seed,
            episode_index=episode_index,
        )
        if not self.trajectories_captured:
            return []

        rows: list[dict[str, Any]] = []
        for index, (state, action, reward, next_state) in enumerate(
            zip(
                self.state_trajectory[:-1],
                self.action_trajectory,
                self.original_reward_trajectory,
                self.state_trajectory[1:],
                strict=True,
            )
        ):
            final_transition = index == self.steps - 1
            rows.append(
                {
                    "condition": condition,
                    "split": split,
                    "training_seed": int(training_seed),
                    "episode": int(episode_index),
                    "environment_seed": self.environment_seed,
                    "step": index + 1,
                    "position": state[0],
                    "velocity": state[1],
                    "action": action,
                    "reward": reward,
                    "original_reward": reward,
                    "next_position": next_state[0],
                    "next_velocity": next_state[1],
                    "terminated": bool(final_transition and self.terminated),
                    "truncated": bool(final_transition and self.truncated),
                }
            )
        return rows


def _validate_record_context(
    *, condition: str, split: str, training_seed: int, episode_index: int
) -> None:
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a nonempty string")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    validate_seed(training_seed)
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode_index must be an integer")
    if episode_index < 0:
        raise ValueError("episode_index must be nonnegative")


def _state(observation: Any) -> State:
    validated = validate_observation(observation)
    return float(validated[0]), float(validated[1])


def _seed_space(space: Any, seed: int) -> None:
    seed_method = getattr(space, "seed", None)
    if callable(seed_method):
        seed_method(seed)


def evaluate_agent(
    agent: Agent,
    *,
    episode_seeds: Iterable[int],
    training_seed: int,
    condition: str,
    max_episode_steps: int = 200,
    env_factory: EnvironmentFactory = make_env,
    capture_trajectories: bool = True,
) -> list[EvaluationEpisode]:
    """Evaluate an agent greedily on fixed seeds and the original objective.

    The returned environments are owned and closed by this function. One
    environment is reused across episode seeds, with the environment and both
    spaces reseeded for every episode. Collection stops on ``terminated or
    truncated`` while retaining both flags independently. Assignment success
    is true only when the observed goal position is first reached within 100
    actual calls to ``env.step``.

    This function never calls ``observe``, ``end_episode``, an optimizer, a
    replay buffer, or a reward shaper. It does not snapshot arbitrary agents;
    implementations of ``select_action(..., explore=False)`` must therefore be
    side-effect free for learned policies.
    """

    valid_training_seed = validate_seed(training_seed)
    del valid_training_seed  # Validation is intentional; the value is metadata only.
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a nonempty string")
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise TypeError("max_episode_steps must be an integer")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if not isinstance(capture_trajectories, bool):
        raise TypeError("capture_trajectories must be a boolean")

    seeds = [validate_seed(seed) for seed in episode_seeds]
    if not seeds:
        raise ValueError("episode_seeds must contain at least one seed")

    env = env_factory(seed=seeds[0], max_episode_steps=max_episode_steps)
    try:
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "goal_position"):
            raise AttributeError(
                "evaluation environment must expose unwrapped.goal_position"
            )
        goal_position = float(getattr(unwrapped, "goal_position"))
        if not math.isfinite(goal_position):
            raise ValueError("environment goal_position must be finite")

        episodes: list[EvaluationEpisode] = []
        for environment_seed in seeds:
            reset_result = env.reset(seed=environment_seed)
            if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                raise TypeError("env.reset(seed=...) must return (observation, info)")
            observation, _ = reset_result
            _seed_space(env.action_space, environment_seed)
            _seed_space(env.observation_space, environment_seed)

            initial_state = _state(observation)
            states: list[State] = [initial_state]
            actions: list[int] = []
            original_rewards: list[float] = []
            terminated = False
            truncated = False
            first_goal_step: int | None = None

            while not (terminated or truncated):
                policy_observation = np.asarray(observation, dtype=np.float32).copy()
                raw_action = agent.select_action(
                    policy_observation,
                    explore=False,
                )
                if isinstance(raw_action, bool) or not isinstance(
                    raw_action, (int, np.integer)
                ):
                    raise TypeError("agent.select_action must return an integer action")
                action = int(raw_action)
                if not env.action_space.contains(action):
                    raise ValueError(
                        f"agent selected action {action}, which is outside action_space"
                    )

                step_result = env.step(action)
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    raise TypeError(
                        "env.step(action) must return "
                        "(observation, reward, terminated, truncated, info)"
                    )
                next_observation, reward, terminated_raw, truncated_raw, _ = step_result
                next_state = _state(next_observation)
                try:
                    original_reward = float(reward)
                except (TypeError, ValueError) as error:
                    raise TypeError("environment reward must be a real number") from error
                if not math.isfinite(original_reward):
                    raise ValueError("environment reward must be finite")

                terminated = bool(terminated_raw)
                truncated = bool(truncated_raw)
                actions.append(action)
                original_rewards.append(original_reward)
                states.append(next_state)
                if first_goal_step is None and next_state[0] >= goal_position:
                    first_goal_step = len(actions)

                observation = next_observation
                if (
                    len(actions) >= max_episode_steps
                    and not (terminated or truncated)
                ):
                    raise RuntimeError(
                        "environment exceeded max_episode_steps without termination "
                        "or truncation; verify the evaluation factory's TimeLimit"
                    )

            if capture_trajectories:
                state_trajectory = tuple(states)
                action_trajectory = tuple(actions)
                position_trajectory = tuple(state[0] for state in states)
                velocity_trajectory = tuple(state[1] for state in states)
                reward_trajectory = tuple(original_rewards)
            else:
                state_trajectory = ()
                action_trajectory = ()
                position_trajectory = ()
                velocity_trajectory = ()
                reward_trajectory = ()

            environment_completion = first_goal_step is not None
            episodes.append(
                EvaluationEpisode(
                    environment_seed=environment_seed,
                    steps=len(actions),
                    original_return=math.fsum(original_rewards),
                    environment_completion=environment_completion,
                    assignment_success=bool(
                        first_goal_step is not None and first_goal_step <= 100
                    ),
                    terminated=terminated,
                    truncated=truncated,
                    state_trajectory=state_trajectory,
                    action_trajectory=action_trajectory,
                    position_trajectory=position_trajectory,
                    velocity_trajectory=velocity_trajectory,
                    original_reward_trajectory=reward_trajectory,
                    shaped_return=None,
                )
            )
        return episodes
    finally:
        env.close()


def summarize_evaluation_episodes(
    episodes: Iterable[EvaluationEpisode],
) -> dict[str, Any]:
    """Summarize evaluation episodes using the existing original-return schema."""

    episode_list = list(episodes)
    rows = [
        {
            "environment_completion": episode.environment_completion,
            "assignment_success": episode.assignment_success,
            "steps": episode.steps,
            "episode_return": episode.original_return,
        }
        for episode in episode_list
    ]
    return summarize_episodes(rows)
