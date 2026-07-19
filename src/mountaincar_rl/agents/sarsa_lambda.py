"""True on-policy tile-coded SARSA(lambda) with eligibility traces."""

from __future__ import annotations

import copy
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.agents.tile_coder import TileCoder
from mountaincar_rl.utils.checkpoints import atomic_save_bytes
from mountaincar_rl.utils.seeding import validate_seed


def _finite_float(value: Any, *, name: str) -> float:
    """Validate a real finite scalar while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


class SarsaLambdaAgent(Agent):
    """On-policy SARSA(lambda) over binary tile-coded state-action features.

    The action value is the sum of the active weights.  For TD error

    ``delta = r + gamma * (0 if terminated else Q(s_next, a_next)) - Q(s, a)``,

    current traces are replaced by one (or accumulated), weights receive
    ``(alpha / num_tilings) * delta * trace``, and traces then decay by
    ``gamma * lambda``.  Artificial time-limit truncation deliberately does
    not mask the bootstrap term; callers still end collection and invoke
    :meth:`end_episode` after either Gymnasium boundary flag.
    """

    _CHECKPOINT_VERSION = 1
    _TRACE_TYPES = frozenset(("replacing", "accumulating"))

    def __init__(
        self,
        tile_coder: TileCoder,
        alpha: float,
        gamma: float,
        lambda_: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        trace_type: str = "replacing",
        trace_clip: float | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize weights, traces, exploration schedule, and private RNG."""

        if not isinstance(tile_coder, TileCoder):
            raise TypeError("tile_coder must be a TileCoder")
        validated_alpha = _finite_float(alpha, name="alpha")
        if validated_alpha <= 0.0:
            raise ValueError("alpha must be positive")
        validated_gamma = _probability(gamma, name="gamma")
        validated_lambda = _probability(lambda_, name="lambda_")
        validated_epsilon_start = _probability(
            epsilon_start, name="epsilon_start"
        )
        validated_epsilon_end = _probability(epsilon_end, name="epsilon_end")
        if validated_epsilon_end > validated_epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        if isinstance(epsilon_decay_steps, bool) or not isinstance(
            epsilon_decay_steps, (int, np.integer)
        ):
            raise TypeError("epsilon_decay_steps must be an integer")
        validated_decay_steps = int(epsilon_decay_steps)
        if validated_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if trace_type not in self._TRACE_TYPES:
            allowed = ", ".join(sorted(self._TRACE_TYPES))
            raise ValueError(f"trace_type must be one of: {allowed}")
        if trace_clip is None:
            validated_trace_clip = None
        else:
            validated_trace_clip = _finite_float(trace_clip, name="trace_clip")
            if validated_trace_clip <= 0.0:
                raise ValueError("trace_clip must be positive when provided")

        self.tile_coder = tile_coder
        self.alpha = validated_alpha
        self.gamma = validated_gamma
        self.lambda_ = validated_lambda
        self.epsilon_start = validated_epsilon_start
        self.epsilon_end = validated_epsilon_end
        self.epsilon_decay_steps = validated_decay_steps
        self.trace_type = trace_type
        self.trace_clip = validated_trace_clip
        self.seed = validate_seed(seed)

        self.weights = np.zeros(self.tile_coder.total_features, dtype=np.float64)
        self.traces = np.zeros_like(self.weights)
        self.update_count = 0
        self.last_terminated: bool | None = None
        self.last_truncated: bool | None = None
        self._rng = np.random.default_rng(self.seed)

    @property
    def current_epsilon(self) -> float:
        """Return linearly decayed epsilon at the current successful update count."""

        fraction = min(self.update_count / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def q_value(self, observation: np.ndarray, action: int) -> float:
        """Return the sum of active weights for one state-action pair."""

        active = self.tile_coder.encode(observation, action)
        value = float(np.sum(self.weights[active], dtype=np.float64))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite SARSA action value")
        return value

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return finite action values in environment action order."""

        state_tiles = self.tile_coder.state_tiles(observation)
        actions = np.arange(self.tile_coder.num_actions, dtype=np.int64)
        indices = state_tiles[np.newaxis, :] + (
            actions[:, np.newaxis] * self.tile_coder.features_per_action
        )
        values = np.sum(self.weights[indices], axis=1, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite SARSA action values")
        return values

    def select_action(
        self, observation: np.ndarray, *, explore: bool = True
    ) -> int:
        """Choose epsilon-greedily for training or deterministically for evaluation.

        Evaluation (``explore=False``) selects the smallest maximizing action
        and advances neither RNG nor schedule state.  Training uses the private
        RNG both for exploration and for ties among greedy actions.
        """

        if not isinstance(explore, (bool, np.bool_)):
            raise TypeError("explore must be a boolean")
        values = self.q_values(observation)
        maximizing = np.flatnonzero(values == np.max(values))
        if not bool(explore):
            return int(maximizing[0])
        if self._rng.random() < self.current_epsilon:
            return int(self._rng.integers(self.tile_coder.num_actions))
        return int(self._rng.choice(maximizing))

    def _clip_and_check_traces(self) -> None:
        if self.trace_clip is not None:
            np.clip(
                self.traces,
                -self.trace_clip,
                self.trace_clip,
                out=self.traces,
            )
        if not np.all(np.isfinite(self.traces)):
            raise FloatingPointError("non-finite SARSA eligibility trace")

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_action: int | None = None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Apply one SARSA(lambda) update and return its scalar TD error.

        ``terminated`` alone masks bootstrapping.  Therefore ``next_action`` is
        optional at a true terminal transition but required for all other
        transitions, including a pure time-limit truncation.
        """

        validated_reward = _finite_float(reward, name="reward")
        is_terminated = _boolean(terminated, name="terminated")
        is_truncated = _boolean(truncated, name="truncated")
        active = self.tile_coder.encode(observation, action)
        current_value = float(np.sum(self.weights[active], dtype=np.float64))
        if not np.isfinite(current_value):
            raise FloatingPointError("non-finite current SARSA action value")

        if is_terminated:
            target = validated_reward
        else:
            if next_action is None:
                raise ValueError(
                    "next_action is required when terminated is false, "
                    "including at pure truncation"
                )
            next_value = self.q_value(next_observation, next_action)
            target = validated_reward + self.gamma * next_value
        td_error = target - current_value
        if not np.isfinite(td_error):
            raise FloatingPointError("non-finite SARSA TD error")
        if not np.all(np.isfinite(self.traces)):
            raise FloatingPointError("non-finite SARSA eligibility trace")

        if self.trace_type == "replacing":
            self.traces[active] = 1.0
        else:
            self.traces[active] += 1.0
        self._clip_and_check_traces()

        scaled_step = self.alpha / self.tile_coder.num_tilings
        with np.errstate(over="ignore", invalid="ignore"):
            increment = scaled_step * td_error * self.traces
            updated_weights = self.weights + increment
        if not np.all(np.isfinite(updated_weights)):
            raise FloatingPointError("SARSA update would produce non-finite weights")
        self.weights[...] = updated_weights

        self.traces *= self.gamma * self.lambda_
        self._clip_and_check_traces()
        self.update_count += 1
        self.last_terminated = is_terminated
        self.last_truncated = is_truncated
        return float(td_error)

    def end_episode(self) -> None:
        """Clear all traces at a collected episode boundary."""

        self.traces.fill(0.0)

    def _hyperparameters(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "trace_type": self.trace_type,
            "trace_clip": self.trace_clip,
            "seed": self.seed,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a complete, independent checkpoint state mapping."""

        return {
            "checkpoint_version": self._CHECKPOINT_VERSION,
            "tile_coder": self.tile_coder.metadata(),
            "hyperparameters": self._hyperparameters(),
            "weights": self.weights.copy(),
            "traces": self.traces.copy(),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "update_count": self.update_count,
            "last_terminated": self.last_terminated,
            "last_truncated": self.last_truncated,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore complete state after validating configuration compatibility."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        version = state.get("checkpoint_version")
        if version != self._CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported SARSA checkpoint version {version!r}; "
                f"expected {self._CHECKPOINT_VERSION}"
            )
        saved_tile_coder = state.get("tile_coder")
        if saved_tile_coder != self.tile_coder.metadata():
            raise ValueError("checkpoint tile-coder metadata is incompatible")
        saved_hyperparameters = state.get("hyperparameters")
        if saved_hyperparameters != self._hyperparameters():
            raise ValueError("checkpoint SARSA hyperparameters are incompatible")

        weights = np.asarray(state.get("weights"), dtype=np.float64)
        traces = np.asarray(state.get("traces"), dtype=np.float64)
        expected_shape = (self.tile_coder.total_features,)
        if weights.shape != expected_shape:
            raise ValueError(
                f"checkpoint weights shape must be {expected_shape}, got {weights.shape}"
            )
        if traces.shape != expected_shape:
            raise ValueError(
                f"checkpoint traces shape must be {expected_shape}, got {traces.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("checkpoint weights must be finite")
        if not np.all(np.isfinite(traces)):
            raise ValueError("checkpoint traces must be finite")

        update_count = state.get("update_count")
        if isinstance(update_count, bool) or not isinstance(
            update_count, (int, np.integer)
        ):
            raise TypeError("checkpoint update_count must be an integer")
        validated_update_count = int(update_count)
        if validated_update_count < 0:
            raise ValueError("checkpoint update_count must be nonnegative")

        last_terminated = state.get("last_terminated")
        last_truncated = state.get("last_truncated")
        for name, flag in (
            ("last_terminated", last_terminated),
            ("last_truncated", last_truncated),
        ):
            if flag is not None and not isinstance(flag, (bool, np.bool_)):
                raise TypeError(f"checkpoint {name} must be a boolean or null")

        rng_state = copy.deepcopy(state.get("rng_state"))
        if not isinstance(rng_state, Mapping):
            raise TypeError("checkpoint rng_state must be a mapping")
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = dict(rng_state)

        self.weights[...] = weights
        self.traces[...] = traces
        self.update_count = validated_update_count
        self.last_terminated = (
            None if last_terminated is None else bool(last_terminated)
        )
        self.last_truncated = None if last_truncated is None else bool(last_truncated)
        self._rng = restored_rng

    def save(self, path: str | Path) -> Path:
        """Atomically save a portable, pickle-free compressed NumPy checkpoint."""

        state = self.state_dict()
        weights = state.pop("weights")
        traces = state.pop("traces")
        serialized_state = json.dumps(state, sort_keys=True, allow_nan=False)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            metadata=np.asarray(serialized_state),
            weights=weights,
            traces=traces,
        )
        return atomic_save_bytes(path, buffer.getvalue())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        tile_coder: TileCoder | None = None,
    ) -> SarsaLambdaAgent:
        """Construct an agent from a checkpoint, optionally checking a coder."""

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            required = {"metadata", "weights", "traces"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"SARSA checkpoint is missing arrays: {sorted(missing)}"
                )
            metadata_array = archive["metadata"]
            if metadata_array.shape != ():
                raise ValueError("SARSA checkpoint metadata must be a scalar string")
            metadata = json.loads(str(metadata_array.item()))
            if not isinstance(metadata, dict):
                raise TypeError("SARSA checkpoint metadata must decode to a mapping")
            metadata["weights"] = np.asarray(archive["weights"], dtype=np.float64)
            metadata["traces"] = np.asarray(archive["traces"], dtype=np.float64)

        tile_metadata = metadata.get("tile_coder")
        if not isinstance(tile_metadata, Mapping):
            raise TypeError("SARSA checkpoint tile_coder metadata must be a mapping")
        if tile_coder is None:
            tile_coder = TileCoder(
                low=tile_metadata.get("low"),
                high=tile_metadata.get("high"),
                num_actions=tile_metadata.get("num_actions"),
                num_tilings=tile_metadata.get("num_tilings"),
                tiles_per_dim=tile_metadata.get("tiles_per_dim"),
            )

        hyperparameters = metadata.get("hyperparameters")
        if not isinstance(hyperparameters, Mapping):
            raise TypeError("SARSA checkpoint hyperparameters must be a mapping")
        agent = cls(tile_coder=tile_coder, **dict(hyperparameters))
        agent.load_state_dict(metadata)
        return agent
