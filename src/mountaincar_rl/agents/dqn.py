"""Shared neural DQN implementation for the controlled replay ablation.

Replay and online DQN use the same observation normalizer, Q-network, Adam
optimizer, Huber objective, target equation, exploration schedule, and target
update rule.  The online condition consumes a batch of exactly one current
transition and retains no previous transitions.  Consequently, removing
replay necessarily changes both minibatch size and temporal decorrelation; it
is not a perfectly isolated memory-only intervention.

Target-network synchronization is measured in **optimizer steps** in both
conditions.  At every ``target_update_frequency`` optimizer steps, ``tau=1``
performs a hard copy and ``0 < tau < 1`` performs a Polyak update.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.agents.replay_buffer import ReplayBatch, ReplayBuffer
from mountaincar_rl.utils.checkpoints import (
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from mountaincar_rl.utils.seeding import validate_seed


class QNetwork(nn.Module):
    """A configurable ReLU MLP mapping normalized states to action values."""

    def __init__(
        self,
        input_dim: int,
        action_count: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        """Construct the network, permitting an empty hidden stack for tests."""

        super().__init__()
        validated_input_dim = _positive_integer(input_dim, name="input_dim")
        validated_action_count = _positive_integer(
            action_count, name="action_count"
        )
        validated_hidden_sizes = _hidden_sizes(hidden_sizes)

        layers: list[nn.Module] = []
        previous_size = validated_input_dim
        for hidden_size in validated_hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, validated_action_count))
        self.model = nn.Sequential(*layers)
        self.input_dim = validated_input_dim
        self.action_count = validated_action_count
        self.hidden_sizes = validated_hidden_sizes

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q value per action for each leading observation row."""

        return self.model(observations)


def compute_td_targets(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute one-step DQN targets using only true termination as the mask.

    Mathematically, for each transition this returns

    ``r + gamma * (1 - terminated) * max_a Q_target(s_next, a)``.

    Artificial time-limit truncation is deliberately absent: a transition for
    which ``truncated=True`` and ``terminated=False`` must bootstrap from its
    retained post-step observation.
    """

    if not isinstance(rewards, torch.Tensor):
        raise TypeError("rewards must be a torch.Tensor")
    if not isinstance(next_q_values, torch.Tensor):
        raise TypeError("next_q_values must be a torch.Tensor")
    if not isinstance(terminated, torch.Tensor):
        raise TypeError("terminated must be a torch.Tensor")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must have shape [B], got {tuple(rewards.shape)}")
    if next_q_values.ndim != 2:
        raise ValueError(
            "next_q_values must have shape [B, action_count], got "
            f"{tuple(next_q_values.shape)}"
        )
    if next_q_values.shape[0] != rewards.shape[0]:
        raise ValueError("rewards and next_q_values batch dimensions must match")
    if next_q_values.shape[1] <= 0:
        raise ValueError("next_q_values must contain at least one action")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must have the same shape as rewards")
    if terminated.dtype != torch.bool:
        raise TypeError("terminated must have boolean dtype")
    if rewards.device != next_q_values.device or rewards.device != terminated.device:
        raise ValueError("TD-target tensors must be on the same device")
    validated_gamma = _bounded_real(gamma, name="gamma", lower=0.0, upper=1.0)

    max_next_q_values = next_q_values.max(dim=1).values
    bootstrap_mask = (~terminated).to(dtype=max_next_q_values.dtype)
    return rewards + validated_gamma * bootstrap_mask * max_next_q_values


class DQNAgent(Agent):
    """Deep Q-learning agent shared by replay and one-transition conditions.

    ``observation_normalizer`` must expose ``normalize(observation)`` and
    ``metadata()``.  Its JSON-safe metadata is embedded in checkpoints and
    reconstructed with ``ObservationNormalizer.from_metadata`` by
    :meth:`load`.
    """

    _CHECKPOINT_VERSION = 1

    def __init__(
        self,
        action_count: int,
        observation_normalizer: Any,
        hidden_sizes: Sequence[int] = (64, 64),
        *,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        gradient_clip_norm: float = 10.0,
        target_update_frequency: int = 1_000,
        tau: float = 1.0,
        use_replay: bool = True,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        warmup_steps: int = 1_000,
        update_frequency: int = 1,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize networks, optimizer, exploration RNG, and optional replay."""

        self.action_count = _positive_integer(action_count, name="action_count")
        self.hidden_sizes = _hidden_sizes(hidden_sizes)
        self.gamma = _bounded_real(gamma, name="gamma", lower=0.0, upper=1.0)
        self.learning_rate = _positive_real(learning_rate, name="learning_rate")
        self.epsilon_start = _bounded_real(
            epsilon_start, name="epsilon_start", lower=0.0, upper=1.0
        )
        self.epsilon_end = _bounded_real(
            epsilon_end, name="epsilon_end", lower=0.0, upper=1.0
        )
        if self.epsilon_start < self.epsilon_end:
            raise ValueError("epsilon_start must be greater than or equal to epsilon_end")
        self.epsilon_decay_steps = _positive_integer(
            epsilon_decay_steps, name="epsilon_decay_steps"
        )
        self.gradient_clip_norm = _positive_real(
            gradient_clip_norm, name="gradient_clip_norm"
        )
        self.target_update_frequency = _positive_integer(
            target_update_frequency, name="target_update_frequency"
        )
        self.tau = _bounded_real(
            tau, name="tau", lower=0.0, upper=1.0, lower_inclusive=False
        )
        if not isinstance(use_replay, bool):
            raise TypeError("use_replay must be a boolean")
        self.use_replay = use_replay
        self.replay_capacity = _positive_integer(
            replay_capacity, name="replay_capacity"
        )
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        if self.use_replay and self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        self.warmup_steps = _nonnegative_integer(
            warmup_steps, name="warmup_steps"
        )
        self.update_frequency = _positive_integer(
            update_frequency, name="update_frequency"
        )
        self.seed = validate_seed(seed)
        self.device = _validated_device(device)

        self.observation_normalizer = observation_normalizer
        self.preprocessing_metadata = _normalizer_metadata(observation_normalizer)
        low = np.asarray(self.preprocessing_metadata.get("low"), dtype=np.float32)
        high = np.asarray(self.preprocessing_metadata.get("high"), dtype=np.float32)
        if low.ndim != 1 or low.shape != high.shape or low.size == 0:
            raise ValueError(
                "Observation normalizer metadata must contain equally shaped "
                "one-dimensional low/high bounds"
            )
        self.observation_shape = (int(low.size),)
        self.input_dim = int(low.size)
        # Exercise the preprocessing contract at construction so bad duck types
        # fail before training rather than midway through a run.
        midpoint = (low.astype(np.float64) + high.astype(np.float64)) / 2.0
        self._normalize(midpoint)

        # Network initialization is deterministic for the agent seed without
        # advancing the process-wide PyTorch RNG used by other experiment code.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed % (2**63))
            online_network = QNetwork(
                self.input_dim, self.action_count, self.hidden_sizes
            )
        self.online_network = online_network.to(self.device)
        # Deep-copying avoids an otherwise unused second random initialization
        # from advancing the process-wide PyTorch RNG before the hard sync.
        self.target_network = deepcopy(self.online_network).to(self.device)
        self.hard_sync_target()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.learning_rate
        )

        self.environment_steps = 0
        self.optimizer_steps = 0
        self.last_loss: float | None = None
        self.last_mean_q_value: float | None = None
        self.last_gradient_norm: float | None = None
        self._rng = np.random.default_rng(self.seed)
        self.replay_buffer: ReplayBuffer | None = None
        if self.use_replay:
            self.replay_buffer = ReplayBuffer(
                capacity=self.replay_capacity,
                observation_shape=self.observation_shape,
                seed=self.seed,
            )

    @property
    def epsilon(self) -> float:
        """Return linearly annealed exploration at the current environment step."""

        fraction = min(self.environment_steps / self.epsilon_decay_steps, 1.0)
        return float(
            self.epsilon_start
            + fraction * (self.epsilon_end - self.epsilon_start)
        )

    def replay_state_coverage(
        self, bins: Sequence[int] = (20, 20)
    ) -> dict[str, int | float]:
        """Summarize occupied bins among states currently retained by replay.

        Replay stores the shared normalized observation in ``[-1, 1]``. Each
        state is assigned to one cell of the configured regular grid; coverage
        is the number and fraction of unique joint cells occupied. This is a
        descriptive state-diversity diagnostic, not a guarantee of trajectory
        independence or useful momentum.
        """

        try:
            bin_counts = tuple(bins)
        except TypeError as error:
            raise TypeError("bins must be a sequence of integers") from error
        if len(bin_counts) != self.input_dim:
            raise ValueError(
                f"bins must contain {self.input_dim} dimensions, got {len(bin_counts)}"
            )
        validated_bins = tuple(
            _positive_integer(value, name="coverage bin count")
            for value in bin_counts
        )
        total_cells = int(np.prod(validated_bins, dtype=np.int64))
        if self.replay_buffer is None:
            return {
                "buffer_size": 0,
                "occupied_cells": 0,
                "total_cells": total_cells,
                "coverage_fraction": 0.0,
            }
        observations = self.replay_buffer.retained_observations()
        if observations.shape[0] == 0:
            occupied_cells = 0
        else:
            fractions = np.clip((observations.astype(np.float64) + 1.0) / 2.0, 0, 1)
            indices = np.empty_like(fractions, dtype=np.int64)
            for dimension, count in enumerate(validated_bins):
                indices[:, dimension] = np.minimum(
                    np.floor(fractions[:, dimension] * count).astype(np.int64),
                    count - 1,
                )
            occupied_cells = int(np.unique(indices, axis=0).shape[0])
        return {
            "buffer_size": len(self.replay_buffer),
            "occupied_cells": occupied_cells,
            "total_cells": total_cells,
            "coverage_fraction": float(occupied_cells / total_cells),
        }

    def select_action(
        self, observation: np.ndarray, *, explore: bool = True
    ) -> int:
        """Select epsilon-greedily, or greedily with smallest-index tie breaking.

        ``explore=False`` performs no RNG draw, changes no counter, and does not
        toggle network training mode, making deterministic evaluation strictly
        non-mutating.
        """

        normalized = self._normalize(observation)
        if not isinstance(explore, bool):
            raise TypeError("explore must be a boolean")
        if explore and float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(self.action_count))
        observation_tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action_values = self.online_network(observation_tensor)
        # torch.argmax returns the first maximum, hence the smallest action.
        return int(torch.argmax(action_values, dim=1).item())

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return a finite CPU copy of online action values without mutation."""

        normalized = self._normalize(observation)
        observation_tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            values = self.online_network(observation_tensor).squeeze(0)
        result = values.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("DQN action values became non-finite")
        return result

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None = None,
    ) -> float | None:
        """Consume one transition and, when scheduled, perform one update.

        Replay mode inserts normalized states and waits for both the configured
        environment-step warm-up and a complete minibatch.  Online mode stores
        nothing and updates from a one-transition batch.  If its
        ``update_frequency`` exceeds one, intervening current transitions are
        deliberately discarded rather than retained in hidden replay memory.

        ``environment_steps`` increments exactly once for each valid consumed
        transition.  Data collection must still stop after either boundary,
        while target masking below depends only on ``terminated``.
        """

        del info
        normalized = self._normalize(observation)
        normalized_next = self._normalize(next_observation)
        validated_action = _bounded_action(action, self.action_count)
        validated_reward = _finite_real(reward, name="reward")
        validated_terminated = _boolean(terminated, name="terminated")
        validated_truncated = _boolean(truncated, name="truncated")

        if self.replay_buffer is not None:
            self.replay_buffer.add(
                normalized,
                validated_action,
                validated_reward,
                normalized_next,
                terminated=validated_terminated,
                truncated=validated_truncated,
            )

        self.environment_steps += 1
        if self.environment_steps % self.update_frequency != 0:
            return None

        if self.replay_buffer is not None:
            if self.environment_steps < self.warmup_steps:
                return None
            if len(self.replay_buffer) < self.batch_size:
                return None
            batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        else:
            batch = ReplayBatch(
                observations=torch.as_tensor(
                    normalized, dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                actions=torch.tensor(
                    [validated_action], dtype=torch.int64, device=self.device
                ),
                rewards=torch.tensor(
                    [validated_reward], dtype=torch.float32, device=self.device
                ),
                next_observations=torch.as_tensor(
                    normalized_next, dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                terminated=torch.tensor(
                    [validated_terminated], dtype=torch.bool, device=self.device
                ),
                truncated=torch.tensor(
                    [validated_truncated], dtype=torch.bool, device=self.device
                ),
            )
        return self._optimize_batch(batch)

    def _optimize_batch(self, batch: ReplayBatch) -> float:
        """Apply one clipped Adam step to the mean Huber TD loss."""

        predicted_q_values = self.online_network(batch.observations).gather(
            1, batch.actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(batch.next_observations)
            targets = compute_td_targets(
                batch.rewards, next_q_values, batch.terminated, self.gamma
            )
        loss = functional.smooth_l1_loss(predicted_q_values, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("DQN loss became non-finite")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.gradient_clip_norm
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("DQN gradient norm became non-finite")
        self.optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in self.online_network.parameters()
        ):
            raise FloatingPointError("DQN optimizer produced non-finite parameters")

        self.optimizer_steps += 1
        self.last_loss = float(loss.detach().cpu().item())
        self.last_mean_q_value = float(
            predicted_q_values.detach().mean().cpu().item()
        )
        self.last_gradient_norm = float(
            torch.as_tensor(gradient_norm).detach().cpu().item()
        )
        if self.optimizer_steps % self.target_update_frequency == 0:
            if self.tau == 1.0:
                self.hard_sync_target()
            else:
                self.soft_sync_target()
        return self.last_loss

    def hard_sync_target(self) -> None:
        """Copy all online parameters and buffers to the target network."""

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.requires_grad_(False)
        self.target_network.eval()

    def soft_sync_target(self, tau: float | None = None) -> None:
        """Polyak-update target parameters with ``tau*online + (1-tau)*target``."""

        amount = self.tau if tau is None else _bounded_real(
            tau,
            name="tau",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_network.parameters(),
                self.online_network.parameters(),
                strict=True,
            ):
                target_parameter.mul_(1.0 - amount).add_(
                    online_parameter, alpha=amount
                )
            # QNetwork currently has no buffers, but keeping this explicit makes
            # synchronization correct if a non-floating diagnostic buffer is added.
            for target_buffer, online_buffer in zip(
                self.target_network.buffers(),
                self.online_network.buffers(),
                strict=True,
            ):
                if target_buffer.is_floating_point():
                    target_buffer.mul_(1.0 - amount).add_(
                        online_buffer, alpha=amount
                    )
                else:
                    target_buffer.copy_(online_buffer)
        self.target_network.requires_grad_(False)
        self.target_network.eval()

    def state_dict(self) -> dict[str, Any]:
        """Return a complete resumable checkpoint, including both private RNGs."""

        return {
            "checkpoint_version": self._CHECKPOINT_VERSION,
            "agent_class": type(self).__name__,
            "config": self._configuration(),
            "preprocessing": deepcopy(self.preprocessing_metadata),
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "optimizer_steps": self.optimizer_steps,
            "last_loss": self.last_loss,
            "last_mean_q_value": self.last_mean_q_value,
            "last_gradient_norm": self.last_gradient_norm,
            "rng_state": deepcopy(self._rng.bit_generator.state),
            "replay_buffer": (
                None if self.replay_buffer is None else self.replay_buffer.state_dict()
            ),
            "online_network_training": self.online_network.training,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint into a configuration-compatible agent."""

        if not isinstance(state, Mapping):
            raise TypeError("DQN checkpoint state must be a mapping")
        if state.get("checkpoint_version") != self._CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported DQN checkpoint version: "
                f"{state.get('checkpoint_version')!r}"
            )
        saved_config = state.get("config")
        if not isinstance(saved_config, Mapping):
            raise TypeError("DQN checkpoint config must be a mapping")
        expected_config = self._configuration()
        if _without_device(saved_config) != _without_device(expected_config):
            raise ValueError(
                "DQN checkpoint configuration does not match the agent instance"
            )
        saved_preprocessing = state.get("preprocessing")
        if saved_preprocessing != self.preprocessing_metadata:
            raise ValueError(
                "DQN checkpoint observation preprocessing does not match the agent"
            )

        environment_steps = _nonnegative_integer(
            state.get("environment_steps"), name="environment_steps"
        )
        optimizer_steps = _nonnegative_integer(
            state.get("optimizer_steps"), name="optimizer_steps"
        )
        rng_state = deepcopy(state.get("rng_state"))
        if not isinstance(rng_state, Mapping):
            raise TypeError("DQN checkpoint rng_state must be a mapping")
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as error:
            raise ValueError("DQN checkpoint rng_state is invalid") from error

        replay_state = state.get("replay_buffer")
        if self.replay_buffer is None and replay_state is not None:
            raise ValueError("Online DQN checkpoint unexpectedly contains replay state")
        if self.replay_buffer is not None and replay_state is None:
            raise ValueError("Replay DQN checkpoint is missing replay state")

        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer.load_state_dict(state["optimizer"])
        _move_optimizer_state(self.optimizer, self.device)
        if self.replay_buffer is not None:
            self.replay_buffer.load_state_dict(replay_state)
        self.environment_steps = environment_steps
        self.optimizer_steps = optimizer_steps
        self.last_loss = _optional_finite_real(state.get("last_loss"), name="last_loss")
        self.last_mean_q_value = _optional_finite_real(
            state.get("last_mean_q_value"), name="last_mean_q_value"
        )
        self.last_gradient_norm = _optional_finite_real(
            state.get("last_gradient_norm"), name="last_gradient_norm"
        )
        self._rng = restored_rng
        if _boolean(
            state.get("online_network_training"), name="online_network_training"
        ):
            self.online_network.train()
        else:
            self.online_network.eval()

    def save(self, path: str | Path) -> Path:
        """Atomically write a complete resumable checkpoint."""

        return save_torch_checkpoint(path, self.state_dict())

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> DQNAgent:
        """Reconstruct an agent and its normalizer from a saved checkpoint."""

        target_device = _validated_device(device)
        payload = load_torch_checkpoint(
            path, map_location=target_device, weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise TypeError("DQN checkpoint payload must be a mapping")
        config = payload.get("config")
        preprocessing = payload.get("preprocessing")
        if not isinstance(config, Mapping):
            raise TypeError("DQN checkpoint config must be a mapping")
        if not isinstance(preprocessing, Mapping):
            raise TypeError("DQN checkpoint preprocessing must be a mapping")

        from mountaincar_rl.environments.observation import ObservationNormalizer

        normalizer = ObservationNormalizer.from_metadata(preprocessing)
        constructor_config = dict(config)
        constructor_config.pop("device", None)
        constructor_config["hidden_sizes"] = tuple(
            constructor_config["hidden_sizes"]
        )
        agent = cls(
            observation_normalizer=normalizer,
            device=target_device,
            **constructor_config,
        )
        agent.load_state_dict(payload)
        return agent

    def diagnostic_state(self) -> dict[str, Any]:
        """Return a compact non-mutating fingerprint for evaluation invariants."""

        replay_state = (
            None if self.replay_buffer is None else self.replay_buffer.state_dict()
        )
        return {
            "environment_steps": self.environment_steps,
            "optimizer_steps": self.optimizer_steps,
            "epsilon": self.epsilon,
            "exploration_rng": repr(self._rng.bit_generator.state),
            "replay_size": 0 if self.replay_buffer is None else len(self.replay_buffer),
            "replay_next_index": (
                None if self.replay_buffer is None else self.replay_buffer.next_index
            ),
            "replay_rng": (
                None if replay_state is None else repr(replay_state["rng_state"])
            ),
            "online_parameters": _module_digest(self.online_network),
            "target_parameters": _module_digest(self.target_network),
            "online_network_training": self.online_network.training,
            "last_loss": self.last_loss,
            "last_mean_q_value": self.last_mean_q_value,
            "last_gradient_norm": self.last_gradient_norm,
        }

    def _normalize(self, observation: Any) -> np.ndarray:
        normalize = getattr(self.observation_normalizer, "normalize", None)
        if not callable(normalize):
            raise TypeError("observation_normalizer must provide normalize()")
        try:
            normalized = np.asarray(normalize(observation), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Failed to normalize DQN observation") from error
        if normalized.shape != self.observation_shape:
            raise ValueError(
                "Normalized observation must have shape "
                f"{self.observation_shape}, got {normalized.shape}"
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Normalized observation must contain only finite values")
        return normalized

    def _configuration(self) -> dict[str, Any]:
        return {
            "action_count": self.action_count,
            "hidden_sizes": list(self.hidden_sizes),
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "target_update_frequency": self.target_update_frequency,
            "tau": self.tau,
            "use_replay": self.use_replay,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "update_frequency": self.update_frequency,
            "seed": self.seed,
            "device": str(self.device),
        }


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative")
    return integer


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _optional_finite_real(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_real(value, name=name)


def _positive_real(value: Any, *, name: str) -> float:
    number = _finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _bounded_real(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
) -> float:
    number = _finite_real(value, name=name)
    lower_ok = number >= lower if lower_inclusive else number > lower
    if not lower_ok or number > upper:
        lower_bracket = "[" if lower_inclusive else "("
        raise ValueError(
            f"{name} must lie in {lower_bracket}{lower}, {upper}], got {number}"
        )
    return number


def _hidden_sizes(hidden_sizes: Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_sizes, (str, bytes)):
        raise TypeError("hidden_sizes must be a sequence of integers")
    try:
        values = tuple(hidden_sizes)
    except TypeError as error:
        raise TypeError("hidden_sizes must be a sequence of integers") from error
    return tuple(
        _positive_integer(size, name="hidden size") for size in values
    )


def _bounded_action(action: Any, action_count: int) -> int:
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("action must be an integer")
    integer = int(action)
    if integer < 0 or integer >= action_count:
        raise ValueError(
            f"action must be in [0, {action_count - 1}], got {integer}"
        )
    return integer


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _normalizer_metadata(normalizer: Any) -> dict[str, Any]:
    metadata_method = getattr(normalizer, "metadata", None)
    if not callable(metadata_method):
        raise TypeError("observation_normalizer must provide metadata()")
    metadata = metadata_method()
    if not isinstance(metadata, Mapping):
        raise TypeError("observation_normalizer.metadata() must return a mapping")
    copied = deepcopy(dict(metadata))
    if "low" not in copied or "high" not in copied:
        raise ValueError("Observation normalizer metadata must include low and high")
    return copied


def _validated_device(device: str | torch.device) -> torch.device:
    try:
        validated = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid PyTorch device: {device!r}") from error
    if validated.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available")
    return validated


def _without_device(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(config)
    copied.pop("device", None)
    return copied


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _module_digest(module: nn.Module) -> str:
    digest = sha256()
    for name, tensor in module.state_dict().items():
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


__all__ = ["DQNAgent", "QNetwork", "compute_td_targets"]
