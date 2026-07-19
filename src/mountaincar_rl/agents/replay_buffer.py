"""Fixed-capacity uniform experience replay for DQN.

The buffer intentionally stores ``terminated`` and ``truncated`` in separate
arrays.  Sampling returns one-dimensional action, reward, and boundary tensors
with shape ``[batch]`` and observation tensors with shape
``[batch, *observation_shape]``.  The DQN target masks only ``terminated``;
``truncated`` remains available for auditing and alternative diagnostics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from mountaincar_rl.utils.seeding import validate_seed


@dataclass(frozen=True)
class ReplayBatch:
    """A minibatch sampled without replacement from :class:`ReplayBuffer`.

    Attributes:
        observations: Float32 tensor with shape ``[B, *observation_shape]``.
        actions: Int64 tensor with shape ``[B]``.
        rewards: Float32 tensor with shape ``[B]``.
        next_observations: Float32 tensor matching ``observations``.
        terminated: Boolean tensor with shape ``[B]`` for true MDP terminals.
        truncated: Boolean tensor with shape ``[B]`` for artificial boundaries.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Store transitions in a circular array and sample them uniformly.

    Sampling uses a private NumPy generator, so its stream is independent of
    action exploration and global NumPy state.  Samples contain distinct
    transitions because minibatch indices are drawn without replacement.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        capacity: int,
        observation_shape: Sequence[int] = (2,),
        seed: int = 0,
    ) -> None:
        """Allocate a fixed-capacity buffer with validated dimensions."""

        self.capacity = _positive_integer(capacity, name="capacity")
        try:
            shape = tuple(observation_shape)
        except TypeError as error:
            raise TypeError("observation_shape must be a sequence of integers") from error
        if not shape:
            raise ValueError("observation_shape must contain at least one dimension")
        self.observation_shape = tuple(
            _positive_integer(dimension, name="observation_shape dimension")
            for dimension in shape
        )
        self.seed = validate_seed(seed)

        storage_shape = (self.capacity, *self.observation_shape)
        self._observations = np.zeros(storage_shape, dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_observations = np.zeros(storage_shape, dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=np.bool_)
        self._truncated = np.zeros(self.capacity, dtype=np.bool_)
        self._size = 0
        self._next_index = 0
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        """Return the number of currently retained transitions."""

        return self._size

    @property
    def next_index(self) -> int:
        """Return the physical slot that the next transition will overwrite."""

        return self._next_index

    def retained_observations(self) -> npt.NDArray[np.float32]:
        """Return an independent copy of currently retained current states.

        Physical circular-buffer order is intentionally irrelevant: this view
        exists for coverage diagnostics, not sequence reconstruction.
        """

        return self._observations[: self._size].copy()

    def add(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one validated transition, overwriting the oldest when full."""

        current = _validated_observation(
            observation, self.observation_shape, name="observation"
        )
        following = _validated_observation(
            next_observation, self.observation_shape, name="next_observation"
        )
        validated_action = _nonnegative_integer(action, name="action")
        validated_reward = _finite_real(reward, name="reward")
        validated_terminated = _boolean(terminated, name="terminated")
        validated_truncated = _boolean(truncated, name="truncated")

        index = self._next_index
        self._observations[index] = current
        self._actions[index] = validated_action
        self._rewards[index] = validated_reward
        self._next_observations[index] = following
        self._terminated[index] = validated_terminated
        self._truncated[index] = validated_truncated

        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: str | torch.device = "cpu"
    ) -> ReplayBatch:
        """Sample a distinct uniform minibatch and copy it to ``device``.

        Raises:
            ValueError: If fewer than ``batch_size`` transitions are retained.
        """

        validated_batch_size = _positive_integer(batch_size, name="batch_size")
        if validated_batch_size > self._size:
            raise ValueError(
                "Cannot sample "
                f"{validated_batch_size} transitions from replay buffer of size "
                f"{self._size}"
            )
        target_device = torch.device(device)
        indices = self._rng.choice(
            self._size, size=validated_batch_size, replace=False
        )

        return ReplayBatch(
            observations=torch.as_tensor(
                self._observations[indices], dtype=torch.float32, device=target_device
            ),
            actions=torch.as_tensor(
                self._actions[indices], dtype=torch.int64, device=target_device
            ),
            rewards=torch.as_tensor(
                self._rewards[indices], dtype=torch.float32, device=target_device
            ),
            next_observations=torch.as_tensor(
                self._next_observations[indices],
                dtype=torch.float32,
                device=target_device,
            ),
            terminated=torch.as_tensor(
                self._terminated[indices], dtype=torch.bool, device=target_device
            ),
            truncated=torch.as_tensor(
                self._truncated[indices], dtype=torch.bool, device=target_device
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return complete storage, cursor, and private-RNG state for resumption."""

        return {
            "version": self._STATE_VERSION,
            "capacity": self.capacity,
            "observation_shape": self.observation_shape,
            "seed": self.seed,
            "size": self._size,
            "next_index": self._next_index,
            "observations": self._observations.copy(),
            "actions": self._actions.copy(),
            "rewards": self._rewards.copy(),
            "next_observations": self._next_observations.copy(),
            "terminated": self._terminated.copy(),
            "truncated": self._truncated.copy(),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict` into this buffer.

        Capacity and observation shape are constructor-time invariants.  A
        mismatch is rejected instead of silently reallocating the buffer.
        """

        if not isinstance(state, Mapping):
            raise TypeError("Replay buffer state must be a mapping")
        if state.get("version") != self._STATE_VERSION:
            raise ValueError(
                f"Unsupported replay buffer state version: {state.get('version')!r}"
            )
        if state.get("capacity") != self.capacity:
            raise ValueError(
                "Replay buffer capacity mismatch: "
                f"checkpoint={state.get('capacity')!r}, instance={self.capacity}"
            )
        saved_shape = tuple(state.get("observation_shape", ()))
        if saved_shape != self.observation_shape:
            raise ValueError(
                "Replay buffer observation_shape mismatch: "
                f"checkpoint={saved_shape!r}, instance={self.observation_shape!r}"
            )
        restored_seed = validate_seed(state.get("seed"))

        size = _nonnegative_integer(state.get("size"), name="state size")
        next_index = _nonnegative_integer(
            state.get("next_index"), name="state next_index"
        )
        if size > self.capacity:
            raise ValueError("Replay buffer state size exceeds capacity")
        if next_index >= self.capacity:
            raise ValueError("Replay buffer state next_index is outside capacity")
        expected_next_index = size if size < self.capacity else next_index
        if size < self.capacity and next_index != expected_next_index:
            raise ValueError(
                "Partially filled replay state must have next_index equal to size"
            )

        storage_shape = (self.capacity, *self.observation_shape)
        observations = _state_array(
            state, "observations", np.float32, storage_shape, finite=True
        )
        actions = _state_array(
            state, "actions", np.int64, (self.capacity,), finite=False
        )
        rewards = _state_array(
            state, "rewards", np.float32, (self.capacity,), finite=True
        )
        next_observations = _state_array(
            state, "next_observations", np.float32, storage_shape, finite=True
        )
        terminated = _state_array(
            state, "terminated", np.bool_, (self.capacity,), finite=False
        )
        truncated = _state_array(
            state, "truncated", np.bool_, (self.capacity,), finite=False
        )
        if np.any(actions[:size] < 0):
            raise ValueError("Replay buffer state contains a negative retained action")

        rng_state = deepcopy(state.get("rng_state"))
        if not isinstance(rng_state, Mapping):
            raise TypeError("Replay buffer rng_state must be a mapping")
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as error:
            raise ValueError("Replay buffer rng_state is invalid") from error

        self._observations[...] = observations
        self._actions[...] = actions
        self._rewards[...] = rewards
        self._next_observations[...] = next_observations
        self._terminated[...] = terminated
        self._truncated[...] = truncated
        self._size = size
        self._next_index = next_index
        self.seed = restored_seed
        self._rng = restored_rng


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


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _validated_observation(
    value: Any, shape: tuple[int, ...], *, name: str
) -> npt.NDArray[np.float32]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _state_array(
    state: Mapping[str, Any],
    key: str,
    dtype: npt.DTypeLike,
    shape: tuple[int, ...],
    *,
    finite: bool,
) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Replay buffer state is missing {key!r}")
    raw = np.asarray(state[key])
    if raw.dtype != np.dtype(dtype):
        raise TypeError(
            f"Replay buffer state {key!r} must have dtype {np.dtype(dtype)}, "
            f"got {raw.dtype}"
        )
    if raw.shape != shape:
        raise ValueError(
            f"Replay buffer state {key!r} must have shape {shape}, got {raw.shape}"
        )
    if finite and not np.all(np.isfinite(raw)):
        raise ValueError(f"Replay buffer state {key!r} must contain finite values")
    return raw


__all__ = ["ReplayBatch", "ReplayBuffer"]
