"""Deterministic-seed random-policy baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.utils.seeding import validate_seed


class RandomAgent(Agent):
    """Sample uniformly from a finite discrete action space.

    A private :class:`numpy.random.Generator` is used instead of
    ``action_space.sample()`` so the policy's random stream is transparent and
    independent from environment randomness.
    """

    def __init__(self, action_space: Any, seed: int) -> None:
        """Initialize a random policy for a Gymnasium-like discrete space."""

        action_count = getattr(action_space, "n", None)
        if action_count is None:
            raise TypeError("RandomAgent requires a discrete action space with 'n'")
        if isinstance(action_count, bool) or not isinstance(
            action_count, (int, np.integer)
        ):
            raise TypeError("action_space.n must be an integer")
        if int(action_count) <= 0:
            raise ValueError("action_space.n must be positive")

        action_start = getattr(action_space, "start", 0)
        if isinstance(action_start, bool) or not isinstance(
            action_start, (int, np.integer)
        ):
            raise TypeError("action_space.start must be an integer when provided")

        self._action_count = int(action_count)
        self._action_start = int(action_start)
        self._rng = np.random.default_rng(validate_seed(seed))

    def select_action(
        self, observation: np.ndarray, *, explore: bool = True
    ) -> int:
        """Return a uniformly sampled action.

        ``observation`` and ``explore`` do not affect a no-learning baseline,
        but remain part of the common interface.
        """

        del observation, explore
        upper = self._action_start + self._action_count
        return int(self._rng.integers(self._action_start, upper))
