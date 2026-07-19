"""Common interface for agents used by experiment runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


class Agent(ABC):
    """Minimal agent interface with optional transition and episode hooks.

    Learning agents can override :meth:`observe` and :meth:`end_episode`.
    Stateless baselines only need to implement :meth:`select_action`.
    """

    @abstractmethod
    def select_action(
        self, observation: np.ndarray, *, explore: bool = True
    ) -> int:
        """Choose a discrete action for ``observation``.

        Args:
            observation: Current environment observation.
            explore: Whether stochastic exploratory behavior is permitted.

        Returns:
            Integer action accepted by the environment.
        """

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
        """Consume one transition; the default implementation does nothing.

        ``terminated`` and ``truncated`` are intentionally separate. Learning
        agents must decide explicitly whether a time-limit truncation should
        bootstrap rather than treating every episode boundary as terminal.
        """

        del (
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
            info,
        )
        return None

    def end_episode(self) -> None:
        """Finish per-episode bookkeeping; stateless agents need no action."""
