"""Bounded potential-based shaping from a MountainCar energy deficit proxy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from mountaincar_rl.environments.observation import (
    GOAL_POSITION,
    VELOCITY_MAX,
    validate_observation,
)


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class RewardComponents:
    """One transition's original, shaping, and learner-facing rewards.

    All three fields use the environment's reward units. ``training_reward``
    is exactly ``original_reward + shaping_reward``.
    """

    original_reward: float
    shaping_reward: float
    training_reward: float

    def __post_init__(self) -> None:
        values = (
            self.original_reward,
            self.shaping_reward,
            self.training_reward,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("reward components must be finite")
        if not np.isclose(
            self.training_reward,
            self.original_reward + self.shaping_reward,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "training_reward must equal original_reward + shaping_reward"
            )


@dataclass(frozen=True, slots=True)
class PotentialBasedRewardShaper:
    """Apply bounded energy-deficit potential shaping.

    This class deliberately calls its state quantity an *energy proxy*, not
    exact mechanical energy. MountainCar exposes dimensionless position and
    velocity coordinates, and its rendered hill-height relationship is

    ``h(x) = 0.45 * sin(3x) + 0.55``.

    The direction-neutral proxy and normalized energy deficit are

    ``E_proxy = h(x) + eta * clip(v / v_max, -1, 1)^2``

    ``D = clip((E_target - E_proxy) / E_scale, 0, 1)``.

    ``h``, ``E_proxy``, ``E_target``, and ``E_scale`` are dimensionless proxy
    units. ``eta`` is a dimensionless relative weight. By default
    ``E_target = h(goal_position)``. We define ``Phi = -D`` so reducing the
    deficit raises the potential. Consequently ``Phi`` is bounded in
    ``[-1, 0]`` and the shaping reward

    ``F = beta * (gamma * Phi_effective(s_next) - Phi(s))``

    is bounded before an optional safety clip by
    ``abs(F) <= beta * (1 + gamma)``. ``beta`` has environment-reward units
    because the potential is dimensionless. The learner receives
    ``r_train = r_environment + F``.

    For a true environment termination ``Phi_effective(s_next) = 0``. For a
    pure time-limit truncation callers pass ``terminated=False`` and the actual
    next-state potential is retained, matching the project's time-limit
    bootstrapping convention. ``beta=0`` disables shaping without changing the
    reward pipeline.
    """

    gamma: float
    beta: float = 1.0
    eta: float = 0.5
    energy_scale: float = 0.9
    goal_position: float = GOAL_POSITION
    target_energy: float | None = None
    velocity_max: float = VELOCITY_MAX
    max_abs_shaping: float | None = None

    def __post_init__(self) -> None:
        gamma = _finite_real("gamma", self.gamma)
        beta = _finite_real("beta", self.beta)
        eta = _finite_real("eta", self.eta)
        energy_scale = _finite_real("energy_scale", self.energy_scale)
        goal_position = _finite_real("goal_position", self.goal_position)
        velocity_max = _finite_real("velocity_max", self.velocity_max)
        target_energy = (
            self.hill_height(goal_position)
            if self.target_energy is None
            else _finite_real("target_energy", self.target_energy)
        )
        max_abs_shaping = (
            None
            if self.max_abs_shaping is None
            else _finite_real("max_abs_shaping", self.max_abs_shaping)
        )
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if beta < 0.0:
            raise ValueError("beta must be nonnegative")
        if eta < 0.0:
            raise ValueError("eta must be nonnegative")
        if energy_scale <= 0.0:
            raise ValueError("energy_scale must be positive")
        if velocity_max <= 0.0:
            raise ValueError("velocity_max must be positive")
        if max_abs_shaping is not None and max_abs_shaping <= 0.0:
            raise ValueError("max_abs_shaping must be positive when provided")

        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "energy_scale", energy_scale)
        object.__setattr__(self, "goal_position", goal_position)
        object.__setattr__(self, "target_energy", target_energy)
        object.__setattr__(self, "velocity_max", velocity_max)
        object.__setattr__(self, "max_abs_shaping", max_abs_shaping)

    @staticmethod
    def hill_height(position: Real) -> float:
        """Return ``0.45*sin(3x)+0.55`` in dimensionless height-proxy units."""

        x = _finite_real("position", position)
        return float(0.45 * np.sin(3.0 * x) + 0.55)

    def energy_proxy(self, state: Any) -> float:
        """Return bounded hill height plus weighted normalized squared speed."""

        position, velocity = validate_observation(state)
        normalized_velocity = float(
            np.clip(velocity / self.velocity_max, -1.0, 1.0)
        )
        proxy = self.hill_height(position) + self.eta * normalized_velocity**2
        if not np.isfinite(proxy):  # defensive guard after transcendental math
            raise FloatingPointError("energy proxy became non-finite")
        return float(proxy)

    def energy_deficit(self, state: Any) -> float:
        """Return the nonnegative target shortfall in energy-proxy units."""

        deficit = max(0.0, float(self.target_energy) - self.energy_proxy(state))
        if not np.isfinite(deficit):
            raise FloatingPointError("energy deficit became non-finite")
        return float(deficit)

    def potential(self, state: Any) -> float:
        """Return negative normalized deficit, clipped to the interval ``[-1, 0]``."""

        normalized_deficit = np.clip(
            self.energy_deficit(state) / self.energy_scale,
            0.0,
            1.0,
        )
        return float(-normalized_deficit)

    @property
    def maximum_unclipped_magnitude(self) -> float:
        """Return the analytic upper bound on ``abs(F)`` in reward units."""

        return float(self.beta * (1.0 + self.gamma))

    def shaping_reward(
        self,
        state: Any,
        next_state: Any,
        *,
        terminated: bool,
    ) -> float:
        """Return only ``F(s,s_next)`` with true-terminal potential handling."""

        if not isinstance(terminated, (bool, np.bool_)):
            raise TypeError("terminated must be a boolean")
        current_potential = self.potential(state)
        next_potential = 0.0 if bool(terminated) else self.potential(next_state)
        shaping = self.beta * (
            self.gamma * next_potential - current_potential
        )
        if self.max_abs_shaping is not None:
            shaping = float(
                np.clip(shaping, -self.max_abs_shaping, self.max_abs_shaping)
            )
        if not np.isfinite(shaping):
            raise FloatingPointError("shaping reward became non-finite")
        return float(shaping)

    def reward_components(
        self,
        reward: Real,
        state: Any,
        next_state: Any,
        *,
        terminated: bool,
    ) -> RewardComponents:
        """Return separately auditable environment, shaping, and total rewards."""

        original = _finite_real("reward", reward)
        shaping = self.shaping_reward(
            state,
            next_state,
            terminated=terminated,
        )
        return RewardComponents(
            original_reward=original,
            shaping_reward=shaping,
            training_reward=float(original + shaping),
        )

    def shaped_reward(
        self,
        reward: Real,
        state: Any,
        next_state: Any,
        terminated: bool,
    ) -> float:
        """Return learner-facing reward; retained as a compatibility convenience."""

        return self.reward_components(
            reward,
            state,
            next_state,
            terminated=terminated,
        ).training_reward

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe description of the complete reward intervention."""

        return {
            "method": "bounded_energy_deficit_potential_v1",
            "gamma": self.gamma,
            "beta": self.beta,
            "eta": self.eta,
            "energy_scale": self.energy_scale,
            "goal_position": self.goal_position,
            "target_energy": self.target_energy,
            "velocity_max": self.velocity_max,
            "max_abs_shaping": self.max_abs_shaping,
            "potential_bounds": [-1.0, 0.0],
            "maximum_unclipped_magnitude": self.maximum_unclipped_magnitude,
            "true_terminal_next_potential": 0.0,
            "truncation_uses_observed_next_potential": True,
        }


def build_reward_shaper(
    config: Mapping[str, Any] | None,
    *,
    gamma: float,
    goal_position: float = GOAL_POSITION,
) -> PotentialBasedRewardShaper:
    """Construct the configured shaper, including the identical disabled path."""

    reward_config: Mapping[str, Any] = {} if config is None else config
    if not isinstance(reward_config, Mapping):
        raise TypeError("reward_shaping configuration must be a mapping")
    method = str(reward_config.get("method", "energy_deficit_potential"))
    if method != "energy_deficit_potential":
        raise ValueError(
            "reward_shaping.method must be 'energy_deficit_potential'"
        )
    enabled = reward_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("reward_shaping.enabled must be a boolean")
    configured_gamma = float(reward_config.get("gamma", gamma))
    if not np.isclose(configured_gamma, float(gamma), rtol=0.0, atol=1e-12):
        raise ValueError(
            "reward_shaping.gamma must equal the learning agent discount"
        )
    configured_beta = float(reward_config.get("beta", 0.0))
    beta = configured_beta if enabled else 0.0
    target_energy_value = reward_config.get("target_energy")
    max_abs_value = reward_config.get("max_abs_shaping")
    return PotentialBasedRewardShaper(
        gamma=float(gamma),
        beta=beta,
        eta=float(reward_config.get("eta", 0.5)),
        energy_scale=float(reward_config.get("energy_scale", 0.9)),
        goal_position=float(goal_position),
        target_energy=(
            None if target_energy_value is None else float(target_energy_value)
        ),
        velocity_max=float(reward_config.get("velocity_max", VELOCITY_MAX)),
        max_abs_shaping=(
            None if max_abs_value is None else float(max_abs_value)
        ),
    )


__all__ = [
    "PotentialBasedRewardShaper",
    "RewardComponents",
    "build_reward_shaper",
]
