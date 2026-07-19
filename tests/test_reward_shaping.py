"""Unit tests for bounded energy-deficit potential shaping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mountaincar_rl.environments.observation import (
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
)
from mountaincar_rl.environments.reward_shaping import (
    PotentialBasedRewardShaper,
    build_reward_shaper,
)


def test_beta_zero_preserves_environment_reward() -> None:
    shaper = PotentialBasedRewardShaper(gamma=0.99, beta=0.0)
    components = shaper.reward_components(
        -1.0,
        np.asarray([-0.5, 0.0]),
        np.asarray([-0.55, -0.03]),
        terminated=False,
    )

    assert components.original_reward == -1.0
    assert components.shaping_reward == 0.0
    assert components.training_reward == -1.0


def test_output_is_deterministic() -> None:
    shaper = PotentialBasedRewardShaper(gamma=0.97, beta=0.5, eta=0.4)
    state = np.asarray([-0.61, -0.02])
    next_state = np.asarray([-0.66, -0.035])

    observed = [
        shaper.reward_components(-1.0, state, next_state, terminated=False)
        for _ in range(5)
    ]

    assert observed == [observed[0]] * 5


def test_finite_bounded_values_throughout_observation_bounds() -> None:
    shaper = PotentialBasedRewardShaper(gamma=0.99, beta=1.0, eta=0.5)
    positions = np.linspace(OBSERVATION_LOW[0], OBSERVATION_HIGH[0], 31)
    velocities = np.linspace(OBSERVATION_LOW[1], OBSERVATION_HIGH[1], 29)
    potentials = [
        shaper.potential(np.asarray([position, velocity]))
        for position in positions
        for velocity in velocities
    ]

    assert np.all(np.isfinite(potentials))
    assert min(potentials) >= -1.0
    assert max(potentials) <= 0.0
    for current, following in zip(potentials[:-1], potentials[1:], strict=True):
        shaping = shaper.beta * (shaper.gamma * following - current)
        assert math.isfinite(shaping)
        assert abs(shaping) <= shaper.maximum_unclipped_magnitude


def test_hand_calculated_transition_matches_documented_formula() -> None:
    gamma = 0.9
    beta = 0.4
    eta = 0.5
    scale = 0.9
    shaper = PotentialBasedRewardShaper(
        gamma=gamma,
        beta=beta,
        eta=eta,
        energy_scale=scale,
    )
    state = np.asarray([-0.5, 0.0])
    next_state = np.asarray([-0.6, 0.035])

    def manual_potential(observation: np.ndarray) -> float:
        position, velocity = observation
        height = 0.45 * math.sin(3.0 * position) + 0.55
        proxy = height + eta * (velocity / 0.07) ** 2
        target = 0.45 * math.sin(3.0 * 0.5) + 0.55
        deficit = min(max((target - proxy) / scale, 0.0), 1.0)
        return -deficit

    expected_shaping = beta * (
        gamma * manual_potential(next_state) - manual_potential(state)
    )
    components = shaper.reward_components(
        -1.0,
        state,
        next_state,
        terminated=False,
    )

    assert components.shaping_reward == pytest.approx(expected_shaping)
    assert components.training_reward == pytest.approx(-1.0 + expected_shaping)


def test_true_terminal_zeroes_next_potential_but_truncation_does_not() -> None:
    shaper = PotentialBasedRewardShaper(gamma=0.97, beta=0.5)
    state = np.asarray([0.45, 0.02])
    next_state = np.asarray([0.51, 0.03])

    terminal = shaper.shaping_reward(state, next_state, terminated=True)
    time_limit = shaper.shaping_reward(state, next_state, terminated=False)

    assert terminal == pytest.approx(-shaper.beta * shaper.potential(state))
    assert time_limit == pytest.approx(
        shaper.beta
        * (shaper.gamma * shaper.potential(next_state) - shaper.potential(state))
    )


def test_reducing_energy_deficit_is_rewarded() -> None:
    shaper = PotentialBasedRewardShaper(gamma=0.99, beta=0.5, eta=0.5)
    valley = np.asarray([-math.pi / 6.0, 0.0])
    faster = np.asarray([-math.pi / 6.0, 0.07])

    assert shaper.energy_deficit(faster) < shaper.energy_deficit(valley)
    assert shaper.shaping_reward(valley, faster, terminated=False) > 0.0


def test_moving_left_is_not_automatically_penalized() -> None:
    shaper = PotentialBasedRewardShaper(gamma=0.99, beta=0.5, eta=0.5)
    state = np.asarray([-0.5, 0.0])
    moving_left_faster = np.asarray([-0.55, -0.04])

    assert moving_left_faster[0] < state[0]
    assert shaper.energy_deficit(moving_left_faster) < shaper.energy_deficit(state)
    assert shaper.shaping_reward(state, moving_left_faster, terminated=False) > 0.0


def test_disabled_builder_uses_same_pipeline_with_zero_beta() -> None:
    shaper = build_reward_shaper(
        {
            "method": "energy_deficit_potential",
            "enabled": False,
            "beta": 99.0,
            "eta": 0.5,
            "energy_scale": 0.9,
        },
        gamma=0.99,
    )

    assert shaper.beta == 0.0
    assert shaper.metadata()["method"] == "bounded_energy_deficit_potential_v1"


def test_builder_rejects_discount_mismatch() -> None:
    with pytest.raises(ValueError, match="must equal"):
        build_reward_shaper(
            {"enabled": True, "beta": 0.5, "gamma": 0.9},
            gamma=0.99,
        )
