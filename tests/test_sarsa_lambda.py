"""Unit tests for on-policy tile-coded SARSA(lambda)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from mountaincar_rl.agents.sarsa_lambda import SarsaLambdaAgent
from mountaincar_rl.agents.tile_coder import TileCoder


def _tile_coder(*, num_tilings: int = 2) -> TileCoder:
    return TileCoder(
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_actions=3,
        num_tilings=num_tilings,
        tiles_per_dim=(2, 2),
    )


def _agent(
    *,
    tile_coder: TileCoder | None = None,
    alpha: float = 0.4,
    gamma: float = 0.9,
    lambda_: float = 0.5,
    epsilon_start: float = 0.0,
    epsilon_end: float = 0.0,
    epsilon_decay_steps: int = 10,
    trace_type: str = "replacing",
    trace_clip: float | None = None,
    seed: int = 7,
) -> SarsaLambdaAgent:
    return SarsaLambdaAgent(
        tile_coder=tile_coder or _tile_coder(),
        alpha=alpha,
        gamma=gamma,
        lambda_=lambda_,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        trace_type=trace_type,
        trace_clip=trace_clip,
        seed=seed,
    )


def test_manual_terminal_transition_delta_weight_change_and_trace_decay() -> None:
    agent = _agent()
    state = np.asarray([0.0, 0.0])
    active = agent.tile_coder.encode(state, 0)

    delta = agent.update(
        state,
        0,
        2.0,
        np.asarray([0.5, 0.5]),
        None,
        terminated=True,
        truncated=False,
    )

    # delta=2, effective step=alpha/m=0.2, so each active weight gains 0.4.
    assert delta == pytest.approx(2.0)
    np.testing.assert_allclose(agent.weights[active], 0.4)
    assert np.count_nonzero(agent.weights) == agent.tile_coder.num_tilings
    # Replacing traces are set to one, then decay by gamma*lambda = 0.45.
    np.testing.assert_allclose(agent.traces[active], 0.45)
    assert agent.update_count == 1
    assert agent.last_terminated is True
    assert agent.last_truncated is False


def test_true_termination_masks_bootstrap_and_does_not_require_next_action() -> None:
    agent = _agent()
    state = np.asarray([-0.25, 0.0])
    current = agent.tile_coder.encode(state, 0)
    next_features = agent.tile_coder.encode(np.asarray([0.5, 0.5]), 2)
    agent.weights[current] = 0.5  # Q(s, a) = 1.0.
    agent.weights[next_features] = 100.0  # Must not enter terminal target.

    delta = agent.update(
        state,
        0,
        -1.0,
        np.asarray([0.5, 0.5]),
        terminated=True,
        truncated=True,
    )

    assert delta == pytest.approx(-2.0)
    assert agent.last_terminated is True
    assert agent.last_truncated is True


def test_pure_truncation_bootstraps_from_supplied_next_action() -> None:
    agent = _agent()
    state = np.asarray([-0.5, 0.0])
    next_state = np.asarray([0.5, 0.5])
    agent.weights[agent.tile_coder.encode(next_state, 1)] = 1.0

    delta = agent.update(
        state,
        0,
        -1.0,
        next_state,
        1,
        terminated=False,
        truncated=True,
    )

    # Q(s_next, 1)=2 and Q(s, 0)=0: -1 + 0.9*2 = 0.8.
    assert delta == pytest.approx(0.8)
    assert agent.last_terminated is False
    assert agent.last_truncated is True


def test_pure_truncation_requires_next_action() -> None:
    agent = _agent()

    with pytest.raises(ValueError, match="pure truncation"):
        agent.update(
            np.zeros(2),
            0,
            -1.0,
            np.ones(2),
            None,
            terminated=False,
            truncated=True,
        )


def test_replacing_and_accumulating_traces_diverge_on_repeated_features() -> None:
    replacing = _agent(gamma=1.0, lambda_=1.0, trace_type="replacing")
    accumulating = _agent(gamma=1.0, lambda_=1.0, trace_type="accumulating")
    state = np.zeros(2)
    active = replacing.tile_coder.encode(state, 0)

    for agent in (replacing, accumulating):
        for _ in range(2):
            agent.update(
                state,
                0,
                0.0,
                state,
                None,
                terminated=True,
                truncated=False,
            )

    np.testing.assert_allclose(replacing.traces[active], 1.0)
    np.testing.assert_allclose(accumulating.traces[active], 2.0)


def test_replacing_trace_propagates_exact_credit_and_lambda_zero_removes_it() -> None:
    state = np.zeros(2)
    next_state = np.asarray([0.5, 0.5])

    for lambda_, expected_old_weight in ((0.5, 0.38), (0.0, 0.2)):
        agent = _agent(gamma=0.9, lambda_=lambda_, trace_type="replacing")
        old_active = agent.tile_coder.encode(state, 0)
        new_active = agent.tile_coder.encode(next_state, 1)

        # First delta is 1.0, so alpha/m=0.2 gives 0.2 on old_active;
        # its post-update eligibility is gamma*lambda.
        agent.update(
            state,
            0,
            1.0,
            next_state,
            1,
            terminated=False,
            truncated=False,
        )
        # The terminal delta is 2.0. With lambda=.5, old_active receives
        # 0.2 * 2 * (0.9 * 0.5) = 0.18 more. Lambda zero removes that credit.
        agent.update(
            next_state,
            1,
            2.0,
            next_state,
            None,
            terminated=True,
            truncated=False,
        )

        np.testing.assert_allclose(agent.weights[old_active], expected_old_weight)
        np.testing.assert_allclose(agent.weights[new_active], 0.4)


def test_trace_clip_and_end_episode_are_applied() -> None:
    agent = _agent(
        gamma=1.0,
        lambda_=1.0,
        trace_type="accumulating",
        trace_clip=1.5,
    )
    state = np.zeros(2)
    active = agent.tile_coder.encode(state, 0)
    for _ in range(2):
        agent.update(
            state,
            0,
            0.0,
            state,
            None,
            terminated=True,
            truncated=False,
        )

    np.testing.assert_allclose(agent.traces[active], 1.5)
    agent.end_episode()
    assert np.count_nonzero(agent.traces) == 0


def test_linear_epsilon_schedule_uses_successful_update_count() -> None:
    agent = _agent(
        gamma=0.0,
        lambda_=0.0,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=4,
    )
    state = np.zeros(2)

    expected = [1.0, 0.8, 0.6, 0.4, 0.2, 0.2]
    observed = [agent.current_epsilon]
    for _ in range(5):
        agent.update(
            state,
            0,
            0.0,
            state,
            None,
            terminated=True,
            truncated=False,
        )
        observed.append(agent.current_epsilon)

    assert observed == pytest.approx(expected)


def test_seeded_training_tie_breaking_is_reproducible() -> None:
    first = _agent(seed=123)
    second = _agent(seed=123)
    observation = np.zeros(2)

    first_actions = [first.select_action(observation, explore=True) for _ in range(30)]
    second_actions = [
        second.select_action(observation, explore=True) for _ in range(30)
    ]

    assert first_actions == second_actions
    assert all(0 <= action < 3 for action in first_actions)


def test_greedy_evaluation_uses_smallest_tie_and_mutates_no_state() -> None:
    agent = _agent(epsilon_start=1.0, epsilon_end=0.1)
    observation = np.zeros(2)
    # Actions 1 and 2 tie for the maximum, so deterministic evaluation picks 1.
    agent.weights[agent.tile_coder.encode(observation, 1)] = 0.5
    agent.weights[agent.tile_coder.encode(observation, 2)] = 0.5
    before = agent.state_dict()

    actions = [agent.select_action(observation, explore=False) for _ in range(10)]
    after = agent.state_dict()

    assert actions == [1] * 10
    np.testing.assert_array_equal(after["weights"], before["weights"])
    np.testing.assert_array_equal(after["traces"], before["traces"])
    assert after["rng_state"] == before["rng_state"]
    assert after["update_count"] == before["update_count"]


def test_nonfinite_values_fail_visibly_without_incrementing_schedule() -> None:
    agent = _agent()
    observation = np.zeros(2)

    with pytest.raises(ValueError, match="reward must be finite"):
        agent.update(
            observation,
            0,
            np.nan,
            observation,
            None,
            terminated=True,
            truncated=False,
        )
    assert agent.update_count == 0

    agent.weights[agent.tile_coder.encode(observation, 0)[0]] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite"):
        agent.q_values(observation)


def test_checkpoint_round_trip_preserves_training_and_rng_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "sarsa_checkpoint.npz"
    agent = _agent(
        epsilon_start=0.8,
        epsilon_end=0.1,
        trace_type="accumulating",
        trace_clip=3.0,
        seed=2025,
    )
    state = np.asarray([-0.2, 0.1])
    next_state = np.asarray([0.3, -0.1])
    agent.select_action(state, explore=True)
    agent.update(
        state,
        1,
        -0.5,
        next_state,
        2,
        terminated=False,
        truncated=True,
    )

    saved_path = agent.save(checkpoint_path)
    restored = SarsaLambdaAgent.load(checkpoint_path)

    assert saved_path == checkpoint_path
    assert checkpoint_path.is_file()
    assert restored.tile_coder.metadata() == agent.tile_coder.metadata()
    assert restored._hyperparameters() == agent._hyperparameters()
    np.testing.assert_array_equal(restored.weights, agent.weights)
    np.testing.assert_array_equal(restored.traces, agent.traces)
    assert restored.update_count == agent.update_count
    assert restored.last_terminated == agent.last_terminated
    assert restored.last_truncated == agent.last_truncated

    # The next stochastic behavior decisions must also continue identically.
    original_actions = [agent.select_action(state, explore=True) for _ in range(20)]
    restored_actions = [restored.select_action(state, explore=True) for _ in range(20)]
    assert restored_actions == original_actions


def test_loading_incompatible_tile_metadata_fails_visibly(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "sarsa_checkpoint.npz"
    agent = _agent()
    agent.save(checkpoint_path)
    incompatible = _tile_coder(num_tilings=4)

    with pytest.raises(ValueError, match="tile-coder metadata"):
        SarsaLambdaAgent.load(checkpoint_path, tile_coder=incompatible)


def test_state_dict_is_independent_and_rejects_incompatible_hyperparameters() -> None:
    agent = _agent()
    state = agent.state_dict()
    copied = copy.deepcopy(state)
    state["weights"][0] = 99.0

    assert agent.weights[0] == 0.0
    incompatible = _agent(alpha=0.2)
    with pytest.raises(ValueError, match="hyperparameters"):
        incompatible.load_state_dict(copied)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"alpha": 0.0}, ValueError, "alpha"),
        ({"gamma": 1.1}, ValueError, "gamma"),
        ({"lambda_": np.nan}, ValueError, "finite"),
        ({"epsilon_start": 0.1, "epsilon_end": 0.2}, ValueError, "epsilon_end"),
        ({"epsilon_decay_steps": 0}, ValueError, "decay_steps"),
        ({"trace_type": "dutch"}, ValueError, "trace_type"),
        ({"trace_clip": 0.0}, ValueError, "trace_clip"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_invalid_agent_configuration_is_rejected(
    kwargs: dict[str, object], error_type: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "tile_coder": _tile_coder(),
        "alpha": 0.4,
        "gamma": 0.9,
        "lambda_": 0.5,
        "epsilon_start": 0.1,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 10,
    }
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        SarsaLambdaAgent(**arguments)  # type: ignore[arg-type]
