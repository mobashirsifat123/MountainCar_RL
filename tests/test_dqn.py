"""Deterministic unit tests for the shared replay/online DQN implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as functional

from mountaincar_rl.agents.dqn import DQNAgent, QNetwork, compute_td_targets
from mountaincar_rl.environments.observation import (
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
    ObservationNormalizer,
)


def _normalizer() -> ObservationNormalizer:
    return ObservationNormalizer(OBSERVATION_LOW, OBSERVATION_HIGH)


def _agent(*, use_replay: bool, **overrides: object) -> DQNAgent:
    config: dict[str, object] = {
        "action_count": 3,
        "observation_normalizer": _normalizer(),
        "hidden_sizes": (),
        "gamma": 0.9,
        "learning_rate": 0.02,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 10,
        "gradient_clip_norm": 1.0,
        "target_update_frequency": 100,
        "tau": 1.0,
        "use_replay": use_replay,
        "replay_capacity": 16,
        "batch_size": 2,
        "warmup_steps": 2,
        "update_frequency": 1,
        "seed": 13,
        "device": "cpu",
    }
    config.update(overrides)
    return DQNAgent(**config)


def _last_linear(agent: DQNAgent) -> nn.Linear:
    layers = [
        module
        for module in agent.online_network.modules()
        if isinstance(module, nn.Linear)
    ]
    return layers[-1]


def test_q_network_output_shape_for_single_and_batched_observations() -> None:
    """The final dimension always contains one value per action."""

    network = QNetwork(input_dim=2, action_count=3, hidden_sizes=(8, 4))

    assert network(torch.zeros(2)).shape == (3,)
    assert network(torch.zeros(7, 2)).shape == (7, 3)


def test_td_target_calculation_and_true_terminal_mask_are_exact() -> None:
    """A true terminal receives reward only; other rows use max target Q."""

    rewards = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    next_q_values = torch.tensor(
        [[2.0, 4.0, 3.0], [8.0, 1.0, -1.0], [-2.0, -3.0, -1.0]],
        dtype=torch.float32,
    )
    terminated = torch.tensor([False, True, False])

    targets = compute_td_targets(rewards, next_q_values, terminated, gamma=0.5)

    torch.testing.assert_close(targets, torch.tensor([3.0, -2.0, 0.0]))


def test_pure_time_limit_truncation_bootstraps_by_convention() -> None:
    """With terminated false, a truncated transition retains its bootstrap."""

    # ``truncated=True`` is intentionally not an argument to the target helper.
    rewards = torch.tensor([-1.0])
    next_q_values = torch.tensor([[2.0, 5.0, 3.0]])
    terminated = torch.tensor([False])
    truncated = torch.tensor([True])

    target = compute_td_targets(rewards, next_q_values, terminated, gamma=0.9)

    assert bool(truncated.item()) is True
    torch.testing.assert_close(target, torch.tensor([3.5]))


def test_hard_and_soft_target_synchronization_have_declared_math() -> None:
    """Hard copy and Polyak interpolation update every target parameter."""

    agent = _agent(use_replay=False, tau=0.25)
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.fill_(2.0)
        for parameter in agent.target_network.parameters():
            parameter.zero_()

    agent.soft_sync_target()
    for parameter in agent.target_network.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))

    agent.hard_sync_target()
    for parameter in agent.target_network.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2.0))
        assert parameter.requires_grad is False


def test_target_sync_frequency_counts_optimizer_steps_and_targets_are_detached() -> None:
    """The target is frozen until the configured optimizer-step boundary."""

    agent = _agent(
        use_replay=False,
        target_update_frequency=2,
        update_frequency=1,
    )
    observation = np.array([-0.5, 0.0], dtype=np.float32)
    next_observation = np.array([-0.49, 0.01], dtype=np.float32)
    initial_target = [
        parameter.detach().clone() for parameter in agent.target_network.parameters()
    ]

    agent.observe(
        observation,
        0,
        -1.0,
        next_observation,
        terminated=False,
        truncated=False,
    )

    assert agent.optimizer_steps == 1
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            initial_target, agent.target_network.parameters(), strict=True
        )
    )
    assert all(parameter.grad is None for parameter in agent.target_network.parameters())

    agent.observe(
        observation,
        0,
        -1.0,
        next_observation,
        terminated=False,
        truncated=False,
    )

    assert agent.optimizer_steps == 2
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            agent.online_network.parameters(),
            agent.target_network.parameters(),
            strict=True,
        )
    )
    assert all(parameter.grad is None for parameter in agent.target_network.parameters())


def test_epsilon_schedule_counts_consumed_environment_transitions() -> None:
    """Action queries and evaluation do not advance the epsilon schedule."""

    agent = _agent(
        use_replay=False,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_decay_steps=10,
        update_frequency=2,
    )
    observation = np.array([-0.5, 0.0], dtype=np.float32)
    next_observation = np.array([-0.49, 0.01], dtype=np.float32)

    for _ in range(7):
        agent.select_action(observation, explore=True)
        agent.select_action(observation, explore=False)
    assert agent.environment_steps == 0
    assert agent.epsilon == pytest.approx(1.0)

    for _ in range(4):
        agent.observe(
            observation,
            0,
            -1.0,
            next_observation,
            terminated=False,
            truncated=False,
        )

    assert agent.environment_steps == 4
    assert agent.optimizer_steps == 2
    assert agent.epsilon == pytest.approx(0.6)


def test_epsilon_zero_is_greedy_with_smallest_argmax_tie_breaking() -> None:
    """With equal Q values, deterministic greedy selection returns action zero."""

    agent = _agent(use_replay=False, epsilon_start=0.0, epsilon_end=0.0)
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
    observation = np.array([-0.5, 0.0], dtype=np.float32)

    assert [agent.select_action(observation, explore=True) for _ in range(20)] == [
        0
    ] * 20


def test_epsilon_one_uses_reproducible_uniform_exploration() -> None:
    """Full exploration ignores a network made to prefer one action."""

    first = _agent(
        use_replay=False, epsilon_start=1.0, epsilon_end=1.0, seed=77
    )
    second = _agent(
        use_replay=False, epsilon_start=1.0, epsilon_end=1.0, seed=77
    )
    with torch.no_grad():
        _last_linear(first).bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
        _last_linear(second).bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
    observation = np.array([-0.5, 0.0], dtype=np.float32)

    first_actions = [first.select_action(observation, explore=True) for _ in range(90)]
    second_actions = [
        second.select_action(observation, explore=True) for _ in range(90)
    ]

    assert first_actions == second_actions
    assert set(first_actions) == {0, 1, 2}


def test_greedy_evaluation_is_strictly_non_mutating() -> None:
    """Repeated evaluation actions leave RNG, counters, buffers, and nets intact."""

    agent = _agent(use_replay=True)
    observation = np.array([-0.5, 0.01], dtype=np.float32)
    before = agent.diagnostic_state()

    actions = [agent.select_action(observation, explore=False) for _ in range(25)]
    after = agent.diagnostic_state()

    assert len(set(actions)) == 1
    assert after == before


def test_replay_waits_for_batch_while_online_updates_current_transition() -> None:
    """The ablation differs in retention, minibatch size, and decorrelation."""

    observation = np.array([-0.6, 0.0], dtype=np.float32)
    next_observation = np.array([-0.59, 0.01], dtype=np.float32)
    replay = _agent(use_replay=True, batch_size=2, warmup_steps=2)
    online = _agent(use_replay=False, batch_size=8, warmup_steps=999)

    first_replay_loss = replay.observe(
        observation,
        0,
        -1.0,
        next_observation,
        terminated=False,
        truncated=False,
    )
    second_replay_loss = replay.observe(
        next_observation,
        1,
        -1.0,
        observation,
        terminated=False,
        truncated=True,
    )
    online_loss = online.observe(
        observation,
        0,
        -1.0,
        next_observation,
        terminated=False,
        truncated=True,
    )

    assert first_replay_loss is None
    assert second_replay_loss is not None and np.isfinite(second_replay_loss)
    assert replay.replay_buffer is not None
    assert len(replay.replay_buffer) == 2
    assert replay.environment_steps == 2
    assert replay.optimizer_steps == 1
    assert online_loss is not None and np.isfinite(online_loss)
    assert online.replay_buffer is None
    assert online.environment_steps == 1
    assert online.optimizer_steps == 1


def test_replay_state_coverage_reports_retained_normalized_cells() -> None:
    agent = _agent(use_replay=True, batch_size=4, warmup_steps=99)
    states = [
        np.asarray([-1.2, -0.07], dtype=np.float32),
        np.asarray([-0.3, -0.07], dtype=np.float32),
        np.asarray([0.6, 0.07], dtype=np.float32),
    ]
    for state in states:
        agent.observe(
            state,
            0,
            -1.0,
            state,
            terminated=False,
            truncated=False,
        )

    coverage = agent.replay_state_coverage((2, 2))

    assert coverage == {
        "buffer_size": 3,
        "occupied_cells": 3,
        "total_cells": 4,
        "coverage_fraction": 0.75,
    }


def test_checkpoint_round_trip_preserves_complete_training_state(
    tmp_path: Path,
) -> None:
    """Networks, optimizer, normalizer, counters, replay, and RNGs all resume."""

    agent = _agent(
        use_replay=True,
        epsilon_start=0.8,
        epsilon_end=0.1,
        target_update_frequency=2,
    )
    observations = [
        np.array([-0.7, -0.01], dtype=np.float32),
        np.array([-0.6, 0.02], dtype=np.float32),
        np.array([-0.5, 0.03], dtype=np.float32),
        np.array([-0.4, 0.01], dtype=np.float32),
    ]
    for index in range(3):
        agent.select_action(observations[index], explore=True)
        agent.observe(
            observations[index],
            index % 3,
            -1.0,
            observations[index + 1],
            terminated=False,
            truncated=(index == 2),
        )
    checkpoint = agent.save(tmp_path / "dqn.pt")

    restored = DQNAgent.load(checkpoint, device="cpu")

    assert restored.preprocessing_metadata == agent.preprocessing_metadata
    assert restored.diagnostic_state() == agent.diagnostic_state()
    assert restored.optimizer.state_dict()["state"]
    batch = torch.tensor(
        np.stack([_normalizer().normalize(item) for item in observations]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            restored.online_network(batch), agent.online_network(batch)
        )
        torch.testing.assert_close(
            restored.target_network(batch), agent.target_network(batch)
        )
    assert restored.select_action(observations[0], explore=True) == agent.select_action(
        observations[0], explore=True
    )
    assert restored.replay_buffer is not None
    assert agent.replay_buffer is not None
    torch.testing.assert_close(
        restored.replay_buffer.sample(2).actions,
        agent.replay_buffer.sample(2).actions,
    )


def test_controlled_optimizer_sequence_reduces_terminal_huber_loss() -> None:
    """Repeated deterministic terminal targets reduce a known supervised loss."""

    agent = _agent(
        use_replay=False,
        gamma=0.99,
        learning_rate=0.05,
        gradient_clip_norm=0.5,
    )
    observation = np.array(
        (OBSERVATION_LOW + OBSERVATION_HIGH) / 2.0, dtype=np.float32
    )
    normalized = torch.tensor(
        agent.observation_normalizer.normalize(observation), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        initial_prediction = agent.online_network(normalized)[0, 0]
        initial_loss = functional.smooth_l1_loss(
            initial_prediction, torch.tensor(1.0)
        ).item()

    losses = [
        agent.observe(
            observation,
            0,
            1.0,
            observation,
            terminated=True,
            truncated=False,
        )
        for _ in range(30)
    ]
    with torch.no_grad():
        final_prediction = agent.online_network(normalized)[0, 0]
        final_loss = functional.smooth_l1_loss(
            final_prediction, torch.tensor(1.0)
        ).item()

    assert all(loss is not None and np.isfinite(loss) for loss in losses)
    assert final_loss < initial_loss * 0.1


def test_optimizer_applies_configured_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite online update routes gradients through the declared clip norm."""

    calls: list[float] = []
    original_clip = nn.utils.clip_grad_norm_

    def recording_clip(
        parameters: object, max_norm: float, *args: object, **kwargs: object
    ) -> torch.Tensor:
        calls.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", recording_clip)
    agent = _agent(use_replay=False, gradient_clip_norm=0.125)
    loss = agent.observe(
        np.array([-0.5, 0.0], dtype=np.float32),
        2,
        1_000.0,
        np.array([-0.4, 0.02], dtype=np.float32),
        terminated=True,
        truncated=False,
    )

    assert loss is not None and np.isfinite(loss)
    assert calls == [0.125]
