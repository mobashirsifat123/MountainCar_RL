"""Contract tests for deterministic fixed-capacity uniform replay."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mountaincar_rl.agents.replay_buffer import ReplayBuffer


def _add_marker(
    buffer: ReplayBuffer,
    marker: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> None:
    observation = np.array([marker, -marker], dtype=np.float32)
    next_observation = observation + np.array([0.25, 0.5], dtype=np.float32)
    buffer.add(
        observation,
        marker,
        float(marker),
        next_observation,
        terminated=terminated,
        truncated=truncated,
    )


def test_circular_buffer_overwrites_only_the_oldest_transitions() -> None:
    """A full buffer retains the most recent ``capacity`` markers."""

    buffer = ReplayBuffer(capacity=3, seed=4)
    for marker in range(5):
        _add_marker(buffer, marker)

    state = buffer.state_dict()

    assert len(buffer) == 3
    assert buffer.next_index == 2
    assert set(state["actions"].tolist()) == {2, 3, 4}
    assert state["actions"].tolist() == [3, 4, 2]


def test_sampling_before_sufficient_data_raises_visible_error() -> None:
    """Replay must never silently sample uninitialized storage."""

    buffer = ReplayBuffer(capacity=4)
    _add_marker(buffer, 0)

    with pytest.raises(ValueError, match="size 1"):
        buffer.sample(2)


def test_equal_seeds_produce_identical_without_replacement_samples() -> None:
    """Sampling has an isolated reproducible RNG stream."""

    first = ReplayBuffer(capacity=8, seed=19)
    second = ReplayBuffer(capacity=8, seed=19)
    for marker in range(8):
        _add_marker(first, marker)
        _add_marker(second, marker)

    first_batch = first.sample(5)
    second_batch = second.sample(5)

    torch.testing.assert_close(first_batch.actions, second_batch.actions)
    assert torch.unique(first_batch.actions).numel() == 5


def test_sample_shapes_dtypes_device_and_boundary_flags_are_preserved() -> None:
    """Every transition field has an explicit tensor contract."""

    buffer = ReplayBuffer(capacity=4, seed=2)
    _add_marker(buffer, 0, terminated=False, truncated=False)
    _add_marker(buffer, 1, terminated=True, truncated=False)
    _add_marker(buffer, 2, terminated=False, truncated=True)
    _add_marker(buffer, 3, terminated=True, truncated=True)

    batch = buffer.sample(4, device="cpu")

    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.next_observations.shape == (4, 2)
    assert batch.terminated.shape == (4,)
    assert batch.truncated.shape == (4,)
    assert batch.observations.dtype == torch.float32
    assert batch.actions.dtype == torch.int64
    assert batch.rewards.dtype == torch.float32
    assert batch.next_observations.dtype == torch.float32
    assert batch.terminated.dtype == torch.bool
    assert batch.truncated.dtype == torch.bool
    assert all(tensor.device.type == "cpu" for tensor in batch.__dict__.values())

    flags_by_action = {
        int(action): (bool(terminated), bool(truncated))
        for action, terminated, truncated in zip(
            batch.actions.tolist(),
            batch.terminated.tolist(),
            batch.truncated.tolist(),
            strict=True,
        )
    }
    assert flags_by_action == {
        0: (False, False),
        1: (True, False),
        2: (False, True),
        3: (True, True),
    }


def test_state_round_trip_preserves_storage_cursor_and_sampling_stream() -> None:
    """Resuming replay yields the same future samples and overwrites."""

    original = ReplayBuffer(capacity=4, seed=31)
    for marker in range(6):
        _add_marker(original, marker, truncated=(marker == 5))
    original.sample(2)  # Advance the private sampling stream before checkpointing.

    restored = ReplayBuffer(capacity=4, seed=999)
    restored.load_state_dict(original.state_dict())

    torch.testing.assert_close(original.sample(3).actions, restored.sample(3).actions)
    _add_marker(original, 8, terminated=True)
    _add_marker(restored, 8, terminated=True)
    original_state = original.state_dict()
    restored_state = restored.state_dict()
    for key in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminated",
        "truncated",
    ):
        np.testing.assert_array_equal(original_state[key], restored_state[key])
    assert original.next_index == restored.next_index


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("observation", [np.nan, 0.0], "finite"),
        ("next_observation", [0.0, np.inf], "finite"),
        ("reward", np.inf, "finite"),
        ("action", -1, "nonnegative"),
        ("terminated", 1, "boolean"),
    ],
)
def test_add_rejects_invalid_transition_fields(
    field: str, value: object, error: str
) -> None:
    """Corrupt values are rejected before any storage cursor advances."""

    transition: dict[str, object] = {
        "observation": [0.0, 0.0],
        "action": 0,
        "reward": -1.0,
        "next_observation": [0.1, 0.0],
        "terminated": False,
        "truncated": False,
    }
    transition[field] = value
    buffer = ReplayBuffer(capacity=2)

    with pytest.raises((TypeError, ValueError), match=error):
        buffer.add(
            transition["observation"],
            transition["action"],
            transition["reward"],
            transition["next_observation"],
            terminated=transition["terminated"],
            truncated=transition["truncated"],
        )

    assert len(buffer) == 0
    assert buffer.next_index == 0
