"""Tests for fixed bounds-based observation preprocessing."""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest

from mountaincar_rl.environments.observation import ObservationNormalizer


def _normalizer(*, clip: bool = True) -> ObservationNormalizer:
    return ObservationNormalizer(
        low=np.array([-1.2, -0.07], dtype=np.float32),
        high=np.array([0.6, 0.07], dtype=np.float32),
        clip=clip,
    )


def test_exact_bounds_and_midpoint_map_to_minus_one_zero_one() -> None:
    normalizer = _normalizer()
    midpoint = (
        normalizer.low.astype(np.float64) + normalizer.high.astype(np.float64)
    ) / 2.0

    normalized = normalizer.normalize(
        np.stack([normalizer.low, midpoint, normalizer.high])
    )

    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(
        normalized,
        np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )


def test_normalization_clips_by_default_and_can_leave_values_unclipped() -> None:
    observation = np.array([-2.1, 0.14], dtype=np.float64)

    np.testing.assert_array_equal(
        _normalizer(clip=True).normalize(observation),
        np.array([-1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _normalizer(clip=False).normalize(observation),
        np.array([-2.0, 2.0], dtype=np.float32),
        rtol=0.0,
        atol=2e-6,
    )


def test_batch_normalization_preserves_all_leading_dimensions() -> None:
    normalizer = ObservationNormalizer(
        low=np.array([[0.0, 10.0], [-2.0, 4.0]], dtype=np.float32),
        high=np.array([[2.0, 14.0], [2.0, 8.0]], dtype=np.float32),
    )
    batch = np.stack(
        [
            normalizer.low,
            (normalizer.low + normalizer.high) / 2.0,
            normalizer.high,
        ]
    )

    normalized = normalizer.normalize(batch)

    assert normalized.shape == (3, 2, 2)
    np.testing.assert_array_equal(normalized[0], -np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(normalized[1], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(normalized[2], np.ones((2, 2), dtype=np.float32))


@pytest.mark.parametrize(
    ("low", "high", "error_type"),
    [
        ([0.0], [1.0, 2.0], ValueError),
        ([0.0, np.nan], [1.0, 2.0], ValueError),
        ([0.0, 1.0], [1.0, np.inf], ValueError),
        ([0.0, 1.0], [0.0, 2.0], ValueError),
        ([0.0, 2.0], [1.0, 1.0], ValueError),
        ([], [], ValueError),
        (0.0, 1.0, ValueError),
    ],
)
def test_invalid_bounds_are_rejected(
    low: object, high: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        ObservationNormalizer(low=low, high=high)  # type: ignore[arg-type]


def test_invalid_clip_and_observation_shapes_are_rejected() -> None:
    with pytest.raises(TypeError, match="clip"):
        ObservationNormalizer(low=np.zeros(2), high=np.ones(2), clip=1)  # type: ignore[arg-type]

    normalizer = _normalizer()
    with pytest.raises(ValueError, match="shape"):
        normalizer.normalize(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        normalizer.normalize(np.array([-0.5, np.nan]))


def test_from_space_uses_the_declared_bounds_without_fitting() -> None:
    space = gym.spaces.Box(
        low=np.array([-3.0, -2.0], dtype=np.float32),
        high=np.array([1.0, 6.0], dtype=np.float32),
        dtype=np.float32,
    )

    normalizer = ObservationNormalizer.from_space(space)

    np.testing.assert_array_equal(normalizer.low, space.low)
    np.testing.assert_array_equal(normalizer.high, space.high)
    np.testing.assert_array_equal(
        normalizer.normalize(np.array([-1.0, 2.0])),
        np.zeros(2, dtype=np.float32),
    )


def test_from_space_rejects_missing_or_nonfinite_bounds() -> None:
    with pytest.raises(TypeError, match="low and high"):
        ObservationNormalizer.from_space(gym.spaces.Discrete(3))
    infinite = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
    with pytest.raises(ValueError, match="finite"):
        ObservationNormalizer.from_space(infinite)


def test_metadata_round_trip_is_json_safe_and_bounds_are_immutable() -> None:
    source_low = np.array([-1.2, -0.07], dtype=np.float32)
    source_high = np.array([0.6, 0.07], dtype=np.float32)
    normalizer = ObservationNormalizer(source_low, source_high, clip=False)
    source_low[:] = 0.0
    source_high[:] = 1.0

    payload = json.loads(json.dumps(normalizer.metadata(), allow_nan=False))
    restored = ObservationNormalizer.from_metadata(payload)

    assert payload == {
        "version": 1,
        "method": "bounds_affine_to_minus_one_one",
        "low": [-1.2000000476837158, -0.07000000029802322],
        "high": [0.6000000238418579, 0.07000000029802322],
        "output_range": [-1.0, 1.0],
        "clip": False,
    }
    assert restored.metadata() == payload
    np.testing.assert_array_equal(restored.low, normalizer.low)
    np.testing.assert_array_equal(restored.high, normalizer.high)
    with pytest.raises(ValueError):
        restored.low[0] = 0.0


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("version", 2, ValueError),
        ("method", "trajectory_standardization", ValueError),
        ("output_range", [0.0, 1.0], ValueError),
        ("clip", 1, TypeError),
    ],
)
def test_invalid_metadata_contract_is_rejected(
    field: str, value: object, error_type: type[Exception]
) -> None:
    metadata = _normalizer().metadata()
    metadata[field] = value

    with pytest.raises(error_type):
        ObservationNormalizer.from_metadata(metadata)

    missing = _normalizer().metadata()
    del missing["high"]
    with pytest.raises(ValueError, match="missing"):
        ObservationNormalizer.from_metadata(missing)
