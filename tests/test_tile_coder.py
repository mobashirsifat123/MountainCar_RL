"""Unit tests for deterministic collision-free MountainCar tile coding."""

import numpy as np
import pytest

from mountaincar_rl.agents.tile_coder import TileCoder


LOW = np.asarray([-1.2, -0.07], dtype=np.float64)
HIGH = np.asarray([0.6, 0.07], dtype=np.float64)


def _coder() -> TileCoder:
    return TileCoder(
        low=LOW,
        high=HIGH,
        num_actions=3,
        num_tilings=8,
        tiles_per_dim=(8, 8),
    )


def test_encode_has_exactly_one_unique_bounded_index_per_tiling() -> None:
    coder = _coder()

    active = coder.encode(np.asarray([-0.5, 0.012]), action=2)

    assert active.dtype == np.int64
    assert active.shape == (coder.num_tilings,)
    assert np.unique(active).size == coder.num_tilings
    assert np.all(active >= 0)
    assert np.all(active < coder.total_features)


def test_encoding_is_deterministic_across_calls_and_instances() -> None:
    state = np.asarray([-0.47, -0.013])
    first = _coder()
    second = _coder()

    np.testing.assert_array_equal(first.encode(state, 1), first.encode(state, 1))
    np.testing.assert_array_equal(first.encode(state, 1), second.encode(state, 1))
    np.testing.assert_array_equal(first.offsets, second.offsets)


def test_lower_midpoint_and_upper_boundaries_have_stable_dense_indices() -> None:
    coder = _coder()
    tiling_bases = np.arange(8, dtype=np.int64) * 64

    np.testing.assert_array_equal(coder.state_tiles(LOW), tiling_bases)
    np.testing.assert_array_equal(
        coder.state_tiles((LOW + HIGH) / 2.0), tiling_bases + 36
    )
    np.testing.assert_array_equal(coder.state_tiles(HIGH), tiling_bases + 63)

    # Boundary clipping is part of the declared deterministic behavior.
    np.testing.assert_array_equal(coder.state_tiles(LOW - 100.0), tiling_bases)
    np.testing.assert_array_equal(coder.state_tiles(HIGH + 100.0), tiling_bases + 63)


def test_nearby_states_share_some_but_not_all_active_tiles() -> None:
    coder = _coder()
    midpoint = (LOW + HIGH) / 2.0
    nearby = midpoint + np.asarray([0.045, 0.0])

    first = coder.state_tiles(midpoint)
    second = coder.state_tiles(nearby)
    shared = np.intersect1d(first, second)

    assert 0 < shared.size < coder.num_tilings
    assert not np.array_equal(first, second)


def test_action_feature_ranges_are_disjoint() -> None:
    coder = _coder()
    state = np.asarray([-0.4, 0.02])
    action_indices = [coder.encode(state, action) for action in range(3)]

    for action, indices in enumerate(action_indices):
        lower = action * coder.features_per_action
        upper = (action + 1) * coder.features_per_action
        assert np.all((indices >= lower) & (indices < upper))
    assert set(action_indices[0]).isdisjoint(action_indices[1])
    assert set(action_indices[1]).isdisjoint(action_indices[2])
    assert set(action_indices[0]).isdisjoint(action_indices[2])


def test_metadata_records_configuration_offsets_and_effective_sizes() -> None:
    coder = _coder()
    metadata = coder.metadata()

    assert metadata["low"] == LOW.tolist()
    assert metadata["high"] == HIGH.tolist()
    assert metadata["num_actions"] == 3
    assert metadata["num_tilings"] == 8
    assert metadata["tiles_per_dim"] == [8, 8]
    assert metadata["tiles_per_tiling"] == 64
    assert metadata["features_per_action"] == 512
    assert metadata["total_features"] == 1536
    assert metadata["offset_scheme"] == "fractional_reverse_half_phase_v1"
    np.testing.assert_allclose(metadata["offsets"], coder.offsets)
    assert not np.array_equal(coder.offsets[:, 0], coder.offsets[:, 1])


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"low": [0.0], "high": [1.0]}, ValueError, "exactly two"),
        ({"low": [0.0, np.nan], "high": [1.0, 1.0]}, ValueError, "finite"),
        ({"low": [0.0, 0.0], "high": [0.0, 1.0]}, ValueError, "strictly"),
        ({"num_actions": 0}, ValueError, "positive"),
        ({"num_tilings": True}, TypeError, "integer"),
        ({"tiles_per_dim": (8, 0)}, ValueError, "positive"),
        ({"tiles_per_dim": (8,)}, ValueError, "exactly two"),
    ],
)
def test_invalid_tile_coder_configuration_is_rejected(
    kwargs: dict[str, object], error_type: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "low": LOW,
        "high": HIGH,
        "num_actions": 3,
        "num_tilings": 8,
        "tiles_per_dim": (8, 8),
    }
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        TileCoder(**arguments)  # type: ignore[arg-type]


def test_invalid_observation_and_action_are_rejected() -> None:
    coder = _coder()

    with pytest.raises(ValueError, match="exactly two"):
        coder.encode(np.asarray([0.0]), 0)
    with pytest.raises(ValueError, match="finite"):
        coder.encode(np.asarray([0.0, np.inf]), 0)
    with pytest.raises(TypeError, match="integer"):
        coder.encode(np.zeros(2), True)
    with pytest.raises(ValueError, match="action"):
        coder.encode(np.zeros(2), 3)
