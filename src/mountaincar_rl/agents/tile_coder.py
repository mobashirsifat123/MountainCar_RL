"""Collision-free dense tile coding for two-dimensional continuous states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _positive_integer(value: Any, *, name: str) -> int:
    """Return ``value`` as an integer after rejecting booleans and nonpositives."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


class TileCoder:
    """Encode a bounded two-dimensional state with overlapping dense tilings.

    Each tiling owns a disjoint block of ``prod(tiles_per_dim)`` features.
    Actions then own disjoint blocks containing every tiling, so the returned
    indices are collision-free and can directly index one flat weight vector.
    The first dimension uses evenly spaced fractional-tile offsets; the second
    uses the same spacing in reverse order with a half-spacing phase shift.
    Consequently both axes move between tilings instead of sharing one
    diagonal offset.

    Observations outside the declared bounds are clipped before indexing.  In
    particular, the exact lower and upper bounds deterministically map to the
    first and last tile, respectively, in every tiling.
    """

    _METADATA_VERSION = 1
    _OFFSET_SCHEME = "fractional_reverse_half_phase_v1"

    def __init__(
        self,
        low: Sequence[float] | np.ndarray,
        high: Sequence[float] | np.ndarray,
        num_actions: int,
        num_tilings: int = 8,
        tiles_per_dim: Sequence[int] = (8, 8),
    ) -> None:
        """Initialize a deterministic tile coder over finite rectangular bounds."""

        low_array = np.asarray(low, dtype=np.float64)
        high_array = np.asarray(high, dtype=np.float64)
        if low_array.shape != (2,) or high_array.shape != (2,):
            raise ValueError("low and high must each contain exactly two values")
        if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
            raise ValueError("low and high must contain only finite values")
        if np.any(high_array <= low_array):
            raise ValueError("each high bound must be strictly greater than low")

        if isinstance(tiles_per_dim, (str, bytes)):
            raise TypeError("tiles_per_dim must be a two-element integer sequence")
        try:
            tile_counts = tuple(tiles_per_dim)
        except TypeError as error:
            raise TypeError(
                "tiles_per_dim must be a two-element integer sequence"
            ) from error
        if len(tile_counts) != 2:
            raise ValueError("tiles_per_dim must contain exactly two values")

        self.low = low_array.copy()
        self.high = high_array.copy()
        self.num_actions = _positive_integer(num_actions, name="num_actions")
        self.num_tilings = _positive_integer(num_tilings, name="num_tilings")
        self.tiles_per_dim = (
            _positive_integer(tile_counts[0], name="tiles_per_dim[0]"),
            _positive_integer(tile_counts[1], name="tiles_per_dim[1]"),
        )
        self.tiles_per_tiling = int(np.prod(self.tiles_per_dim, dtype=np.int64))
        self.features_per_action = self.num_tilings * self.tiles_per_tiling
        self.total_features = self.num_actions * self.features_per_action

        tiling_ids = np.arange(self.num_tilings, dtype=np.float64)
        offset_x = tiling_ids / self.num_tilings
        offset_y = (
            np.mod(-tiling_ids + 0.5, self.num_tilings) / self.num_tilings
        )
        self._offsets = np.column_stack((offset_x, offset_y))
        self._tile_counts = np.asarray(self.tiles_per_dim, dtype=np.int64)
        self._tiling_bases = (
            np.arange(self.num_tilings, dtype=np.int64) * self.tiles_per_tiling
        )

    @property
    def offsets(self) -> np.ndarray:
        """Return a defensive copy of the fractional-tile offsets."""

        return self._offsets.copy()

    def _validated_observation(
        self, observation: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float64)
        if array.shape != (2,):
            raise ValueError("observation must contain exactly two values")
        if not np.all(np.isfinite(array)):
            raise ValueError("observation must contain only finite values")
        return np.clip(array, self.low, self.high)

    def _validated_action(self, action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        result = int(action)
        if not 0 <= result < self.num_actions:
            raise ValueError(
                f"action must be in [0, {self.num_actions - 1}], got {result}"
            )
        return result

    def state_tiles(
        self, observation: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Return one collision-free, action-local feature index per tiling.

        The returned indices lie in ``[0, features_per_action)``.  Use
        :meth:`encode` to obtain global indices for a particular action.
        """

        clipped = self._validated_observation(observation)
        normalized = (clipped - self.low) / (self.high - self.low)
        coordinates = np.floor(
            normalized[np.newaxis, :] * self._tile_counts + self._offsets
        ).astype(np.int64)
        coordinates = np.clip(coordinates, 0, self._tile_counts - 1)
        flat_tiles = coordinates[:, 0] * self.tiles_per_dim[1] + coordinates[:, 1]
        indices = self._tiling_bases + flat_tiles
        return indices.astype(np.int64, copy=False)

    def encode(
        self, observation: Sequence[float] | np.ndarray, action: int
    ) -> np.ndarray:
        """Return exactly one bounded global feature index per tiling/action."""

        validated_action = self._validated_action(action)
        return self.state_tiles(observation) + (
            validated_action * self.features_per_action
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible configuration and effective feature sizes."""

        return {
            "version": self._METADATA_VERSION,
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "num_actions": self.num_actions,
            "num_tilings": self.num_tilings,
            "tiles_per_dim": list(self.tiles_per_dim),
            "tiles_per_tiling": self.tiles_per_tiling,
            "features_per_action": self.features_per_action,
            "total_features": self.total_features,
            "offset_scheme": self._OFFSET_SCHEME,
            "offsets": self._offsets.tolist(),
        }
