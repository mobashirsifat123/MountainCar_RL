"""MountainCar state bounds and shared neural observation preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
GOAL_POSITION = 0.5

OBSERVATION_LOW = np.asarray([POSITION_MIN, VELOCITY_MIN], dtype=np.float32)
OBSERVATION_HIGH = np.asarray([POSITION_MAX, VELOCITY_MAX], dtype=np.float32)
OBSERVATION_LOW.setflags(write=False)
OBSERVATION_HIGH.setflags(write=False)


def _immutable_float32_array(value: Any, *, name: str) -> npt.NDArray[np.float32]:
    """Return a finite float32 array backed by immutable bytes."""

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an array of real numbers") from error
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty array with at least one axis")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    # A read-only view of an owning ndarray can be made writeable again. Using
    # immutable ``bytes`` as the ultimate buffer prevents checkpoint metadata or
    # caller-owned arrays from mutating preprocessing after construction.
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float32)
    return immutable.reshape(array.shape)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationNormalizer:
    """Fixed affine observation transform derived only from declared bounds.

    Each component is mapped as

    ``z = 2 * (observation - low) / (high - low) - 1``.

    The bounds are copied into immutable float32 arrays. They are never fitted
    from trajectory data, so every DQN condition can share an identical,
    checkpointable preprocessing transform. Leading batch dimensions are
    accepted as long as the trailing dimensions match the bounds.
    """

    low: npt.NDArray[np.float32]
    high: npt.NDArray[np.float32]
    clip: bool = True

    METADATA_VERSION: ClassVar[int] = 1
    METHOD: ClassVar[str] = "bounds_affine_to_minus_one_one"
    OUTPUT_RANGE: ClassVar[tuple[float, float]] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.clip, bool):
            raise TypeError("clip must be a boolean")
        low = _immutable_float32_array(self.low, name="low")
        high = _immutable_float32_array(self.high, name="high")
        if low.shape != high.shape:
            raise ValueError(
                f"low and high must have the same shape, got {low.shape} and "
                f"{high.shape}"
            )
        if not np.all(high > low):
            raise ValueError("every high bound must be strictly greater than low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @classmethod
    def from_space(
        cls, space: Any, *, clip: bool = True
    ) -> "ObservationNormalizer":
        """Build a normalizer from a finite bounds-declaring space.

        Gymnasium ``Box`` spaces satisfy this interface. Duck typing keeps the
        serialization utility lightweight while constructor validation rejects
        missing, non-finite, mismatched, or degenerate bounds.
        """

        if not hasattr(space, "low") or not hasattr(space, "high"):
            raise TypeError("space must declare low and high bounds")
        return cls(low=getattr(space, "low"), high=getattr(space, "high"), clip=clip)

    def normalize(self, observation: Any) -> npt.NDArray[np.float32]:
        """Normalize one observation or a batch with matching trailing shape."""

        try:
            array = np.asarray(observation, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("observation must be an array of real numbers") from error

        bound_shape = self.low.shape
        if array.ndim < len(bound_shape) or array.shape[-len(bound_shape) :] != bound_shape:
            raise ValueError(
                "observation shape must equal the bounds shape or have matching "
                f"trailing dimensions {bound_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("observation must contain only finite values")

        low = self.low.astype(np.float64, copy=False)
        high = self.high.astype(np.float64, copy=False)
        normalized = 2.0 * (array - low) / (high - low) - 1.0
        if self.clip:
            normalized = np.clip(normalized, *self.OUTPUT_RANGE)
        return normalized.astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        """Return strict-JSON-compatible preprocessing checkpoint metadata."""

        return {
            "version": self.METADATA_VERSION,
            "method": self.METHOD,
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "output_range": list(self.OUTPUT_RANGE),
            "clip": self.clip,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ObservationNormalizer":
        """Restore and validate a normalizer from checkpoint metadata."""

        if not isinstance(metadata, Mapping):
            raise TypeError("normalizer metadata must be a mapping")
        missing = {
            "version",
            "method",
            "low",
            "high",
            "output_range",
            "clip",
        }.difference(metadata)
        if missing:
            raise ValueError(
                "normalizer metadata is missing required fields: "
                + ", ".join(sorted(missing))
            )

        version = metadata["version"]
        if isinstance(version, bool) or version != cls.METADATA_VERSION:
            raise ValueError(
                f"unsupported normalizer metadata version {version!r}; "
                f"expected {cls.METADATA_VERSION}"
            )
        if metadata["method"] != cls.METHOD:
            raise ValueError(
                f"unsupported normalizer method {metadata['method']!r}; "
                f"expected {cls.METHOD!r}"
            )
        try:
            output_range = tuple(float(value) for value in metadata["output_range"])
        except (TypeError, ValueError) as error:
            raise ValueError("normalizer output_range must contain two numbers") from error
        if output_range != cls.OUTPUT_RANGE:
            raise ValueError(
                f"normalizer output_range must be {list(cls.OUTPUT_RANGE)}, "
                f"got {list(output_range)}"
            )
        clip = metadata["clip"]
        if not isinstance(clip, bool):
            raise TypeError("normalizer metadata clip must be a boolean")
        return cls(low=metadata["low"], high=metadata["high"], clip=clip)


_DEFAULT_NORMALIZER = ObservationNormalizer(
    low=OBSERVATION_LOW,
    high=OBSERVATION_HIGH,
    clip=True,
)


def validate_observation(observation: Any) -> npt.NDArray[np.float64]:
    """Convert a MountainCar state to a validated two-element array.

    Values slightly beyond nominal bounds are accepted here so diagnostics can
    still inspect them; normalization performs the documented clipping.
    """

    array = np.asarray(observation, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(
            f"MountainCar observation must have shape (2,), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("MountainCar observation must contain only finite values")
    return array


def normalize_observation(observation: Any) -> npt.NDArray[np.float32]:
    """Map position and velocity linearly to the shared ``[-1, 1]`` scale.

    Clipping makes preprocessing robust to floating-point excursions at the
    declared Gymnasium bounds. Both replay and non-replay DQN conditions should
    use this exact transform for a controlled comparison.
    """

    array = validate_observation(observation)
    return _DEFAULT_NORMALIZER.normalize(array)
