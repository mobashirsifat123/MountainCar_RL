"""MountainCar environment creation, preprocessing, and reward treatment."""

from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.environments.observation import (
    GOAL_POSITION,
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
    ObservationNormalizer,
    normalize_observation,
    validate_observation,
)
from mountaincar_rl.environments.reward_shaping import (
    PotentialBasedRewardShaper,
    RewardComponents,
    build_reward_shaper,
)

__all__ = [
    "GOAL_POSITION",
    "OBSERVATION_HIGH",
    "OBSERVATION_LOW",
    "ObservationNormalizer",
    "PotentialBasedRewardShaper",
    "RewardComponents",
    "build_reward_shaper",
    "make_env",
    "normalize_observation",
    "validate_observation",
]
