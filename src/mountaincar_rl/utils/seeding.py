"""Centralized random-number seeding for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Any, Mapping

import numpy as np


def validate_seed(seed: int) -> int:
    """Validate and return a nonnegative integer seed.

    Boolean values are rejected even though ``bool`` subclasses ``int``.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    validated = int(seed)
    if validated < 0:
        raise ValueError("seed must be nonnegative")
    return validated


def seed_everything(
    seed: int, deterministic_torch: bool = True
) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch and return an independent NumPy RNG.

    NumPy's legacy global RNG accepts only 32-bit seeds, while
    :func:`numpy.random.default_rng` accepts the full validated integer. The
    returned generator should be preferred by new code. PyTorch configuration
    is applied when PyTorch is installed; its absence does not prevent use of
    non-neural baselines.
    """

    validated = validate_seed(seed)
    if not isinstance(deterministic_torch, bool):
        raise TypeError("deterministic_torch must be a boolean")

    os.environ["PYTHONHASHSEED"] = str(validated)
    random.seed(validated)
    np.random.seed(validated % (2**32))

    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        torch_seed = validated % (2**63)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    return np.random.default_rng(validated)


def seed_env(env: Any, seed: int) -> tuple[Any, Mapping[str, Any]]:
    """Seed a Gymnasium environment and both of its spaces, then reset it."""

    validated = validate_seed(seed)
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        seed_method = getattr(space, "seed", None)
        if callable(seed_method):
            seed_method(validated)

    reset_result = env.reset(seed=validated)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise TypeError("env.reset(seed=...) must return (observation, info)")
    observation, info = reset_result
    if not isinstance(info, Mapping):
        raise TypeError("env.reset(seed=...) info must be a mapping")
    return observation, info
