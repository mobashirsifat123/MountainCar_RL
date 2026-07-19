"""Small, dependency-light statistical summaries used by experiment runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize episode records without conflating the two success definitions.

    Successful-step summaries are conditional on environment completion. When no
    episode completes, those fields are ``None`` rather than an invented sentinel.
    """

    if not episodes:
        raise ValueError("At least one episode is required to compute a summary.")

    completed = np.asarray([bool(row["environment_completion"]) for row in episodes])
    assignment = np.asarray([bool(row["assignment_success"]) for row in episodes])
    returns = np.asarray([float(row["episode_return"]) for row in episodes], dtype=float)
    steps = np.asarray([int(row["steps"]) for row in episodes], dtype=int)
    successful_steps = steps[completed]

    return {
        "episodes": len(episodes),
        "environment_completion_rate": float(completed.mean()),
        "assignment_success_rate": float(assignment.mean()),
        "mean_steps_among_completions": (
            float(successful_steps.mean()) if successful_steps.size else None
        ),
        "median_steps_among_completions": (
            float(np.median(successful_steps)) if successful_steps.size else None
        ),
        "mean_episode_return": float(returns.mean()),
    }


def bootstrap_mean_confidence_interval(
    seed_level_values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 17_071,
) -> dict[str, float | int]:
    """Return a percentile bootstrap CI over independent training-seed values.

    Resampling operates at the training-seed level, never at the episode level,
    so repeated evaluation episodes from one learned policy are not treated as
    independent agent replications.
    """

    values = np.asarray(seed_level_values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two scalar training-seed values are required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if resamples < 1:
        raise ValueError("resamples must be positive.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return {
        "n_training_seeds": int(values.size),
        "mean": float(values.mean()),
        "confidence": float(confidence),
        "lower": float(lower),
        "upper": float(upper),
        "resamples": int(resamples),
    }

