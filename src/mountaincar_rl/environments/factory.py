"""Gymnasium environment factory with explicit reproducibility controls."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mountaincar_rl.utils.seeding import seed_env, validate_seed


def make_env(
    *,
    seed: int,
    render_mode: str | None = None,
    max_episode_steps: int = 200,
) -> gym.Env[Any, Any]:
    """Create and seed ``MountainCar-v0``.

    ``max_episode_steps`` is passed directly to :func:`gymnasium.make`, which
    configures the environment's single ``TimeLimit`` wrapper. This avoids
    stacking ambiguous time-limit wrappers. Callers must continue to preserve
    Gymnasium's distinction between ``terminated`` (goal reached) and
    ``truncated`` (including this artificial time limit).

    The factory performs an initial seeded reset and seeds both spaces. A
    caller may reset again with a per-episode seed when using a predefined
    evaluation seed set.
    """

    valid_seed = validate_seed(seed)
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise TypeError("max_episode_steps must be an integer")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if render_mode is not None and not isinstance(render_mode, str):
        raise TypeError("render_mode must be a string or None")

    env = gym.make(
        "MountainCar-v0",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    seed_env(env, valid_seed)
    return env
