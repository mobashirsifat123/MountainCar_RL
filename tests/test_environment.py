"""Fast contract tests for the MountainCar environment factory."""

import numpy as np

from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.environments.observation import normalize_observation


def test_make_env_repeats_same_seed_state_and_space_samples() -> None:
    """Equal seeds should control reset, action-space, and observation-space RNGs."""
    first = make_env(seed=123)
    second = make_env(seed=123)
    try:
        first_observation, first_info = first.reset()
        second_observation, second_info = second.reset()

        np.testing.assert_array_equal(first_observation, second_observation)
        assert first_info == second_info
        assert [first.action_space.sample() for _ in range(8)] == [
            second.action_space.sample() for _ in range(8)
        ]
        for _ in range(3):
            np.testing.assert_array_equal(
                first.observation_space.sample(), second.observation_space.sample()
            )
    finally:
        first.close()
        second.close()


def test_step_preserves_terminated_and_truncated_fields() -> None:
    """The factory must retain Gymnasium's five-value step API."""
    env = make_env(seed=9)
    try:
        env.reset()
        observation, reward, terminated, truncated, info = env.step(1)

        assert observation.shape == (2,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_mountaincar_action_mapping_and_original_reward_are_numerically_correct() -> None:
    """Actions 0/1/2 push left/neutral/right and every step returns -1."""

    env = make_env(seed=9)
    try:
        velocities: list[float] = []
        rewards: list[float] = []
        for action in (0, 1, 2):
            env.unwrapped.state = np.asarray([-0.5, 0.0], dtype=np.float32)
            observation, reward, terminated, truncated, _ = env.step(action)
            velocities.append(float(observation[1]))
            rewards.append(float(reward))
            assert terminated is False
            assert truncated is False
        assert velocities[0] < velocities[1] < velocities[2]
        np.testing.assert_allclose(np.diff(velocities), [0.001, 0.001], atol=1e-7)
        np.testing.assert_array_equal(rewards, [-1.0, -1.0, -1.0])
    finally:
        env.close()


def test_custom_time_limit_truncates_without_environment_termination() -> None:
    """A one-step artificial horizon is truncation, not a true terminal state."""
    env = make_env(seed=11, max_episode_steps=1)
    try:
        env.reset()
        _, _, terminated, truncated, _ = env.step(1)

        assert terminated is False
        assert truncated is True
    finally:
        env.close()


def test_normalize_observation_maps_bounds_to_unit_box() -> None:
    """MountainCar bounds should map component-wise to approximately [-1, 1]."""
    low = np.array([-1.2, -0.07], dtype=np.float32)
    high = np.array([0.6, 0.07], dtype=np.float32)
    midpoint = (low + high) / 2.0

    normalized_low = normalize_observation(low)
    normalized_high = normalize_observation(high)
    normalized_midpoint = normalize_observation(midpoint)

    assert normalized_low.dtype == np.float32
    assert normalized_high.dtype == np.float32
    assert normalized_midpoint.dtype == np.float32
    np.testing.assert_allclose(normalized_low, -np.ones(2), atol=1e-6)
    np.testing.assert_allclose(normalized_high, np.ones(2), atol=1e-6)
    np.testing.assert_allclose(normalized_midpoint, np.zeros(2), atol=1e-6)
