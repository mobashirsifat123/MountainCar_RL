"""Construct configured agents while keeping train/evaluation setup identical."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.agents.dqn import DQNAgent
from mountaincar_rl.agents.random_agent import RandomAgent
from mountaincar_rl.agents.sarsa_lambda import SarsaLambdaAgent
from mountaincar_rl.agents.tile_coder import TileCoder
from mountaincar_rl.environments.observation import ObservationNormalizer


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required nested configuration mapping with a useful error."""

    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration key '{key}' must be a mapping.")
    return value


def configured_agent_type(config: Mapping[str, Any]) -> str:
    """Return the declared implementation type used by checkpoint validation."""

    agent_type = require_mapping(config, "agent").get("type")
    if not isinstance(agent_type, str) or not agent_type:
        raise ValueError("Configuration agent.type must be a nonempty string.")
    return agent_type


def _epsilon(config: Mapping[str, Any]) -> tuple[float, float, int]:
    epsilon = config.get("epsilon")
    if not isinstance(epsilon, Mapping):
        raise ValueError("Configuration agent.epsilon must be a mapping.")
    return (
        float(epsilon["start"]),
        float(epsilon["end"]),
        int(epsilon["decay_steps"]),
    )


def _discrete_action_count(env: gym.Env[Any, Any]) -> int:
    action_count = getattr(env.action_space, "n", None)
    if action_count is None:
        raise TypeError("Configured agents require a discrete action space.")
    return int(action_count)


def build_agent(
    config: Mapping[str, Any], env: gym.Env[Any, Any]
) -> Agent:
    """Build one agent from config and environment-declared spaces.

    Neural preprocessing is derived only from ``env.observation_space``. Both
    replay and no-replay DQN pass through this exact factory and therefore share
    architecture and normalization code.
    """

    experiment = require_mapping(config, "experiment")
    agent_config = require_mapping(config, "agent")
    reproducibility = require_mapping(config, "reproducibility")
    agent_type = configured_agent_type(config)
    training_seed = int(experiment["training_seed"])
    action_count = _discrete_action_count(env)

    if agent_type == "random":
        return RandomAgent(env.action_space, seed=training_seed)

    epsilon_start, epsilon_end, epsilon_decay_steps = _epsilon(agent_config)
    if agent_type == "sarsa_lambda":
        tile_config = agent_config.get("tile_coder")
        if not isinstance(tile_config, Mapping):
            raise ValueError("Configuration agent.tile_coder must be a mapping.")
        tiles = tile_config.get("tiles_per_dim", (8, 8))
        if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
            raise ValueError("agent.tile_coder.tiles_per_dim must be a sequence.")
        tile_coder = TileCoder(
            low=env.observation_space.low,
            high=env.observation_space.high,
            num_actions=action_count,
            num_tilings=int(tile_config.get("num_tilings", 8)),
            tiles_per_dim=tuple(int(value) for value in tiles),
        )
        return SarsaLambdaAgent(
            tile_coder=tile_coder,
            alpha=float(agent_config["alpha"]),
            gamma=float(agent_config["gamma"]),
            lambda_=float(agent_config["lambda"]),
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            trace_type=str(agent_config.get("trace_type", "replacing")),
            trace_clip=(
                None
                if agent_config.get("trace_clip") is None
                else float(agent_config["trace_clip"])
            ),
            seed=training_seed,
        )

    if agent_type == "dqn":
        hidden_sizes = agent_config.get("hidden_sizes", (64, 64))
        if not isinstance(hidden_sizes, Sequence) or isinstance(
            hidden_sizes, (str, bytes)
        ):
            raise ValueError("agent.hidden_sizes must be a sequence.")
        normalizer = ObservationNormalizer.from_space(env.observation_space)
        return DQNAgent(
            action_count=action_count,
            observation_normalizer=normalizer,
            hidden_sizes=tuple(int(size) for size in hidden_sizes),
            gamma=float(agent_config["gamma"]),
            learning_rate=float(agent_config["learning_rate"]),
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            gradient_clip_norm=float(agent_config["gradient_clip_norm"]),
            target_update_frequency=int(agent_config["target_update_frequency"]),
            tau=float(agent_config.get("tau", 1.0)),
            use_replay=bool(agent_config["use_replay"]),
            replay_capacity=int(agent_config["replay_capacity"]),
            batch_size=int(agent_config["batch_size"]),
            warmup_steps=int(agent_config["warmup_steps"]),
            update_frequency=int(agent_config["update_frequency"]),
            seed=training_seed,
            device=str(reproducibility.get("device", "cpu")),
        )

    raise ValueError(
        f"Unsupported agent.type {agent_type!r}; expected random, sarsa_lambda, or dqn."
    )


def preprocessing_metadata(agent: Agent) -> dict[str, Any] | None:
    """Extract the preprocessing/feature mapping needed to interpret weights."""

    normalizer = getattr(agent, "observation_normalizer", None)
    if normalizer is not None and hasattr(normalizer, "metadata"):
        return dict(normalizer.metadata())
    tile_coder = getattr(agent, "tile_coder", None)
    if tile_coder is not None and hasattr(tile_coder, "metadata"):
        return {"method": "tile_coding", "tile_coder": dict(tile_coder.metadata())}
    return None

