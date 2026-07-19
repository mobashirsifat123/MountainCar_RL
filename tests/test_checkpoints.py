"""Tests for the shared versioned checkpoint envelope."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mountaincar_rl.utils.checkpoints import (
    config_fingerprint,
    load_agent_checkpoint,
    save_agent_checkpoint,
)


def test_config_fingerprint_is_order_independent() -> None:
    assert config_fingerprint({"b": 2, "a": {"x": 1}}) == config_fingerprint(
        {"a": {"x": 1}, "b": 2}
    )


def test_agent_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "agent.pt"
    config = {"agent": {"type": "synthetic"}, "seed": 3}
    save_agent_checkpoint(
        path,
        agent_type="synthetic",
        agent_state={
            "weights": torch.tensor([1.0, 2.0]),
            "rng": np.random.default_rng(4).bit_generator.state,
        },
        config=config,
        training_state={"completed_episodes": 7},
        preprocessing={"method": "affine_bounds", "low": [-1.0, -2.0]},
        selection={"score": [0.0, 0.5, -150.0]},
    )

    loaded = load_agent_checkpoint(
        path, expected_agent_type="synthetic", expected_config=config
    )

    torch.testing.assert_close(
        loaded["agent_state"]["weights"], torch.tensor([1.0, 2.0])
    )
    assert loaded["training_state"]["completed_episodes"] == 7
    assert loaded["preprocessing"]["method"] == "affine_bounds"


def test_agent_checkpoint_rejects_config_or_agent_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "agent.pt"
    config = {"agent": {"type": "synthetic"}}
    save_agent_checkpoint(
        path,
        agent_type="synthetic",
        agent_state={},
        config=config,
        training_state={},
    )

    with pytest.raises(ValueError, match="agent_type"):
        load_agent_checkpoint(path, expected_agent_type="other")
    with pytest.raises(ValueError, match="fingerprint"):
        load_agent_checkpoint(path, expected_config={"agent": {"type": "other"}})

