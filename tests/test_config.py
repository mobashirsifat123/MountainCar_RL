"""Tests for loading experiment configuration files."""

from pathlib import Path

import pytest

from mountaincar_rl.utils.config import load_config


def test_load_config_returns_nested_mapping(tmp_path: Path) -> None:
    """A valid YAML mapping should be returned without type coercion surprises."""
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "seed: 7\nexperiment:\n  name: random-smoke\n  episodes: 2\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == {
        "seed": 7,
        "experiment": {"name": "random-smoke", "episodes": 2},
    }


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    """Missing configuration files should produce the standard useful error."""
    missing_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        load_config(missing_path)


def test_load_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    """A parser diagnostic should be preserved for malformed YAML."""
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("experiment: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed.yaml") as error:
        load_config(config_path)

    assert "parse" in str(error.value).lower()


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    """A valid YAML sequence is not a valid top-level experiment config."""
    config_path = tmp_path / "sequence.yaml"
    config_path.write_text("- random\n- dqn\n", encoding="utf-8")

    with pytest.raises(TypeError, match="mapping"):
        load_config(config_path)


def test_config_inheritance_deep_merges_and_replaces_sequences(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "agent:\n  gamma: 0.99\n  hidden: [64, 64]\n"
        "evaluation:\n  seeds: [1000, 1001]\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: parent.yaml\nagent:\n  hidden: [32, 32]\n"
        "experiment:\n  name: child\n",
        encoding="utf-8",
    )

    resolved = load_config(child)

    assert "extends" not in resolved
    assert resolved["agent"] == {"gamma": 0.99, "hidden": [32, 32]}
    assert resolved["evaluation"] == {"seeds": [1000, 1001]}
    assert resolved["experiment"] == {"name": "child"}


def test_config_inheritance_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\nvalue: 1\n", encoding="utf-8")
    second.write_text("extends: first.yaml\nvalue: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        load_config(first)
