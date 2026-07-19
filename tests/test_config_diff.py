"""Tests for machine-readable resolved condition comparisons."""

from pathlib import Path

from mountaincar_rl.experiments.config_diff import (
    build_comparison_report,
    configuration_differences,
)


ROOT = Path(__file__).resolve().parents[1]


def test_leaf_configuration_differences_are_sorted() -> None:
    differences = configuration_differences(
        {"agent": {"gamma": 0.99, "hidden": [64, 64]}, "seed": 0},
        {"agent": {"gamma": 0.99, "hidden": [32, 32]}, "extra": True},
    )

    assert [difference["key"] for difference in differences] == [
        "agent.hidden",
        "extra",
        "seed",
    ]


def test_project_manifest_exposes_only_intended_shaping_core_difference() -> None:
    report = build_comparison_report(ROOT / "configs/comparison_manifest.yaml")
    weak = report["groups"]["shaping_validation"]["candidates"]["weak"]
    changed_keys = {item["key"] for item in weak["differences"]}

    assert changed_keys == {
        "experiment.condition",
        "experiment.name",
        "logging.checkpoint_dir",
        "logging.output_dir",
        "reward_shaping.beta",
        "reward_shaping.enabled",
    }
    assert "agent.hidden_sizes" not in changed_keys
    assert "evaluation.validation_episode_seeds" not in changed_keys


def test_rapid_decay_changes_epsilon_but_not_other_agent_settings() -> None:
    report = build_comparison_report(ROOT / "configs/comparison_manifest.yaml")
    rapid = report["groups"]["rapid_decay_validation"]["candidates"][
        "rapid_decay"
    ]
    agent_changes = [
        item["key"] for item in rapid["differences"] if item["key"].startswith("agent.")
    ]

    assert agent_changes == ["agent.epsilon.decay_steps"]
