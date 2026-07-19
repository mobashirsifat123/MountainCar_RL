"""Integration checks for the on-policy action order in the training loop."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mountaincar_rl.agents.sarsa_lambda import SarsaLambdaAgent
from mountaincar_rl.experiments.train import _write_jsonl_atomic
from mountaincar_rl.experiments.training import run_learning_training
from mountaincar_rl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_training_selects_next_action_before_update_and_executes_it(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The sampled SARSA next action must become the next behavior action."""

    config = deepcopy(load_config(ROOT / "configs/smoke/sarsa_lambda.yaml"))
    config["experiment"]["episodes"] = 1
    config["environment"]["max_episode_steps"] = 3
    config["evaluation"]["validation_episode_seeds"] = [1000]
    events: list[tuple[object, ...]] = []
    training_selections = 0
    original_update = SarsaLambdaAgent.update

    def prescribed_action(
        self: SarsaLambdaAgent, observation: object, *, explore: bool = True
    ) -> int:
        nonlocal training_selections
        if not explore:
            return 0
        action = training_selections % self.tile_coder.num_actions
        training_selections += 1
        events.append(("select", action))
        return action

    def recording_update(
        self: SarsaLambdaAgent,
        observation: object,
        action: int,
        reward: float,
        next_observation: object,
        next_action: int | None = None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        events.append(("update", action, next_action, terminated, truncated))
        return original_update(
            self,
            observation,  # type: ignore[arg-type]
            action,
            reward,
            next_observation,  # type: ignore[arg-type]
            next_action,
            terminated=terminated,
            truncated=truncated,
        )

    monkeypatch.setattr(SarsaLambdaAgent, "select_action", prescribed_action)
    monkeypatch.setattr(SarsaLambdaAgent, "update", recording_update)

    run_learning_training(
        config,
        output_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "checkpoints",
        overwrite=False,
        write_jsonl_atomic=_write_jsonl_atomic,
    )

    updates = [event for event in events if event[0] == "update"]
    assert updates == [
        ("update", 0, 1, False, False),
        ("update", 1, 2, False, False),
        # Pure TimeLimit truncation still samples the behavior-policy action
        # used for the required SARSA bootstrap, though collection then ends.
        ("update", 2, 0, False, True),
    ]
    for update_index in (2, 4, 6):
        assert events[update_index - 1][0] == "select"
        assert events[update_index - 1][1] == events[update_index][2]

    transitions = [
        json.loads(line)
        for line in (tmp_path / "run" / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["action"] for row in transitions] == [0, 1, 2]
