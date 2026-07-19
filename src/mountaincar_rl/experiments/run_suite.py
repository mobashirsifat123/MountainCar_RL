"""Run the locked five-seed training and isolated held-out evaluation suite."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from mountaincar_rl.experiments.evaluate import run_evaluation
from mountaincar_rl.experiments.train import run_random_training, _write_jsonl_atomic
from mountaincar_rl.experiments.training import run_learning_training
from mountaincar_rl.utils.config import load_config
from mountaincar_rl.utils.checkpoints import config_fingerprint
from mountaincar_rl.utils.logging import write_json


SEEDS = (0, 1, 2, 3, 4)
TEMPLATES = {
    "random": Path("configs/full/random.yaml"),
    "sarsa_lambda": Path("configs/full/sarsa_lambda.yaml"),
    "dqn_no_replay": Path("configs/full/dqn_no_replay.yaml"),
    "dqn_replay": Path("configs/full/dqn_replay.yaml"),
    "dqn_replay_shaped": Path("configs/full/dqn_shaped.yaml"),
    "dqn_replay_fast_decay": Path("configs/full/dqn_fast_decay.yaml"),
}


def _config_for_seed(template: dict[str, Any], condition: str, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["experiment"]["training_seed"] = seed
    config["experiment"]["condition"] = condition
    config["experiment"]["name"] = f"{condition}_full_seed{seed}"
    config["logging"]["output_dir"] = f"results/raw/full/{condition}_seed{seed}"
    if config["agent"]["type"] != "random":
        config["logging"]["checkpoint_dir"] = f"checkpoints/full/{condition}_seed{seed}"
    return config


def run_suite(*, output: Path, conditions: list[str] | None = None,
              seeds: list[int] | None = None, evaluate: bool = True,
              train: bool = True) -> dict[str, Any]:
    """Train every locked condition for all seeds, then evaluate on held-out seeds."""
    final_seed_config = load_config("configs/full/final_test.yaml")
    test_seeds = [int(seed) for seed in final_seed_config["episode_seeds"]]
    if set(test_seeds) & set(SEEDS):
        raise ValueError("Final-test seeds overlap training seeds")
    manifest: dict[str, Any] = {"schema_version": 1, "training_seeds": list(SEEDS),
                                "final_test_episode_seeds": test_seeds, "conditions": []}
    selected_conditions = list(TEMPLATES) if conditions is None else conditions
    selected_seeds = list(SEEDS) if seeds is None else [int(seed) for seed in seeds]
    unknown_seeds = sorted(set(selected_seeds) - set(SEEDS))
    if unknown_seeds:
        raise ValueError(f"Seeds must be drawn from {list(SEEDS)}: {unknown_seeds}")
    unknown = sorted(set(selected_conditions) - set(TEMPLATES))
    if unknown:
        raise ValueError(f"Unknown condition(s): {unknown}")
    for condition in selected_conditions:
        template_path = TEMPLATES[condition]
        template = load_config(template_path)
        template_hash = config_fingerprint(template)
        condition_row: dict[str, Any] = {"condition": condition, "template": str(template_path),
                                         "template_hash": template_hash, "seeds": []}
        for seed in selected_seeds if train else []:
            config = _config_for_seed(template, condition, seed)
            config_hash = config_fingerprint(config)
            output_dir = Path(config["logging"]["output_dir"])
            if config["agent"]["type"] == "random":
                train_summary = run_random_training(config, output_dir=output_dir, overwrite=True)
                checkpoint = None
            else:
                train_summary = run_learning_training(
                    config, output_dir=output_dir,
                    checkpoint_dir=Path(config["logging"]["checkpoint_dir"]),
                    overwrite=True, write_jsonl_atomic=_write_jsonl_atomic,
                )
                if train_summary.get("status") != "complete":
                    raise RuntimeError(f"Incomplete final training run: {condition} seed {seed}")
                checkpoint = Path(config["logging"]["checkpoint_dir"]) / "best.pt"
            condition_row["seeds"].append({"seed": seed, "config_hash": config_hash,
                                           "training_summary": train_summary,
                                           "checkpoint": None if checkpoint is None else str(checkpoint)})
        if not train:
            for seed in selected_seeds:
                config = _config_for_seed(template, condition, seed)
                run_dir = Path(config["logging"]["output_dir"])
                summary_path = run_dir / "summary.json"
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Missing completed training summary: {summary_path}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("status", "complete") != "complete":
                    raise RuntimeError(f"Training is not complete: {summary_path}")
                checkpoint = None if config["agent"]["type"] == "random" else Path(config["logging"]["checkpoint_dir"]) / "best.pt"
                condition_row["seeds"].append({"seed": seed, "config_hash": config_fingerprint(config),
                                               "training_summary": summary,
                                               "checkpoint": None if checkpoint is None else str(checkpoint)})
        # Held-out evaluation starts only after all selected training seeds for this locked condition.
        for seed in selected_seeds if evaluate else []:
            config = _config_for_seed(template, condition, seed)
            eval_dir = Path(config["logging"]["output_dir"]) / "final_test"
            checkpoint = None if config["agent"]["type"] == "random" else Path(config["logging"]["checkpoint_dir"]) / "best.pt"
            evaluation = run_evaluation(
                config, output_dir=eval_dir, split="final_test", episodes=None,
                overwrite=True, checkpoint=checkpoint, episode_seeds_override=test_seeds,
                capture_trajectories_override=True,
            )
            for row in condition_row["seeds"]:
                if row["seed"] == seed:
                    row["final_test_summary"] = evaluation
                    break
        manifest["conditions"].append(condition_row)
    write_json(output, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/processed/final_manifest.json"))
    parser.add_argument("--condition", action="append", choices=sorted(TEMPLATES),
                        help="Run only these conditions; repeat for a resumable bounded invocation.")
    parser.add_argument("--seed", action="append", type=int,
                        help="Run only these training seeds; repeat for a resumable invocation.")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Train only; use after preflight or when batching long runs.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate existing locked checkpoints without rerunning training.")
    args = parser.parse_args()
    if args.skip_evaluation and args.evaluate_only:
        raise ValueError("--skip-evaluation and --evaluate-only are mutually exclusive")
    report = run_suite(output=args.output, conditions=args.condition, seeds=args.seed,
                       evaluate=not args.skip_evaluation, train=not args.evaluate_only)
    print(json.dumps({"conditions": len(report["conditions"]), "training_seeds": list(SEEDS),
                      "final_test_episodes": len(report["final_test_episode_seeds"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
