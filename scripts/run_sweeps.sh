#!/usr/bin/env bash
set -euo pipefail

# Validation-only seed-0 intervention trials. This script never reads the
# held-out final-test seed registry.
for config in \
  configs/sweeps/shaping/no_shaping.yaml \
  configs/sweeps/shaping/weak.yaml \
  configs/sweeps/shaping/moderate.yaml \
  configs/sweeps/shaping/strong.yaml \
  configs/sweeps/dqn_fast_decay.yaml
do
  uv run --python 3.12 mountaincar-train \
    --config "$config" \
    --overwrite
done

uv run --python 3.12 mountaincar-config-diff \
  --manifest configs/comparison_manifest.yaml \
  --output results/processed/config_differences.json

uv run --python 3.12 mountaincar-aggregate-interventions \
  --run-dir results/raw/validation/shaping_none_seed0 \
  --run-dir results/raw/validation/shaping_weak_seed0 \
  --run-dir results/raw/validation/shaping_moderate_seed0 \
  --run-dir results/raw/validation/shaping_strong_seed0 \
  --run-dir results/raw/validation/dqn_replay_fast_decay_seed0 \
  --output results/processed/intervention_validation_seed0.json
