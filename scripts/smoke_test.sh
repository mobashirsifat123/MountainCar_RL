#!/usr/bin/env bash
set -euo pipefail

uv run --python 3.12 pytest
uv run --python 3.12 mountaincar-config-diff \
  --manifest configs/comparison_manifest.yaml \
  --output results/processed/config_differences.json >/dev/null
uv run --python 3.12 mountaincar-train \
  --config configs/smoke/random.yaml \
  --overwrite
uv run --python 3.12 mountaincar-evaluate \
  --config configs/smoke/random.yaml \
  --split validation \
  --overwrite

for config in \
  configs/smoke/sarsa_lambda.yaml \
  configs/smoke/dqn_no_replay.yaml \
  configs/smoke/dqn_replay.yaml \
  configs/smoke/dqn_shaped.yaml \
  configs/smoke/dqn_fast_decay.yaml
do
  uv run --python 3.12 mountaincar-train \
    --config "$config" \
    --overwrite
  uv run --python 3.12 mountaincar-evaluate \
    --config "$config" \
    --split validation \
    --overwrite
done
