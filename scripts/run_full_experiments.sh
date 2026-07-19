#!/usr/bin/env bash
set -euo pipefail

# Locked templates and seeds are resolved by the suite runner. Final-test seeds
# are loaded only inside the runner after every training seed has completed.
uv run --python 3.12 mountaincar-run-suite \
  --output results/processed/final_manifest.json
