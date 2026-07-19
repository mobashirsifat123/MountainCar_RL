# Learning to Build Momentum

A reproducible empirical study of on-policy learning, experience replay, and
potential-based reward shaping in Gymnasium `MountainCar-v0`. All agents are
implemented directly with NumPy and PyTorch; no packaged reinforcement-learning
algorithm is used.

- **Assignment goal:** reach the `MountainCar-v0` flag within **100** actual
  environment steps (assignment success). Environment completion (reaching the
  flag within the normal 200-step limit) is tracked separately. A completion in
  101–200 steps is **not** assignment success.
- **Methods compared:** random policy; tile-coded SARSA(λ) with eligibility
  traces; DQN without replay; DQN with experience replay and a target network;
  replay DQN with potential-based energy-deficit shaping; and a replay DQN with
  deliberately rapid exploration decay as a controlled failure intervention.
- **Best verified result:** tile-coded **SARSA(λ)** — 1.000 [1.000, 1.000]
  completion within 200 steps and **0.380 [0.330, 0.420]** assignment success
  within 100 steps, with a 104-step median completed length and mean original
  return −100.940 [−102.840, −99.310]. Intervals are 95% bootstrap intervals
  over the five training-seed summaries.
- **≤100-step target achieved?** **No.** No condition reliably met the
  100-step target; 0.38 mean success is progress over the random baseline (0.000)
  but not reliable task fulfilment.
- **Training seeds:** 5 (seeds 0–4), with 20 held-out evaluation episodes per
  trained policy (seeds 2000–2019).

![Final performance comparison](results/figures/figure_2_final_performance.png)

**Figure: held-out completion (≤200 steps) and assignment success (≤100 steps)
with uncertainty across the five training seeds.**

```bash
# Installation (Python 3.12; locked environment)
uv sync --frozen --python 3.12 --extra dev

# Quick evaluation of the best SARSA checkpoint on 2 validation episodes
uv run --python 3.12 mountaincar-evaluate \
  --config checkpoints/best/sarsa_lambda_seed0/config.snapshot.json \
  --split validation --episodes 2 \
  --checkpoint checkpoints/best/sarsa_lambda_seed0/best.pt \
  --output-dir results/reproduction/sarsa_lambda_seed0_validation \
  --capture-trajectories
```

## Overview

MountainCar tests whether an agent can learn that reaching a goal to its right
requires first moving away to build momentum. The engine is too weak to climb
the right hill directly, so a successful controller alternates acceleration with
the direction of travel, climbs one slope, reverses, and converts height into
speed until it can cross the goal at x = 0.5. This repository contains the full
training, validation, held-out evaluation, checkpointing, trajectory logging,
statistical aggregation, and plotting workflows as command-line tools, plus the
final report, figures, tables, and compact best checkpoints.

## Task

`MountainCar-v0` has a two-dimensional continuous state (position x, velocity v)
and three discrete actions (accelerate left, coast, accelerate right). An
episode normally lasts at most 200 steps. This study keeps two outcomes
distinct:

- **Environment completion:** reach the flag within the normal 200-step limit.
- **Assignment success:** reach the flag in at most 100 actual environment
  steps.

## Methods

1. **Random policy** — no-learning sanity check.
2. **Tile-coded SARSA(λ)** — 16 overlapping 8×8 tile codings, replacing
   eligibility traces, ε-greedy behavior, γ = 0.99, λ = 0.9. True goal
   termination removes the bootstrap term; time-limit truncation does not.
3. **DQN without replay** — online one-transition updates, batch size one.
4. **DQN with replay** — uniform experience replay (50,000-item buffer, 64-item
   batches after a 1,000-transition warm-up) with a periodically copied target
   network. Removing replay also changes batch size and temporal decorrelation,
   so the replay ablation is not memory-only.
5. **Shaped replay DQN** — replay DQN trained with a bounded potential-based
   energy-deficit shaping term, but evaluated on the original reward.
6. **Rapid-epsilon-decay DQN** — valid replay DQN with ε decaying from 1.0 to
   0.05 over 200 interactions rather than 20,000, intended as a controlled
   failure intervention rather than broken code.

SARSA and DQN use different function approximators (tile coding with traces
versus a neural network), so their comparison is descriptive rather than a
controlled on-policy/off-policy causal test. See `report/report.md` for the full
equations and protocol.

## Main Results

Held-out evaluation used greedy policies and the original MountainCar reward.
Intervals are 95% percentile bootstrap intervals across the five training seeds,
not across the 100 pooled episode records within a condition.

| Condition | Completion ≤200 | Assignment success ≤100 | Median completed steps |
|---|---:|---:|---:|
| Random policy | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | NA |
| SARSA(λ) | 1.000 [1.000, 1.000] | 0.380 [0.330, 0.420] | 104.0 |
| DQN without replay | 0.400 [0.000, 0.800] | 0.000 [0.000, 0.000] | 151.5 |
| DQN with replay | 0.180 [0.000, 0.540] | 0.000 [0.000, 0.000] | 132.0 |
| Shaped DQN with replay | 0.430 [0.030, 0.830] | 0.020 [0.000, 0.060] | 169.0 |
| Rapid-epsilon-decay DQN | 0.210 [0.030, 0.390] | 0.030 [0.000, 0.090] | 189.0 |

No condition reliably met the ≤100-step target. SARSA(λ) was strongest.
Standard replay DQN did not improve on the no-replay DQN at this 60,000
interaction budget. Shaping was associated with occasional earlier discoveries
and a small nonzero final success rate but not a robust improvement. The
rapid-decay condition did not exhibit the preregistered systematic
valley-trapping failure, so that hypothesis was not supported. Full metrics,
learning curves, phase-space analysis, and failure diagnostics are in
`results/tables/final_results.md`, `results/processed/results_summary.md`, the
four figures under `results/figures/`, and `report/report.md`.

## Repository Structure

```text
src/mountaincar_rl/      agents, environments, experiment CLIs, and analysis
tests/                   unit and integration tests
scripts/                 smoke, sweep, full-suite, reproduction, and PDF build
configs/                 base/, full/ (locked final), smoke/, sweeps/, rescue/
report/                  report.md, report.css, and the final PDF
results/figures/         four final figures (PNG + PDF)
results/tables/          final_results.csv and final_results.md
results/processed/       results summary, data-validation report, and manifests
checkpoints/best/        25 compact validation-selected best checkpoints
docs/                    algorithm notes
experiments/             preregistered hypotheses and experiment ledger
```

Raw per-run evidence (`results/raw/`, ~408 MiB) and full checkpoint trees
(`checkpoints/full/`, ~121 MiB) are intentionally excluded from Git due to size;
the committed figures, tables, processed summaries, and `checkpoints/best/` are
sufficient to inspect every reported result.

## Installation

Python 3.10 or newer is required (3.12 recommended). With `uv`:

```bash
uv sync --frozen --python 3.12 --extra dev
```

Without `uv`, use a Python 3.10+ virtual environment:

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Evaluation

Evaluate the published best SARSA seed-0 checkpoint on two validation episodes.
Evaluation is greedy, does not train, and uses the original environment reward:

```bash
uv run --python 3.12 mountaincar-evaluate \
  --config checkpoints/best/sarsa_lambda_seed0/config.snapshot.json \
  --split validation --episodes 2 \
  --checkpoint checkpoints/best/sarsa_lambda_seed0/best.pt \
  --output-dir results/reproduction/sarsa_lambda_seed0_validation \
  --capture-trajectories
```

To evaluate on all 20 held-out final-test seeds instead:

```bash
uv run --python 3.12 mountaincar-evaluate \
  --config checkpoints/best/sarsa_lambda_seed0/config.snapshot.json \
  --split final_test --seeds-config configs/full/final_test.yaml \
  --checkpoint checkpoints/best/sarsa_lambda_seed0/best.pt \
  --output-dir results/reproduction/sarsa_lambda_seed0_final_test \
  --capture-trajectories
```

## Smoke Training

The smoke workflow runs the tests and short versions of all six conditions. It
writes only to smoke output directories and does not touch the final results:

```bash
bash scripts/smoke_test.sh
```

## Full Training

The complete locked five-seed experiment is expensive (five learned conditions ×
five seeds × 60,000 interactions, plus the random baseline) and writes to the
full-result paths. Do not run it merely to regenerate the report:

```bash
bash scripts/run_full_experiments.sh
```

## Regenerating Figures

The analysis command reads only `results/raw/full` and regenerates the final
CSV/Markdown table, results summary, analysis manifest, and four PNG/PDF figures
without any training or evaluation:

```bash
uv run --python 3.12 python -m mountaincar_rl.analysis.make_all_figures
```

Note: `results/raw/full` is excluded from Git due to size. To regenerate figures
in a fresh clone, first run the full training above (or obtain the raw-evidence
archive from the GitHub release). The committed figures and tables are the
canonical artifacts for inspection.

## Testing

```bash
uv run --python 3.12 pytest
```

Tests cover configuration resolution, seeding, environment and time-limit
semantics, tile coding, replay, DQN/SARSA targets, reward shaping, checkpoint
resumption, evaluation isolation, and artifact aggregation.

## Reproducibility

Final training seeds are `0, 1, 2, 3, 4`; validation episode seeds are
`1000`–`1019`; held-out episode seeds are `2000`–`2019` (in
`configs/full/final_test.yaml`). These held-out seeds were not used for tuning
or checkpoint selection. All learned runs used a locked 60,000-environment
interaction budget. Best checkpoints were selected by a deterministic
validation-only rule: assignment-success rate, completion rate, median completed
length, episode-length standard deviation, then earliest checkpoint on an exact
tie. Configuration hashes are in `configs/full/locked_manifest.yaml`. Exact
commands for every step are in `reproduction_commands.txt`.

## Final Models

The 25 compact validation-selected best checkpoints (one per learned condition
and training seed) are under `checkpoints/best/<condition>_seed<N>/`. Each
directory contains `best.pt`, its resolved `config.snapshot.json`, and a
`metadata.json` record (algorithm, seed, configuration hash, validation and
final evaluation metrics, and a checksum). The strongest-condition seed-0
checkpoint is `checkpoints/best/sarsa_lambda_seed0/best.pt`.

## Limitations

The study uses five training seeds, a simple mostly deterministic benchmark,
bounded validation tuning, and a practical 60,000-interaction budget. SARSA and
DQN use different function approximators, so their difference is not a
controlled on-policy/off-policy causal effect. The replay ablation also changes
batch size and temporal correlation. The shaping potential is an interpretable
energy proxy rather than exact mechanical energy. These findings do not
establish a universal algorithm ranking or imply performance on larger or
noisier reinforcement-learning tasks.

## AI-Assistance Disclosure

Coding agents assisted with implementation, testing, analysis, and editing. All
reported experimental metrics were produced by running this repository and were
regenerated from saved raw result files; no result was manually invented. The
author is responsible for understanding, verifying, and defending the methods,
code, and conclusions.

## License

Released under the MIT License; see `LICENSE`.
