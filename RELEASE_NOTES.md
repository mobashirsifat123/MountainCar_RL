## MountainCar RL Recruitment Submission v1.0

### Objective

Reach the `MountainCar-v0` flag within 100 environment steps, while separately
tracking ordinary completion within the environment's 200-step time limit.

### Methods

- Tile-coded SARSA(lambda)
- DQN without replay
- DQN with experience replay
- Potential-shaped replay DQN
- Controlled rapid-exploration-decay failure intervention
- Random-policy sanity check

### Main verified result

No condition reliably met the at-most-100-step target. The strongest measured
condition was SARSA(lambda), evaluated over five training seeds and 20 held-out
episodes per seed. It completed 1.000 [1.000, 1.000] of episodes within 200
steps, achieved 0.380 [0.330, 0.420] assignment success within 100 steps, and
had a pooled median completed length of 104.0 steps. Intervals are 95% bootstrap
intervals over training-seed summaries.

### Included materials

- Complete NumPy/PyTorch implementation (`src/mountaincar_rl/`)
- 133 unit and integration tests (`tests/`)
- Locked final configurations and configuration hashes (`configs/full/`)
- 25 compact validation-selected best checkpoints (`checkpoints/best/`)
- Markdown and six-page PDF report (`report/`)
- Four essential figures (`results/figures/`) and final result tables
  (`results/tables/`)
- Processed summaries, data-validation report, and manifests
  (`results/processed/`)
- Preregistered hypotheses and experiment ledger (`experiments/`)
- Formal scientific audit (`AUDIT_REPORT.md`)
- Exact reproduction commands (`reproduction_commands.txt`, `README.md`)

Raw per-run evidence (~408 MiB under `results/raw/`) and full checkpoint trees
(~121 MiB under `checkpoints/full/`) are excluded from Git due to size; the
committed figures, tables, processed summaries, and compact best checkpoints are
sufficient to inspect every reported result.

### Reproducibility

```bash
uv sync --frozen --python 3.12 --extra dev   # install
uv run --python 3.12 pytest                   # 133 tests
bash scripts/smoke_test.sh                    # all six conditions, short budget
uv run --python 3.12 mountaincar-evaluate \   # evaluate a published best checkpoint
  --config checkpoints/best/sarsa_lambda_seed0/config.snapshot.json \
  --split validation --episodes 2 \
  --checkpoint checkpoints/best/sarsa_lambda_seed0/best.pt \
  --output-dir results/reproduction/sarsa_lambda_seed0_validation \
  --capture-trajectories
```

A fresh clean clone (tracked files only) passed installation, all 133 tests, the
six-condition smoke workflow, and best-checkpoint evaluation. Regenerating
figures from raw requires `results/raw/full/` (excluded from Git); obtain it by
running `bash scripts/run_full_experiments.sh` or from the release archive, then
`uv run --python 3.12 python -m mountaincar_rl.analysis.make_all_figures`.

### Limitations

The study uses five training seeds, a simple mostly deterministic benchmark,
bounded validation tuning, and 60,000 interactions per learned final run. SARSA
and DQN use different function approximators, so their difference is not a
controlled on-policy/off-policy causal effect. The energy quantity is an
interpretable proxy, not exact physical energy. Results do not establish a
universal algorithm ranking.
