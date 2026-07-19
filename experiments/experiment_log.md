# Experiment log

This append-only log records executed empirical runs. Setup commands and software
verification belong in `PROGRESS.md`; scientific runs, including failures, belong
here. No final experiments have been run in phase 1.

## 2026-07-18 — intervention smoke runs

The shaped replay-DQN and fixed rapid-decay replay-DQN configurations each ran
for three 50-step training episodes at training seed 0, with periodic validation
on seeds 1000 and 1001. Both train and checkpoint-backed validation commands
completed. Neither smoke condition completed an episode; these short runs are
mechanical checks only. The shaped run's generated mean absolute
shaping/environment-reward ratio was `0.00503473753178782`, below the
preregistered `0.5` numeric-dominance threshold. The rapid schedule reached
epsilon `0.1` after eight interactions, before replay warm-up at interaction 16.
All reward components, epsilon values, replay coverage, kinematic extrema,
valley labels, and phase-space trajectories were present in raw artifacts.

## 2026-07-18 — validation-only seed-0 interventions

Executed `configs/sweeps/shaping/{no_shaping,weak,moderate,strong}.yaml` and
`configs/sweeps/dqn_fast_decay.yaml`. Every run used training seed 0, validation
seeds 1000--1019, 300 training episodes, and original-objective checkpoint
selection. No held-out seed was loaded or evaluated. The table below is copied
from the generated artifact only for readability; the authoritative
machine-derived record is `results/processed/intervention_validation_seed0.json`.

| Condition | First validation completion interaction | First validation <=100 interaction | Selected completion rate | Selected <=100 rate | Selected median completed steps | Mean original return | Absolute shaping ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| no shaping | 15,000 | censored | 0.10 | 0.00 | 195.0 | -199.50 | 0.000000 |
| weak (`beta=0.1`) | 15,000 | censored | 1.00 | 0.00 | 179.5 | -179.55 | 0.001107 |
| moderate (`beta=0.5`) | 15,000 | 44,499 | 1.00 | 0.05 | 124.0 | -139.40 | 0.005966 |
| strong (`beta=1.0`) | 10,000 | censored | 1.00 | 0.00 | 128.0 | -127.70 | 0.012866 |
| rapid epsilon decay | 4,748 | 9,448 | 1.00 | 0.10 | 112.0 | -119.85 | 0.000000 |

None of the shaping levels crossed the numeric reward-dominance threshold.
Moderate shaping was the only shaping setting with a seed-0 validation
assignment success, while strong shaping discovered ordinary completion earlier.
This single seed is not used to claim reliability or freeze a final setting.
Behavioral reward-seeking review remains part of the multi-seed phase-space
analysis.

The rapid-decay outcome is contrary to H6's intended failure mechanism at seed
0: it had final replay-grid coverage `0.7425`, only 6/300 valley-retained
episodes, and better selected validation outcomes than the standard schedule.
The schedule will not be redesigned after this observation. H6 remains a fixed
multi-seed hypothesis and may be falsified; no condition is called failed merely
because it was intended to fail.

## Bounded validation sweep — 2026-07-18T21:41:01.719420+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/tilings_16.yaml` | 0.100 | 1.000 | 106.0 | 4.912 | 60000 | 5.92 |
| 2 | `configs/sweeps/sarsa/alpha_low.yaml` | 0.100 | 1.000 | 106.5 | 13.668 | 60000 | 5.69 |
| 3 | `configs/sweeps/sarsa/base.yaml` | 0.100 | 1.000 | 108.5 | 6.296 | 60000 | 5.83 |
| 4 | `configs/sweeps/sarsa/decay_long.yaml` | 0.100 | 1.000 | 112.0 | 24.753 | 60000 | 5.80 |
| 5 | `configs/sweeps/sarsa/lambda_high.yaml` | 0.100 | 0.950 | 107.0 | 25.109 | 60000 | 5.81 |
| 6 | `configs/sweeps/sarsa/resolution_12.yaml` | 0.050 | 1.000 | 113.0 | 5.578 | 60000 | 5.95 |

## Bounded validation sweep — 2026-07-18T21:42:31.440960+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/base.yaml` | 0.100 | 1.000 | 108.5 | 6.296 | 60000 | 5.79 |

## Bounded validation sweep — 2026-07-18T21:42:38.441261+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/alpha_low.yaml` | 0.100 | 1.000 | 106.5 | 13.668 | 60000 | 5.76 |

## Bounded validation sweep — 2026-07-18T21:42:45.435105+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/lambda_high.yaml` | 0.100 | 0.950 | 107.0 | 25.109 | 60000 | 5.89 |

## Bounded validation sweep — 2026-07-18T21:42:52.663546+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/decay_long.yaml` | 0.100 | 1.000 | 112.0 | 24.753 | 60000 | 6.10 |

## Bounded validation sweep — 2026-07-18T21:42:59.863936+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/resolution_12.yaml` | 0.050 | 1.000 | 113.0 | 5.578 | 60000 | 6.00 |

## Bounded validation sweep — 2026-07-18T21:43:07.018430+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/sarsa/tilings_16.yaml` | 0.100 | 1.000 | 106.0 | 4.912 | 60000 | 6.07 |

## Bounded validation sweep — 2026-07-18T21:44:34.329166+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/base.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 30.44 |

## Bounded validation sweep — 2026-07-18T21:45:05.591271+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/decay_long.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 29.97 |

## Bounded validation sweep — 2026-07-18T21:45:40.510450+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/lr_high.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 33.57 |

## Bounded validation sweep — 2026-07-18T21:45:46.590014+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/lr_high.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 33.81 |

## Bounded validation sweep — 2026-07-18T21:46:15.649746+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/target_slow.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 33.64 |

## Bounded validation sweep — 2026-07-18T21:46:25.609996+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/target_slow.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 33.41 |

## Bounded validation sweep — 2026-07-18T21:46:39.693845+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/update_2.yaml` | 0.000 | 1.000 | 132.5 | 1.526 | 60000 | 22.45 |

## Bounded validation sweep — 2026-07-18T21:46:49.275025+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/update_2.yaml` | 0.000 | 1.000 | 132.5 | 1.526 | 60000 | 22.19 |

## Bounded validation sweep — 2026-07-18T21:47:14.605325+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/warmup_long.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 33.50 |

## Bounded validation sweep — 2026-07-18T21:47:23.576859+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/warmup_long.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 32.89 |

## Bounded validation sweep — 2026-07-18T21:47:43.007760+08:00

Generated from actual run artifacts; final-test seeds were rejected by code.

| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/sweeps/dqn/warmup_long.yaml` | 0.000 | 0.000 | None | 0.000 | 60000 | 38.51 |

## Locked full suite and held-out evaluation — 2026-07-18

All six configurations were locked after validation in
`configs/full/locked_manifest.yaml`. Every planned training seed (`0`--`4`)
completed. Original-objective held-out evaluation used the preregistered 20
episode seeds `2000`--`2019` after training and validation checkpoint selection.
Generated provenance and outcomes are in `results/processed/final_manifest.json`
and `results/processed/final_results.json`.

The 60,000-interaction neural conditions did not reliably satisfy the at-most-
100-step target. SARSA(lambda) had mean held-out assignment success 0.380;
shaped replay and rapid-decay DQN were 0.020 and 0.030 respectively. The rapid
condition consequently did not uniformly exhibit the hypothesized failure and
was not redesigned after this contrary observation.

## Controlled rescue 1/6 — slower epsilon decay — 2026-07-19

**Hypothesis:** retaining exploration longer may prevent premature commitment
and improve momentum discovery. **Only scientific difference:** epsilon decay
steps `20,000 -> 40,000` relative to locked replay DQN. Training seeds were 0
and 1; validation episode seeds were 1000--1019. No held-out seed was loaded.

Commands:

```bash
uv run --python 3.12 mountaincar-train --config configs/rescue/epsilon_slow_seed0.yaml
uv run --python 3.12 mountaincar-train --config configs/rescue/epsilon_slow_seed1.yaml
```

| Seed | Best <=100 | Best completion | Median completed steps | First validation completion | First validation <=100 | Final replay coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.100 | 178.0 | 40,000 | censored | 0.4375 |
| 1 | 0.000 | 0.000 | NA | censored | censored | 0.3200 |

Conclusion: rejected. Neither seed achieved strict validation success; seed 0
was worse than the locked reference's 1.0 completion and seed 1 remained at the
zero-completion reference floor. Longer exploration alone was not a repeatable
fix.

## Controlled rescue 2/6 — longer replay warm-up — 2026-07-19

**Hypothesis:** collecting a more diverse buffer before optimization may reduce
early correlated/low-energy updates. **Only scientific difference:** replay
warm-up `1,000 -> 5,000` transitions. Seeds and validation protocol were
unchanged; no held-out seed was loaded.

Commands:

```bash
uv run --python 3.12 mountaincar-train --config configs/rescue/warmup_long_seed0.yaml
uv run --python 3.12 mountaincar-train --config configs/rescue/warmup_long_seed1.yaml
```

| Seed | Best <=100 | Best completion | Median completed steps | First training completion | First validation completion | Final replay coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | NA | 29,164 | censored | 0.5600 |
| 1 | 0.000 | 0.000 | NA | censored | censored | 0.4325 |

Conclusion: rejected. Increased coverage and one seed-0 behavior-policy
completion did not transfer to any scheduled greedy validation completion, and
neither seed achieved strict success. Warm-up length alone was not a fix.

## Controlled rescue 3/6 — smaller learning rate — 2026-07-19

**Hypothesis:** smaller optimizer steps may reduce policy/value instability.
**Only scientific difference:** Adam learning rate `0.0005 -> 0.00025`. Seeds
and validation protocol were unchanged; no held-out seed was loaded.

Commands:

```bash
uv run --python 3.12 mountaincar-train --config configs/rescue/learning_rate_small_seed0.yaml
uv run --python 3.12 mountaincar-train --config configs/rescue/learning_rate_small_seed1.yaml
```

| Seed | Best <=100 | Best completion | Median completed steps | First training completion | First validation completion | Final replay coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.050 | 193.0 | 16,168 | 30,108 | 0.6275 |
| 1 | 0.000 | 0.000 | NA | censored | censored | 0.4150 |

Conclusion: rejected. Seed 0 explored broadly and sometimes completed during
training, but greedy validation completion was only 0.05; seed 1 remained at
zero. Halving the learning rate did not produce repeatable strict success.

## Controlled rescue 4/6 — slower target synchronization — 2026-07-19

**Hypothesis:** a less frequently copied target network may provide a more
stationary bootstrap target. **Only scientific difference:** hard target copy
interval `500 -> 1,000` optimizer steps. Seeds and validation protocol were
unchanged; no held-out seed was loaded.

Commands:

```bash
uv run --python 3.12 mountaincar-train --config configs/rescue/target_update_slow_seed0.yaml
uv run --python 3.12 mountaincar-train --config configs/rescue/target_update_slow_seed1.yaml
```

| Seed | Best <=100 | Best completion | Median completed steps | First training completion | First validation completion | Final replay coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.850 | 153.0 | 21,160 | 60,000 | 0.6450 |
| 1 | 0.000 | 0.000 | NA | censored | censored | 0.4525 |

Conclusion: rejected. Seed 0 improved ordinary completion at the final
validation point but did not achieve the strict target, while seed 1 remained
at zero completion. The change therefore failed the preregistered repeatability
criterion and did not justify a corrected lock.

## Controlled rescue 5/6 — moderate energy-deficit shaping — 2026-07-19

**Hypothesis:** a moderate potential coefficient may make useful momentum states
easier to learn without dominating the original reward. **Only scientific
difference:** shaping coefficient `beta: 0.0 -> 0.5`; the energy-proxy
coefficient remained `eta: 0.5`. Seeds and validation protocol were unchanged;
no held-out seed was loaded.

Commands:

```bash
uv run --python 3.12 mountaincar-train --config configs/rescue/shaping_moderate_seed0.yaml
uv run --python 3.12 mountaincar-train --config configs/rescue/shaping_moderate_seed1.yaml
```

| Seed | Best <=100 | Best completion | Median completed steps | First training <=100 | First validation completion | Final replay coverage | abs(shaping)/abs(environment) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 168.0 | censored | 20,183 | 0.7250 | 0.00620 |
| 1 | 0.000 | 0.050 | 179.0 | 59,161 | 30,197 | 0.7450 | 0.00702 |

Conclusion: rejected. Shaping substantially expanded replay coverage and both
seeds discovered ordinary completions, while its absolute contribution stayed
below 0.8% of the environmental reward magnitude. However, neither seed
achieved the strict target under greedy validation. The isolated seed-1 strict
training episode is behavior-policy evidence, not model-selection evidence, so
this setting did not meet the repeatability rule.

## Controlled rescue 6/6 — longer interaction budget — 2026-07-19

**Hypothesis:** the locked replay DQN may simply need more samples to consolidate
the transient successful behavior. **Only scientific difference:** maximum
environment interactions `60,000 -> 120,000` (with the episode ceiling raised
only so that the interaction budget could be reached). Seeds and validation
protocol were unchanged; no held-out seed was loaded.

Commands:

```bash
uv run --python 3.12 mountaincar-train --config configs/rescue/budget_long_seed0.yaml
uv run --python 3.12 mountaincar-train --config configs/rescue/budget_long_seed1.yaml
```

| Seed | Best <=100 | Best completion | Median completed steps | Best checkpoint step | First validation <=100 | Final replay coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 132.5 | 40,000 | censored | 0.4450 |
| 1 | 0.000 | 0.000 | NA | 10,000 | censored | 0.2575 |

Conclusion: rejected. Seed 0 deterministically reproduced the locked run's
transient best checkpoint at 40,000 interactions and did not improve during the
additional 80,000 interactions. Seed 1 remained unsuccessful through 120,000.
More interactions alone did not produce strict success or repeatable validation
improvement.

## Controlled rescue decision — 2026-07-19

None of the six one-variable interventions met the preregistered correction
criterion: both validation seeds had to improve and both had to show nonzero
greedy success within 100 steps. Therefore no corrected configuration was
locked, no five-seed retraining was initiated, and no held-out final-test seed
was used. The original unsuccessful replay DQN remains preserved as the
reference failure rather than being overwritten by a post-hoc configuration.
