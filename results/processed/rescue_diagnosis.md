# Controlled rescue diagnosis

**Scope:** training and periodic validation artifacts only. No held-out final-test artifact or seed was read. Diagnostics at 6k, 15k, 30k, and 60k interactions correspond to 10%, 25%, 50%, and 100% of the locked 60k DQN budget. Position and velocity summaries are seed means of each seed's trailing-10-episode median; update, Q, and gradient summaries use the same trailing window.

## Failure classification

| Candidate cause | Assessment | Artifact evidence |
|---|---|---|
| Implementation defect | Unlikely | The focused numerical tests cover action mapping, observation normalization, reward, termination/truncation, DQN targets and target copies, replay sampling, SARSA updates, checkpoints, and deterministic evaluation. All passed before rescue runs. |
| Inadequate exploration | Not the primary cause | Slower epsilon decay was rejected on both validation seeds. The rapid-decay reference nevertheless had higher 60k replay coverage and momentum than standard replay, contradicting a simple 'too little coverage' account. |
| Unstable optimization / policy transfer | Most supported | Standard replay briefly reached 1.0 ordinary validation completion for seed 0 at 40k, then returned to zero at later checkpoints. Multiple rescue runs discovered completions or broad replay coverage without producing a stable greedy <=100 policy. |
| Harmful shaping | Not primary | Moderate shaping kept abs(shaping)/abs(environment) below 0.008, expanded coverage in both seeds, and produced ordinary completions, but did not yield greedy <=100 validation success. |
| Insufficient budget | Not sufficient by itself | Extending only the budget to 120k reproduced seed 0's 40k transient best and left seed 1 unsuccessful; neither seed improved after 60k. |
| Evaluation bug | Unlikely | Deterministic evaluation, exact success threshold, original-reward accounting, and non-training evaluation are covered by passing tests. Behavior-policy successes that fail to transfer to greedy validation remain scientifically plausible. |
| Checkpoint-selection bug | Unlikely | The deterministic validation selector retained seed 0's transient 40k checkpoint over later regressions; it did not consult held-out seeds. |

## Locked-run diagnostics

| Condition | epsilon @6k/15k/30k/60k | coverage @60k | max position @60k | max abs velocity @60k | update metric @60k | mean Q @60k | grad norm @60k |
|---|---|---:|---:|---:|---:|---:|---:|
| Replay DQN | 0.715/0.288/0.050/0.050 | 0.4175 | -0.3549 | 0.0199 | 0.0017 | -44.3070 | 0.3571 |
| Shaped replay DQN | 0.715/0.288/0.050/0.050 | 0.5885 | 0.0115 | 0.0306 | 0.0235 | -43.7938 | 0.5054 |
| Rapid-decay replay DQN | 0.050/0.050/0.050/0.050 | 0.5945 | 0.2217 | 0.0424 | 0.0447 | -42.3882 | 0.7761 |
| No-replay DQN | 0.715/0.288/0.050/0.050 | NA | -0.2644 | 0.0228 | 0.1021 | -39.1534 | 6.4858 |
| SARSA(lambda) | 0.715/0.290/0.050/0.050 | NA | 0.5081 | 0.0463 | -0.0267 | -43.5981 | NA |

For DQN, `update metric` is loss; for SARSA(lambda), it is TD error and is therefore not directly comparable. Missing replay coverage for no-replay DQN and SARSA(lambda) is expected.

## Controlled validation interventions

| Intervention | Seed | Best <=100 | Best completion | Median completed steps | Best step | First validation <=100 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| epsilon_slow | 0 | 0.000 | 0.100 | 178.0 | 40000 | NA | 0.4375 |
| epsilon_slow | 1 | 0.000 | 0.000 | NA | 10000 | NA | 0.3200 |
| warmup_long | 0 | 0.000 | 0.000 | NA | 10000 | NA | 0.5600 |
| warmup_long | 1 | 0.000 | 0.000 | NA | 10000 | NA | 0.4325 |
| learning_rate_small | 0 | 0.000 | 0.050 | 193.0 | 30108 | NA | 0.6275 |
| learning_rate_small | 1 | 0.000 | 0.000 | NA | 10000 | NA | 0.4150 |
| target_update_slow | 0 | 0.000 | 0.850 | 153.0 | 60000 | NA | 0.6450 |
| target_update_slow | 1 | 0.000 | 0.000 | NA | 10000 | NA | 0.4525 |
| shaping_moderate | 0 | 0.000 | 1.000 | 168.0 | 30138 | NA | 0.7250 |
| shaping_moderate | 1 | 0.000 | 0.050 | 179.0 | 30197 | NA | 0.7450 |
| budget_long | 0 | 0.000 | 1.000 | 132.5 | 40000 | NA | 0.4450 |
| budget_long | 1 | 0.000 | 0.000 | NA | 10000 | NA | 0.2575 |

## Decision

No intervention achieved nonzero greedy <=100 validation success in both seeds. Under the preregistered rule, no corrected configuration was locked, no five-seed rerun was started, and no new held-out evaluation was performed.

The evidence supports a bounded conclusion: within this implementation, representation, and tested budget, neural value/policy stability and transfer from exploratory behavior to the greedy policy are the leading failure mode. This is not proof of a universal DQN limitation. The original unsuccessful configuration and contrary rapid-decay evidence remain preserved.
