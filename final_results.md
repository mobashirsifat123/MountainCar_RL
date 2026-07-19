# Final result summary

## Task and success definition

The task is Gymnasium `MountainCar-v0`. Environment completion means reaching
the flag within the normal 200-step limit. Assignment success is stricter:
reaching the flag in at most 100 actual environment steps. A completion in
101--200 steps is not assignment success.

## Best measured condition

The strongest aggregate condition was **SARSA(λ)**:

- training seeds: 5;
- held-out episodes per trained policy: 20;
- environment completion within 200 steps: 1.000 [1.000, 1.000];
- assignment success within 100 steps: 0.380 [0.330, 0.420];
- median length among completed held-out episodes: 104.0 steps;
- mean original environment return: -100.940 [-102.840, -99.310].

Intervals are 95% percentile bootstrap intervals over the five training-seed
summaries, not over pooled evaluation episodes. The completed-episode median is
conditional and pooled; failed episodes remain in the rate denominators.

## Evaluation protocol

Training seeds were 0--4. Checkpoints were selected only with validation episode
seeds 1000--1019. Greedy held-out evaluation then used the same 20 disjoint seeds
2000--2019 for every trained policy and always used the original MountainCar
reward, including for the shaped agent.

## Main findings

- **Replay:** Online DQN completed
  0.400 [0.000, 0.800] of held-out episodes versus
  0.180 [0.000, 0.540] for replay DQN; both had
  0.000 [0.000, 0.000] assignment success.
  Replay did not establish a stability or sample-efficiency advantage at the
  locked 60,000-interaction budget. Removing replay also changed batch size and
  temporal decorrelation.
- **Shaping:** Shaped replay completed
  0.430 [0.030, 0.830] and achieved
  0.020 [0.000, 0.060] assignment success.
  Its first training completion was
  19196 [14784, 23146]; 2/5 censored; sparse successes
  and broad seed intervals do not establish a robust final-performance benefit.
- **Rapid-decay intervention:** Rapid-epsilon-decay DQN completed
  0.210 [0.030, 0.390] and achieved
  0.030 [0.000, 0.090] assignment success.
  Aggregate diagnostics did not support the preregistered claim that rapid decay
  systematically trapped the agent in low-momentum valley trajectories.

No condition reliably met the at-most-100-step assignment target.

## Validation-selected best checkpoints

There is no checkpoint selected across training seeds using held-out results.
The five validation-selected checkpoints for the strongest condition are:

- `best_models/sarsa_lambda_seed0/best.pt` (repository source: `checkpoints/full/sarsa_lambda_seed0/best.pt`)
- `best_models/sarsa_lambda_seed1/best.pt` (repository source: `checkpoints/full/sarsa_lambda_seed1/best.pt`)
- `best_models/sarsa_lambda_seed2/best.pt` (repository source: `checkpoints/full/sarsa_lambda_seed2/best.pt`)
- `best_models/sarsa_lambda_seed3/best.pt` (repository source: `checkpoints/full/sarsa_lambda_seed3/best.pt`)
- `best_models/sarsa_lambda_seed4/best.pt` (repository source: `checkpoints/full/sarsa_lambda_seed4/best.pt`)

The exact submitted seed-0 best-checkpoint path is
`best_models/sarsa_lambda_seed0/best.pt`. Best checkpoints for every other
learned condition and training seed are also included under `best_models/`.

## Limitations

The study uses five training seeds, a simple deterministic benchmark, bounded
validation tuning, and a practical 60,000-interaction budget. SARSA and DQN use
different function approximators, so their comparison is not a controlled
on-policy/off-policy causal test. The shaping potential is an interpretable
energy proxy rather than exact mechanical energy. Conclusions may not
generalize to larger or stochastic reinforcement-learning tasks.
