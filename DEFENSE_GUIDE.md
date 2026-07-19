# MountainCar Project Defense Guide

Use this guide to explain the checked-in study, not an idealized version of it.
The authoritative numbers are in `results/tables/final_results.md`; the locked
protocol is in `configs/full/locked_manifest.yaml`.

## 1. Two-minute explanation

This project asks an agent to solve Gymnasium's `MountainCar-v0`. The state is
position and velocity, the actions are accelerate left, coast, or accelerate
right, and the standard episode limit is 200 steps. I tracked two outcomes:
ordinary environment completion within 200 steps and the stricter assignment
target of reaching the flag within 100 steps.

The car cannot climb the right hill by continuously accelerating right because
its engine is too weak. It has to move away from the goal, climb the left side,
reverse, and use the resulting velocity to make progressively larger swings.
That is why momentum discovery, not simply reducing distance to the flag, is the
central learning problem.

I implemented a random baseline, tile-coded SARSA(lambda), DQN with online
one-transition updates, DQN with uniform replay, replay DQN with potential-based
energy-deficit shaping, and replay DQN with epsilon decaying much too quickly.
The primary algorithms use NumPy and PyTorch directly. Each learned condition
received 60,000 interactions for each of five training seeds. Validation seeds
1000--1019 selected checkpoints; greedy final evaluation used disjoint seeds
2000--2019, 20 episodes per trained policy, with the original reward.

SARSA(lambda) was strongest: completion was 1.000 [1.000, 1.000], assignment
success was 0.380 [0.330, 0.420], and the median completed length was 104.0 steps.
That is meaningful learning, but not reliable satisfaction of the 100-step
target. Replay DQN had lower completion than online DQN, 0.180 versus 0.400, and
both had zero assignment success, so replay did not establish the predicted
benefit at this budget. Shaped replay reached 0.020 assignment success, but the
events were sparse and did not establish a robust improvement. The rapid-decay
condition also did not validate the planned valley-trapping explanation: it
reached 0.030 assignment success, and aggregate coverage and momentum were often
higher than standard replay. The main limitations are only five training seeds,
a 60,000-step budget, and the fact that SARSA and DQN use different function
approximators, so their difference is not a clean on-policy/off-policy causal
comparison.

## 2. Technical concepts as implemented

- **State and actions.** `MountainCar-v0` exposes `(position, velocity)` and
  actions `0=accelerate left`, `1=coast`, and `2=accelerate right`.
- **Original reward.** The environment supplies `-1` per step, including the
  final transition. Earlier completion therefore produces a less-negative
  return. Final evaluation always logs this original reward.
- **SARSA(lambda).** `SarsaLambdaAgent.update` uses the action actually selected
  by the current epsilon-greedy policy in its next-state target. The training
  loop selects that action before updating and then executes the same action on
  the next step, making the update genuinely on-policy.
- **Eligibility traces.** A trace records recently active state-action features
  so one TD error can update more than the current transition. The locked agent
  uses replacing traces, lambda `0.9`, gamma `0.99`, and clears traces at every
  collected episode boundary.
- **Tile coding.** Sixteen offset `8 x 8` tilings turn the continuous state into
  16 active binary features per action. Nearby observations share some features,
  giving local generalization without a neural network. Action blocks are
  disjoint and collision-free in this implementation.
- **DQN.** The network maps normalized position and velocity to three Q-values.
  Locked DQNs use two 64-unit ReLU layers, Adam, Huber loss, gradient clipping,
  and a greedy max over target-network values.
- **Target network.** A separate frozen network supplies the bootstrap values.
  With the locked `tau=1`, it receives a hard copy every 500 optimizer steps,
  limiting the moving-target feedback loop.
- **Experience replay.** Standard replay retains up to 50,000 historical
  transitions, waits for 1,000 interactions, and samples uniform batches of 64.
  The no-replay agent retains no buffer and updates from a batch containing only
  the current scheduled transition.
- **Epsilon-greedy exploration.** Training chooses a random action with
  probability epsilon and otherwise a maximizing action. Standard DQN decays
  from 1.0 to 0.05 over 20,000 consumed transitions; the deliberate intervention
  uses only 200. Greedy evaluation passes `explore=False` and does not advance
  RNG or schedule state.
- **Potential-based shaping.** The shaped condition adds
  `beta * (gamma * Phi(s_next) - Phi(s))` to the training reward. `Phi` is the
  negative bounded deficit of a hill-height-plus-squared-speed energy proxy.
  The locked values are `beta=0.5`, `eta=0.5`, and `energy_scale=0.9`.
- **Termination versus truncation.** Goal termination is a true terminal state,
  so TD targets do not bootstrap and shaping uses zero next potential. The
  200-step `TimeLimit` is artificial truncation: collection stops, but TD targets
  bootstrap from the observed next state and shaping retains its potential.
- **Validation versus final evaluation.** Validation on seeds 1000--1019 chose
  the checkpoint lexicographically by 100-step success, completion, completed
  length, length variability, and earliest exact tie. Held-out seeds 2000--2019
  were used only after training and selection, with greedy policies and original
  reward.

## 3. Code walkthrough

| Responsibility | Exact path | Main functions/classes and what to point out |
|---|---|---|
| Environment creation | `src/mountaincar_rl/environments/factory.py` | `make_env` creates exactly one `MountainCar-v0` `TimeLimit`, seeds it, and preserves separate boundary flags. |
| Reproducible seeding | `src/mountaincar_rl/utils/seeding.py` | `seed_everything` seeds Python, NumPy, and PyTorch; `seed_env` seeds the environment and both spaces. |
| Tile coding | `src/mountaincar_rl/agents/tile_coder.py` | `TileCoder.state_tiles` computes one tile per tiling; `encode` adds the disjoint action offset. |
| SARSA action/update | `src/mountaincar_rl/agents/sarsa_lambda.py` | `SarsaLambdaAgent.select_action` is epsilon-greedy; `update` computes the on-policy TD error and applies trace-weighted updates. |
| On-policy loop | `src/mountaincar_rl/experiments/training.py` | `run_learning_training` selects `next_action` before `update` and carries it into the next environment step. |
| Eligibility traces | `src/mountaincar_rl/agents/sarsa_lambda.py` | `update` replaces/accumulates active traces, updates weights, then decays by `gamma * lambda`; `end_episode` clears them. |
| Replay buffer | `src/mountaincar_rl/agents/replay_buffer.py` | `ReplayBuffer.add` writes the circular buffer; `sample` draws distinct uniform historical indices and preserves both boundary flags. |
| DQN TD target | `src/mountaincar_rl/agents/dqn.py` | `compute_td_targets` implements `r + gamma*(1-terminated)*max Q_target`; `_optimize_batch` computes target values under `torch.no_grad`. |
| Target-network update | `src/mountaincar_rl/agents/dqn.py` | `_optimize_batch` checks optimizer-step frequency; `hard_sync_target` and `soft_sync_target` update the frozen target network. |
| Reward shaping | `src/mountaincar_rl/environments/reward_shaping.py` | `PotentialBasedRewardShaper.energy_proxy`, `potential`, `shaping_reward`, and `reward_components` implement and separately log the proxy intervention. |
| Success calculation | `src/mountaincar_rl/experiments/evaluation.py` | `evaluate_agent` records the first observed goal step; completion is goal reached, assignment success is `first_goal_step <= 100`. It never trains. |
| Checkpointing | `src/mountaincar_rl/utils/checkpoints.py` | `save_agent_checkpoint` writes a versioned envelope with config fingerprint and training state; `load_agent_checkpoint` validates it. Atomic writers prevent partial replacement. |
| Analysis/table | `src/mountaincar_rl/analysis/make_all_figures.py` | `build_final_table` summarizes policies at training-seed level; `generate_all` regenerates the table, summary, manifest, and four figures from `results/raw/full`. |
| Learning/final figures | `results/figures/figure_1_learning_curves.png`, `results/figures/figure_2_final_performance.png` | Figure 1 shows validation success versus interactions; Figure 2 shows held-out completion and 100-step success with seed-level uncertainty. |
| Behavior/failure figures | `results/figures/figure_3_phase_space.png`, `results/figures/figure_4_failure_diagnostics.png` | Figure 3 compares deterministic representative phase-space trajectories; Figure 4 compares epsilon, peak position, speed, and replay coverage. |

The locked condition files are `configs/full/sarsa_lambda.yaml`,
`configs/full/dqn_no_replay.yaml`, `configs/full/dqn_replay.yaml`,
`configs/full/dqn_shaped.yaml`, and `configs/full/dqn_fast_decay.yaml`.

## 4. Twenty likely questions

1. **Why must the car first move left?**  From the valley, direct rightward
   thrust cannot overcome the hill. Moving left gains height, then reversing
   converts that height into rightward speed. Figure 3 shows expanding loops in
   position-velocity space for completed episodes.

2. **Why use SARSA(lambda)?**  It is a compact on-policy baseline that can learn
   directly from epsilon-greedy behavior, while traces propagate sparse progress
   through recent features. In this study it was the strongest measured agent,
   with 1.000 completion and 0.380 strict success.

3. **Why use tile coding?**  Position and velocity are continuous. Overlapping
   tiles provide local generalization, sparse updates, and an interpretable
   linear value function. The validated setting used 16 `8 x 8` tilings.

4. **Why not use a basic Q-table?**  Exact floating-point states rarely repeat,
   so a table keyed by raw observations would not generalize. Discretizing into
   one coarse grid would introduce hard boundaries; overlapping tiles soften
   them.

5. **Why is DQN off-policy?**  Data come from an epsilon-greedy behavior policy,
   but the TD target uses the greedy `max_a Q_target(s',a)`. The target policy is
   therefore different from the exploratory behavior policy.

6. **Why use experience replay?**  The intended benefit is to reuse transitions
   and reduce temporal correlation through random minibatches. That was a
   hypothesis, not an assumed result; standard replay did not outperform online
   DQN on the locked endpoints here.

7. **Why use a target network?**  Without it, the same rapidly changing network
   defines both predictions and bootstrap targets. A frozen copy makes each
   optimization interval more stable; this code hard-copies every 500 optimizer
   steps.

8. **Why compare replay and no replay?**  It is the closest available replay
   ablation: observation normalization, 64-64 network, optimizer family, target
   equation, exploration schedule, budget, and evaluation cadence are shared.

9. **Why is that comparison not perfectly controlled?**  Removing replay also
   changes batch size from 64 to 1 and removes minibatch decorrelation and data
   reuse. A performance difference cannot be attributed to memory alone.

10. **Why distinguish termination and truncation?**  Reaching the goal is a true
    task terminal. Hitting 200 steps is an external time limit, not evidence that
    the next state has zero value. The implementation stops both episodes but
    bootstraps only through pure truncation.

11. **Why evaluate with epsilon zero?**  The question is what the learned policy
    can do, not how lucky its evaluation-time random actions are. `explore=False`
    also makes comparisons deterministic and prevents evaluation from changing
    agent RNG or schedule state.

12. **Why use multiple training seeds?**  Exploration and neural optimization
    are seed-sensitive. Five seeds expose failures and allow uncertainty over
    independently trained policies; the 20 episodes within one policy are not 20
    independent training experiments.

13. **Why separate validation and test seeds?**  Validation legitimately chooses
    checkpoints and bounded settings. Reusing those episodes as final evidence
    would bias the estimate. Final seeds 2000--2019 were isolated from training
    and validation seeds.

14. **Why report 100-step success separately?**  The normal task allows 200
    steps, but the assignment requires at most 100. A 150-step goal reach is a
    completion and explicitly not assignment success.

15. **Why use potential-based shaping?**  A distance reward would penalize the
    necessary initial movement left. The potential instead rewards reductions
    in a direction-neutral energy deficit through a potential difference, while
    retaining original rewards separately.

16. **Is the energy formula exact physical energy?**  No. It combines the
    rendered hill-height relation with normalized squared velocity. The code and
    report deliberately call it an interpretable energy proxy.

17. **Could reward shaping alter behavior?**  Yes in finite training with
    function approximation, even when the potential form has policy-invariance
    motivation. That is why the study bounds the term, checks its magnitude, and
    evaluates only on original reward. The measured 0.020 strict success was too
    sparse to claim robust benefit.

18. **Why did rapid epsilon decay fail?**  The honest answer is that it did not
    fail in the preregistered aggregate sense. It performed poorly overall, but
    obtained 0.210 completion and 0.030 strict success, while standard replay had
    0.180 and 0.000. The intended mechanism was not supported.

19. **What evidence supports that explanation?**  The selected failed episode
    in Figure 3 stays near its start, and epsilon reaches its minimum before the
    replay warm-up. But Figure 4 shows rapid decay had broader coverage, larger
    peak position, and greater absolute velocity than standard replay for much
    of training. Therefore the evidence supports one real failure trajectory,
    not systematic valley trapping.

20. **What is the project's largest limitation?**  For causal interpretation,
    the biggest limitation is that SARSA uses tile coding and traces while DQN
    uses a neural approximator and Adam; its stronger result cannot be credited
    solely to being on-policy. Statistically, five training seeds also make rare
    success estimates coarse.

## 5. Live coding preparation: purpose and invariants

### Epsilon-greedy action selection

Purpose: balance discovery and exploitation during training. Compute all action
values, select a uniform random action with probability epsilon, otherwise pick
a maximizer. Invariants: the action is valid; epsilon is in `[0,1]`; evaluation
uses the deterministic smallest maximizing action; evaluation must not consume
RNG state or advance the schedule. DQN epsilon is indexed by consumed
environment transitions; SARSA uses its successful update count.

### SARSA TD update

Purpose: move `Q(s,a)` toward the return implied by the current behavior policy.
The nonterminal target is `r + gamma*Q(s_next,a_next)` using the already selected
`a_next`; true termination uses only `r`. Pure truncation still requires a next
action and bootstrap. Invariants: the selected `a_next` must also be the action
executed next, values must remain finite, and the learning rate is divided by
the number of active tilings.

### Eligibility-trace update

Purpose: assign one TD error to a recency-weighted set of prior active features.
The implementation first marks current active traces, applies
`(alpha/tilings) * delta * trace` to all weights, then decays traces by
`gamma*lambda`. Replacing traces set active entries to one. Invariants: traces
stay finite and within the optional clip, and `end_episode` clears them so credit
does not cross collection boundaries.

### Replay sampling

Purpose: train on a uniform historical minibatch rather than an adjacent stream.
`ReplayBuffer` keeps aligned arrays for state, action, reward, next state,
terminated, and truncated, overwriting the oldest slot at capacity. Invariants:
sampling cannot exceed current size, indices are distinct within a batch,
sampling uses the buffer's private seeded RNG, and no-replay mode must never
instantiate or retain this buffer.

### DQN TD target

Purpose: construct a stable one-step regression target. The code computes
`r + gamma*(1-terminated)*max Q_target(next_state)` under `torch.no_grad`.
Invariants: target and online networks are distinct, target values receive no
gradient, only true termination masks the bootstrap, and target synchronization
uses optimizer-step units.

### Shaping reward

Purpose: make progress in momentum/height visible without penalizing leftward
motion by distance. The potential is a finite value in `[-1,0]`; shaping is
`beta*(gamma*Phi_next-Phi_current)`, optionally clipped, and training reward is
exactly original plus shaping. Invariants: `beta=0` produces exactly zero
shaping, true terminal next potential is zero, truncation uses the observed next
potential, and original/shaping/combined rewards remain separately auditable.

### Success-within-100 metric

Purpose: prevent ordinary 101--200-step completion from being mislabeled as the
assignment target. During evaluation, record the first step whose next position
reaches the environment's goal. `environment_completion = first_goal_step is not
None`; `assignment_success = completion and first_goal_step <= 100`. Invariants:
count real `env.step` calls, retain both Gymnasium flags, stop on either flag, and
include all failed episodes in rate denominators.

## 6. Honest limitations

- **Five seeds:** This is enough to show severe seed dependence but not enough
  for precise inference about rare events. The response is to report seed-level
  bootstrap intervals and avoid treating 100 pooled episodes as 100 agents.
- **Simple benchmark:** MountainCar is low-dimensional and mostly deterministic.
  The project demonstrates implementation and experimental discipline, not
  readiness for high-dimensional or stochastic control.
- **Bounded search:** Tuning was intentionally small and validation-only. A
  stronger configuration may exist; the result is conditional on the declared
  candidates and 60,000-interaction budget.
- **Representation confound:** SARSA and DQN differ in approximation, traces,
  optimization, and policy regime. The replay/no-replay DQN comparison is more
  controlled, but it still changes batch size and decorrelation.
- **Approximate proxy:** Hill height plus normalized squared velocity is useful
  and direction-neutral but is not the simulator's exact mechanical energy.
- **Limited generalization:** Nothing here supports universal claims that SARSA
  beats DQN, replay is harmful, or shaping is ineffective. Those are measured
  outcomes for this implementation, protocol, and budget.

## 7. Three future experiments

| Experiment | Hypothesis | Controlled change | Primary metrics | Main risk |
|---|---|---|---|---|
| Double DQN | Separating action selection from target evaluation will reduce max-operator overestimation and improve DQN stability. | Starting from locked replay DQN, change only the target to choose the action with the online network and evaluate it with the target network; keep architecture, replay, schedule, seeds, and budget fixed. | Validation/final 100-step success, completion, seed variability, mean Q-value, and TD loss. | At 60,000 interactions the effect may be too small or optimization noise may dominate. |
| Prioritized replay | Sampling high-TD-error transitions will improve sample efficiency by revisiting rare momentum-building transitions more often. | Replace only uniform sampling with proportional priorities plus importance-sampling correction; keep capacity, batch size, warm-up, network, and interaction budget fixed. | Time to first completion/100-step success, learning-curve area, final original-reward success, and effective sampling diversity. | Priorities can amplify noisy TD errors, reduce coverage, or introduce bias if correction is weak. |
| Stronger exploration | A count-based bonus over position-velocity bins will increase useful state coverage and discovery more reliably than the fixed epsilon schedule. | Add a preregistered count bonus to standard replay DQN while keeping core DQN/replay settings fixed; evaluate without the bonus. A separate follow-up could test noisy networks rather than mixing both changes. | Time to first success, replay coverage, maximum position/velocity, valley-retained episodes, and final original-objective rates. | Bin resolution may create an arbitrary novelty signal, and the bonus may reward visitation without producing a goal-reaching policy. |
