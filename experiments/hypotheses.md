# Pre-results Hypotheses and Decision Rules

**Registered:** 2026-07-18 (Asia/Shanghai)  
**Status:** Written before the five-seed final comparison and before inspecting
held-out test results. Phase-1 random smoke output, if any, is infrastructure
validation and is not evidence for selecting final hypotheses or settings.

## Fixed protocol

The final comparison uses training seeds `0`–`4`, validation episode seeds
`1000`–`1019`, and disjoint final-test episode seeds `2000`–`2019` loaded from
an isolated configuration. Hyperparameters and checkpoints are selected using
validation only.
Assignment success means reaching the flag in at most 100 steps; ordinary
completion within the 200-step limit is a separate outcome. Primary inference
uses seed-level aggregates and paired seed differences where conditions share
training seeds. Held-out results will be shown for all seeds.

The primary reliability endpoint is the held-out at-most-100-step rate. Also
report completion rate, original return, and mean/median steps among completed
episodes. Learning sample efficiency is measured against environment
interactions, not optimizer updates or wall-clock time. “Time to first
success” is the number of training interactions before the first scheduled
greedy validation evaluation containing an at-most-100-step completion; runs
without one by budget are right-censored rather than assigned an invented
time.

## Phase-3 intervention registration (before shaped/failure runs)

**Registered:** 2026-07-18 (Asia/Shanghai), before executing either the shaped
DQN smoke run or the rapid-decay smoke run. Smoke runs are mechanical checks,
not evidence for accepting a hypothesis or selecting a final setting.

The reward intervention uses the simulator's dimensionless hill-height
relationship

```text
h(x) = 0.45 * sin(3x) + 0.55
E_proxy(x, v) = h(x) + eta * (v / v_max)^2
E_target = h(x_goal)
D(x, v) = clip((E_target - E_proxy(x, v)) / E_scale, 0, 1)
Phi(x, v) = -D(x, v)
F(s, s_next) = beta * (gamma * Phi_effective(s_next) - Phi(s))
r_train = r_environment + F.
```

This is an interpretable, dimensionless **energy proxy**, not exact mechanical
energy for the simulator. Squared normalized velocity is direction-neutral.
The negative sign makes lower energy deficit a higher potential, so an ordinary
transition that reduces deficit is generally rewarded. True goal termination
uses `Phi_effective(s_next) = 0`; an artificial time-limit truncation retains
the observed next potential. `Phi` is bounded in `[-1, 0]`, so before an
optional numerical safety cap `|F| <= beta * (1 + gamma)`. The learner and
shaper use the same `gamma`.

The bounded shaping validation ablation fixes `eta = 0.5`,
`E_scale = 0.9` proxy units, and changes only `beta`: no shaping `0.0`, weak
`0.1`, moderate `0.5`, and strong `1.0`. The target is derived from the
configured environment goal position, not fitted to trajectories. A setting is
ineligible if its mean absolute per-step shaping magnitude exceeds `0.5` times
the mean absolute environmental reward, if validation original-objective
success is worse than the unshaped reference without compensating discovery
benefit, or if trajectories show repeated high proxy-energy/reward collection
without flag progress. Shaped return is never a selection or final evaluation
objective.

The deliberate failure condition is the valid replay-DQN base with only linear
epsilon decay shortened. In the mechanical smoke pair it decays from `1.0` to
`0.1` in 8 environment steps while replay warm-up is 16 transitions. In the
validation pair it decays in 200 interactions while replay warm-up is 1,000;
the standard comparison schedule decays over 20,000 interactions. Thus epsilon
is near its floor before the first replay update, without breaking the agent.
Architecture, optimizer, preprocessing, replay capacity/minibatch, target
updates, budget, validation cadence, and validation seeds remain inherited from
the same base. Diagnostics are realized epsilon by interaction, replay-bin
coverage, position range, maximum position, peak absolute velocity, normalized
action entropy, first training and scheduled-validation successes, the count of
episodes whose states remain within `0.20` position units of the valley minimum
`-pi/6` with peak speed at most `0.025`, and position--velocity trajectories.
No secondary failure control is preregistered for this phase because it is not
needed to identify the rapid-decay mechanism and would add a separate causal
intervention.

## Hypotheses

### H1 — reliability within 100 steps

The random policy will provide a low sanity-check floor, and at least one
learning condition will improve the held-out at-most-100-step rate. We do not
predeclare which learned condition will have the highest final reliability,
and we do not assume the assignment target is achievable by every seed.

Interpretation: support requires the learned condition's seed-level advantage
over random to be consistently positive and practically visible with its
uncertainty. If no learned condition improves, or confidence intervals and
per-seed values show no credible separation, H1 is unsupported. Completion in
101–200 steps can support learning progress but cannot support H1's assignment
claim.

### H2 — effect of experience replay

With the shared DQN preprocessing and network, uniform replay plus minibatches
is expected to improve validation-curve stability and sample efficiency over
online one-transition updates. Stability will be summarized by across-seed
dispersion and catastrophic loss/performance regressions; sample efficiency by
area under the validation assignment-success curve versus interactions and
interactions to first validation success.

Interpretation: the stability and efficiency claims are evaluated separately.
Positive paired seed differences in curve area and lower seed/curve
variability support the respective claims; mixed directions are reported as
mixed evidence. Equal or worse replay results falsify the directional part.
This is not a pure “memory” intervention: replay also introduces batch size
greater than one and breaks short-range transition correlations.

### H3 — tile-coded on-policy learning versus neural DQN

Tile-coded SARSA(lambda) is expected to be competitive at modest interaction
budgets because overlapping local features and traces propagate sparse reward
without fitting a neural network. Neural replay DQN may improve later, but no
final winner is hypothesized.

Interpretation: compare full learning curves, final seed-level reliability,
and successful-step distributions. Early SARSA advantage supports the stated
sample-efficiency expectation; a consistently better DQN curve falsifies it.
Any difference is descriptive: function approximators, update rules, and
optimization differ, so it is not a controlled causal test of on-policy versus
off-policy learning.

### H4 — potential shaping and discovery time

Replay DQN trained with the preregistered potential-based energy shaping term
is expected to reach its first greedy validation assignment success in fewer
environment interactions than otherwise identical replay DQN. The mechanism
hypothesis is broader early position–velocity coverage and higher useful speed
near turning points, not merely a larger shaped numeric return.

Interpretation: compare paired, right-censored first-success observations and
early validation curve area. Earlier discovery across most paired seeds plus
the predicted phase-space diagnostics supports H4. Similar/later discovery or
only larger shaped returns falsifies it. Censored runs and contrary seeds stay
in the analysis.

### H5 — shaping without final-objective harm

Because the shaping term is potential-based with matched discount and terminal
conventions, it is expected to improve exploration without materially reducing
the final greedy policy's performance under the original objective. “No
material harm” is preregistered as a lower 95% confidence bound above `-0.05`
for the paired difference (shaped minus unshaped) in held-out assignment-success
rate. The five-percentage-point margin represents at most one additional miss
per twenty evaluation episodes and will not be changed after test inspection.

Interpretation: H5 needs both exploration evidence from H4 diagnostics and the
non-inferiority criterion under original reward. A lower bound at or below the
margin is inconclusive or harmful, not proof of equivalence. Any shaped-return
gain without original-objective gain is irrelevant to this claim.

### H6 — rapid exploration decay and valley trapping

The deliberately fast-decay DQN is expected to commit prematurely to a narrow
action pattern before observing enough high-energy trajectories. Relative to
standard replay DQN it should show lower position–velocity coverage, lower
behavior-policy action entropy after decay, smaller position range or peak
speed, and fewer/ later validation successes—consistent with oscillating near
the valley rather than learning alternating acceleration.

Interpretation: trajectory, visitation, action-map, and epsilon diagnostics
must agree with the proposed mechanism. Failure without those signatures is
not evidence for the mechanism. If the fixed fast-decay condition reliably
learns momentum and matches the standard schedule, H6 is falsified; it will
not be redefined or replaced post hoc to manufacture a failure.

## Reporting rules fixed before results

- Show individual seed points and uncertainty; distinguish “unsupported” or
  “inconclusive” from proof of no effect.
- Use the same planned interaction budget and validation cadence for relevant
  comparisons. Curves include all scheduled observations and visibly mark
  missing/incomplete runs.
- Checkpoints are chosen by a documented validation rule. The test set is
  evaluated only after condition configurations and selected checkpoints are
  locked.
- Phase-space diagrams use held-out or fixed diagnostic trajectories selected
  without reference to favorable outcomes. Report both successes and failures.
- Multiple secondary metrics are explanatory, not opportunities to replace a
  failed primary endpoint.

## Post-submission controlled rescue protocol

**Registered:** 2026-07-19 (Asia/Shanghai), after the audited final comparison
showed that no condition reliably met the at-most-100-step target and before any
rescue run. Existing held-out results motivate diagnosis only; they are not read
by the rescue runner, used to rank candidates, or reused as validation data.

The locked replay DQN in `configs/full/dqn_replay.yaml` is the reference because
the requested interventions concern replay warm-up, target updates, and shaping.
Initial rescue trials use only training seeds 0 and 1 and validation episode
seeds 1000--1019. Every candidate inherits the locked reference and changes one
scientific variable; experiment names, condition labels, seed, and output paths
are provenance differences rather than algorithm changes. Candidates run in
this fixed order:

1. epsilon decay steps: 20,000 to 40,000;
2. replay warm-up: 1,000 to 5,000 transitions;
3. learning rate: 0.0005 to 0.00025;
4. target hard-copy interval: 500 to 1,000 optimizer steps;
5. shaping beta: 0.0 to 0.5, with the registered eta and scale unchanged;
6. interaction budget: 60,000 to 120,000.

The reference validation outcomes are retained from the original runs: seed 0
had zero strict success and 1.0 completion at its selected checkpoint; seed 1
had zero strict success and zero completion. A candidate counts as repeatable
improvement only if both seeds improve the lexicographic validation score and
both have nonzero at-most-100 validation success. This deliberately demanding
rule matches the rescue objective and prevents locking a setting because of one
unusual episode. If no candidate meets it, no corrected configuration will be
locked and no new held-out evaluation will be run.

For each trial, report the selected validation score, censored first completion
and strict-success interactions, coverage, reward-shaping ratio, and training
diagnostics. No final-test seed file may be passed to training or diagnostic
code. The original locked condition and its contrary rapid-decay result remain
unchanged regardless of the rescue outcome.
