# Algorithm and Mathematical Notes

These notes define the implemented algorithms and their tested behavior.
Random, tile-coded SARSA(lambda), online one-transition DQN, and uniform-replay
DQN execute through the same train/validation/checkpoint pipeline. Short smoke
runs verify mechanics only; they are **not evidence of learning performance**.
The shaped replay-DQN and rapid-decay replay-DQN conditions also execute through
the shared pipeline. Their smoke runs verify mechanics only; validation sweeps
and final experiments remain later gates.

## Environment semantics

Let state `s = (x, v)` be position and velocity and action `a` be one of the
three discrete accelerations. The original environment reward `r` and goal
termination are used unchanged for all evaluation. Gymnasium returns
`(s_next, r, terminated, truncated, info)`.

Data collection stops after `terminated or truncated`, but the learning mask
is

```text
b = 0  if terminated
    1  otherwise.
```

Thus a pure 200-step `TimeLimit` truncation still bootstraps. The transition's
post-step observation is retained and used in the target. Goal attainment is
recorded from the actual goal condition independently of truncation, so a
wrapper emitting both flags cannot erase a genuine final-step goal. Evaluation
counts actual calls to `env.step` and never bootstraps or continues after an
episode boundary.

## Observation preprocessing

DQN inputs use a fixed affine transformation derived from the environment
observation bounds, for example

```text
z_i = 2 * (s_i - low_i) / (high_i - low_i) - 1.
```

The transform is fitted from no trajectory data and is identical for online,
replay, shaped, and rapid-decay DQN conditions. Bounds and transform parameters
belong in checkpoints. Tile coding uses the same declared bounds but its own
tiling coordinates.

`ObservationNormalizer.from_space` constructs the mapping directly from the
environment's declared finite bounds. It supports batches, clips to `[-1, 1]`
by default, and serializes a versioned record containing method, bounds, output
range, and clipping choice. No trajectory is used to fit preprocessing.

## Tile coding

Use `m` overlapping rectangular tilings over `(x, v)`. Each tiling has the same
declared tile widths and a deterministic fractional offset; offsets must cover
both axes rather than shift every tiling identically. For state `s`, one tile
index per tiling is active:

```text
I(s) = {i_1(s), ..., i_m(s)}.
Q(s, a) = sum_{i in I(s)} w[a, i].
```

A collision-free dense index is preferred at MountainCar's scale. If hashing
is used, table size, hash function, and collision behavior are configuration
and checkpoint state. Tests must verify exactly `m` active features, stable
boundary clipping, deterministic offsets/indices, and distinct nearby-state
activation patterns. Because `m` weights are updated together, the configured
per-decision step size is distributed across tilings (`alpha / m`).

The implementation uses a collision-free dense table with action-specific
blocks. With eight tilings and `8 x 8` tiles, each action has
`8 * 8 * 8 = 512` parameters and three actions have 1,536 total. Exactly eight
indices are active for one state-action pair. Dividing `alpha` by eight keeps
the aggregate update near the configured step size rather than eight times it.

## SARSA(lambda) with replacing traces

Behavior and target are the same epsilon-greedy policy. Ties among maximizing
actions are broken with the agent RNG, not a global RNG. On transition
`(s_t, a_t, r_{t+1}, s_{t+1})`, select `a_{t+1}` when bootstrapping and compute

```text
delta_t = r_{t+1}
          + gamma * b_t * Q(s_{t+1}, a_{t+1})
          - Q(s_t, a_t).
```

For binary tile features, use replacing traces: set traces for the current
action's active features to one, apply

```text
w <- w + (alpha / m) * delta_t * e,
e <- gamma * lambda * e,
```

and clear traces at the observed episode boundary. At pure truncation,
`a_{t+1}` is sampled only to form the bootstrap target; collection still ends.
At true termination `b_t = 0` and no next action is needed. Tests cover lambda
zero (one-step SARSA), true terminal targets, truncation bootstrapping, trace
replacement/decay, and epsilon-greedy tie handling.

Replacing and accumulating traces are both configurable. Optional clipping and
finite guards fail before corrupting weights. Epsilon is annealed linearly over
successful update count. Training uses a private seeded RNG for exploration and
ties; `explore=False` chooses the smallest maximizing action without advancing
RNG or schedule. Checkpoints preserve weights, traces, hyperparameters, update
count, both last boundary flags, tile metadata, and RNG state.

## DQN target and update

Let `Q_theta` be the online network and `Q_target` the periodically synchronized
target network. The one-step target is

```text
y_t = r_t + gamma * (1 - terminated_t)
                  * max_a Q_target(s_next_t, a).
```

`truncated_t` is deliberately absent from the stop-bootstrap mask. Optimize a
declared robust TD loss (Huber by default) between `Q_theta(s_t, a_t)` and a
detached `y_t`, with configured optimizer and optional documented gradient
clipping. Epsilon-greedy behavior uses the training RNG; greedy validation
does not advance training exploration state. Checkpoints include online and
target weights, optimizer, exploration progress, counters, RNG state, and
configuration identity.

`QNetwork` is a configurable ReLU multilayer perceptron from normalized state to
action values. Both DQN variants use Adam, mean Huber loss, gradient clipping,
the same online/target architecture, and the same target function. Hard updates
copy parameters when `tau = 1`; otherwise the tested Polyak update is
`target <- tau * online + (1 - tau) * target`. Greedy evaluation uses
`torch.no_grad()`, smallest-index argmax ties, and mutates no network mode, RNG,
counter, replay, or optimizer state.

## Replay ablation

Replay DQN stores transitions including separate `terminated` and `truncated`
flags in a bounded uniform buffer, begins optimization only after a declared
warm-up, samples with its own seeded RNG, and trains on configured minibatches.
Target synchronization is measured in optimizer steps for every DQN condition.
Because the locked configurations update every two environment transitions,
one interval of 500 optimizer steps normally spans 1,000 environment
transitions (after replay warm-up where applicable).

Online DQN consumes each new transition immediately and performs an update on
that single transition without storing/sampling prior experience. It shares
the observation transform, architecture, target formula/network, optimizer
family, exploration schedule, and comparable update-to-data convention with
replay DQN wherever feasible. This ablation necessarily changes both temporal
decorrelation and batch size (one online versus a replay minibatch); conclusions
must describe both rather than claim a perfectly isolated replay-memory effect.

The replay buffer is a fixed-capacity circular array. Entries contain float32
observations/next observations, integer action, float reward, and separate
boolean `terminated`/`truncated` flags. Sampling is without replacement from a
private seeded RNG and is rejected until a full requested batch exists. Replay
storage, cursor, size, and RNG are checkpointed.

Replay DQN inserts every transition, waits for warm-up and a full minibatch,
then updates at the configured frequency. Online DQN stores nothing and builds a
batch of one from only the current transition. Thus removal of replay also
changes batch size and temporal decorrelation; this is not described as a pure
memory-only intervention.

## Potential-based reward shaping

Training uses

```text
r_train(s, a, s_next) = r_environment
                        + beta * (gamma * Phi_effective(s_next) - Phi(s)).
```

The preregistered state quantity is an interpretable dimensionless **energy
proxy**, not exact physical energy for the simulator:

```text
h(x) = 0.45 * sin(3x) + 0.55
E_proxy(x, v) = h(x) + eta * clip(v / v_max, -1, 1)^2
E_target = h(x_goal)
D(x, v) = clip((E_target - E_proxy(x, v)) / E_scale, 0, 1)
Phi(x, v) = -D(x, v).
```

`h`, `E_proxy`, `E_target`, and `E_scale` are in dimensionless proxy units.
`eta` is dimensionless, and `beta` has environment-reward units because `Phi`
is dimensionless. Squared speed is direction-neutral, so useful leftward
acceleration can reduce the deficit rather than being punished for increasing
distance to the flag. The negative deficit makes a deficit reduction an
increase in potential. `Phi` is bounded in `[-1, 0]`, giving the analytic guard
`|F| <= beta * (1 + gamma)` before an optional configured numerical clip.
`beta=0` disables shaping through the identical reward-component pipeline.

For a true goal terminal, define `Phi_effective(s_next) = 0` so no residual
terminal potential changes episodic preferences. For a pure artificial
truncation, use the actual `Phi(s_next)` consistently with time-limit
bootstrapping. The shaping discount must equal the agent discount. Tests cover
the telescoping discounted shaping sum, goal-terminal zeroing, truncation,
bounded finite values, and separation of original versus shaped reward logs.

Potential-based invariance is a theoretical guide, not a guarantee that finite
training with function approximation and checkpoint selection yields identical
policies. Every model—including the shaped model—is selected and evaluated on
original goal outcomes and original environment return; shaped return is a
training diagnostic only.

## Deliberately rapid epsilon decay

The failure condition changes only the predeclared linear epsilon-decay horizon
relative to its inherited replay-DQN configuration, apart from run identity and
output paths. Smoke decay is 8 interactions versus a 16-transition replay
warm-up. Validation decay is 200 interactions versus a 1,000-transition
warm-up; standard validation decay is 20,000. It is not selected by searching
for a seed that fails. Realized per-transition epsilon, per-episode replay-grid
coverage, normalized action entropy, position range, maximum position, peak
speed, success timing, valley-retention labels, and position--velocity
trajectories are recorded. If it succeeds, that is a valid falsification rather
than grounds to redesign the condition after seeing final results.

## Shared evaluation

`evaluate_agent` is the sole shared evaluation rollout. For each fixed seed it
calls `select_action(..., explore=False)` and never calls `observe`,
`end_episode`, a shaper, replay sampling, or an optimizer. It returns immutable
records containing original return, optional separate shaped return, both success
labels and boundary flags, actual steps, and complete state/action/original-reward/
position/velocity trajectories. State trajectories include reset and every next
state. Goal position is checked independently of flags, including when termination
and truncation coincide. Every primary evaluation metric uses original rewards.

## Shared checkpoint envelope

Best-validation and last checkpoints use an atomic versioned envelope:

```text
checkpoint_schema_version, agent_type, config_fingerprint, config,
preprocessing, agent_state, training_state, selection
```

The SHA-256 configuration fingerprint prevents loading a policy under a different
architecture or reward/config contract. SARSA state holds linear weights/traces
and RNG state. DQN state holds online/target networks, Adam, counters, epsilon RNG,
preprocessing, and replay state when enabled. Evaluation validates both the shared
envelope and algorithm state before running.

## Structured metric schemas

Training `episodes.csv` contains condition/split/seeds, actual steps, original,
shaping, and total training returns, both flags and success labels, realized
epsilon, finite update metrics, counters, kinematic extrema, action entropy,
valley retention, replay coverage, and shaping-magnitude ratios.
`trajectories.jsonl` retains original, shaping, and total reward, both state
potentials, realized epsilon, state/action/next-state, flags, and update metric
per transition. `diagnostics.json` records first training and scheduled-
validation successes, final replay coverage, valley-episode count, intervention
metadata, and shaping dominance status.

Periodic `validation_episodes.csv` and standalone evaluation `episodes.csv` use
the same original-return success fields. Evaluation trajectory JSONL retains each
original-reward transition. `validation_history.json` stores every scheduled
score. Best policy selection is lexicographic by at-most-100 rate, completion
rate, then mean original return; the last policy is also always saved. Each run
adds `summary.json`, `policy.json`, config snapshot, and runtime metadata.
