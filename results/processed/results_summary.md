# Results summary

All final metrics are regenerated from `results/raw/full`; the independent unit
is the training seed (`n=5`), never the 20 held-out episodes within one policy.
See [Table 1](../tables/final_results.md) and Figures 1--4.

## Target outcome

**No model reliably met the ≤100-step target.** The strongest condition was
SARSA(lambda): 0.380 [0.330, 0.420].
This is progress over random (0.000 [0.000, 0.000]),
not reliable task fulfilment (Table 1; Figure 2).

## Hypotheses

### H1 — reliability within 100 steps: partially supported

At least one learned policy exceeded random, but no condition approached a
reliable 100-step completion rate (Table 1; Figures 1--2).

### H2 — replay improves stability/sample efficiency: not supported

Replay DQN ended at completion 0.180 [0.000, 0.540]
and success 0.000 [0.000, 0.000];
the no-replay values were 0.400 [0.000, 0.800]
and 0.000 [0.000, 0.000].
Figure 1 shows neither DQN produced sustained ≤100 validation success at the
locked budget. This comparison remains limited because removing replay also
changes minibatch decorrelation and batch size.

### H3 — SARSA(lambda) versus neural DQN: partially supported

SARSA(lambda) had stronger saved outcomes than all neural conditions (Table 1;
Figures 1--2). This is descriptive only: tile coding, traces, and neural
optimization differ, so it is not a controlled on-policy/off-policy causal test.

### H4 — potential shaping accelerates discovery: partially supported

Shaped replay's first completion was
19196 [14784, 23146]; 2/5 censored,
whereas standard replay was
not observed (5/5 censored). Its
first ≤100 training success was
59161 [59161, 59161]; 4/5 censored.
The censoring and sparse successes shown in Table 1/Figure 1 support occasional
earlier discovery, not a reliable acceleration conclusion.

### H5 — shaping without final-objective harm: partially supported

Shaped replay's held-out success was
0.020 [0.000, 0.060]
versus 0.000 [0.000, 0.000]
for standard replay. The paired seed-level difference (shaped minus unshaped)
was 0.020 [0.000, 0.060], above the preregistered
-0.05 lower-margin rule. The broad interval and low absolute success mean this
does not establish a robust final-performance improvement (Table 1; Figure 2).

### H6 — rapid exploration decay causes valley trapping: not supported

Rapid decay reached success
0.030 [0.000, 0.090]
and completion 0.210 [0.030, 0.390].
Figure 4 confirms immediate epsilon reduction but does not show the proposed
systematic loss of momentum/coverage; Figure 3 provides the fixed representative
phase-space comparison. The registered failure mechanism is therefore not
supported and is not redefined after observing the result.
