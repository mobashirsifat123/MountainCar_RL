# Reinforcement-Learning Laboratory Audit

Audit date: 2026-07-18 (Asia/Shanghai)

## Verdict

The saved study is scientifically usable with the limitations already stated in
the report. No Critical or High algorithmic or experimental-validity defect was
found. All 30 planned condition/seed runs and all 600 held-out episode records
remain present. No locked configuration, full-training artifact, checkpoint, or
final-test metric was changed, and no full experiment was rerun.

Two Medium presentation issues were corrected: the learning-curve aggregator
discarded intermediate checkpoints whose episode-boundary interaction counts did
not match exactly across seeds, and parts of the prose used causal wording that
the five-seed observational comparison did not support. These corrections affect
the displayed learning curve and interpretation, not the final metrics table.

## Issues and corrections

| ID | Severity | Evidence | Affected file | Correction | Results affected? | Experiment rerun? |
|---|---|---|---|---|---|---|
| A-01 | Medium | Validation occurs at episode boundaries. Exact intersection of recorded interaction counts left only two points for SARSA and shaped DQN and three for no-replay DQN, even though every seed had six scheduled validations. This could conceal instability and distort sample-efficiency interpretation. | `src/mountaincar_rl/analysis/make_all_figures.py`; Figure 1 | Align each seed by linear interpolation to the locked 10,000-interaction grid before the documented three-checkpoint display smoother. Added alignment regression tests and regenerated Figure 1 from raw files. | Figure 1 changed; raw data and Table 1 did not. | No. Analysis only. |
| R-01 | Medium | “Shaping therefore enabled…” and “momentum-aware representation … matter” implied causal conclusions not isolated by the design. Only sparse shaped successes were observed, and representation was not an intervention isolated across all conditions. | `report/report.md`; `README.md` | Replaced causal language with association-based wording and limited the conclusion to observed trajectories and seed-level evidence. | Interpretation only. | No. |
| S-01 | Low | Table brackets consistently denoted 95% seed bootstrap intervals except in first-success columns, where they were minima/maxima. The conditional median also did not explicitly say that successful episodes were pooled. | `src/mountaincar_rl/analysis/make_all_figures.py`; `results/tables/final_results.md` | The generated note now labels first-success brackets as observed ranges, states censoring, defines the pooled conditional median, and confirms failed episodes remain in rate denominators. | No numeric value changed. | No. Analysis only. |
| D-01 | Low | Algorithm notes described target synchronization as possibly counted in environment or optimizer steps, while the implementation and locked runs use optimizer steps. | `docs/algorithm_notes.md` | Documented optimizer-step counting and its approximate environment-step equivalent for the locked replay configuration. Added exact synchronization/detachment tests. | No. | No. |
| T-01 | Low | The original tests covered SARSA algebra but not the complete select-before-update/carry-forward training-loop sequence or exact two-step trace credit. | `tests/test_sarsa_training_semantics.py`; `tests/test_sarsa_lambda.py` | Added focused on-policy sequencing and eligibility-credit regression tests. | No defect was exposed; no result changed. | No. |
| P-01 | Low | Five historical random-policy training summaries predate `config_hash` logging. | `results/raw/full/random_seed*/summary.json` | Preserved immutable evidence. The validator instead checks each saved configuration snapshot and held-out evaluation hash and reports a visible warning. | No; hashes agree through the alternative evidence. | No. |

## A. Algorithm correctness

| Area | Status | Audit evidence |
|---|---|---|
| SARSA next action and on-policy sequencing | PASS | `training.py` selects the epsilon-greedy next action before calling `update`, passes that action into the TD target, and carries the same action into the next environment step. The new integration test covers ordinary and pure-truncation paths. |
| SARSA TD error and terminal handling | PASS | `sarsa_lambda.py` computes `r + gamma * Q(s',a') - Q(s,a)` and masks only genuine termination. Pure time-limit truncation retains the bootstrap but still ends collection and clears traces at the episode boundary. |
| SARSA eligibility traces | PASS | Active features use configured replacing or accumulating traces, the weight step is scaled by the number of tilings, and traces decay by `gamma * lambda`. Exact two-step credit and `lambda=0` are tested. |
| Tile coding/action parameters | PASS | Every tiling contributes one state feature; action feature blocks are disjoint, offset by `features_per_action`, collision-free for the configured dense coder, and bounds/actions are validated. |
| DQN network separation and targets | PASS | The target is a deep copy, target parameters are frozen, the optimizer owns online parameters only, and target Q values are computed under `torch.no_grad()`. True termination alone masks bootstrap. |
| Replay ablation | PASS | Replay DQN samples historical transitions from a bounded uniform buffer. No-replay DQN has `replay_buffer=None`, builds a batch of one current transition, and retains no historical transition state. The change in batch size/decorrelation is disclosed. |
| Epsilon and target-update schedules | PASS | Epsilon advances once per consumed environment transition. Target synchronization occurs exactly every configured optimizer step. Evaluation action selection advances neither schedule nor RNG. |
| Evaluation isolation | PASS | Evaluation calls greedy action selection only and does not invoke learning hooks. Unit tests compare replay, optimizer, counters, RNG and parameters before/after evaluation; an actual checkpoint-backed audit rollout reproduced identical trajectories. |
| Reward shaping | PASS | The implementation stores original, shaping, and combined training rewards separately. `beta=0` follows the same pipeline and yields zero shaping. The report equation matches the bounded negative deficit potential. True termination uses zero next potential; time truncation retains observed next potential. Final evaluation logs original reward and null shaped fields. |

For the five shaped full runs, the maximum absolute discrepancy in
`training_reward - (original_reward + shaping_reward)` was `2.84e-13`, consistent
with floating-point roundoff. All shaped final-test rows use original return.

## B. Experimental validity

| Check | Status | Evidence |
|---|---|---|
| Seed isolation | PASS | Training seeds 0–4, validation seeds 1000–1019, and held-out seeds 2000–2019 are pairwise disjoint. Sweep code rejects held-out seeds, and full training configs contain validation seeds only. |
| No final-test tuning | PASS | The append-only experiment ledger records validation selection before locking. Checkpoint scoring reads validation outcomes only; held-out seeds are supplied through the separate final-test seed file. No final metric enters sweep or selection code. |
| Comparable learned budgets | PASS | Every seed of SARSA and all four DQN variants (including rapid decay) has exactly 60,000 saved environment interactions. The five learned conditions total 1,500,000 interactions. |
| Complete seeds | PASS | All six conditions contain seeds 0–4; no unsuccessful seed is missing. Every policy has exactly 20 held-out episodes. |
| Model selection | PASS | Best checkpoint priority is validation assignment success, validation completion, lower conditional median length, lower episode-length SD, then the earliest exact tie. Final test is not consulted. |
| Success semantics | PASS | Evaluation stops at termination or truncation, counts actual steps, and defines assignment success as goal completion with `steps <= 100`. Boundary tests cover 100, 150, 200, pure truncation, and simultaneous termination/truncation. |
| Results and figures provenance | PASS | Table 1 and all figures are regenerated by `mountaincar_rl.analysis.make_all_figures` from `results/raw/full`. The raw-tree aggregate SHA-256 before and after regeneration was identical: `1a31f6889579ac6228ea4db85bdeee506eba244828feb603cb31d5a54c1532aa`. |

## C. Statistical validity

- Rates and returns are first summarized per trained policy. The 95% percentile
  bootstrap resamples the five training-seed summaries 10,000 times; held-out
  episodes are not treated as independent training replicates.
- Failed episodes remain in completion and assignment-success denominators.
  Median length is explicitly conditional on completion and pooled over the
  completed held-out episodes; no artificial failure length is inserted.
- Confidence intervals use the same seed-level bootstrap implementation and
  deterministic RNG scheme throughout. First-success ranges are now clearly
  distinguished from confidence intervals and report censored seeds.
- Figure 1 shows faint individual seed curves, a seed-level uncertainty band,
  and a documented three-checkpoint centered smoother. Figure 4 aligns each seed
  to a common 500-interaction grid before a five-point display smoother. The raw
  trajectories and unsmoothed episode metrics remain available.
- The conclusions do not establish universal superiority. The report identifies
  the function-approximator confound in SARSA versus DQN and the minibatch and
  decorrelation changes in the replay ablation.

## D. Reproducibility verification

| Verification | Result | Evidence |
|---|---:|---|
| Complete unit/integration suite | PASS | `130 passed in 3.65s`; the smoke workflow independently repeated all 130 tests in 2.91s. |
| Six-condition smoke workflow | PASS | Random, SARSA, no-replay DQN, replay DQN, shaped DQN, and rapid-decay DQN each trained for the smoke budget and completed checkpoint-backed validation. These are mechanical checks, not new study results. |
| Checkpoint loading | PASS | Focused checkpoint suite: `3 passed in 1.25s`; the raw validator also loaded all 100 learned full-run checkpoint envelopes. |
| Deterministic short evaluation | PASS | Two validation episodes from `checkpoints/smoke/dqn_replay_seed0/best.pt` exactly matched the smoke evaluation in trajectory JSONL and every non-timing episode field. |
| Analysis regeneration | PASS | The raw-only command regenerated the CSV/Markdown table, summary, manifest, and four PNG/PDF figure pairs. It did not modify `results/raw/full`. |
| Result validator | PASS with one documented warning | 10 PASS, 1 WARNING, 0 FAIL. The warning is P-01 above. |

The audit did not reinstall dependencies. The recorded environment is Python
3.12.13 on macOS arm64, CPU, Gymnasium 1.3.0, NumPy 2.5.1, and PyTorch 2.13.0.
Git provenance is unavailable because this directory is not a Git work tree;
metadata records `null` rather than inventing a commit.

## E. Report-claim audit

| Section / important claim | Classification after correction | Basis |
|---|---|---|
| Abstract: SARSA completed 1.000 and achieved 0.380 [0.330, 0.420] assignment success | Supported | Exact match to the generated table; interval unit is training seed. |
| Abstract: replay did not improve the locked DQN outcome | Supported for this implementation and budget | Replay had 0.180 completion versus 0.400 without replay; both had zero assignment success. The report does not generalize beyond this setup. |
| Abstract: shaped replay had sparse discovery and 0.020 [0.000, 0.060] final success | Supported / reasonable interpretation | Exact final rate; three of five seeds had a first completion and one a first within-100 success. Causal wording was removed. |
| Results: reported rates, medians, returns, first-success times and variability | Supported | Values agree with `results/tables/final_results.md`; first-success brackets are now identified as observed ranges. |
| Failure analysis: the shown rapid-decay trajectory is valley-confined | Supported for the selected episode | Deterministic phase-space selection uses the failed episode nearest the pooled failed median maximum position. |
| Failure analysis: rapid decay did not systematically reduce momentum or coverage | Supported | Figure 4 shows greater aggregate maximum position, absolute velocity, and replay coverage for rapid decay over much of training. The text explicitly rejects treating one episode as condition-wide proof. |
| Alternative explanations for the failed/unstable DQNs | Reasonable interpretation | Optimization instability, action bias, checkpoint transfer, and budget are presented as unresolved alternatives, not findings. |
| Conclusion: no condition reliably met the ≤100 target | Supported | The best mean held-out assignment success was SARSA at 0.380; no reliability threshold was claimed as reached. |
| Conclusion: successful displayed policies use oscillatory momentum-building paths | Supported descriptively | Figure 3 shows expanding position–velocity loops for the deterministically selected completed episodes. It is not claimed as a causal algorithm comparison. |
| Former causal shaping/representation language | Overclaim, corrected | See R-01. No unsupported claim remains in the audited abstract, results, failure analysis, or conclusion. |

## Residual limitations

The five training seeds yield coarse intervals, especially for rare successes.
The conditional step median is descriptive and pooled rather than a seed-level
estimand with its own interval. The 60,000-interaction budget is much smaller
than the initially suggested 200,000–500,000 range but is openly locked and equal
across learned conditions. SARSA versus DQN remains confounded by function
approximation, traces, and optimization. The potential is an interpretable
energy proxy, not exact mechanical energy. These are limitations, not hidden
correctness failures.
