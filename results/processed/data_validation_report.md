# Final Result Data Validation

Generated only from existing `results/raw/full` artifacts and locked configurations; no training or evaluation was rerun.

Overall: **PASS** — 10 pass, 1 warning, 0 failure.

| Check | Status | Evidence |
|---|---|---|
| Required experimental conditions | PASS | all six preregistered conditions are present |
| Final-test seed registry | PASS | configured seeds=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]; expected=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019] |
| Planned training seeds and required raw artifacts | PASS | 30/30 condition-seed runs present and complete |
| Duplicate or unexpected final runs | PASS | exactly one raw run per condition and training seed |
| Locked configuration hashes and per-seed snapshots | WARNING | random_seed0: random training summary predates config_hash logging; snapshot and held-out evaluation hash were checked instead; random_seed1: random training summary predates config_hash logging; snapshot and held-out evaluation hash were checked instead; random_seed2: random training summary predates config_hash logging; snapshot and held-out evaluation hash were checked instead; random_seed3: random training summary predates config_hash logging; snapshot and held-out evaluation hash were checked instead; random_seed4: random training summary predates config_hash logging; snapshot and held-out evaluation hash were checked instead |
| Finite numeric values | PASS | all parsed CSV, JSON, and JSONL numeric values are finite |
| Episode lengths and <=100 success semantics | PASS | all episodes have 1--200 steps and valid completion/success flags |
| Disjoint training, validation, and held-out seeds | PASS | training=0--4, validation=1000--1019, final test=2000--2019 |
| Original-objective final evaluation | PASS | all final-test episode_return values equal original_return; shaped fields are empty |
| Equal final-test episode counts | PASS | each of 30 policies has 20 held-out episodes |
| Checkpoint completeness | PASS | all 100 learned-agent checkpoint envelopes load and match their run configuration |

Interpretation: a `WARNING` flags a non-fatal audit concern; a `FAIL` means the affected result must not be treated as validated until the saved artifact is repaired or its exclusion is explicitly documented. Confidence intervals and final comparisons must use training seeds, not the 20 within-policy held-out episodes as independent runs.
