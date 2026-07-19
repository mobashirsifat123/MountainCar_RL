# Validation-only sweep configurations

`shaping/` contains the preregistered no/weak/moderate/strong energy-deficit
shaping ablation. `dqn_fast_decay.yaml` is the fixed rapid-exploration-decay
failure condition. Every file inherits the same replay-DQN base and uses only
validation seeds `1000`--`1019`; none loads the held-out final-test registry
(`2000`--`2019`). These files are not final frozen settings until the bounded
validation phase and its rejection rules are complete.
