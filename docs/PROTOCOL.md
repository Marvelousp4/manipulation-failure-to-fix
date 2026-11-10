# Experiment protocol

**Question:** How do controlled observation noise and action delay affect policy rollouts?

- Compare clean vs shifted with shared seeds; verify initial states match.
- Record true simulator success; unknown is not failure.
- Report success, episode length, return, sample count, uncertainty and regressions.
- Freeze policy/environment; change one variable at a time.
- `ToyReach` is a smoke test. Noise/delay are interface—not contact—shifts.
- For research claims: trained policies, multiple tasks, strong baselines, equal budgets.

**Hypotheses (untested):** repairs differ by failure mode; targeted data may outperform random data.
