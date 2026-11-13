# Clean baseline review

`160c11e` is preserved. No new robustness experiments.

- 50/50 sampled demonstration action replays succeed; state/action alignment and normalized
  `pd_ee_delta_pos` control agree. Only 1/1,000 demonstrations exceeds 100 steps.
- 999/1,000 stored initial grasp flags were stale. Fresh-scene replay removes them; corrected
  replay also succeeds 50/50. This is a data defect, not proof of the main performance cause.
- Old early termination conflated success at first arrival with fixed-horizon success.
  New metrics follow the [official IL protocol](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) at 200 steps.
- Old BC: offline action MSE 0.000646 on training and 0.001147 on 50 new planner references.
  Closed-loop success: 2/50 training seeds, 1/50 diagnostic held-out seeds; neither retains
  success at step 200. Translation fit is close; gripper error dominates MSE.
- Videos show repeated grasp/lift/lower cycles, stalled reaching, failed grasps and failure
  to retain the goal. Selected BC ends outside the goal in 45/50 training and 44/50 diagnostic
  held-out episodes; most robots are static. The remaining failures are not merely a velocity
  threshold artifact. Phase labels are symptoms, not controller-defect diagnoses.
- There are 4,314 supervised states following prior success and 3,762 near-zero arm actions;
  absence of all stop examples is ruled out. Coverage, phase ambiguity and accumulated error
  remain hypotheses. No architecture ablation establishes their causal contributions.
- Planner generation crashes in the original NumPy/SciPy environment. Isolated NumPy 1.26.4
  and SciPy 1.14.1 work; evaluation/training dependencies remain unchanged.

The committed [plan](../artifacts/clean-review/plan.json) fixes validation seeds 20000–20049,
untouched test seeds 40000–40099, checkpoint selection and credibility targets.
New BC training starts fresh because preprocessing changed and v0.2 saved no optimizer state.
It validates every 10k through 100k.
Validation selects 80k (84% success once, 6% at step 200); the final 100k checkpoint scores 66%/6%
despite lower offline MSE. Selected BC scores 86%/10% on training and 58%/12% on diagnostic
held-out seeds. The official Diffusion Policy comparison uses the recorded fallback protocol;
its compute and model size differ. Test seeds are used only after selection.

| Policy | Validation success once / at 200 | Untouched test once / at 200 |
|---|---:|---:|
| BC, selected 80k | 84% / 6% (n=50) | Not evaluated; DP selected before test |
| Official DP, selected 20k | 100% / 100% (n=50) | 100% / 100% (n=100) |

DP 30k ties 20k, so the earlier checkpoint wins. Test Wilson 95% intervals are 96.3–100%
for both metrics. One task, training seed and workstation do not establish general robustness.
The trained EMA weights are published; all ten BC snapshots include optimizer state.
DP inference tensors were verified equal to the parent checkpoint before publication.
Official-recorder videos play at 30 fps; simulation control is 20 Hz (one frame per step,
plus the reset frame). Other diagnostic videos play at 20 fps.

[Learning curve](../artifacts/clean-review/bc-learning-curve.png) ·
[Failure videos](../artifacts/clean-review/phase-videos/) ·
[Test results/videos](../artifacts/clean-review/test/) ·
[Checkpoint provenance](../artifacts/clean-review/dp-20000-ema-provenance.json).

```bash
source .venv/bin/activate
pip install -r requirements-clean.txt
F2F_CLEAN_RUN=runs/clean-reproduce bash scripts/prepare_clean.sh
F2F_CLEAN_RUN=runs/clean-reproduce bash scripts/run_clean_training.sh
```

The script preserves existing runs, selects using validation only, and guards against repeated
test use. Frozen diagnostic references and all BC checkpoints are saved under `artifacts/clean-review/`.
