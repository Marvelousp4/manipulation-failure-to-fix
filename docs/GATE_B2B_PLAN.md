# Gate B2b: preregistered independent confirmation

Committed and published before confirmation; machine protocol:
[plan.json](../artifacts/gate-b2b/plan.json). B2 remains historically STOPPED; B2a remains
accepted within its scope. Gate C is NO-GO until this protocol passes.

## Fixed experiment

Exactly 50 fresh paired seeds **70000–70049**. No earlier range, exclusion, replacement,
severity selection, threshold adjustment or policy training. Official DP **20k EMA**,
SHA256 `7a6af7f4e51389a54bc834807348c6866a7babd702c9be7a8ea655011cb9fd53`.
PickCube-v1, state observations, `pd_ee_delta_pos`, CPU PhysX/renderer, neural GPU,
200 control steps, 20 Hz control / 100 Hz physics.

Original Gate B trigger: first three consecutive grasp reports, at least eight steps
remaining. Select the next normal command, then one native world **+x COM impulse**
before its first physics substep. No pose/velocity setters or queue/history resets.
Primary **0.256 N·s versus paired clean**; negative control zero-impulse sham versus
clean; secondary **0.064 and 0.128 N·s** descriptive only. Keep all unsuccessful cases.

Run all 50 clean/sham pairs first; only after all pass run the 50 primary pairs, then
both fixed secondary levels on all seeds. No-grasp/undefined-trigger cases stop rather
than disappear. Free-body API calibration precedes rollouts; it is not contact validity.

## Corrected exact criterion

B2a established a representation artifact from pointer-dependent enumeration and tied
sort keys. Replace list identity with exact native manifold multisets, strengthening
scope to **all scene contacts**. Logical identity is unique body name/component, shape
index and a digest of typed geometry/material values; never a pointer address.
Canonicalize body-pair orientation and sort complete records. Preserve every point,
empty manifold, duplicate, inactive contact, position, normal, separation, impulse,
precision and signed-zero bit representation. No approximate comparison or rounding.

Require exact env/controller/global RNG before/after every physics substep and exact
control-frame observations, action, policy history/queue and RNG. Policy runs only at
normal control boundaries. Compare the pre-impulse command and substep; disturbances
may differ only afterward. Inference schedule and common policy random streams remain
identical throughout. First updated observation enters history after one control step;
normal inference is recorded, not forced early. Sham and deterministic repeats must
match for the complete horizon, including transient substep states.

## Unchanged physical stops and measurements

Cube audit window: two control steps before the event through ten from the kick (0.5 s).
Stop on nonfinite data, cube bottom or reported cube-contact separation below **-5 mm**,
nonzero cube linear damping, or first-substep inferred unreported contact impulse beyond
`sum(max(pair friction)*reported normal impulse norm) + 0.0002 + 0.005*J` N·s.
Stop if no primary paired cube displacement exceeds 1 µm. Friction compatibility is
necessary, not sufficient: native normal reports omit tangential friction. Keep measured
pose/velocity, displacement, native contact response and momentum-inferred response
separate from free-body J/m; do not equate this idealized impulse with a real collision.

After fixed severities, audit first three seeds at 200 and 400 Hz physics, with independent
clean/sham/primary pairs and unchanged 20 Hz control. Apply the same stops; do not pool
these cases with n=50. Outcome sensitivity alone is reported, not declared invalidity.
Repeat first five seeds at all five 100 Hz branches; any mismatch stops. Record first
seed live in all branches; then, only if valid, record predeclared representative reruns
(first seed/regression/preserved success/successful regrasp), deduplicated and checked.
Every stop saves full/partial typed failure-pair evidence; never resume or rerun after stop.

## Endpoints, uncertainty and decision

Preserve B2's statistical rule. Final success at step 200; paired improvement/regression
counts. Risk-difference 95% conservative interval: Bonferroni simultaneous 97.5% exact
Clopper–Pearson intervals for discordant improvement/regression probabilities, subtract
endpoints. Primary two-sided exact discordant binomial test; secondary results descriptive.
Success/grasp-loss Wilson 95% intervals. Repeats/timestep cases never increase n=50.

Report any and early (ten-step) grasp loss, first three-report sustained regrasp from
first loss, first five-report stable completion from event, and right censoring.
Report observed medians and Kaplan–Meier medians, verified-onset cutoff horizon-2 for
regrasp/horizon-4 for completion. Conditional regrasp is not a causal recovery estimate.
Exploratory chunk association: Spearman ranks and fixed remaining-queue strata 0–3
versus 4–7; keep censored cases in outcome rates, label observed-only latency association.

**GO only if:** every validity/physical/repeat/video check passes, primary paired-risk
95% upper bound is below zero, at least five regressions and five preserved successes,
and measured primary displacement. Otherwise **NO-GO**. GO is limited to this idealized
simulated near-table fault; no real-world impact or repair-benefit claim.

```bash
source .venv/bin/activate
F2F_SIM_TESTS=1 F2F_UPSTREAM=/tmp/f2f-ManiSkill OMP_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 python scripts/probe_gate_b2b.py --output runs/gate-b2b/confirmation-v1
```
