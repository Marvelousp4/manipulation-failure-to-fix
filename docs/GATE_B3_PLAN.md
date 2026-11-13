# Gate B3: exploratory physical-failure feasibility

This protocol and tested implementation are committed/published before any pilot seed.
[Machine plan](../artifacts/gate-b3/plan.json) is authoritative. Historical B2/B2b stay
STOPPED; B2a's accepted diagnosis is unchanged. This is not another confirmation.

**Frozen:** official 20k EMA DP (SHA256
`7a6af7f4e51389a54bc834807348c6866a7babd702c9be7a8ea655011cb9fd53`),
PickCube-v1 state, original `pd_ee_delta_pos`, 200 control steps at 20 Hz, original
three-consecutive-grasp trigger, unchanged action chunks/history/RNG, native one-shot
world +x COM impulse. CPU PhysX/rendering, CUDA policy. Never retrain or modify weights.

**Development pilot:** exactly **80000–80011**, 12 fresh seeds, all at **0.064, 0.128,
0.192 N·s**, plus paired clean/sham. 0.256 is only the archived stress reference.
Exclude all earlier ranges; no exclusions/replacements or additional severity search.
Run all clean/sham first, require >=11/12 clean final successes, then each severity
on all seeds in ascending order. Keep every off-table case in overall rates.

**Physical validity:** native cube/table collision boxes and floor plane, their local
poses and current body poses. Fifteen-axis oriented-box SAT gives table penetration;
actual transformed cube vertices give floor-halfspace penetration. Clip their projected
convex footprint against the finite table rectangle to identify full/partial/no overlap.
Footprint is potential support, recorded separately from active native table impulse.
No center-position threshold and no global z=0 veto. Geometric arithmetic epsilon is
1e-8 m. Ballistic requires no active cube-contact impulse and delta-v=gravity*dt within
atol=rtol=1e-4; keep this flag separate from workspace location.

Stop on any nonfinite exact data, **native contact separation < -5 mm**, table/floor
geometric penetration **>5 mm** anywhere in the horizon, nonzero linear damping, or
first-step momentum/friction-envelope excess >`0.0002+0.005*J` N·s. The last audit
retains B2's necessary friction compatibility check; normal reports omit tangential
friction. Stop on unsupported geometry, sham/repeat mismatch or unexpected error,
preserve full/partial failed pair and return C/PIVOT. Valid departure/free fall is
an outcome, not invalidity. Thresholds remain fixed after the pilot starts.

**Pairing:** retain B2b's exact semantic contact comparison (all scene manifolds,
precision/zero signs/multiplicities), before/after-substep dynamics/controller/RNG,
control observations/actions/policy history/queue/RNG and unchanged inference schedule.
Check disturbance prefix through pre-impulse command/substep. No snapshot restoration.

**Outcomes:** any loss plus verified sustained loss (three consecutive false reports),
verified regrasp (three true after sustained loss), completion (five successes), onset
latencies and right censoring. Eligible fault loss starts within ten post-kick control
steps. Final noncompletion is adjudicated at step 200; record loss/departure order and
20 Hz timing ambiguity, rather than inventing early irreversible task-failure times.
Departure means projected collision footprint is completely outside the tabletop or
all cube collision vertices are below its top plane, including later returns. Report in-workspace noncompletion and departure noncompletion
separately, with all 12 in each severity's denominator. Record actual pose/velocity,
displacement/contact response and time until updated observations enter normal inference.

**Exploratory A criterion:** at least one severity has >=2 clean-success paired cases
with early sustained loss, verified regrasp and final success, and >=2 with early
sustained loss and final noncompletion; all four must stay inside the support workspace
throughout. At most 4/12 departures. Lowest eligible severity is the candidate. No
confirmatory p-values, new severity or post-hoc threshold adjustment.

**Reproducibility/timestep:** repeat first three seeds at all five 100 Hz branches.
Then first three seeds at clean/sham/all severities at **200 and 400 Hz** physics,
keeping 20 Hz control; combine with existing 100 Hz cases (nine disturbed triples).
Pair within each frequency, and report pre-event pose/trigger/queue differences across
frequencies: these can confound integration effects with closed-loop history changes.
A large change is any final-success/departure/sustained-loss-regrasp category change,
or a between-frequency difference in the paired response at common +50 ms exceeding
**10 mm**, or velocity difference >max(**0.5 m/s**, **50%** of max response-vector speed).
Do not compare first substeps as equal durations. >=2/9 unstable triples, or any
fine-rate clean noncompletion, yields C. No causal recovery interpretation.

If candidate and fixed audit remain viable, audit first candidate recovered and first
candidate noncompleted seed: exact 100 Hz clean/sham/candidate repeat and paired
200/400 Hz cases, skipping already audited configurations. Any key-case instability
also yields C. No additional seed search. First-seed videos are recorded live; no
outcome-chosen policy reruns after a stop. Repeats/audits never increase n=12.

**Decision:** A/CONTINUE requires mixed in-workspace outcomes and all validity,
repeatability and timestep checks. C/PIVOT for invalid/unreproducible/timestep-sensitive
physical results; otherwise B/PIVOT, no suitable mixed regime. B/C explicitly means
**abandon this instantaneous impulse configuration for recovery research** and choose
another fault mechanism/setup, not another debugging gate. No real-world fidelity claim.

Offline validation uses archived 70001 only, with native geometry SHA256 verification
and recorded per-substep poses. Seed 90000 is reserved solely for a two-step zero-action
API smoke test without policy inference; it is not a pilot or baseline result.

```bash
source .venv/bin/activate
python scripts/audit_gate_b3_geometry.py --output runs/gate-b3/offline-check.json
F2F_SIM_TESTS=1 F2F_UPSTREAM=/tmp/f2f-ManiSkill OMP_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 python scripts/probe_gate_b3.py --output runs/gate-b3/pilot-v1
```
