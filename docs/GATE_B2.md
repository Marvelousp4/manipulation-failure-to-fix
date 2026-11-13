# Gate B2: independent confirmation and physical plausibility audit

**STOP / NO-GO for Gate C.** No independent impulse effect was estimated.
The frozen 20k EMA checkpoint and all previous results retain their original checksums.
[Protocol](../artifacts/gate-b2/plan.json) and tested implementation were published in
[`c09e6fa`](https://github.com/Marvelousp4/manipulation-failure-to-fix/commit/c09e6fa36f195af120e7576bf6a556434fc718a6)
before execution. [Partial evidence](../artifacts/gate-b2/confirmation-v1/summary.json).

## Protocol and observed stop

Fixed fresh seeds 60000–60049; all previously used ranges below 60000 excluded.
Primary: paired clean/sham/0.256 N·s at 200 steps. Secondary: 0.064 and 0.128 N·s,
without severity selection. Same Gate B grasp trigger, native +x COM impulse, CPU
PhysX 100 Hz, control 20 Hz, unchanged policy/action chunks and common random streams.
Preregistered uncertainty used conservative paired discordance intervals; planned finer
200/400 Hz checks, five-seed repeats and representative videos followed validity checks.

Four 200-step control rollouts ran: clean/sham at 60000 and 60001. Both pairs matched
all recorded simulator/controller state, observations, actions, grasp/success flags,
policy histories/queues and RNG hashes with **zero numerical error**. Both branches
were successful at step 200. Native physics-step contact fingerprints matched for
60000 but differed for sham seed **60001**. The runner exited with code 1 immediately.

**Zero nonzero-impulse rollouts ran.** No secondary, timestep, repeat or video stage ran.
Only three branch payloads were saved: the original runner persisted branches after
comparison, losing the failed sham payload. Its failed check and successful full-state
comparison survive in the summary. Failure-pair capture and first-mismatch diagnostics
are now tested; no threshold was relaxed and no experiment was restarted.

## Confirmed effects and rejected hypotheses

- Independent severity-dependent effects are **unconfirmed**. Primary paired final-success
  difference/interval, impulse-induced grasp-loss frequency, regrasp/completion latency,
  censoring rates and chunk-duration associations are **not estimable** from this run.
  Gate B's earlier 4/6 versus 6/6 result remains exploratory evidence only.
- Matching exposed state/action traces does **not** guarantee identical strict serialized
  native-contact fingerprints: the stronger universal-equivalence hypothesis failed.
  This does not establish that sham changed the physical trajectory.
- Synthetic table-contact tests reject treating SAPIEN's reported normal impulses as the
  full contact momentum transfer with friction. Frictionless momentum balance closes;
  with friction, the unreported tangential component is measurable. The audit separates
  native normal impulses from net contact response inferred through cube momentum.

## Physical plausibility and remaining uncertainties

The two saved clean cases triggered at cube center heights 20.76/20.40 mm: near the
z=0 table surface, not airborne. Their contact-window maximum reported penetration was
2.29/2.80 mm; minimum oriented cube bottom was -0.012/-0.004 mm. These passed the
predeclared 5 mm bound. Gripper material friction reaches 2; a conservative friction
compatibility bound is loose and is only a necessary check, not proof of solver fidelity.
No in-contact nonzero-impulse velocity/displacement or contact response was confirmed.

The contact mismatch could involve ordering, signed zero or another representation
change, or native contact nondeterminism. A unit test demonstrates equal numerical
signed zeros can hash differently; it does not diagnose this particular failure.
Without the failed native-contact payload, its magnitude and first differing substep
are unknown. Timestep/contact-offset sensitivity, collision fidelity and real-world
impact validity remain unproven. No recovery, causal chunking or novelty claim is made.

## Decision and executable checks

**NO-GO:** resolve and classify native-contact equivalence with retained failing pairs
before a newly preregistered independent confirmation. Do not silently weaken the
failed criterion or promote Gate B's selected pilot severity to a confirmed result.
Reserve 60000–60049 for this protocol; use a new range for a replacement confirmation.

```bash
source .venv/bin/activate
F2F_SIM_TESTS=1 F2F_UPSTREAM=/tmp/f2f-ManiSkill OMP_NUM_THREADS=1 python -m pytest -q
sha256sum -c artifacts/gate-b2/SHA256SUMS
# Fixed-protocol runner; requires a new output directory, never resumes a stopped run:
OMP_NUM_THREADS=1 python scripts/probe_gate_b2.py --output runs/gate-b2/new-run
```

`F2F_UPSTREAM` identifies the preserved original BC checkout matching its provenance;
the DP runner uses `.cache/clean-ManiSkill`. Current full suite: **43 passed**.
The runner command executes the entire fixed protocol; it has not been rerun after stop.
