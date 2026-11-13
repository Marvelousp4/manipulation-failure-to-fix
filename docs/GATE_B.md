# Gate B: controlled physical fault feasibility

**PASS on this workstation.** Frozen official 20k EMA DP; no training or weight changes.
[Plan](../artifacts/gate-b/plan.json) was published in `565ffe3` before execution.
[Summary](../artifacts/gate-b/pilot-v1/summary.json), [cases](../artifacts/gate-b/selected-cases.json),
[traces and videos](../artifacts/gate-b/pilot-v1/). Existing result checksums still pass.

## Protocol

Six fresh development seeds, 52000–52005, every severity, no exclusions. Locked validation
20000–20049 and previous test 40000–40099 were unused. State observations, CPU PhysX,
`pd_ee_delta_pos`, 200 control steps (50 ms each; physics 10 ms), neural inference on GPU.
Checkpoint SHA256: `7a6af7f4e51389a54bc834807348c6866a7babd702c9be7a8ea655011cb9fd53`.

After three consecutive grasp reports, select the next normal action, then apply one
world +x impulse at the cube COM before its first physics substep. Native SAPIEN
`add_force_at_point(..., mode="impulse")`; no pose/velocity setters. Cube mass 0.064 kg.
Free-body checks at every severity verified one-shot Δv=J/m, zero COM torque and no
immediate pose mutation. This validates API units, not real impact fidelity.

Branches replay from identical resets rather than restoring snapshots. Clean makes no
impulse API call; sham calls it with zero impulse. Prefix checks include actions,
observations, simulator/controller state, policy history/queue and RNG hashes. The policy
queue is never cleared. Updated observations first enter policy history 50 ms after the
kick; normal inference consumes them after 50–350 ms, depending on the existing chunk.
Inference schedules and random draws match across branches throughout the episode.

## Results

| Branch / impulse (N·s) | Success at step 200 | Wilson 95% interval |
|---|---:|---:|
| Clean | 6/6 | 61–100% |
| Sham | 6/6 | 61–100% |
| 0.004 | 6/6 | 61–100% |
| 0.016 | 6/6 | 61–100% |
| 0.064 | 6/6 | 61–100% |
| 0.256 (selected) | 4/6 | 30–90% |

The predeclared rule selected 0.256 N·s: two paired regressions, four preserved successes,
no improvements; no midpoint refinement. Its free-body Δv would be 4 m/s; constrained
contact response is measured, not prescribed. All 6 full-horizon sham comparisons and
24 disturbance prefix checks had zero numerical error. All 24 RNG/schedule checks passed.
All 18 selected-configuration repeat traces were byte-identical. Six representative video
reruns matched their original traces exactly; each video contains 200 frames at 20 fps.
Repeats do not increase the six-seed sample size. Automated suite: **34 passed**.

At selected severity, grasp loss occurred in three seeds. Seed 52005 regained sustained
grasp in 2.9 s and completed, 3.3 s later than paired clean. Seed 52002 regained grasp in
6.2 s but did not complete by step 200; its video shows displacement at grasp, delayed
reacquisition and incomplete transport. Seed 52004 never regained sustained grasp by the
cutoff (right-censored). Other seeds retained grasp. Regrasp means three consecutive grasp
reports; completion latency uses five consecutive success reports, separately from final
success. All events precede first goal success, so completion is not restoration of a
previously successful goal state.

## Limits and reproduction

The event is **post-grasp, pre-lift**: cube center height 20.3–21.6 mm, near the table.
This establishes a reproducible idealized bump, not airborne disturbance validity.
One task, policy, direction and timestep; no real-impact calibration or timestep convergence
study. Grasp flags are contact predicates. Severity selection used this same exploratory
pilot: intervals are descriptive, not confirmatory. Generalization, repair benefit and
scientific novelty remain unproven. No repair or further benchmark was implemented.

Prepared WSL environment and upstream checkout from the clean baseline are required:

```bash
source .venv/bin/activate
pytest -q
OMP_NUM_THREADS=1 python scripts/probe_gate_b.py --output runs/gate-b/reproduce-1
sha256sum -c artifacts/gate-b/SHA256SUMS
```

Output must be new; invalid physics, sham/prefix divergence or nonreproduction stops the
runner. Full configurations, pre/post impulse pose/velocity, grasp/action/success timelines,
RNG fingerprints, queues and latency censoring are archived with the videos.
