# Gate B2b: independent confirmation

**STOP / NO-GO for Gate C.** Historical B2/B2a verdicts remain unchanged.
[Protocol](GATE_B2B_PLAN.md) and tested implementation were published in
[`5a2fa49`](https://github.com/Marvelousp4/manipulation-failure-to-fix/commit/5a2fa49984f802a66ad625e0e957f80687ef8f20)
before seeds 70000–70049 ran. Frozen official 20k EMA checkpoint; no training or repairs.
[Original summary](../artifacts/gate-b2b/confirmation-v1/summary.json).

## Validity and stop

All **50 clean/sham pairs passed** exact control-state, action, policy-buffer, RNG,
schedule and before/after-substep dynamics/contact checks. Shape-aware native manifold
comparison retained precision, signed zeros, inactive points, empty manifolds and
multiplicities. Both branches achieved **50/50** success at step 200 (Wilson 95%:
92.9–100%). No clean grasp loss; median verified stable-completion onset latency after
trigger was 1.40 s, with no censoring. Trigger cube heights were 20.03–21.40 mm.

The primary condition ran seeds 70000 and 70001, then stopped at the unchanged
**cube-bottom height < -5 mm** guard on 70001. No thresholds/seeds were adjusted and
nothing resumed. 102 rollouts completed; 101 ordinary records plus the full stopped
pair are preserved. First-primary prefix passed; stopped-case prefix was subsequently
verified offline through substep 235, including the pre-impulse state. That diagnostic
does not alter the recorded STOP.

## Primary and secondary results

**The planned 50-seed primary risk difference and 95% interval are not estimable.**
The two attempted cases are descriptive only, not a confirmed success/failure rate:

| Seed, J=0.256 N·s | Success at 200 | Verified regrasp onset latency | Stable completion latency |
|---|---:|---:|---:|
| 70000 | Yes | 1.15 s from first loss | 3.40 s from trigger |
| 70001 (stopped case) | No | Right-censored | Right-censored |

Seed 70000 completed 1.60 s later than paired clean. Regrasp means three consecutive
reports; completion means five consecutive successes, using their verified onset.
Seed 70001 had neither verified regrasp nor completion by step 200; this is not proof
of permanent inability to recover. **Neither secondary severity, finer-timestep audit,
nor planned deterministic repeats was reached. Severity dependence remains unconfirmed.**

## Physical response and guard limitation

Measured first-10-ms x displacements were **21.59/34.52 mm**, and native post-step x
velocities **0.685/3.496 m/s** for 70000/70001. These are measured in-contact responses;
the separate free-body J/m estimate is approximately 4 m/s. Maximum reported contact
penetrations were 4.78/2.02 mm; the first case approached the 5 mm limit.

On 70001, the first height-guard violation occurred **0.21 s** after the impulse:
cube bottom -5.89 mm, cube x-min **0.571 m**, finite table collision-box x-max
**0.485 m**. The cube was entirely beyond the table. Cube-contact reports were empty
and subsequent vertical acceleration was approximately -9.81 m/s². The inherited
global-height guard rejects off-table falling and cannot distinguish it from penetration.
This stop does **not** demonstrate invalid table contact physics. The guard remains
unchanged; its spatial applicability needs a new preregistered audit.
[Saved-state analysis](../artifacts/gate-b2b/stopped-case-audit/analysis.json) ·
[Trajectory plot](../artifacts/gate-b2b/stopped-case-audit/trajectory.png).

Coarse 100 Hz collision fidelity, timestep sensitivity and nonzero-impulse repeatability
remain unproven. The cube traveled 34.5 mm in the first stopped-case physics step.
A near-table idealized impulse is not a calibrated real-world collision. Updated
observations entered history after 50 ms; normal updated-observation inference occurred
after 300/50 ms in the two cases. Association analysis is incomplete; different grasp
states and contact response preclude attributing recovery to chunk timing.

## Decision and reproduction

**NO-GO:** retain this STOP, assess spatially appropriate penetration/contact validity
under a new protocol, and obtain complete independent confirmation before Gate C.
No claim about repair benefit or scientific novelty follows from these observations.

Prepared WSL environment, pinned `.cache/clean-ManiSkill`, and stored checkpoint required:

```bash
source .venv/bin/activate
F2F_SIM_TESTS=1 F2F_UPSTREAM=/tmp/f2f-ManiSkill OMP_NUM_THREADS=1 python -m pytest -q
# Reproduction of this fixed protocol, not a new independent confirmation:
OMP_NUM_THREADS=1 python scripts/probe_gate_b2b.py --output runs/gate-b2b/reproduce-1
# Offline only; no simulator/policy execution, new output directory required:
python scripts/analyze_gate_b2b_stop.py --input artifacts/gate-b2b/confirmation-v1/failure-pair.json.gz --output runs/gate-b2b/offline-audit-1
sha256sum -c artifacts/gate-b2b/SHA256SUMS
mkdir -p runs/gate-b2b/extracted
for part in artifacts/gate-b2b/confirmation-v1/records/*.tar.xz; do tar -xJf "$part" -C runs/gate-b2b/extracted; done
```

[Record manifest](../artifacts/gate-b2b/confirmation-v1/records-manifest.json) maps every
record to its archive and original hashes; all 101 gzip records reconstruct byte-exactly.
[Three preregistered first-seed videos](../artifacts/gate-b2b/confirmation-v1/videos/)
contain 200 frames at 20 fps; the stopped seed has saved states, not a rerun video.
**65 tests passed before confirmation; 66 now pass**, including the offline geometry
calculation test. Operational criteria stayed frozen throughout. Checkpoint and all
254 earlier result-file checksum checks pass; upstream inference classes are unchanged.
