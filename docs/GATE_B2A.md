# Gate B2a: native contact fingerprint forensics

**A. REPRESENTATION ARTIFACT**, established for the reproduced failure.
**Gate B2 remains STOPPED; Gate C remains NO-GO.** No confirmation, nonzero impulse,
training or recovery experiment ran. Seed 60001 was replayed only under the explicit
forensic authorization; 60000–60049 remain consumed for future confirmation.

## Exact first mismatch

[Captured pair and six raw runs](../artifacts/gate-b2a/forensic-v1/record-index.json),
[first mismatch](../artifacts/gate-b2a/first-mismatch.json),
[float32 bit patterns](../artifacts/gate-b2a/precision-evidence.json).
First mismatch: physics substep **142 zero-based**, control step 29/substep 2, ending
at **1.43 s**, before the sham API call at 2.35 s. Both runs have **4 cube contact
shape-pairs and 9 points**. Raw shape-pair enumeration switches left-finger shapes
**2,3 → 3,2**; body identities/orientations and within-shape point ordering agree.

Legacy sorted points 1 and 2 share the exact key `(other body, position, impulse)`:
`panda_leftfinger`, position approximately `(0.063734487,-0.024591167,0.039999593)` m,
cube-oriented impulse `(+0,+0,-0)` N·s. Their other fields are distinct:

| Finger shape | Cube-oriented normal (rounded) | Separation (m) |
|---|---|---:|
| 2 | (0.015453968, 0.519109428, -0.854568005) | 0.03934623673558235 |
| 3 | (0.013581237, 0.453812003, -0.890994012) | 0.02054712548851967 |

The unchanged serializer omits shape identity, normal and separation from its sort
key. Stable sorting therefore preserves the incoming order for these ties, exchanging
list positions and changing SHA256. It is **not a numerical change in either point**.
All raw vector fields are float32; matched-point signed-zero bits and precision agree.
Later list-index signed-zero differences also disappear when matched by shape identity.
Replaying the captured fields through the original serializer, then swapping only the
two input shape-pair records, reproduces the differing output exactly without physics.

## Three equivalence levels

Six rollouts in fixed order `clean, sham, sham, clean, clean, sham`, 200 control steps
and 1000 physics substeps each; frozen original DP and unchanged Gate B2 serializer.
All 15 pairwise comparisons are diagnostic, not independent statistical samples.

| Comparison | Recorded execution | Substep dynamics/contact response | Strict legacy hashes |
|---|---|---|---|
| Clean–clean (3 pairs) | All exact | All exact | 2/3 differ |
| Sham–sham (3 pairs) | All exact | All exact | 2/3 differ |
| Clean–sham (9 pairs) | All exact | All exact | 5/9 differ |

Every before/after-substep simulator/controller field matches numerically with maximum
error **0**. Control-step observation/action/history/queue/RNG execution and inference
schedules match exactly. Native point multisets, including body/shape geometry identity,
position, normal, separation, impulse, precision and zero signs, match exactly at every
substep. Measured cube momentum-transfer difference and aggregate normal-impulse
response difference are **0**. Each differing pair has 829 legacy hash mismatches.
The recreated clean run also matches all 201 archived original clean state hashes and
its action/grasp/success timeline. [Full comparison matrix](../artifacts/gate-b2a/forensic-v1/summary.json).

## Cause, limits and proposed replacement

SAPIEN 3.0.3 enumerates a map keyed by native shape pointers, rather than logical
shape identity; those addresses are not stable identities across recreated scenes.
[Version-pinned callback source](https://github.com/haosulab/SAPIEN/blob/731622eac5b140b320076c8a1b6eb4b553c3ccd4/include/sapien/physx/simulation_callback.hpp#L78-L87).
PhysX defines world-space points, normals oriented from shape 1 toward shape 0, and
negative separation as penetration; the audit preserves those raw conventions before
orienting onto the cube. [PhysX 5.3.1 API](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/struct_px_contact_pair_point.html).

**Observed:** enumeration order plus incomplete tie-breaking causes the reproduced
failure; same-branch differences and pre-sham onset exclude a zero-impulse explanation
for that mismatch. Neither reduction roundoff nor signed-zero drift is its cause.
**Limit:** the historical failed sham payload was lost, so its exact differing bytes
cannot be recovered. This establishes physical equivalence for these six diagnostic
rollouts on one workstation, not all seeds, nonzero impulses or real impacts.

For a **new preregistration only**, propose exact, shape-aware unordered point-multiset
comparison: stable body/shape/geometry IDs; full native field values and bit patterns;
retain duplicates, inactive zero-impulse points and signed zeros. Canonically order all
fields before hashing/reduction, while independently requiring exact substep dynamics,
policy buffers/RNG and schedules. Reject numeric/manifold changes; do not introduce a
post-hoc tolerance. No replacement criterion was applied to Gate B2. New independent
confirmation and physical-fault plausibility checks are still prerequisites for Gate C.

## Reproduce

```bash
source .venv/bin/activate
python scripts/probe_gate_b2a.py --analyse-only artifacts/gate-b2a/forensic-v1
# Explicit forensic replay only; new output directory, consumed seed 60001:
OMP_NUM_THREADS=1 python scripts/probe_gate_b2a.py --output runs/gate-b2a/diagnostic-2
F2F_SIM_TESTS=1 F2F_UPSTREAM=/tmp/f2f-ManiSkill OMP_NUM_THREADS=1 python -m pytest -q
sha256sum -c artifacts/gate-b2a/SHA256SUMS
```

**47 tests passed**, including captured-field serializer reproduction and detection of
transient substep motion. Original code/results/checkpoint checksums are unchanged.
Executed-source snapshots and hashes pin the run; subsequent edits add offline analysis.
