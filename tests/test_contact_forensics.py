import copy
import json
from pathlib import Path
from types import SimpleNamespace as S

import numpy as np

from failure2fix.contact_forensics import exact_dynamics, logical_points
from failure2fix.gate_b2 import canonical, compare_contact, contact_sample

FIRST = Path(__file__).resolve().parents[1] / "artifacts/gate-b2a/first-mismatch.json"


def replay_serializer(raw_contacts, coefficients):
    """Feed captured float32 native fields back into the unchanged B2 serializer."""
    bodies = {p["body"]: S(entity=S(name=p["body"])) for c in raw_contacts for p in c["pair"]}
    cube = bodies["scene-0_cube"]
    contacts = []
    for contact in raw_contacts:
        other = contact["pair"][1 - contact["cube_index"]]["body"]
        mu = coefficients[other]
        material = S(static_friction=mu, dynamic_friction=mu)
        contacts.append(
            S(
                bodies=[bodies[p["body"]] for p in contact["pair"]],
                shapes=[S(physical_material=material)] * 2,
                points=[
                    S(
                        position=np.asarray(p["position"], np.float32),
                        normal=np.asarray(p["normal"], np.float32),
                        impulse=np.asarray(p["impulse"], np.float32),
                        separation=p["separation"],
                    )
                    for p in contact["points"]
                ],
            )
        )
    return contact_sample(S(scene=S(px=S(get_contacts=lambda: contacts))), cube)


def test_captured_shape_order_swap_reproduces_original_sha_failure_without_physics():
    f = json.loads(FIRST.read_text())
    coefficients = {
        p["other"]: p["friction_bound_mu"] for p in f["legacy_reference"]["contacts"]["points"]
    }
    left = replay_serializer(f["raw_reference"], coefficients)
    right = replay_serializer(f["raw_actual"], coefficients)
    assert canonical(left) == canonical(f["legacy_reference"]["contacts"])
    assert canonical(right) == canonical(f["legacy_actual"]["contacts"])
    assert canonical(left) != canonical(right)
    # Reorder only the two native shape-pair records; their field values stay fixed.
    swapped = f["raw_reference"].copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert canonical(replay_serializer(swapped, coefficients)) == canonical(right)
    assert canonical(logical_points(f["raw_reference"])) == canonical(
        logical_points(f["raw_actual"])
    )
    assert not compare_contact(
        dict(full_fingerprints=[canonical(left)]), dict(full_fingerprints=[canonical(right)])
    )


def test_original_sort_key_collides_for_distinct_shapes_with_zero_impulse():
    f = json.loads(FIRST.read_text())
    a, b = f["legacy_reference"]["contacts"]["points"][1:3]

    def key(p):
        return (p["other"], *p["position"], *p["impulse"])

    assert key(a) == key(b)
    assert a["normal"] != b["normal"] and a["separation"] != b["separation"]
    assert np.linalg.norm(a["impulse"]) == np.linalg.norm(b["impulse"]) == 0
    assert f["substep_zero_based"] < f["event_substep"]


def test_shape_aware_diagnostic_preserves_zero_impulse_geometry_differences():
    f = json.loads(FIRST.read_text())
    changed = copy.deepcopy(f["raw_actual"])
    changed[0]["pair"][0]["shape_index"] += 1
    assert canonical(logical_points(changed)) != canonical(logical_points(f["raw_reference"]))
    changed = copy.deepcopy(f["raw_actual"])
    changed[0]["points"][0]["separation"] += 1e-6
    assert canonical(logical_points(changed)) != canonical(logical_points(f["raw_reference"]))


def test_substep_check_detects_transient_motion_between_matching_control_frames():
    left = [
        dict(before_state={"env/cube": np.zeros(3)}, after_state={"env/cube": np.zeros(3)})
        for _ in range(5)
    ]
    right = copy.deepcopy(left)
    right[2]["after_state"]["env/cube"][0] = 1e-6
    d = exact_dynamics(left, right)
    assert not d["numeric_equal"] and d["first_mismatch"]["substep"] == 2
    assert d["max_abs_error"] == 1e-6
