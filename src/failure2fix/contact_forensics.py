"""Forensic diagnostics only; the Gate B2 serializer and validity checks stay unchanged."""

import hashlib
import math
from collections import Counter

import numpy as np

from failure2fix.gate_b2 import ContactRecorder, canonical
from failure2fix.replay_gate import _leaves


def raw_contacts(raw, cube, shapes):
    result = []
    for contact in raw.scene.px.get_contacts():
        if cube not in contact.bodies:
            continue
        identity = []
        for body, shape in zip(contact.bodies, contact.shapes):
            shape_index = body.collision_shapes.index(shape)
            key = (body.entity.name, shape_index)
            if key not in shapes:
                geometry = dict(
                    type=type(shape).__name__,
                    local_p=shape.local_pose.p,
                    local_q=shape.local_pose.q,
                )
                for attr in [
                    "half_size",
                    "vertices",
                    "triangles",
                    "scale",
                    "radius",
                    "half_length",
                ]:
                    if hasattr(shape, attr):
                        geometry[attr] = getattr(shape, attr)
                shapes[key] = hashlib.sha256(canonical(geometry)).hexdigest()
            identity.append(
                dict(body=body.entity.name, shape_index=shape_index, geometry_sha256=shapes[key])
            )
        points = []
        for point in contact.points:
            fields = {k: getattr(point, k).copy() for k in ["position", "normal", "impulse"]}
            points.append(
                dict(
                    **fields,
                    separation=float(point.separation),
                    precision={k: str(v.dtype) for k, v in fields.items()},
                )
            )
        result.append(dict(pair=identity, cube_index=contact.bodies.index(cube), points=points))
    return result


class ForensicRecorder(ContactRecorder):
    def __init__(self):
        super().__init__()
        self.substeps = []
        self.shapes = {}

    def attach(self, raw):
        super().attach(raw)
        before, after = raw._before_simulation_step, raw._after_simulation_step

        def state():
            return {
                **_leaves(raw.get_state_dict(), "env"),
                **_leaves(raw.agent.get_controller_state(), "controller"),
            }

        def pre():
            before()
            self.pre_state = state()

        def post():
            after()
            self.substeps.append(
                dict(
                    before_state=self.pre_state,
                    after_state=state(),
                    raw_contacts=raw_contacts(raw, self.body, self.shapes),
                )
            )

        raw._before_simulation_step, raw._after_simulation_step = pre, post


def differences(a, b, path=""):
    """Every JSON-level difference, preserving signed zero and exact float hex values."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [dict(path=path, kind="keys", left=list(a), right=list(b))]
        return [d for k in a for d in differences(a[k], b[k], path + "/" + str(k))]
    if isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray)):
        if len(a) != len(b):
            return [dict(path=path, kind="length", left=len(a), right=len(b))]
        return [
            d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, path + "/" + str(i))
        ]
    if canonical(a) == canonical(b):
        return []
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        x, y = float(a), float(b)
        return [
            dict(
                path=path,
                kind="signed_zero" if x == y == 0 else "numeric",
                left=x,
                right=y,
                abs_error=abs(x - y),
                left_hex=x.hex(),
                right_hex=y.hex(),
                ulps=abs(x - y) / max(abs(np.spacing(x)), abs(np.spacing(y))),
            )
        ]
    return [dict(path=path, kind="value", left=a, right=b)]


def stable_response(sample):
    """Diagnostic order-independent sums; never used to pass the original gate."""
    points = sample["points"]
    return dict(
        net_impulse_Ns=[math.fsum(float(p["impulse"][i]) for p in points) for i in range(3)],
        friction_bound_Ns=math.fsum(
            p["friction_bound_mu"] * float(np.linalg.norm(p["impulse"])) for p in points
        ),
    )


def logical_points(raw):
    """Retain logical body/shape identity while comparing unordered native manifolds."""
    result = []
    for contact in raw:
        index = contact["cube_index"]
        sign = 1 if index == 0 else -1
        for point in contact["points"]:
            result.append(
                dict(
                    cube=contact["pair"][index],
                    other=contact["pair"][1 - index],
                    position=point["position"],
                    normal=sign * np.asarray(point["normal"]),
                    impulse=sign * np.asarray(point["impulse"]),
                    separation=point["separation"],
                    precision=point["precision"],
                )
            )
    return sorted(result, key=canonical)


def exact_dynamics(a, b):
    result = dict(
        numeric_equal=True, max_abs_error=0.0, first_mismatch=None, signed_zero_differences=0
    )
    for step, (x, y) in enumerate(zip(a, b)):
        for phase in ["before_state", "after_state"]:
            if x[phase].keys() != y[phase].keys():
                raise RuntimeError("State schema changed")
            for key in x[phase]:
                p, q = np.asarray(x[phase][key]), np.asarray(y[phase][key])
                if p.shape != q.shape:
                    raise RuntimeError("State shape changed")
                error = float(np.max(np.abs(p.astype(float) - q.astype(float)))) if p.size else 0.0
                result["max_abs_error"] = max(error, result["max_abs_error"])
                result["signed_zero_differences"] += int(
                    np.sum((p == 0) & (q == 0) & (np.signbit(p) != np.signbit(q)))
                )
                if not np.array_equal(p, q):
                    result["numeric_equal"] = False
                    if result["first_mismatch"] is None:
                        result["first_mismatch"] = dict(
                            substep=step, phase=phase, field=key, max_abs_error=error
                        )
    if len(a) != len(b):
        raise RuntimeError("Incomplete substep dynamics comparison")
    return result


def compare_forensic(reference, actual):
    a, b = reference[1], actual[1]
    rows_a, rows_b = a["_substeps"], b["_substeps"]
    indices = [
        i for i, (p, q) in enumerate(zip(a["full_fingerprints"], b["full_fingerprints"])) if p != q
    ]
    field_counts = Counter()
    point_differences, stable_differences, momentum_difference = 0, 0, 0.0
    raw_order_differences = 0
    native_multiset_mismatches = []
    body_orientation_changes, within_shape_point_changes = 0, 0
    for i, (p, q) in enumerate(zip(rows_a, rows_b)):
        if i in indices:
            field_counts.update(d["path"] for d in differences(p, q))
        for phase in ["before_contacts", "contacts"]:
            point_differences += canonical(p[phase]["points"]) != canonical(q[phase]["points"])
            stable_differences += canonical(stable_response(p[phase])) != canonical(
                stable_response(q[phase])
            )
        # Measured net momentum transfer, independent of contact-report aggregation.
        pa = (np.asarray(p["after"][7:10], float) - p["before"][7:10]) * a["mass_kg"]
        pb = (np.asarray(q["after"][7:10], float) - q["before"][7:10]) * b["mass_kg"]
        momentum_difference = max(momentum_difference, float(np.max(np.abs(pa - pb))))
        raw_a = a["_forensic"][i]["raw_contacts"]
        raw_b = b["_forensic"][i]["raw_contacts"]
        raw_order_differences += canonical(raw_a) != canonical(raw_b)
        if canonical(logical_points(raw_a)) != canonical(logical_points(raw_b)):
            native_multiset_mismatches.append(i)

        def by_shape_pair(raw):
            return {canonical(sorted(c["pair"], key=canonical)): c for c in raw}

        ma, mb = by_shape_pair(raw_a), by_shape_pair(raw_b)
        if ma.keys() == mb.keys():
            for key in ma:
                body_orientation_changes += canonical(ma[key]["pair"]) != canonical(mb[key]["pair"])
                within_shape_point_changes += canonical(ma[key]["points"]) != canonical(
                    mb[key]["points"]
                )
        else:
            within_shape_point_changes += 1

    first = indices[0] if indices else None
    evidence = None
    if first is not None:
        ra, rb = a["_forensic"][first]["raw_contacts"], b["_forensic"][first]["raw_contacts"]
        evidence = dict(
            substep_zero_based=first,
            control_step=first // 5 + 1,
            substep_within_control_zero_based=first % 5,
            end_time_s=(first + 1) * 0.01,
            event_substep=a["event_index"],
            legacy_differences=differences(rows_a[first], rows_b[first]),
            raw_contact_counts=[len(ra), len(rb)],
            raw_point_counts=[sum(len(c["points"]) for c in r) for r in [ra, rb]],
            normalized_native_points_equal=canonical(logical_points(ra))
            == canonical(logical_points(rb)),
            raw_reference=ra,
            raw_actual=rb,
            legacy_reference=rows_a[first],
            legacy_actual=rows_b[first],
        )
    return dict(
        strict_fingerprints_equal=not indices,
        mismatching_substeps=indices,
        mismatching_fields=dict(field_counts),
        sorted_point_sample_differences=point_differences,
        order_independent_response_differences=stable_differences,
        raw_order_or_field_differences=raw_order_differences,
        native_point_multiset_mismatches=native_multiset_mismatches,
        native_body_pair_orientation_changes=body_orientation_changes,
        native_within_shape_point_changes=within_shape_point_changes,
        measured_momentum_max_abs_difference_Ns=momentum_difference,
        dynamics=exact_dynamics(a["_forensic"], b["_forensic"]),
        first=evidence,
    )
