"""Exact semantic contact and substep checks for the separately preregistered Gate B2b."""

import gzip
import hashlib
from pathlib import Path

import numpy as np

from failure2fix.gate_b import compare_branches, compare_schedule, rng_state
from failure2fix.gate_b2 import ContactRecorder, canonical
from failure2fix.replay_gate import _leaves


def bits(value):
    a = np.asarray(value)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"Unsupported numeric dtype: {a.dtype}")
    return dict(
        dtype=a.dtype.str,
        shape=list(a.shape),
        bits=a.tobytes(order="C").hex(),
        finite=bool(np.isfinite(a).all()),
    )


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_evidence(value):
    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        return dict(array_bits=bits(value))
    if isinstance(value, float) and not np.isfinite(value):
        return dict(array_bits=bits(value))
    if isinstance(value, dict):
        return {k: safe_evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_evidence(v) for v in value]
    return value


def save_failure(path, reference, actual):
    def encode(pair):
        if pair is None:
            return None
        run, audit = pair
        return dict(
            run=safe_evidence({k: v for k, v in run.items() if k != "frames"}),
            frames=[safe_evidence(vars(f)) for f in run.get("frames", [])],
            audit=safe_evidence(audit),
        )

    Path(path).write_bytes(
        gzip.compress(
            canonical(
                dict(schema="typed-arrays-v1", reference=encode(reference), actual=encode(actual))
            ),
            mtime=0,
        )
    )


def all_finite(value):
    if isinstance(value, dict):
        return value["finite"] if "finite" in value else all(all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v) for v in value)
    return True


def contact_multiset(contacts):
    """Full contact records, sorted without dropping empty manifolds or multiplicities."""
    result = []
    for contact in contacts:
        pair = contact["pair"]
        reverse = canonical(pair[0]) > canonical(pair[1])
        points = []
        for point in contact["points"]:
            fields = {}
            for key in ["position", "normal", "impulse"]:
                original = np.asarray(point[key])
                a = np.asarray(point[key], dtype=point["precision"][key])
                if not np.array_equal(a, original, equal_nan=True):
                    raise ValueError("Native field cannot be represented in its declared precision")
                if key != "position" and reverse:
                    a = -a  # Exact orientation change, including zero sign bits.
                fields[key] = bits(a)
            sep = float(point["separation"])
            if not (np.isnan(sep) or float(np.float32(sep)) == sep):
                raise ValueError("Native separation is not an exact float32 value")
            fields["separation"] = bits(np.float32(sep))
            points.append(fields)
        result.append(
            dict(pair=pair[::-1] if reverse else pair, points=sorted(points, key=canonical))
        )
    return sorted(result, key=canonical)


def read_contacts(raw, identities, bodies):
    """All scene contacts. Native object equality only checks within-scene name uniqueness."""
    result = []
    for contact in raw.scene.px.get_contacts():
        pair = []
        for body, shape in zip(contact.bodies, contact.shapes):
            name = body.entity.name
            if name in bodies and bodies[name] != body:
                raise RuntimeError(f"Ambiguous logical body name: {name}")
            bodies[name] = body
            index = body.collision_shapes.index(shape)
            key = (name, index)
            if key not in identities:
                geometry = dict(
                    type=type(shape).__name__,
                    local_p=bits(shape.local_pose.p),
                    local_q=bits(shape.local_pose.q),
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
                        geometry[attr] = bits(getattr(shape, attr))
                material = shape.physical_material
                geometry["material"] = {
                    k: bits(np.float32(getattr(material, k)))
                    for k in ["static_friction", "dynamic_friction", "restitution"]
                }
                identities[key] = dict(
                    body=name,
                    component=type(body).__name__,
                    shape_index=index,
                    geometry_sha256=digest(geometry),
                )
            pair.append(identities[key])
        points = []
        for p in contact.points:
            vectors = {k: getattr(p, k).copy() for k in ["position", "normal", "impulse"]}
            points.append(
                dict(
                    **vectors,
                    separation=float(p.separation),
                    precision={k: str(v.dtype) for k, v in vectors.items()},
                )
            )
        result.append(dict(pair=pair, points=points))
    return contact_multiset(result)


class ExactRecorder(ContactRecorder):
    def __init__(self):
        super().__init__()
        self.exact = []
        self.identities, self.bodies = {}, {}

    def attach(self, raw):
        super().attach(raw)
        before, after = raw._before_simulation_step, raw._after_simulation_step

        def snapshot():
            fields = {
                **_leaves(raw.get_state_dict(), "env"),
                **_leaves(raw.agent.get_controller_state(), "controller"),
                **_leaves(rng_state(), "rng"),
            }
            return dict(
                state={k: bits(v) for k, v in fields.items()},
                contacts=read_contacts(raw, self.identities, self.bodies),
            )

        def pre():
            before()
            self.pre_exact = snapshot()

        def post():
            try:
                after()
            finally:
                self.exact.append(dict(before=self.pre_exact, after=snapshot()))

        raw._before_simulation_step, raw._after_simulation_step = pre, post

    def evidence(self, run):
        result = super().evidence(run)
        del result["full_fingerprints"]  # Historical list-order hashes are not a B2b criterion.
        result["_substeps"] = self.rows
        result["_exact"] = self.exact
        result["exact_hashes"] = [
            dict(before=digest(r["before"]), after=digest(r["after"])) for r in self.exact
        ]
        if len(self.exact) != len(self.rows):
            raise RuntimeError("Incomplete exact substep recording")
        result["finite_exact_states_and_contacts"] = all_finite(self.exact)
        result["valid"] = result["valid"] and result["finite_exact_states_and_contacts"]
        result["logical_shape_inventory"] = sorted(self.identities.values(), key=canonical)
        return result


def compare_exact(reference, actual, prefix=False):
    c, a = reference[0], actual[0]
    check = compare_branches(c, a, prefix_only=prefix, atol=0)
    if not check["pass"]:
        return dict(pass_=False, kind="recorded-execution", detail=check)
    control_n = c["trigger"] + 1 if prefix else len(c["frames"])
    for x, y in zip(c["frames"][:control_n], a["frames"][:control_n]):
        if digest({k: bits(v) for k, v in x.state.items()}) != digest(
            {k: bits(v) for k, v in y.state.items()}
        ):
            return dict(pass_=False, kind="control-state-bits", control_step=x.step)
    left, right = reference[1]["exact_hashes"], actual[1]["exact_hashes"]
    n = reference[1]["event_index"] if prefix else len(left)
    if len(left) != len(right):
        return dict(pass_=False, kind="substep-count")
    for i in range(n + int(prefix)):
        phases = ["before"] if prefix and i == n else ["before", "after"]
        for phase in phases:
            if left[i][phase] != right[i][phase]:
                return dict(pass_=False, kind="substep", substep=i, phase=phase)
    schedule = compare_schedule(c, a)
    if not schedule["pass"]:
        return dict(pass_=False, kind="RNG-inference-schedule", detail=schedule)
    return dict(
        pass_=True, control_frames=control_n, substeps=n, pre_impulse_substep_checked=prefix
    )
