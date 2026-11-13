"""Finite collision geometry and exploratory failure categories for Gate B3 only."""

import itertools

import numpy as np

from failure2fix.gate_b import metrics
from failure2fix.gate_b2b import ExactRecorder, bits, compare_exact, digest

EPS = 1e-8


def decode(value):
    if isinstance(value, dict):
        if "array_bits" in value:
            a = value["array_bits"]
            return (
                np.frombuffer(bytes.fromhex(a["bits"]), dtype=a["dtype"]).reshape(a["shape"]).copy()
            )
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def rotation(q):
    q = np.asarray(q, float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def obb(pose, shape):
    r = rotation(pose[3:7])
    local = shape["local_pose"]
    return (
        np.asarray(pose[:3]) + r @ np.asarray(local[:3]),
        r @ rotation(local[3:7]),
        np.asarray(shape["half_size"]),
    )


def vertices(box):
    center, r, half = box
    return np.array(
        [center + r @ (np.asarray(s) * half) for s in itertools.product([-1, 1], repeat=3)]
    )


def penetration(a, b):
    """OBB minimum translation depth by all 15 separating axes; zero when separated."""
    ca, ra, ha = a
    cb, rb, hb = b
    axes = [*ra.T, *rb.T, *[np.cross(x, y) for x in ra.T for y in rb.T]]
    depth = float("inf")
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < EPS:
            continue
        axis = axis / length
        overlap = float(np.abs(ra.T @ axis) @ ha + np.abs(rb.T @ axis) @ hb - abs((cb - ca) @ axis))
        if overlap <= 0:
            return 0.0
        depth = min(depth, overlap)
    return depth


def hull(points):
    points = sorted(set(map(tuple, points)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


def footprint(cube_vertices, table):
    center, r, half = table
    xy = (np.asarray(cube_vertices) - center) @ r
    polygon = hull(xy[:, :2])
    full = bool(np.all(np.abs(xy[:, :2]) <= half[:2] + EPS))
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        if not len(polygon):
            break
        clipped = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            da = sign * a[axis] - half[axis]
            db = sign * b[axis] - half[axis]
            inside_a, inside_b = da <= EPS, db <= EPS
            if inside_a:
                clipped.append(a)
            if inside_a != inside_b:
                clipped.append(a + (b - a) * (da / (da - db)))
        polygon = np.asarray(clipped)
    return "full" if full else ("partial" if len(polygon) else "none")


def spatial(cube_pose, table_pose, cube_shape, table_shape, floor_pose, floor_shape):
    cube, table = obb(cube_pose, cube_shape), obb(table_pose, table_shape)
    points = vertices(cube)
    fr = rotation(floor_pose[3:7])
    local = floor_shape["local_pose"]
    floor_point = np.asarray(floor_pose[:3]) + fr @ np.asarray(local[:3])
    normal = (fr @ rotation(local[3:7]))[:, 0]  # SAPIEN plane's local normal is +x.
    if normal[2] < 0.99:
        raise RuntimeError("Unsupported floor orientation")
    footprint_class = footprint(points, table)
    below_top = bool(np.max((points - table[0]) @ table[1], axis=0)[2] < table[2][2] - EPS)
    return dict(
        footprint=footprint_class,
        support_departed=footprint_class == "none" or below_top,
        entirely_below_tabletop=below_top,
        table_penetration_m=penetration(cube, table),
        floor_penetration_m=max(0.0, float(-np.min((points - floor_point) @ normal))),
    )


def geometry(body, box=True):
    shapes = body.collision_shapes
    expected = "PhysxCollisionShapeBox" if box else "PhysxCollisionShapePlane"
    if len(shapes) != 1 or type(shapes[0]).__name__ != expected:
        raise RuntimeError(f"Unsupported collision geometry: expected one {expected}")
    shape = shapes[0]
    local = shape.local_pose
    native = dict(type=type(shape).__name__, local_p=bits(local.p), local_q=bits(local.q))
    result = dict(body=body.entity.name, local_pose=np.concatenate([local.p, local.q]))
    if box:
        result["half_size"] = shape.half_size.copy()
        native["half_size"] = bits(shape.half_size)
    material = shape.physical_material
    native["material"] = {
        k: bits(np.float32(getattr(material, k)))
        for k in ["static_friction", "dynamic_friction", "restitution"]
    }
    result["geometry_sha256"] = digest(native)
    return result


def pose(body):
    p = body.entity_pose
    return np.concatenate([p.p, p.q]).copy()


def is_ballistic(delta_velocity, active_contacts, gravity, dt):
    return active_contacts == 0 and bool(
        np.allclose(delta_velocity, np.asarray(gravity) * dt, atol=1e-4, rtol=1e-4)
    )


def valid_physics(finite, native_separation, table_depth, floor_depth, friction_excess, impulse):
    return (
        finite
        and native_separation >= -0.005
        and table_depth <= 0.005
        and floor_depth <= 0.005
        and friction_excess <= 0.0002 + 0.005 * impulse
    )


class SpatialRecorder(ExactRecorder):
    def __init__(self):
        super().__init__()
        self.spatial_rows = []
        self.collision_geometry = {}

    def attach(self, raw):
        super().attach(raw)
        old = raw._after_simulation_step

        def after():
            old()
            table = raw.table_scene.table._bodies[0]
            floor = raw.table_scene.ground._bodies[0]
            if not self.collision_geometry:
                self.collision_geometry.update(
                    cube=geometry(self.body), table=geometry(table), floor=geometry(floor, False)
                )
            g = self.collision_geometry
            cp, tp, fp = pose(self.body), pose(table), pose(floor)
            entry = spatial(cp, tp, g["cube"], g["table"], fp, g["floor"])
            row = self.rows[-1]
            active = [p for p in row["contacts"]["points"] if np.linalg.norm(p["impulse"]) > 0]
            delta = row["after"][7:10].astype(float) - row["before"][7:10].astype(float)
            entry.update(
                index=len(self.spatial_rows),
                cube_pose=cp,
                table_pose=tp,
                floor_pose=fp,
                native_min_separation_m=min(
                    (p["separation"] for p in row["contacts"]["points"]), default=0.0
                ),
                native_net_normal_impulse_Ns=row["contacts"]["net_impulse_Ns"],
                active_cube_contacts=len(active),
                native_table_support=any(
                    p["other"] == g["table"]["body"] and p["impulse"][2] > 0 for p in active
                ),
                contact_free_ballistic=is_ballistic(delta, len(active), self.gravity, self.dt),
            )
            self.spatial_rows.append(entry)
            if (
                entry["native_min_separation_m"] < -0.005
                or entry["table_penetration_m"] > 0.005
                or entry["floor_penetration_m"] > 0.005
            ):
                raise RuntimeError(
                    f"Physical invalidity at substep {entry['index']}: native={entry['native_min_separation_m']}, table={entry['table_penetration_m']}, floor={entry['floor_penetration_m']}"
                )

        raw._after_simulation_step = after

    def evidence(self, run):
        result = super().evidence(run)
        result["legacy_global_height_would_reject"] = result["min_cube_bottom_m"] < -0.005
        result["collision_geometry"] = self.collision_geometry
        result["_spatial"] = self.spatial_rows
        result["spatial_hashes"] = [digest(r) for r in self.spatial_rows]
        result["min_native_separation_all_m"] = min(
            (p["separation"] for r in self.rows for p in r["contacts"]["points"]), default=0.0
        )
        result["max_table_penetration_m"] = max(r["table_penetration_m"] for r in self.spatial_rows)
        result["max_floor_penetration_m"] = max(r["floor_penetration_m"] for r in self.spatial_rows)
        result["valid"] = valid_physics(
            result["finite_exact_states_and_contacts"],
            result["min_native_separation_all_m"],
            result["max_table_penetration_m"],
            result["max_floor_penetration_m"],
            result["friction_bound_excess_Ns"],
            run["impulse_Ns"],
        )
        if len(self.spatial_rows) != len(self.rows):
            raise RuntimeError("Incomplete spatial audit")
        return result


def compare(reference, actual, prefix=False):
    check = compare_exact(reference, actual, prefix)
    if not check["pass_"]:
        return check
    a, b = reference[1], actual[1]
    if digest(a["collision_geometry"]) != digest(b["collision_geometry"]):
        return dict(pass_=False, kind="collision-geometry")
    n = a["event_index"] if prefix else len(a["spatial_hashes"])
    if a["spatial_hashes"][:n] != b["spatial_hashes"][:n]:
        return dict(pass_=False, kind="spatial-replay")
    return check


def episode(run, audit):
    result = metrics(run)
    del result["event"]
    event = audit["event_index"]
    dt = audit["dt_s"]
    rows = audit["_spatial"][event:]
    departed = next((r for r in rows if r["support_departed"]), None)
    departure_time = None if departed is None else (departed["index"] + 1) * dt
    loss_time = None if result["grasp_loss_step"] is None else result["grasp_loss_step"] * 0.05
    frames = run["frames"]
    verified_loss = next(
        (
            i
            for i in range(run["trigger"] + 1, len(frames) - 2)
            if all(not f.grasped for f in frames[i : i + 3])
        ),
        None,
    )
    verified_regrasp = (
        None
        if verified_loss is None
        else next(
            (
                i
                for i in range(verified_loss + 3, len(frames) - 2)
                if all(f.grasped for f in frames[i : i + 3])
            ),
            None,
        )
    )
    ordering = (
        "no_reported_loss"
        if loss_time is None
        else (
            "no_departure"
            if departure_time is None
            else (
                "loss_before_departure"
                if loss_time < departure_time
                else (
                    "departure_before_loss_interval"
                    if departure_time < (result["grasp_loss_step"] - 1) * 0.05
                    else "same_control_interval_unresolved"
                )
            )
        )
    )
    result.update(
        verified_loss_step=verified_loss,
        early_verified_loss=verified_loss is not None and verified_loss <= run["trigger"] + 10,
        verified_regrasp_step=verified_regrasp,
        verified_regrasp_latency_s=None
        if verified_regrasp is None
        else (verified_regrasp - verified_loss) * 0.05,
        verified_regrasp_right_censored=verified_loss is not None and verified_regrasp is None,
        first_loss_departure_order=ordering,
        departed=departed is not None,
        departure_time_s=departure_time,
        edge_interaction=any(r["footprint"] == "partial" for r in rows),
        ballistic=any(r["contact_free_ballistic"] for r in rows),
        loss_before_departure=None
        if loss_time is None
        else departure_time is None or loss_time < departure_time,
        failure_adjudicated_at_s=10.0,
        failure_after_departure=(not result["success_at_200"] and departure_time is not None),
        category=("success_after_departure" if departed else "in_workspace_success")
        if result["success_at_200"]
        else ("departure_noncompletion" if departed else "in_workspace_noncompletion"),
        queue_remaining=run["event"]["queue_remaining"],
        inference_feedback_delay_s=(
            run["event"]["first_inference_with_updated_observation"] - run["trigger"] - 1
        )
        * 0.05,
        regrasp_latency_s=None
        if result["regrasp_latency_steps"] is None
        else result["regrasp_latency_steps"] * 0.05,
        completion_latency_s=None
        if result["post_event_goal_latency_steps"] is None
        else result["post_event_goal_latency_steps"] * 0.05,
    )
    return result


def response(run, audit, clean_audit):
    n = audit["event_index"]
    m = round(0.05 / audit["dt_s"])
    actual = audit["_substeps"][n + m - 1]["after"]
    clean = clean_audit["_substeps"][n + m - 1]["after"]
    return dict(
        position_m=(actual[:3] - clean[:3]).tolist(),
        velocity_mps=(actual[7:10] - clean[7:10]).tolist(),
        pre_pose=run["event"]["physics"]["before"][:7],
    )


def select_level(rows):
    levels = []
    for J in [0.064, 0.128, 0.192]:
        group = [r for r in rows if r["freq"] == 100 and r["impulse_Ns"] == J]
        recovered = [
            r["seed"]
            for r in group
            if not r["departed"]
            and r["early_verified_loss"]
            and r["verified_regrasp_step"] is not None
            and r["success_at_200"]
            and r["clean_success"]
        ]
        nonrecovered = [
            r["seed"]
            for r in group
            if not r["departed"]
            and r["early_verified_loss"]
            and not r["success_at_200"]
            and r["clean_success"]
        ]
        departures = sum(r["departed"] for r in group)
        eligible = (
            len(group) == 12 and len(recovered) >= 2 and len(nonrecovered) >= 2 and departures <= 4
        )
        levels.append(
            dict(
                impulse_Ns=J,
                n=len(group),
                successes=sum(r["success_at_200"] for r in group),
                departures=departures,
                recovered=recovered,
                noncompleted=nonrecovered,
                eligible=eligible,
            )
        )
    selected = next((r for r in levels if r["eligible"]), None)
    return levels, selected


def qualitative(row):
    return (
        row["success_at_200"],
        row["departed"],
        "retained"
        if row["verified_loss_step"] is None
        else ("regrasped" if row["verified_regrasp_step"] is not None else "lost"),
    )


def timestep_case(rows):
    from itertools import combinations

    unstable = False
    comparisons = []
    for a, b in combinations(rows, 2):
        x, y = a["response_50ms"], b["response_50ms"]
        dp = float(np.linalg.norm(np.asarray(x["position_m"]) - y["position_m"]))
        dv = float(np.linalg.norm(np.asarray(x["velocity_mps"]) - y["velocity_mps"]))
        speed = max(np.linalg.norm(x["velocity_mps"]), np.linalg.norm(y["velocity_mps"]))
        large = dp > 0.01 or dv > max(0.5, 0.5 * speed)
        different = qualitative(a) != qualitative(b)
        unstable = unstable or large or different
        comparisons.append(
            dict(
                frequencies=[a["freq"], b["freq"]],
                response_delta_position_m=dp,
                response_delta_velocity_mps=dv,
                large_response=bool(large),
                qualitative_changed=different,
                pre_position_difference_m=float(
                    np.linalg.norm(np.asarray(x["pre_pose"][:3]) - y["pre_pose"][:3])
                ),
                trigger_steps=[a["trigger"], b["trigger"]],
                queues=[a["queue_remaining"], b["queue_remaining"]],
            )
        )
    return dict(unstable=bool(unstable), comparisons=comparisons)
