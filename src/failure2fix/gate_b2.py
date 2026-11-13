"""Read-only contact audit and paired confirmation statistics for Gate B2."""

import hashlib
import json

import numpy as np

from failure2fix.gate_b import cube_state, json_default


def canonical(value):
    return json.dumps(value, default=json_default, sort_keys=True, allow_nan=False).encode()


def contact_sample(raw, body):
    contacts = []
    total = np.zeros(3)
    friction_bound = 0.0
    for contact in raw.scene.px.get_contacts():
        if body not in contact.bodies:
            continue
        index = contact.bodies.index(body)
        other = contact.bodies[1 - index].entity.name
        sign = 1 if index == 0 else -1  # ManiSkill sapien_utils contact convention.
        materials = [shape.physical_material for shape in contact.shapes]
        mu = max(max(m.static_friction, m.dynamic_friction) for m in materials)
        for point in contact.points:
            impulse = sign * np.asarray(point.impulse, dtype=float)
            total += impulse
            friction_bound += mu * float(np.linalg.norm(impulse))
            contacts.append(
                dict(
                    other=other,
                    friction_bound_mu=mu,
                    impulse=impulse,
                    position=point.position.copy(),
                    normal=sign * point.normal,
                    separation=float(point.separation),
                )
            )
    contacts.sort(key=lambda p: (p["other"], *p["position"], *p["impulse"]))
    return dict(points=contacts, net_impulse_Ns=total, friction_bound_Ns=friction_bound)


def bottom_height(pose, half_size):
    # Quaternion is w,x,y,z. Exact support of an oriented cube along world z.
    w, x, y, z = pose[3:7]
    row = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return float(pose[2] - half_size * np.abs(row).sum())


class ContactRecorder:
    """Installed before Gate B hooks; never changes state or consumes policy RNG."""

    def __init__(self):
        self.rows = []

    def attach(self, raw):
        self.raw = raw
        self.body = raw.cube._bodies[0]
        self.dt = float(raw.sim_timestep)
        self.control_dt = float(raw.control_timestep)
        self.mass = float(self.body.mass)
        self.half_size = float(raw.cube_half_size)
        self.gravity = np.asarray(raw.sim_config.scene_config.gravity, dtype=float)
        if self.body.linear_damping != 0:
            raise RuntimeError("STOP: momentum audit requires zero cube linear damping")
        before, after = raw._before_simulation_step, raw._after_simulation_step

        def pre():
            before()
            self.body = raw.cube._bodies[0]
            self.pre = cube_state(self.body)
            self.pre_contact = contact_sample(raw, self.body)

        def post():
            after()
            state = cube_state(self.body)
            contact = contact_sample(raw, self.body)
            if not np.isfinite(state).all():
                raise RuntimeError("STOP: nonfinite physics state")
            self.rows.append(
                dict(
                    index=len(self.rows),
                    before=self.pre.copy(),
                    after=state,
                    before_contacts=self.pre_contact,
                    contacts=contact,
                    bottom_m=bottom_height(state, self.half_size),
                )
            )

        raw._before_simulation_step, raw._after_simulation_step = pre, post

    def evidence(self, run):
        per_control = round(self.control_dt / self.dt)
        event_index = run["trigger"] * per_control
        expected = (len(run["frames"]) - 1) * per_control
        if len(self.rows) != expected:
            raise RuntimeError("STOP: incomplete substep audit")
        window = self.rows[max(0, event_index - 2 * per_control) : event_index + 10 * per_control]
        first = self.rows[event_index]
        J = run["impulse_Ns"] if run["branch"] == "disturbance" else 0.0
        residual = (
            self.mass * (first["after"][7:10] - first["before"][7:10])
            - np.array([J, 0, 0])
            - self.mass * self.gravity * self.dt
            - first["contacts"]["net_impulse_Ns"]
        )
        min_separation = min(
            (p["separation"] for r in window for p in r["contacts"]["points"]), default=0.0
        )
        min_bottom = min(r["bottom_m"] for r in window)
        residual_inf = float(np.max(np.abs(residual)))
        valid = (
            min_separation >= -0.005
            and min_bottom >= -0.005
            and np.linalg.norm(residual)
            <= first["contacts"]["friction_bound_Ns"] + 0.0002 + 0.005 * J
        )
        fingerprint = [hashlib.sha256(canonical(r)).hexdigest() for r in self.rows]
        before = first["before"]
        post_window = self.rows[event_index : event_index + 10 * per_control]
        return dict(
            valid=bool(valid),
            dt_s=self.dt,
            mass_kg=self.mass,
            event_index=event_index,
            full_fingerprints=fingerprint,
            window=window,
            unreported_contact_impulse_Ns=residual,
            inferred_net_contact_impulse_Ns=self.mass
            * (first["after"][7:10] - first["before"][7:10])
            - np.array([J, 0, 0])
            - self.mass * self.gravity * self.dt,
            friction_bound_Ns=first["contacts"]["friction_bound_Ns"],
            friction_bound_excess_Ns=max(
                0.0, float(np.linalg.norm(residual)) - first["contacts"]["friction_bound_Ns"]
            ),
            momentum_residual_inf_Ns=residual_inf,
            min_contact_separation_m=min_separation,
            min_cube_bottom_m=min_bottom,
            event_cube_center_z_m=float(before[2]),
            pre_event_contact_names=sorted(
                {p["other"] for p in first["before_contacts"]["points"]}
            ),
            first_velocity_mps=first["after"][7:10],
            first_delta_velocity_mps=first["after"][7:10] - before[7:10],
            first_displacement_m=first["after"][:3] - before[:3],
            peak_speed_0_5s_mps=max(float(np.linalg.norm(r["after"][7:10])) for r in post_window),
            max_displacement_0_5s_m=max(
                float(np.linalg.norm(r["after"][:3] - before[:3])) for r in post_window
            ),
            first_contact_impulse_Ns=first["contacts"]["net_impulse_Ns"],
        )


def compare_contact(a, b, prefix=False):
    n = a["event_index"] if prefix else len(a["full_fingerprints"])
    return a["full_fingerprints"][:n] == b["full_fingerprints"][:n]


def paired_effect(clean, actual):
    """Conservative paired risk-difference CI, including zero discordances."""
    from scipy.stats import beta, binomtest

    clean, actual = np.asarray(clean, bool), np.asarray(actual, bool)
    if clean.shape != actual.shape or clean.ndim != 1 or not len(clean):
        raise ValueError("Expected nonempty equal-length paired outcomes")
    n = len(clean)
    improve = int(np.sum(~clean & actual))
    regress = int(np.sum(clean & ~actual))

    def cp(k):
        return [
            0.0 if k == 0 else float(beta.ppf(0.0125, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.9875, k + 1, n - k)),
        ]

    i, r = cp(improve), cp(regress)
    return dict(
        n=n,
        improvements=improve,
        regressions=regress,
        preserved_successes=int(np.sum(clean & actual)),
        difference=(improve - regress) / n,
        conservative95=[max(-1, i[0] - r[1]), min(1, i[1] - r[0])],
        discordant_exact_p=(
            float(binomtest(improve, improve + regress).pvalue) if improve + regress else 1.0
        ),
    )


def km_median(times, events):
    """Median time to verified sustained-onset; ties process events before censoring."""
    times, events = np.asarray(times), np.asarray(events, bool)
    survival = 1.0
    for time in sorted(set(times)):
        risk = np.sum(times >= time)
        deaths = np.sum((times == time) & events)
        survival *= 1 - deaths / risk
        if survival <= 0.5:
            return float(time)
    return None


def latency_summary(rows, kind):
    regrasp = kind == "regrasp"
    selected = [r for r in rows if r["grasp_loss_step"] is not None] if regrasp else rows
    times, events = [], []
    for r in selected:
        value = r["regrasp_latency_steps" if regrasp else "post_event_goal_latency_steps"]
        cutoff = 200 - (2 if regrasp else 4)
        start = r["grasp_loss_step"] if regrasp else r["trigger"]
        times.append((value if value is not None else max(0, cutoff - start)) * 0.05)
        events.append(value is not None)
    observed = [t for t, e in zip(times, events) if e]
    return dict(
        n=len(selected),
        observed=len(observed),
        censored=len(selected) - len(observed),
        observed_median_s=float(np.median(observed)) if observed else None,
        km_median_s=km_median(times, events) if times else None,
    )


def chunk_association(rows):
    from scipy.stats import spearmanr

    duration = [r["queue_remaining"] * 0.05 for r in rows]

    def corr(y, x=duration):
        if len(set(y)) < 2 or len(set(x)) < 2:
            return None
        return float(spearmanr(x, y).statistic)

    complete = [r for r in rows if r["post_event_goal_latency_steps"] is not None]
    strata = []
    for low, high in [(0, 3), (4, 7)]:
        group = [r for r in rows if low <= r["queue_remaining"] <= high]
        strata.append(
            dict(
                queue_range=[low, high],
                n=len(group),
                successes=sum(r["success_at_200"] for r in group),
                early_losses=sum(r["early_grasp_loss"] for r in group),
                completion=latency_summary(group, "completion"),
            )
        )
    return dict(
        success_spearman=corr([r["success_at_200"] for r in rows]),
        early_loss_spearman=corr([r["early_grasp_loss"] for r in rows]),
        observed_completion_spearman=corr(
            [r["post_event_goal_latency_steps"] for r in complete],
            [r["queue_remaining"] * 0.05 for r in complete],
        ),
        strata=strata,
    )
