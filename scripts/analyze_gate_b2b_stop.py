"""Offline stopped-case audit; never changes B2b validity or executes the policy."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from failure2fix.gate_b import metrics
from failure2fix.gate_b2 import canonical
from failure2fix.gate_b2b import compare_exact
from failure2fix.replay_gate import Frame


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
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="runs/gate-b2b/confirmation-v1/failure-pair.json.gz")
    p.add_argument("--output", default="runs/gate-b2b/stopped-case-audit")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    saved = decode(json.loads(gzip.decompress(Path(a.input).read_bytes())))
    pairs = []
    for key in ["reference", "actual"]:
        v = saved[key]
        run = v["run"]
        run["frames"] = [Frame(**f) for f in v["frames"]]
        pairs.append((run, v["audit"]))
    clean, actual = pairs
    run, audit = actual
    # ManiSkill 3.0.1 TableSceneBuilder's collision box, not its visual mesh.
    table = np.asarray(run["frames"][0].state["env/actors/table-workspace"]).reshape(-1)
    half = np.array([2.418 / 2, 1.209 / 2, 0.9196429 / 2], np.float32).astype(float)
    center = table[:3] + rotation(table[3:7]) @ np.array([0, 0, half[2]])
    table_hi = center + np.abs(rotation(table[3:7])) @ half
    table_lo = center - np.abs(rotation(table[3:7])) @ half
    window = audit["window"]
    post = [r for r in window if r["index"] >= audit["event_index"]]
    violation = next(r for r in post if r["bottom_m"] < -0.005)
    cube = violation["after"]
    cube_x_lo = float(cube[0] - 0.02 * np.abs(rotation(cube[3:7])[0]).sum())
    flight = [
        r
        for r in post
        if not any(np.linalg.norm(p["impulse"]) > 0 for p in r["contacts"]["points"])
    ]
    dvz = [
        (b["after"][9] - a["after"][9]) / 0.01
        for a, b in zip(flight, flight[1:])
        if b["index"] == a["index"] + 1
    ]
    summary = dict(
        diagnostic_only=True,
        original_verdict="STOP / NO-GO; unchanged",
        offline_prefix=compare_exact(clean, actual, prefix=True),
        actual_seed=run["seed"],
        actual_success_at_200=run["frames"][-1].success,
        metrics={k: v for k, v in metrics(run).items() if k != "event"},
        table_collision_aabb_min_m=table_lo,
        table_collision_aabb_max_m=table_hi,
        first_height_violation=dict(
            substep=violation["index"],
            time_after_impulse_s=(violation["index"] - audit["event_index"] + 1) * 0.01,
            cube_center_m=cube[:3],
            cube_bottom_m=violation["bottom_m"],
            cube_x_min_m=cube_x_lo,
            table_x_max_m=float(table_hi[0]),
            cube_entirely_outside_table_x=bool(cube_x_lo > table_hi[0]),
        ),
        no_contact_vertical_acceleration_mps2_range=[float(min(dvz)), float(max(dvz))],
        first_native_velocity_mps=audit["first_velocity_mps"],
        first_native_displacement_m=audit["first_displacement_m"],
        max_displacement_0_5s_m=audit["max_displacement_0_5s_m"],
        max_reported_contact_penetration_m=-audit["min_contact_separation_m"],
        interpretation="Global z-height guard conflates finite-table ejection/free fall with penetration; no criterion changed or experiment resumed",
    )
    (out / "analysis.json").write_bytes(canonical(summary))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([(r["index"] - audit["event_index"] + 1) * 0.01 for r in post])
    xyz = np.array([r["after"][:3] for r in post])
    vel = np.array([r["after"][7:10] for r in post])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    axes[0].plot(xyz[:, 0], xyz[:, 2], label="Saved cube center")
    axes[0].plot(
        [table_lo[0], table_hi[0]], [table_hi[2]] * 2, "k-", lw=3, label="Finite collision tabletop"
    )
    axes[0].scatter(cube[0], cube[2], color="red", label="First height-guard violation")
    axes[0].set(
        xlabel="World x (m)", ylabel="World z (m)", title="Stopped seed 70001: recorded trajectory"
    )
    axes[0].legend(fontsize=7)
    axes[1].plot(t, vel[:, 0], label="vx")
    axes[1].plot(t, vel[:, 2], label="vz")
    axes[1].axvline(
        summary["first_height_violation"]["time_after_impulse_s"],
        color="red",
        ls="--",
        label="Height-guard violation",
    )
    axes[1].set(
        xlabel="Time after impulse (s)",
        ylabel="Native cube velocity (m/s)",
        title="Measured response; no policy rerun",
    )
    axes[1].legend(fontsize=7)
    fig.savefig(out / "trajectory.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k in ["actual_seed", "first_height_violation", "offline_prefix"]
            },
            default=lambda x: x.tolist(),
        )
    )


if __name__ == "__main__":
    main()
