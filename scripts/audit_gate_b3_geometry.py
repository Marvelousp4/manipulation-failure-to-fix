"""OFFLINE 70001 geometry audit: reconstruct primitives and verify archived shape digests."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from failure2fix.gate_b2 import canonical
from failure2fix.gate_b3 import decode, geometry, spatial


def audit_archive():
    import sapien

    path = Path("artifacts/gate-b2b/confirmation-v1/failure-pair.json.gz")
    saved = decode(json.loads(gzip.decompress(path.read_bytes())))["actual"]
    audit = saved["audit"]
    table_pose = saved["frames"][0]["state"]["env/actors/table-workspace"].reshape(-1)[:7]
    # Reconstruct pinned TableSceneBuilder primitives; equality with recorded native
    # geometry SHA256s is mandatory. No reset, physics step, policy call or seed replay.
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    material = sapien.physx.PhysxMaterial(0.3, 0.3, 0)

    def build(half, local, name, plane=False):
        b = scene.create_actor_builder()
        if plane:
            b.add_plane_collision(sapien.Pose(local[:3], local[3:]), material=material)
        else:
            b.add_box_collision(
                sapien.Pose(local[:3], local[3:]), half_size=half, material=material
            )
        entity = b.build_static(name=name)
        body = entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        return geometry(body, box=not plane)

    cube = build([0.02] * 3, [0, 0, 0, 1, 0, 0, 0], "cube")
    table = build([2.418 / 2, 1.209 / 2, 0.9196429 / 2], [0, 0, 0.9196429 / 2, 1, 0, 0, 0], "table")
    floor = build(
        None, [0, 0, -(0.91964257 + 3.5762787e-7), 0.7071068, 0, -0.7071068, 0], "floor", True
    )
    expected = {x["body"]: x["geometry_sha256"] for x in audit["logical_shape_inventory"]}
    for name, shape in [
        ("scene-0_cube", cube),
        ("scene-0_table-workspace", table),
        ("scene-0_ground", floor),
    ]:
        assert shape["geometry_sha256"] == expected[name], (
            name,
            shape["geometry_sha256"],
            expected[name],
        )
    result = []
    ballistic = 0
    for row in audit["window"]:
        encoded = audit["_exact"][row["index"]]["after"]["state"]["env/actors/table-workspace"]
        table_pose = np.frombuffer(bytes.fromhex(encoded["bits"]), dtype=encoded["dtype"]).reshape(
            -1
        )[:7]
        s = spatial(
            row["after"][:7], table_pose, cube, table, np.array([0, 0, 0, 1, 0, 0, 0]), floor
        )
        s.update(index=row["index"], old_bottom_m=row["bottom_m"])
        if s["footprint"] == "none" and not any(
            np.linalg.norm(p["impulse"]) > 0 for p in row["contacts"]["points"]
        ):
            dv = row["after"][7:10].astype(float) - row["before"][7:10].astype(float)
            ballistic += bool(np.allclose(dv, [0, 0, -0.0981], atol=1e-4, rtol=1e-4))
        result.append(s)
    return dict(
        geometry_hashes_verified=True,
        table_pose_source="archived per-physics-step state",
        old_verdict_unchanged="B2b STOPPED",
        collision_geometry=dict(cube=cube, table=table, floor=floor),
        first_old_height_violation=next(r for r in result if r["old_bottom_m"] < -0.005),
        ballistic_rows_after_departure=ballistic,
        window=result,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/gate-b3/offline-70001.json")
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical(audit_archive()))
    print(out)
