import copy

import numpy as np
import pytest

from failure2fix.gate_b3 import footprint, obb, penetration, spatial, vertices


def shape(half):
    return dict(half_size=np.asarray(half), local_pose=np.array([0, 0, 0, 1, 0, 0, 0.0]))


def p(x=0, y=0, z=0, q=(1, 0, 0, 0)):
    return np.array([x, y, z, *q], float)


CUBE = shape([0.02] * 3)
TABLE = shape([1, 1, 0.5])
TABLE_POSE = p(z=-0.5)
FLOOR = dict(local_pose=p(z=-1, q=(np.sqrt(0.5), 0, -np.sqrt(0.5), 0)))


def category(cube):
    return spatial(cube, TABLE_POSE, CUBE, TABLE, p(), FLOOR)


def test_true_penetration_is_detected_inside_the_finite_table():
    s = category(p(z=0.012))
    assert s["footprint"] == "full"
    assert s["table_penetration_m"] == pytest.approx(0.008)
    assert s["floor_penetration_m"] == 0


def test_partial_support_is_not_center_based_departure():
    # Center outside table while collision footprint still overlaps the rim.
    s = category(p(x=1.01, z=0.02))
    assert s["footprint"] == "partial" and s["table_penetration_m"] == 0


def test_off_table_fall_below_global_plane_is_not_table_penetration():
    s = category(p(x=1.1, z=-0.3))
    assert s["footprint"] == "none"
    assert s["table_penetration_m"] == s["floor_penetration_m"] == 0


def test_floor_halfspace_still_rejects_real_floor_penetration():
    assert category(p(x=1.1, z=-0.99))["floor_penetration_m"] == pytest.approx(0.01)


def test_sat_and_polygon_reject_corner_false_overlap_from_world_aabb():
    q = (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8))
    cube = obb(p(x=1.02, y=1.02, z=0.01, q=q), CUBE)
    assert np.min(vertices(cube)[:, 0]) < 1 and np.min(vertices(cube)[:, 1]) < 1
    assert penetration(cube, obb(TABLE_POSE, TABLE)) == 0
    assert footprint(vertices(cube), obb(TABLE_POSE, TABLE)) == "none"


def test_rotated_table_and_offset_collision_shape_use_local_geometry():
    table = shape([1, 1, 0.5])
    table["local_pose"] = p(x=0.2, z=0.5)
    body = p(x=3, z=-1, q=(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))
    c, r, h = obb(body, table)
    np.testing.assert_allclose(c, [3, 0.2, -0.5], atol=1e-14)
    cube = obb(p(x=3, y=0.2, z=0.012), CUBE)
    assert penetration(cube, (c, r, h)) == pytest.approx(0.008)


def test_geometry_does_not_mutate_poses_or_shapes():
    table = copy.deepcopy(TABLE)
    pose = TABLE_POSE.copy()
    category(p(z=0.02))
    np.testing.assert_array_equal(TABLE["half_size"], table["half_size"])
    np.testing.assert_array_equal(TABLE_POSE, pose)


def test_archived_70001_departure_offline_with_verified_collision_geometry():
    from scripts.audit_gate_b3_geometry import audit_archive

    result = audit_archive()
    assert result["geometry_hashes_verified"]
    assert result["old_verdict_unchanged"] == "B2b STOPPED"
    assert result["first_old_height_violation"]["footprint"] == "none"
    assert result["first_old_height_violation"]["table_penetration_m"] == 0
    assert result["ballistic_rows_after_departure"] > 0


def test_contact_free_ballistic_requires_both_force_and_motion_evidence():
    from failure2fix.gate_b3 import is_ballistic

    assert is_ballistic([0, 0, -0.0981], 0, [0, 0, -9.81], 0.01)
    assert not is_ballistic([0, 0, -0.0981], 1, [0, 0, -9.81], 0.01)
    assert not is_ballistic([1, 0, 0], 0, [0, 0, -9.81], 0.01)


def test_native_and_geometric_penetration_checks_remain_strict():
    from failure2fix.gate_b3 import valid_physics

    assert valid_physics(True, -0.002, 0, 0, 0, 0.192)
    assert not valid_physics(True, -0.00501, 0, 0, 0, 0.064)
    assert not valid_physics(True, 0, 0.00501, 0, 0, 0.064)
    assert not valid_physics(True, 0, 0, 0.00501, 0, 0.064)
    assert not valid_physics(True, 0, 0, 0, 0.01, 0.064)


def test_preregistered_viability_requires_mixed_in_workspace_outcomes():
    from failure2fix.gate_b3 import select_level

    rows = [
        dict(
            seed=i,
            freq=100,
            impulse_Ns=0.128,
            departed=False,
            grasp_loss_step=5,
            early_verified_loss=True,
            regrasp_step=10 if i < 10 else None,
            verified_regrasp_step=10 if i < 10 else None,
            success_at_200=i < 10,
            clean_success=True,
        )
        for i in range(12)
    ]
    _, selected = select_level(rows)
    assert selected["impulse_Ns"] == 0.128
    rows[-1]["departed"] = True
    _, selected = select_level(rows)
    assert selected is None  # Departure failure stays in denominator, not an in-workspace signal.


def test_timestep_response_threshold_is_fixed_and_checks_qualitative_changes():
    from failure2fix.gate_b3 import timestep_case

    base = dict(
        freq=100,
        success_at_200=True,
        departed=False,
        grasp_loss_step=None,
        regrasp_step=None,
        verified_loss_step=None,
        verified_regrasp_step=None,
        trigger=40,
        queue_remaining=0,
        response_50ms=dict(
            position_m=[0, 0, 0], velocity_mps=[0, 0, 0], pre_pose=[0, 0, 0, 1, 0, 0, 0]
        ),
    )
    other = copy.deepcopy(base)
    other["freq"] = 200
    assert not timestep_case([base, other])["unstable"]
    other["response_50ms"]["position_m"][0] = 0.01001
    assert timestep_case([base, other])["unstable"]
    other = copy.deepcopy(base)
    other["freq"] = 200
    other["departed"] = True
    assert timestep_case([base, other])["unstable"]


@pytest.mark.skipif(
    __import__("os").getenv("F2F_SIM_TESTS") != "1", reason="opt-in simulator API smoke test"
)
def test_live_spatial_recorder_reads_actual_box_and_plane_poses():
    from failure2fix.envs import make_env
    from failure2fix.gate_b3 import SpatialRecorder

    env = make_env(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        sim_backend="cpu",
        render_backend="none",
        max_episode_steps=200,
    )
    recorder = SpatialRecorder()
    try:
        recorder.attach(env.unwrapped)
        env.reset(seed=90000)  # API-only smoke test: no policy inference or research outcome.
        for _ in range(2):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert len(recorder.spatial_rows) == 10
        assert all(r["footprint"] == "full" for r in recorder.spatial_rows)
        assert recorder.collision_geometry["cube"]["half_size"][0] == np.float32(0.02)
    finally:
        env.close()


def test_verified_loss_and_departure_are_distinct_from_final_task_failure():
    from test_gate_b import fake_run

    from failure2fix.gate_b3 import episode

    run = fake_run()
    run["event"]["queue_remaining"] = 0
    spatial = [
        dict(
            index=i,
            footprint="none" if i >= 38 else "full",
            support_departed=i >= 38,
            contact_free_ballistic=i >= 38,
        )
        for i in range(100)
    ]
    row = episode(run, dict(event_index=20, dt_s=0.01, _spatial=spatial))
    assert row["verified_loss_step"] == 6 and row["verified_regrasp_step"] == 9
    assert row["first_loss_departure_order"] == "loss_before_departure"
    assert row["category"] == "success_after_departure" and not row["failure_after_departure"]


def test_body_below_the_rim_is_departed_even_with_tiny_projected_overlap():
    row = category(p(x=1.0199, z=-0.1))
    assert row["footprint"] == "partial" and row["support_departed"]
    assert 0 < row["table_penetration_m"] < 0.005
