import numpy as np
import pytest

from failure2fix.gate_b2 import (
    bottom_height,
    compare_contact,
    contact_sample,
    km_median,
    paired_effect,
)


def test_paired_difference_uses_discordance_and_non_degenerate_zero_ci():
    effect = paired_effect([True] * 50, [False] * 20 + [True] * 30)
    assert effect["difference"] == -0.4
    assert effect["regressions"] == 20 and effect["improvements"] == 0
    assert effect["conservative95"][0] < -0.4 < effect["conservative95"][1] < 0
    zero = paired_effect([True] * 50, [True] * 50)
    assert zero["conservative95"][0] < 0 < zero["conservative95"][1]
    with pytest.raises(ValueError):
        paired_effect([], [])


def test_cube_bottom_accounts_for_orientation():
    assert bottom_height([0, 0, 0.02, 1, 0, 0, 0], 0.02) == 0
    angle = np.pi / 8
    assert bottom_height([0, 0, 0.03, np.cos(angle), np.sin(angle), 0, 0], 0.02) == pytest.approx(
        0.03 - 0.02 * np.sqrt(2)
    )


def test_contact_sign_is_on_cube_not_other_actor():
    from types import SimpleNamespace as S

    cube, table = S(entity=S(name="cube")), S(entity=S(name="table"))
    point = S(
        impulse=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        position=np.zeros(3),
        separation=-0.001,
    )
    raw = S(
        scene=S(
            px=S(
                get_contacts=lambda: [
                    S(
                        bodies=[table, cube],
                        points=[point],
                        shapes=[S(physical_material=S(static_friction=0.3, dynamic_friction=0.3))]
                        * 2,
                    )
                ]
            )
        )
    )
    sample = contact_sample(raw, cube)
    np.testing.assert_array_equal(sample["net_impulse_Ns"], [-1, -2, -3])
    assert sample["points"][0]["other"] == "table"


def test_contact_prefix_does_not_hide_sham_suffix_divergence():
    a = dict(event_index=2, full_fingerprints=["a", "b", "c"])
    b = dict(event_index=2, full_fingerprints=["a", "b", "d"])
    assert compare_contact(a, b, prefix=True)
    assert not compare_contact(a, b)


def test_km_median_respects_censoring_and_event_ties():
    assert km_median([1, 2, 3], [False, False, True]) == 3
    assert km_median([1, 2, 3], [True, False, False]) is None
    assert km_median([1, 1], [True, False]) == 1


@pytest.mark.parametrize("friction", [0.0, 0.3])
def test_native_contact_impulses_and_friction_account_for_momentum(friction):
    sapien = pytest.importorskip("sapien")
    from types import SimpleNamespace as S

    px = sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([px])
    scene.set_timestep(0.01)
    floor_builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(friction, friction, 0)
    floor_builder.add_box_collision(half_size=[1, 1, 0.01], material=material)
    floor_builder.initial_pose = sapien.Pose([0, 0, -0.01])
    floor_builder.build_static(name="table")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02] * 3, density=1000, material=material)
    builder.initial_pose = sapien.Pose([0, 0, 0.019])
    entity = builder.build(name="cube")
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    body.linear_damping = 0
    body.add_force_at_point([0.008, 0, 0], body.entity_pose.p, mode="impulse")
    scene.step()
    sample = contact_sample(S(scene=S(px=px)), body)
    assert sample["points"]
    residual = (
        body.mass * body.linear_velocity
        - np.array([0.008, 0, -0.0981 * body.mass])
        - sample["net_impulse_Ns"]
    )
    assert np.linalg.norm(residual) <= sample["friction_bound_Ns"] + 1e-5
    assert abs(residual[2]) < 1e-5
    if friction == 0:
        np.testing.assert_allclose(residual, 0, atol=1e-5)
    else:
        assert abs(residual[0]) > 0.001  # Native normal impulses omit tangential friction.
