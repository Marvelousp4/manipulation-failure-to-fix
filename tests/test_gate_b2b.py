import copy
import json
from pathlib import Path

import numpy as np
import pytest

from failure2fix.gate_b2b import bits, contact_multiset, digest


@pytest.fixture
def captured():
    path = Path(__file__).resolve().parents[1] / "artifacts/gate-b2a/first-mismatch.json"
    return json.loads(path.read_text())


def test_record_and_point_enumeration_reordering_passes(captured):
    a = captured["raw_reference"]
    b = copy.deepcopy(a[::-1])
    for c in b:
        c["points"].reverse()
    assert contact_multiset(a) == contact_multiset(b)
    assert contact_multiset(a) == contact_multiset(captured["raw_actual"])


@pytest.mark.parametrize("field", ["position", "normal", "impulse", "separation"])
def test_genuine_native_field_change_fails(captured, field):
    a = captured["raw_reference"]
    b = copy.deepcopy(a)
    point = b[0]["points"][0]
    if field == "separation":
        point[field] = float(np.nextafter(np.float32(point[field]), np.float32(np.inf)))
    else:
        point[field][0] = float(np.nextafter(np.float32(point[field][0]), np.float32(np.inf)))
    assert contact_multiset(a) != contact_multiset(b)


@pytest.mark.parametrize("field", ["body", "shape_index", "geometry_sha256"])
def test_shape_identity_mismatch_fails(captured, field):
    a = captured["raw_reference"]
    b = copy.deepcopy(a)
    b[0]["pair"][0][field] = "changed"
    assert contact_multiset(a) != contact_multiset(b)


@pytest.mark.parametrize("operation", ["remove", "duplicate", "empty_contact"])
def test_missing_and_duplicate_points_and_empty_contacts_fail(captured, operation):
    a = captured["raw_reference"]
    b = copy.deepcopy(a)
    if operation == "remove":
        b[0]["points"].pop()
    elif operation == "duplicate":
        b[0]["points"].append(copy.deepcopy(b[0]["points"][0]))
    else:
        b.append(dict(pair=copy.deepcopy(b[0]["pair"]), points=[]))
    assert contact_multiset(a) != contact_multiset(b)


def test_precision_and_zero_sign_changes_fail(captured):
    a = captured["raw_reference"]
    b = copy.deepcopy(a)
    b[0]["points"][0]["precision"]["position"] = "float64"
    assert contact_multiset(a) != contact_multiset(b)
    b = copy.deepcopy(a)
    b[0]["points"][0]["impulse"][0] = 0.0
    assert np.signbit(a[0]["points"][0]["impulse"][0])
    assert contact_multiset(a) != contact_multiset(b)


def test_consistent_pair_orientation_change_is_exact(captured):
    a = captured["raw_reference"]
    b = copy.deepcopy(a)
    for c in b:
        c["pair"].reverse()
        for p in c["points"]:
            p["normal"] = (-np.asarray(p["normal"])).tolist()
            p["impulse"] = (-np.asarray(p["impulse"])).tolist()
    assert contact_multiset(a) == contact_multiset(b)


def test_transient_substep_state_difference_is_not_hidden_by_endpoints():
    from test_gate_b import fake_run

    from failure2fix.gate_b2b import compare_exact

    x = [
        dict(before=digest(bits(np.zeros(3))), after=digest(bits(np.zeros(3)))) for _ in range(100)
    ]
    left = (fake_run(), dict(event_index=20, exact_hashes=x))
    right = copy.deepcopy(left)
    right[1]["exact_hashes"][22]["after"] = digest(bits(np.array([1e-12, 0.0, 0.0])))
    assert left[1]["exact_hashes"][0] == right[1]["exact_hashes"][0]
    assert left[1]["exact_hashes"][-1] == right[1]["exact_hashes"][-1]
    assert compare_exact(left, left)["pass_"]
    failure = compare_exact(left, right)
    assert not failure["pass_"] and failure["substep"] == 22


def test_native_precision_is_not_silently_rounded(captured):
    a = copy.deepcopy(captured["raw_reference"])
    a[0]["points"][0]["position"][0] += 1e-12
    with pytest.raises(ValueError, match="declared precision"):
        contact_multiset(a)


def test_live_native_geometry_and_precision_are_serializable():
    sapien = pytest.importorskip("sapien")
    from types import SimpleNamespace as S

    from failure2fix.gate_b2b import read_contacts

    px = sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([px])
    scene.set_timestep(0.01)
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[1, 1, 0.01])
    b.initial_pose = sapien.Pose([0, 0, -0.01])
    b.build_static(name="table")
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[0.02] * 3)
    b.initial_pose = sapien.Pose([0, 0, 0.019])
    b.build(name="cube")
    scene.step()
    sample = read_contacts(S(scene=S(px=px)), {}, {})
    assert sample and sample[0]["points"]
    assert sample[0]["points"][0]["position"]["dtype"] == "<f4"


def test_nonfinite_and_signed_zero_failure_payloads_are_retained(tmp_path):
    import gzip

    from failure2fix.gate_b2b import all_finite, save_failure
    from failure2fix.replay_gate import Frame

    state = np.array([-0.0, np.nan], dtype=np.float32)
    assert not all_finite(bits(state))
    run = dict(seed=1, frames=[Frame(0, None, {"x": state}, False, False)], incomplete=True)
    path = tmp_path / "failure.gz"
    save_failure(path, None, (run, dict(_exact=[bits(state)])))
    data = json.loads(gzip.decompress(path.read_bytes()))
    encoded = data["actual"]["frames"][0]["state"]["x"]["array_bits"]
    restored = np.frombuffer(bytes.fromhex(encoded["bits"]), dtype=encoded["dtype"])
    assert np.signbit(restored[0]) and np.isnan(restored[1])


def test_rollout_exception_exposes_partial_actions_policy_and_frames(monkeypatch):
    pytest.importorskip("torch")
    from types import SimpleNamespace as S

    import failure2fix.gate_b as gate
    from failure2fix.replay_gate import Frame

    class Policy:
        def __init__(self, *args):
            self.inferences = []

        def act(self, obs):
            return np.zeros(4)

        def state(self):
            return {"queue": np.zeros((1, 4))}

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.gpu_sim_enabled = False
            self.cube = S(_bodies=[None])
            self._before_simulation_step = lambda: None
            self._after_simulation_step = lambda: None
            self.action_space = S(shape=(4,))
            self.count = 0
            self.closed = False

        def reset(self, seed):
            return np.zeros(2), {}

        def step(self, action):
            self.count += 1
            raise RuntimeError("injected failure")

        def close(self):
            self.closed = True

    env = Env()
    capture = {}
    monkeypatch.setattr(gate, "TrackedChunks", Policy)
    monkeypatch.setattr(gate, "state_frame", lambda e, o, a, s, i, p: Frame(s, a, {}, False, False))
    with pytest.raises(RuntimeError, match="injected"):
        gate.run_branch(lambda: env, None, 1, "clean", capture=capture)
    assert env.closed and len(capture["frames"]) == 1
    assert capture["control_step"] == 1 and "policy_state" in capture and "rng_state" in capture


def test_offline_box_rotation_normalizes_native_float32_quaternion_in_double():
    from scripts.analyze_gate_b2b_stop import rotation

    q = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], dtype=np.float32)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rotation(q), expected, atol=1e-14)
