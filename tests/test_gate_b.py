import copy

import numpy as np
import pytest

from failure2fix.gate_b import compare_branches, compare_schedule, metrics, physical_impulse
from failure2fix.replay_gate import Frame


def fake_run():
    frames = []
    for step in range(21):
        frames.append(
            Frame(
                step,
                np.zeros(4) if step else None,
                {
                    "obs": np.zeros(2),
                    "policy/queue": np.zeros((2, 4)),
                    "policy/history": np.zeros((2, 2)),
                    "rng/test": np.zeros(32, dtype=np.uint8),
                },
                step >= 2 and step not in [6, 7, 8],
                step >= 12,
            )
        )
    return dict(
        frames=frames,
        seed=1,
        branch="clean",
        impulse_Ns=0.0,
        trigger=4,
        inference_steps=[1, 9, 17],
        event=dict(
            pre_state=copy.deepcopy(frames[4].state),
            selected_action=np.zeros(4),
            first_updated_observation_policy_call=6,
            first_inference_with_updated_observation=9,
        ),
    )


def test_sham_divergence_and_hidden_queue_changes_are_rejected():
    reference = fake_run()
    same = copy.deepcopy(reference)
    assert compare_branches(reference, same)["pass"]
    same["frames"][7].state["policy/queue"][0, 0] = 1e-8
    assert not compare_branches(reference, same)["pass"]


def test_suffix_may_change_but_pre_feedback_actions_must_not():
    reference = fake_run()
    changed = copy.deepcopy(reference)
    changed["frames"][9].action[0] = 0.2
    assert compare_branches(reference, changed, prefix_only=True)["pass"]
    assert compare_schedule(reference, changed)["pass"]
    changed["frames"][7].action[0] = 0.1
    assert not compare_schedule(reference, changed)["pass"]


def test_common_random_numbers_are_checked_after_disturbance():
    reference = fake_run()
    changed = copy.deepcopy(reference)
    changed["frames"][10].state["rng/test"][3] = 1
    assert not compare_schedule(reference, changed)["pass"]


def test_latencies_distinguish_loss_regrasp_and_completion():
    row = metrics(fake_run())
    assert row["grasp_loss_step"] == 6
    assert row["regrasp_step"] == 9 and row["regrasp_latency_steps"] == 3
    assert row["stable_goal_step"] == 12 and row["post_event_goal_latency_steps"] == 8
    assert not row["success_before_event"]
    run = fake_run()
    for f in run["frames"][6:]:
        f.grasped = False
        f.success = False
    row = metrics(run)
    assert row["regrasp_right_censored"] and row["goal_right_censored"]
    assert row["regrasp_latency_steps"] is None


def test_native_impulse_is_momentum_not_pose_assignment():
    class Pose:
        p = np.zeros(3, dtype=np.float32)
        q = np.array([1, 0, 0, 0], dtype=np.float32)

        def __mul__(self, other):
            return self

    class Body:
        entity_pose = Pose()
        cmass_local_pose = Pose()
        linear_velocity = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        mass = 0.064

        def add_force_at_point(self, force, point, mode):
            self.called = (force.copy(), point.copy(), mode)

    body = Body()
    event = physical_impulse(body, 0.064)
    assert body.called[2] == "impulse"
    np.testing.assert_allclose(body.called[0], [0.064, 0, 0])
    assert event["before"] == event["after_api"]


def test_free_cube_calibrates_impulse_units_and_one_shot():
    pytest.importorskip("sapien")
    from failure2fix.gate_b import validate_impulse_api

    results = validate_impulse_api([0.004, 0.016, 0.064, 0.256])
    assert len(results) == 5 and all(r["pass"] for r in results)


def test_chunk_observation_clock_preserves_queued_actions():
    torch = pytest.importorskip("torch")
    from failure2fix.gate_b import TrackedChunks

    class Agent:
        obs_horizon = 2

        def get_action(self, obs):
            return torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)

    p = TrackedChunks(Agent(), device="cpu")
    for i in range(10):
        p.act(np.full(42, i, dtype=np.float32))
    assert p.inferences == [1, 9]
    assert p.state()["history_frames"].tolist() == [8, 9]
    assert p.state()["queue_source_inference"] == 9
    assert len(p.chunks.actions) == 6


def test_trace_files_preserve_shapes_and_repeat_byte_exactly(tmp_path):
    import gzip
    import json

    from failure2fix.gate_b import save_run

    run = fake_run()
    run["frames"][0].state["policy/queue"] = np.empty((0, 4), dtype=np.float32)
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    save_run(a, run)
    save_run(b, run)
    assert a.read_bytes() == b.read_bytes()
    first = json.loads(gzip.decompress(a.read_bytes()).decode().splitlines()[0])
    assert first["state"]["policy/queue"]["shape"] == [0, 4]
    assert first["state"]["policy/queue"]["dtype"] == "float32"
