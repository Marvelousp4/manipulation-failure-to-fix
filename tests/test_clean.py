import numpy as np
import pytest

from failure2fix.clean import offline_mse, phase_label, rollup


def test_failure_phase_does_not_turn_grasp_or_goal_proximity_into_success():
    assert phase_label(True, False, False, True) == "transport"
    assert phase_label(True, True, False, True) == "placement/settling"
    assert phase_label(False, False, False, False) == "control/reach"
    assert phase_label(False, False, False, True) == "grasp"


def test_success_at_end_is_distinct_from_success_once():
    result = rollup(
        [
            dict(
                success_once=True,
                success_at_end=False,
                success_once_100=True,
                failure_phase="success",
            )
        ]
    )
    assert result["success_once"] == 1
    assert result["success_at_end"] == 0


def test_offline_alignment_uses_current_observation(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")
    obs = np.arange(20, dtype=np.float32).reshape(5, 4)
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("traj_0")
        g["obs"] = obs
        g["actions"] = obs[:-1]
    result = offline_mse(torch.nn.Identity(), path, device="cpu")
    assert result["mse"] == 0
    assert result["deliberately_misaligned_next_obs_mse"] > 0


def test_fixed_horizon_and_stochastic_seed_isolation(monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    import failure2fix.clean as clean

    actions = []

    class Env:
        action_space = SimpleNamespace(shape=(4,))

        def __init__(self):
            self.unwrapped = self
            self.cube = SimpleNamespace(pose=SimpleNamespace(p=np.array([[0.0, 0.0, 0.02]])))
            self.agent = SimpleNamespace(tcp_pose=SimpleNamespace(p=np.array([[0.0, 0.0, 0.02]])))
            self.goal_site = SimpleNamespace(pose=SimpleNamespace(p=np.array([[0.0, 0.0, 0.1]])))

        def reset(self, seed):
            self.t = 0
            return np.zeros((1, 42), dtype=np.float32), {}

        def step(self, action):
            actions.append(action.copy())
            self.t += 1
            info = dict(
                success=self.t == 1,
                is_grasped=True,
                is_obj_placed=self.t == 1,
                is_robot_static=self.t == 1,
            )
            return np.zeros((1, 42), dtype=np.float32), 0, self.t == 1, self.t == 3, info

        def close(self):
            pass

    monkeypatch.setattr(clean, "make_env", lambda *a, **kw: Env())
    torch.manual_seed(1337)
    before = torch.get_rng_state().clone()

    def sample(obs):
        return torch.rand(4)

    first = clean.rollout(sample, [12], horizon=3)
    second = clean.rollout(sample, [12], horizon=3)
    assert first == second
    assert first[0]["success_once"] and not first[0]["success_at_end"]
    assert first[0]["steps"] == 3
    assert torch.equal(before, torch.get_rng_state())
    np.testing.assert_array_equal(actions[:3], actions[3:])


def test_action_chunk_history_and_episode_reset():
    torch = pytest.importorskip("torch")
    from failure2fix.clean_dp import ActionChunks

    class Agent:
        obs_horizon = 2

        def __init__(self):
            self.inputs = []

        def get_action(self, obs):
            self.inputs.append(obs.clone())
            return torch.arange(8, dtype=torch.float32).reshape(1, 8, 1).repeat(1, 1, 4)

    agent = Agent()
    policy = ActionChunks(agent, device="cpu")
    for i in range(9):
        assert policy(np.full((1, 42), i, dtype=np.float32))[0] == i % 8
    assert agent.inputs[0][0, :, 0].tolist() == [0, 0]
    assert agent.inputs[1][0, :, 0].tolist() == [7, 8]
    policy.reset(3)
    policy(np.full((1, 42), 99, dtype=np.float32))
    assert agent.inputs[-1][0, :, 0].tolist() == [99, 99]


def test_vector_fixed_horizon_handles_gymnasium_object_info():
    torch = pytest.importorskip("torch")
    from failure2fix.clean_dp import vector_evaluate

    class Agent(torch.nn.Module):
        act_horizon = 8

        def get_action(self, obs):
            return torch.zeros((2, 8, 4))

    class Env:
        num_envs = 2

        def reset(self, seed):
            self.t = 0
            return np.zeros((2, 2, 42)), {}

        def step(self, action):
            self.t += 1
            ep = [
                dict(success_once=True, success_at_end=False),
                dict(success_once=False, success_at_end=False),
            ]
            return (
                np.zeros((2, 2, 42)),
                np.zeros(2),
                np.zeros(2, bool),
                np.full(2, self.t == 200),
                dict(
                    episode=np.array(ep, dtype=object),
                    final_info=np.array([dict(episode=e) for e in ep], dtype=object),
                ),
            )

    rows = vector_evaluate(Agent(), Env(), [20000, 20001], device="cpu")
    assert rows[0]["success_once_100"] and not rows[1]["success_once_100"]
    assert not rows[0]["success_at_end"] and rows[0]["steps"] == 200


def test_selection_uses_validation_success_not_last_checkpoint_or_mse():
    from failure2fix.clean_selection import best_candidate

    low_loss = dict(
        updates=100000,
        validation=dict(success_once=0.66, success_at_end=0.06, training_mse=0.00001),
    )
    higher_success = dict(
        updates=80000, validation=dict(success_once=0.84, success_at_end=0.06, training_mse=0.00004)
    )
    assert best_candidate([low_loss, higher_success]) is higher_success


def test_selection_rejects_nonvalidation_seeds(tmp_path):
    import json

    from failure2fix.clean_selection import candidate

    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            dict(summary=dict(updates=10000), episodes=[dict(seed=i) for i in range(40000, 40050)])
        )
    )
    with pytest.raises(ValueError, match="reserved validation seeds"):
        candidate(path, "bc", tmp_path)


def test_test_guard_blocks_reuse_before_policy_or_environment_loading(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"guard fixture, never loaded")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            dict(
                selected=dict(
                    method="bc",
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                ),
                test_seeds=list(range(40000, 40100)),
                horizon=200,
            )
        )
    )
    guard = selection.with_suffix(".test-started.json")
    guard.write_text("already consumed")
    script = Path(__file__).resolve().parents[1] / "scripts/test_clean.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--selection",
            str(selection),
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert guard.read_text() == "already consumed"
    assert not (tmp_path / "output/episodes.json").exists()
