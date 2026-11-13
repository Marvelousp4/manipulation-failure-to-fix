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
