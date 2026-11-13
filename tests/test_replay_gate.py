"""CPU-only replay-gate tests."""

import numpy as np
import pytest

from failure2fix.replay_gate import replay_gate


class Agent:
    def get_controller_state(self):
        return {"goal": np.array([0.2], dtype=np.float32)}


class Env:
    def __init__(self):
        self.unwrapped = self
        self.action_space = type("Space", (), {"shape": (1,)})()
        self.agent = Agent()

    def reset(self, seed=None):
        self.x, self.elapsed_steps = 0.0, 0
        return np.array([self.x], dtype=np.float32), {}

    def get_state_dict(self):
        return {"actors": {"cube": np.array([self.x], dtype=np.float32)}}

    def step(self, action):
        self.x += float(action[0])
        self.elapsed_steps += 1
        return np.array([self.x], dtype=np.float32), 0.0, False, False, {
            "is_grasped": self.elapsed_steps >= 2,
            "success": self.x > 0.5,
        }

    def close(self):
        pass


class Policy:
    def reset(self, seed):
        pass

    def act(self, obs):
        return np.array([0.01], dtype=np.float32)


def test_clean_replay_gate():
    result = replay_gate(Env, Policy, [7, 8], horizon=20)
    assert result["pass"]
    assert result["cases"][0]["branch_step"] == 4
    assert result["cases"][0]["suffix"]["max_error"] == 0


def test_detects_bad_policy_action():
    class Unseeded(Policy):
        call = 0

        def act(self, obs):
            Unseeded.call += 1
            return np.array([0.01 + Unseeded.call * 0.0001], dtype=np.float32)

    result = replay_gate(Env, Unseeded, [9], horizon=20)
    assert not result["pass"]
    assert result["cases"][0]["first_mismatch"]["field"] == "action"


def test_rejects_missing_contact_phase():
    class NoGrasp(Env):
        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            info["is_grasped"] = False
            return obs, reward, done, truncated, info

    with pytest.raises(RuntimeError, match="No sustained grasp"):
        replay_gate(NoGrasp, Policy, [8], horizon=20)


def test_refuses_insufficient_horizon():
    with pytest.raises(ValueError):
        replay_gate(Env, Policy, [1], horizon=5)
