import numpy as np

from failure2fix.envs import ToyReachEnv


def test_toy_reset_reproducible():
    env = ToyReachEnv()
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    np.testing.assert_equal(a["state"], b["state"])


def test_zero_action_does_not_move_toy():
    env = ToyReachEnv()
    before, _ = env.reset(seed=22)
    after, _, _, _, _ = env.step(np.zeros(2))
    np.testing.assert_allclose(before["state"][:2], after["state"][:2])
