import numpy as np

from failure2fix.perturbations import ActionDelay, noisy_observation


def test_action_delay_fifo_and_reset():
    d = ActionDelay(2, (1,))
    assert d.apply(np.array([5])).tolist() == [0]
    assert d.apply(np.array([6])).tolist() == [0]
    assert d.apply(np.array([7])).tolist() == [5]
    d.reset()
    assert d.apply(np.array([8])).tolist() == [0]


def test_noise_reproducibility_and_integer_preservation():
    obs = {
        "state": np.array([1.0, 2.0], dtype=np.float32),
        "image": np.array([5, 20], dtype=np.uint8),
    }
    a = noisy_observation(obs, 0.2, np.random.default_rng(4))
    b = noisy_observation(obs, 0.2, np.random.default_rng(4))
    np.testing.assert_equal(a["state"], b["state"])
    np.testing.assert_equal(a["image"], obs["image"])
    assert not np.array_equal(a["state"], obs["state"])
