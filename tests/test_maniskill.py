"""Opt-in simulator test: F2F_SIM_TESTS=1 pytest -q tests/test_maniskill.py."""

import os

import numpy as np
import pytest

from failure2fix.envs import make_env
from failure2fix.policies import as_numpy

pytestmark = pytest.mark.skipif(os.getenv("F2F_SIM_TESTS") != "1", reason="opt-in simulator")


def test_pickcube_state_reset_and_step():
    env = make_env("PickCube-v1", control_mode="pd_ee_delta_pos", render_backend="none")
    try:
        first, _ = env.reset(seed=10000)
        again, _ = env.reset(seed=10000)
        np.testing.assert_array_equal(as_numpy(first), as_numpy(again))
        assert as_numpy(first).shape == (1, 42)
        _, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert as_numpy(reward).size == 1
        assert "success" in info
        assert as_numpy(terminated).size == as_numpy(truncated).size == 1
    finally:
        env.close()


def test_official_batch_patch_preserves_samples():
    import importlib.util
    import sys
    from pathlib import Path

    import torch

    upstream = os.getenv("F2F_UPSTREAM")
    if not upstream:
        pytest.skip("set F2F_UPSTREAM to patched checkout")
    directory = Path(upstream) / "examples/baselines/bc"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("f2f_test_official_bc", directory / "bc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ds = module.ManiSkillDataset.__new__(module.ManiSkillDataset)
    ds.observations = np.arange(420, dtype=np.float32).reshape(10, 42)
    ds.actions = np.arange(40, dtype=np.float32).reshape(10, 4)
    ds.dones = np.zeros((10, 1), dtype=bool)
    ds.device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = [2, 7, 2, 0]
    original = torch.utils.data.default_collate([ds[i] for i in indices])
    batched = torch.utils.data.default_collate(ds.__getitems__(indices))
    for a, b in zip(original, batched):
        assert torch.equal(a, b)
