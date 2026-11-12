import json
import subprocess
import sys

import numpy as np

from failure2fix.envs import BoxLike
from failure2fix.evaluation import Condition, evaluate_condition


class TransientSuccess:
    action_space = BoxLike(-np.ones(2), np.ones(2))

    def reset(self, seed):
        self.step_count = 0
        return np.zeros(4), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(4), 0, False, self.step_count == 2, {"success": self.step_count == 1}

    def close(self):
        pass


def test_success_once_survives_later_failure():
    result = evaluate_condition(TransientSuccess, "test", "zero", Condition("clean"), [1], 5)[0]
    assert result.success is True
    assert result.success_at_end is False


def test_cli_separates_shifts_and_preserves_outputs(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "failure2fix.cli",
        "evaluate",
        "--episodes",
        "3",
        "--output",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["conditions"] == [
        dict(name="clean", obs_noise=0, action_delay=0),
        dict(name="observation_noise", obs_noise=0.12, action_delay=0),
        dict(name="action_delay", obs_noise=0, action_delay=2),
    ]
    records = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 9
    for seed in manifest["seeds"]:
        assert len({r["initial_observation_sha256"] for r in records if r["seed"] == seed}) == 1
    assert subprocess.run(cmd, capture_output=True).returncode != 0


def test_checkpoint_hash_mismatch_rejected_before_loading(tmp_path):
    import hashlib
    from types import SimpleNamespace

    import pytest

    from failure2fix.baseline import official_bc_loader

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "incorrect",
                "upstream_diff_sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
    )
    args = SimpleNamespace(
        obs_mode="state",
        checkpoint=str(checkpoint),
        provenance=str(provenance),
        upstream=str(tmp_path),
        env="PickCube-v1",
        control_mode="pd_ee_delta_pos",
        sim_backend="cpu",
    )
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        official_bc_loader(args)
