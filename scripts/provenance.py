"""Record a locally trained official BC checkpoint."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--upstream", required=True)
p.add_argument("--checkpoint", required=True)
a = p.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


record = dict(
    checkpoint_sha256=sha(a.checkpoint),
    dataset_sha256=sha(a.dataset),
    dataset_metadata=json.loads(Path(a.dataset).with_suffix(".json").read_text()),
    upstream_commit=subprocess.check_output(
        ["git", "-C", a.upstream, "rev-parse", "HEAD"], text=True
    ).strip(),
    upstream_diff_sha256=hashlib.sha256(
        subprocess.check_output(["git", "-C", a.upstream, "diff", "HEAD"])
    ).hexdigest(),
    upstream_url="https://github.com/mani-skill/ManiSkill",
    env_id="PickCube-v1",
    obs_mode="state",
    control_mode="pd_ee_delta_pos",
    sim_backend="cpu",
    training_seed=1,
    training_command=Path("artifacts/training-command.txt").read_text(),
    checkpoint_path=a.checkpoint,
    raw_dataset_sha256=sha(Path(a.dataset).parent / "trajectory.h5"),
    dataset_revision="d674485bbffdd533914e52d272fdda34c0515608",
    dataset_url="https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations",
    replay_command="python scripts/run_official.py --module mani_skill.trajectory.replay_trajectory --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 --use-first-env-state -c pd_ee_delta_pos -o state --save-traj --num-envs 1 -b cpu",
    total_updates=10000,
    checkpoint_iteration=9999,
    normalize_states=False,
)
Path("artifacts/checkpoint-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
