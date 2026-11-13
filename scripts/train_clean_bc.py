"""Run the official trainer with fixed clean validation and MSE diagnostics."""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from failure2fix.clean import offline_mse, rollout, rollup, sha256, write_json

p = argparse.ArgumentParser()
p.add_argument("--upstream", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--reference")
p.add_argument("--audit", required=True)
p.add_argument("--output", required=True)
p.add_argument("--updates", type=int, default=100000)
p.add_argument("--run-name", default="clean-bc-100k")
a = p.parse_args()
if (Path("runs") / a.run_name).exists():
    raise FileExistsError("Training run already exists; choose a new --run-name")
audit = json.loads(Path(a.audit).read_text())
if (
    audit["dataset_sha256"] != sha256(a.dataset)
    or audit["replay_success"] != 1
    or len(audit["records"]) < 50
    or audit["initial_grasp_flags"] != 0
):
    raise ValueError("Clean replay audit failed or does not match this dataset")
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
write_json(out / "config.json", vars(a))
write_json(
    out / "provenance.json",
    {
        "dataset_sha256": sha256(a.dataset),
        "reference_sha256": sha256(a.reference) if a.reference else None,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", a.upstream, "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_diff": subprocess.check_output(
            ["git", "-C", a.upstream, "diff", "HEAD"], text=True
        ),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
    },
)
sys.path.insert(0, str(Path(a.upstream) / "examples/baselines/bc"))
import behavior_cloning.evaluate as evaluation  # noqa: E402


def validate(n, sample_fn, envs):
    state = sample_fn.__globals__
    update = int(state["iteration"])
    actor = state["actor"]
    records = rollout(sample_fn, list(range(20000, 20050)), horizon=200)
    summary = rollup(records)
    summary.update(updates=update, training_mse=offline_mse(actor, a.dataset))
    if a.reference:
        summary["heldout_mse"] = offline_mse(actor, a.reference)
    for record in records:
        del record["trace"]
    write_json(out / f"{update:06d}.json", dict(summary=summary, episodes=records))
    print("CLEAN_VALIDATION " + json.dumps(summary), flush=True)
    return {
        "success_once": np.array([r["success_once"] for r in records]),
        "success_at_end": np.array([r["success_at_end"] for r in records]),
    }


evaluation.evaluate = validate
make = gym.make


def configured(*args, **kwargs):
    kwargs["render_backend"] = "none"
    return make(*args, **kwargs)


gym.make = configured
sys.argv = [
    "bc.py",
    "--env-id",
    "PickCube-v1",
    "--demo-path",
    a.dataset,
    "--control-mode",
    "pd_ee_delta_pos",
    "--sim-backend",
    "cpu",
    "--max-episode-steps",
    "200",
    "--total-iters",
    str(a.updates),
    "--seed",
    "1",
    "--num-eval-envs",
    "1",
    "--num-eval-episodes",
    "50",
    "--no-capture-video",
    "--save-freq",
    "10000",
    "--eval-freq",
    "10000",
    "--log-freq",
    "1000",
    "--exp-name",
    a.run_name,
]
runpy.run_path(str(Path(a.upstream) / "examples/baselines/bc/bc.py"), run_name="__main__")
