"""Run the official 30k-update Diffusion Policy recipe with fixed clean validation."""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from failure2fix.clean import sha256, write_json
from failure2fix.clean_dp import vector_evaluate

p = argparse.ArgumentParser()
p.add_argument("--upstream", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--output", required=True)
p.add_argument("--updates", type=int, default=30000)
p.add_argument("--run-name", default="clean-dp-30k")
p.add_argument("--eval-every", type=int, default=10000)
p.add_argument("--batch-size", type=int, default=1024)
a = p.parse_args()
if (Path("runs") / a.run_name).exists():
    raise FileExistsError(a.run_name)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
write_json(
    out / "config.json",
    dict(
        **vars(a),
        dataset_sha256=sha256(a.dataset),
        validation_seeds=[20000, 20049],
        horizon=200,
        vector_envs=10,
        noise_seed="first seed of each fixed batch",
    ),
)
write_json(
    out / "provenance.json",
    {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", a.upstream, "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_diff": subprocess.check_output(
            ["git", "-C", a.upstream, "diff", "HEAD"], text=True
        ),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "diffusion_policy": "official default 4.39M-parameter UNet, EMA, 100 DDPM steps; architecture unchanged",
    },
)
sys.path.insert(0, str(Path(a.upstream) / "examples/baselines/diffusion_policy"))
import diffusion_policy.evaluate as evaluation  # noqa: E402

calls = 0


def validate(n, agent, envs, device, sim_backend):
    global calls
    calls += 1
    records = vector_evaluate(agent, envs, list(range(20000, 20050)), 200, str(device))
    summary = dict(
        updates=calls * a.eval_every,
        episodes=len(records),
        success_once=float(np.mean([r["success_once"] for r in records])),
        success_at_end=float(np.mean([r["success_at_end"] for r in records])),
        success_once_100=float(np.mean([r["success_once_100"] for r in records])),
    )
    write_json(out / f"{summary['updates']:06d}.json", dict(summary=summary, episodes=records))
    print("CLEAN_DP_VALIDATION " + json.dumps(summary), flush=True)
    return {k: np.asarray([r[k] for r in records]) for k in ["success_once", "success_at_end"]}


evaluation.evaluate = validate
make = gym.make


def configured(*args, **kwargs):
    kwargs["render_backend"] = "none"
    return make(*args, **kwargs)


gym.make = configured


# Ten synchronous CPU scenes avoid multiprocessing memory duplication on the WSL workstation.
class SyncCPUVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(self, factories, **kwargs):
        super().__init__(factories)


gym.vector.AsyncVectorEnv = SyncCPUVectorEnv
sys.argv = [
    "train.py",
    "--env-id",
    "PickCube-v1",
    "--demo-path",
    a.dataset,
    "--control-mode",
    "pd_ee_delta_pos",
    "--sim-backend",
    "physx_cpu",
    "--max-episode-steps",
    "200",
    "--total-iters",
    str(a.updates),
    "--batch-size",
    str(a.batch_size),
    "--seed",
    "1",
    "--num-eval-envs",
    "10",
    "--num-eval-episodes",
    "50",
    "--no-capture-video",
    "--save-freq",
    str(a.eval_every),
    "--eval-freq",
    str(a.eval_every),
    "--log-freq",
    "100",
    "--exp-name",
    a.run_name,
]
runpy.run_path(
    str(Path(a.upstream) / "examples/baselines/diffusion_policy/train.py"), run_name="__main__"
)
