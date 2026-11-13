"""One-shot test evaluation of a frozen validation-selected policy."""

import argparse
import gzip
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from failure2fix.clean import load_actor, rollout, sha256, write_json
from failure2fix.clean_dp import load_dp, vector_evaluate

p = argparse.ArgumentParser()
p.add_argument("--selection", default="artifacts/clean-review/selection.json")
p.add_argument("--upstream", default=".cache/clean-ManiSkill")
p.add_argument("--output", default="runs/clean-review/test")
a = p.parse_args()
selection_path = Path(a.selection)
selection = json.loads(selection_path.read_text())
policy = selection["selected"]
if sha256(policy["checkpoint"]) != policy["checkpoint_sha256"]:
    raise ValueError("Selected checkpoint changed")
seeds = selection["test_seeds"]
if seeds != list(range(40000, 40100)) or selection["horizon"] != 200:
    raise ValueError("Unexpected reserved test split or horizon")
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
# Exclusive creation prevents accidental repeated test use with another output directory.
guard = selection_path.with_suffix(".test-started.json")
with guard.open("x") as f:
    json.dump(
        dict(
            checkpoint_sha256=policy["checkpoint_sha256"],
            started_utc=datetime.now(timezone.utc).isoformat(),
        ),
        f,
    )
write_json(
    out / "config.json",
    dict(
        selection=selection,
        arguments=vars(a),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    ),
)
if policy["method"] == "bc":
    import torch

    actor = load_actor(a.upstream, policy["checkpoint"])

    def sample(obs):
        return actor(torch.as_tensor(obs, device="cuda", dtype=torch.float32)).cpu().numpy()

    records = rollout(sample, seeds, 200, out / "videos", seeds[:2])
    traces = []
    for r in records:
        traces.append(dict(seed=r["seed"], trace=r.pop("trace")))
    (out / "traces.json.gz").write_bytes(gzip.compress(json.dumps(traces).encode(), mtime=0))
else:
    import gymnasium as gym

    from failure2fix.clean_dp import official_dp

    official_dp(a.upstream)
    from diffusion_policy.make_env import make_eval_envs

    make = gym.make
    created = 0

    def configured(*args, **kwargs):
        global created
        kwargs["render_backend"] = "cpu" if created == 0 else "none"
        created += 1
        return make(*args, **kwargs)

    gym.make = configured

    class SyncCPUVectorEnv(gym.vector.SyncVectorEnv):
        def __init__(self, factories, **kwargs):
            super().__init__(factories)

    gym.vector.AsyncVectorEnv = SyncCPUVectorEnv
    agent = load_dp(a.upstream, policy["checkpoint"])
    envs = make_eval_envs(
        "PickCube-v1",
        10,
        "physx_cpu",
        dict(
            obs_mode="state",
            control_mode="pd_ee_delta_pos",
            max_episode_steps=200,
            render_mode="rgb_array",
        ),
        dict(obs_horizon=2),
        video_dir=str(out / "videos"),
    )
    try:
        records = vector_evaluate(agent, envs, seeds, 200)
    finally:
        envs.close()
write_json(out / "episodes.json", records)
summary = dict(
    episodes=len(records),
    success_once=float(np.mean([r["success_once"] for r in records])),
    success_at_end=float(np.mean([r["success_at_end"] for r in records])),
    success_once_100=float(np.mean([r["success_once_100"] for r in records])),
    method=policy["method"],
    updates=policy["updates"],
)
summary["credible_target_met"] = summary["success_once"] >= 0.8 and summary["success_at_end"] >= 0.7
write_json(out / "summary.json", summary)
print(summary)
