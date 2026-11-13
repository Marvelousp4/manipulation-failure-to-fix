"""WSL replay gate using the frozen 20k-EMA PickCube Diffusion Policy."""

import argparse
import json
from pathlib import Path

from failure2fix.clean import sha256
from failure2fix.clean_dp import ActionChunks, load_dp
from failure2fix.envs import make_env
from failure2fix.replay_gate import replay_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=".cache/clean-ManiSkill")
    p.add_argument("--checkpoint", default="artifacts/clean-review/dp-20000-ema.pt")
    p.add_argument("--provenance", default="artifacts/clean-review/dp-20000-ema-provenance.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[50000, 50001, 50002])
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--output", default="runs/replay-gate/summary.json")
    args = p.parse_args()
    if any(20000 <= s < 20050 or 40000 <= s < 40100 for s in args.seeds):
        p.error("Do not reuse locked validation/test seeds")

    provenance = json.loads(Path(args.provenance).read_text())
    if sha256(args.checkpoint) != provenance["published_sha256"]:
        raise ValueError("Frozen DP checkpoint checksum mismatch")
    agent = load_dp(args.upstream, args.checkpoint)

    class Policy:
        def __init__(self):
            self.chunks = ActionChunks(agent)

        def reset(self, seed):
            self.chunks.reset(seed)

        def act(self, observation):
            return self.chunks(observation)

    def make():
        return make_env(
            "PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pos",
            sim_backend="cpu", render_backend="none", max_episode_steps=200,
        )

    result = replay_gate(make, Policy, args.seeds, args.horizon, args.atol)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "cases": result["cases"]}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
