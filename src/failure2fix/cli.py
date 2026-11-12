"""Run paired-seed policy evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

from failure2fix.envs import make_env
from failure2fix.evaluation import Condition, evaluate_condition, summarize, to_dict


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="f2f", description="Reproducible robot policy failure evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate", help="Compare clean vs shifted execution with paired seeds")
    p.add_argument("--env", default="toy", help="toy or registered ManiSkill env ID")
    p.add_argument(
        "--policy",
        choices=["toy-oracle", "zero", "random", "custom", "official-bc"],
        default="toy-oracle",
    )
    p.add_argument("--factory", help="For custom policy: module.path:create_policy")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--obs-noise", type=float, default=0.12)
    p.add_argument("--action-delay", type=int, default=2)
    p.add_argument("--obs-mode", default="state")
    p.add_argument("--control-mode", default="pd_joint_delta_pos")
    p.add_argument("--output", default="runs/toy-v0")
    p.add_argument("--checkpoint")
    p.add_argument("--provenance")
    p.add_argument("--upstream", help="Pinned official ManiSkill checkout")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sim-backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--render-backend", default="gpu")
    p.add_argument("--video-episodes", type=int, default=1)
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        if args.episodes < 1 or args.max_steps < 1:
            parser.error("--episodes and --max-steps must be positive")
        if args.video_episodes < 0:
            parser.error("--video-episodes must be non-negative")
        if not np.isfinite(args.obs_noise) or args.obs_noise < 0 or args.action_delay < 0:
            parser.error("--obs-noise and --action-delay must be non-negative")
        if args.env != "toy" and args.policy == "toy-oracle":
            parser.error("--policy toy-oracle only works with --env toy")
        out = Path(args.output)
        if out.exists() and any(out.iterdir()):
            parser.error("--output must be empty to preserve previous experiments")
        out.mkdir(parents=True, exist_ok=True)

        def env_factory():
            return make_env(
                args.env,
                args.obs_mode,
                args.control_mode,
                args.sim_backend,
                "rgb_array" if args.env != "toy" and args.video_episodes else None,
                args.max_steps,
                args.render_backend,
            )

        loader = None
        provenance = None
        if args.policy == "official-bc":
            from failure2fix.baseline import official_bc_loader

            loader, provenance = official_bc_loader(args)
        seeds = list(range(args.seed, args.seed + args.episodes))
        conditions = [
            Condition("clean"),
            Condition("observation_noise", obs_noise=args.obs_noise),
            Condition("action_delay", action_delay=args.action_delay),
        ]
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            diff = subprocess.check_output(["git", "diff", "HEAD"])
        except (OSError, subprocess.CalledProcessError):
            revision, diff = None, b""
        runtime = {
            "source_commit": revision,
            "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "command": sys.argv,
        }
        (out / "config.json").write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "runtime": runtime,
                    "seeds": seeds,
                    "conditions": [vars(c) for c in conditions],
                    "checkpoint_provenance": provenance,
                },
                indent=2,
            )
            + "\n"
        )
        results = []
        from failure2fix.video import RolloutVideos

        videos = RolloutVideos(out / "videos", seeds[: args.video_episodes], args.env != "toy")
        with closing(videos), (out / "steps.jsonl").open("w", encoding="utf-8") as traces:
            for condition in conditions:
                results += evaluate_condition(
                    env_factory=env_factory,
                    env_id=args.env,
                    policy_name=args.policy,
                    condition=condition,
                    seeds=seeds,
                    max_steps=args.max_steps,
                    factory=args.factory,
                    policy_loader=loader,
                    on_frame=videos,
                    on_step=lambda record: traces.write(json.dumps(record) + "\n"),
                )
        videos.close()
        _write_jsonl(out / "episodes.jsonl", [to_dict(r) for r in results])
        hashes = {}
        for result in results:
            previous = hashes.setdefault(result.seed, result.initial_observation_sha256)
            if previous != result.initial_observation_sha256:
                raise RuntimeError("Paired initial observations differ")
        clean = {r.seed: r for r in results if r.condition == "clean"}
        paired = {}
        for condition in conditions[1:]:
            group = [r for r in results if r.condition == condition.name]
            known = [
                r for r in group if r.success is not None and clean[r.seed].success is not None
            ]
            paired[condition.name] = {
                "known_pairs": len(known),
                "regressions": sum(clean[r.seed].success and not r.success for r in known),
                "improvements": sum(not clean[r.seed].success and r.success for r in known),
                "success_rate_delta": (
                    sum(int(r.success) - int(clean[r.seed].success) for r in known) / len(known)
                    if known
                    else None
                ),
            }
        (out / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
        summaries = summarize(results)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        manifest = {
            "experiment": "paired-seed separate shifts",
            "initial_observations_matched": True,
            "runtime": runtime,
            "arguments": vars(args),
            "checkpoint_provenance": provenance,
            "env_id": args.env,
            "policy": args.policy,
            "policy_factory": args.factory,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "obs_mode": args.obs_mode,
            "control_mode": args.control_mode,
            "conditions": [vars(c) for c in conditions],
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "notes": [
                "ToyReach is a smoke test, not a manipulation result.",
                "Shared seeds do not guarantee identical initial states in every simulator.",
                "Unknown success stays null; termination is not success.",
                "Observation and action interface shifts are not physical dynamics shifts.",
            ],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {len(results)} episodes to {out}")
        for item in summaries:
            rate = "unknown" if item["success_rate"] is None else f"{item['success_rate']:.1%}"
            print(f"  {item['condition']}: success={rate}, n={item['episodes']}")


if __name__ == "__main__":
    main()
