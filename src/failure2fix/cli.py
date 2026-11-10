"""Run paired-seed policy evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
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
    p.add_argument("--policy", choices=["toy-oracle", "zero", "random", "custom"],
                   default="toy-oracle")
    p.add_argument("--factory", help="For custom policy: module.path:create_policy")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument("--obs-noise", type=float, default=0.12)
    p.add_argument("--action-delay", type=int, default=2)
    p.add_argument("--obs-mode", default="state")
    p.add_argument("--control-mode", default="pd_joint_delta_pos")
    p.add_argument("--output", default="runs/toy-v0")
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.episodes < 1 or args.max_steps < 1:
            parser.error("--episodes and --max-steps must be positive")
        if args.obs_noise < 0 or args.action_delay < 0:
            parser.error("--obs-noise and --action-delay must be non-negative")
        if args.env != "toy" and args.policy == "toy-oracle":
            parser.error("--policy toy-oracle only works with --env toy")
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        env_factory = lambda: make_env(args.env, args.obs_mode, args.control_mode)
        seeds = list(range(args.seed, args.seed + args.episodes))
        conditions = [
            Condition("clean"),
            Condition("shifted", obs_noise=args.obs_noise, action_delay=args.action_delay),
        ]
        results = []
        with (out / "steps.jsonl").open("w", encoding="utf-8") as traces:
            for condition in conditions:
                results += evaluate_condition(
                    env_factory=env_factory, env_id=args.env, policy_name=args.policy,
                    condition=condition, seeds=seeds, max_steps=args.max_steps,
                    factory=args.factory,
                    on_step=lambda record: traces.write(json.dumps(record) + "\n"),
                )
        _write_jsonl(out / "episodes.jsonl", [to_dict(r) for r in results])
        summaries = summarize(results)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        manifest = {
            "experiment": "paired-seed clean vs shifted execution",
            "env_id": args.env, "policy": args.policy, "policy_factory": args.factory,
            "seeds": seeds, "max_steps": args.max_steps,
            "obs_mode": args.obs_mode, "control_mode": args.control_mode,
            "conditions": [vars(c) for c in conditions],
            "python": sys.version.split()[0], "numpy": np.__version__,
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
            rate = "unknown" if item["success_rate"] is None else f'{item["success_rate"]:.1%}'
            print(f'  {item["condition"]}: success={rate}, n={item["episodes"]}')


if __name__ == "__main__":
    main()
