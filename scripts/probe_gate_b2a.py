"""Isolated forensic replays, never Gate B2 confirmation or nonzero disturbance."""

import argparse
import importlib.metadata
import json
import subprocess
from itertools import combinations
from pathlib import Path

from failure2fix.clean import sha256
from failure2fix.clean_dp import load_dp
from failure2fix.contact_forensics import ForensicRecorder, compare_forensic
from failure2fix.gate_b import compare_branches, compare_schedule, run_branch
from failure2fix.gate_b2 import canonical, retain_failure


def analyse(out, runs, schedule):
    comparisons = []
    for i, j in combinations(range(len(runs)), 2):
        result = compare_forensic(runs[i], runs[j])
        result.update(
            left=i,
            right=j,
            branches=[schedule[i], schedule[j]],
            recorded_execution=compare_branches(runs[i][0], runs[j][0], atol=0),
            schedule=compare_schedule(runs[i][0], runs[j][0]),
        )
        if result["first"] is not None:
            (out / f"first-{i}-{j}.json").write_bytes(canonical(result["first"]))
            result["first"] = str(f"first-{i}-{j}.json")
        comparisons.append(result)
    summary = dict(
        diagnostic_only=True,
        gate_B2="STOPPED",
        gate_C="NO-GO",
        comparisons=comparisons,
        analysis_source_files={
            f: sha256(f)
            for f in ["scripts/probe_gate_b2a.py", "src/failure2fix/contact_forensics.py"]
        },
    )
    (out / "summary.json").write_bytes(canonical(summary))


def load_records(out):
    import gzip

    import numpy as np

    from failure2fix.replay_gate import Frame

    schedule = json.loads((out / "config.json").read_text())["order"]
    index_path = out / "record-index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {
            str(i): dict(file=f"{i}-{b}.json.gz", member="actual") for i, b in enumerate(schedule)
        }
    )
    runs = []
    for i in range(len(schedule)):
        entry = index[str(i)]
        payload = json.loads(gzip.decompress((out / entry["file"]).read_bytes()))[entry["member"]]
        run = payload["run"]
        run["frames"] = [
            Frame(
                f["step"],
                None if f["action"] is None else np.asarray(f["action"]),
                {k: np.asarray(v) for k, v in f["state"].items()},
                f["grasped"],
                f["success"],
            )
            for f in payload["frames"]
        ]
        runs.append((run, payload["audit"]))
    return runs, schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/gate-b2a/forensic-v1")
    p.add_argument("--upstream", default=".cache/clean-ManiSkill")
    p.add_argument(
        "--analyse-only",
        type=Path,
        help="Reanalyse saved forensic records; no simulator or policy execution",
    )
    args = p.parse_args()
    if args.analyse_only:
        runs, schedule = load_records(args.analyse_only)
        analyse(args.analyse_only, runs, schedule)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path("artifacts/gate-b2/plan.json").read_text())
    assert sha256(plan["checkpoint"]) == plan["checkpoint_sha256"]
    schedule = ["clean", "sham", "sham", "clean", "clean", "sham"]
    config = dict(
        purpose="FORENSIC DEBUGGING ONLY; Gate B2 STOPPED, Gate C NO-GO",
        seed=60001,
        diagnostic_exception="Explicitly authorized replay of consumed seed",
        order=schedule,
        nonzero_impulses=False,
        horizon=200,
        physics_hz=100,
        checkpoint_sha256=plan["checkpoint_sha256"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_files={
            f: sha256(f)
            for f in [
                "scripts/probe_gate_b2a.py",
                "src/failure2fix/contact_forensics.py",
                "src/failure2fix/gate_b2.py",
            ]
        },
        packages={k: importlib.metadata.version(k) for k in ["sapien", "mani_skill", "torch"]},
    )
    (out / "config.json").write_bytes(canonical(config))
    agent = load_dp(args.upstream, plan["checkpoint"])
    agent.requires_grad_(False)
    runs = []
    trigger = None
    for i, branch in enumerate(schedule):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        recorder = ForensicRecorder()

        def factory():
            env = gym.make(
                "PickCube-v1",
                obs_mode="state",
                control_mode="pd_ee_delta_pos",
                sim_backend="cpu",
                render_backend="cpu",
                render_mode="rgb_array",
                num_envs=1,
                reconfiguration_freq=1,
                max_episode_steps=200,
                sim_config=dict(sim_freq=100, control_freq=20),
            )
            recorder.attach(env.unwrapped)
            return env

        run = run_branch(factory, agent, 60001, branch, 0.0, trigger, 200)
        trigger = run["trigger"]
        audit = recorder.evidence(run)
        audit["_substeps"], audit["_forensic"] = recorder.rows, recorder.substeps
        pair = (run, audit)
        retain_failure(out / f"{i}-{branch}.json.gz", None, pair)
        runs.append(pair)
        if i and audit["full_fingerprints"] != runs[0][1]["full_fingerprints"]:
            path = out / "failure-pair.json.gz"
            if not path.exists():
                retain_failure(path, runs[0], pair)
        print(
            json.dumps(
                dict(
                    run=i,
                    branch=branch,
                    trigger=trigger,
                    strict_equal_to_first=audit["full_fingerprints"]
                    == runs[0][1]["full_fingerprints"],
                )
            ),
            flush=True,
        )
    analyse(out, runs, schedule)
    assert sha256(plan["checkpoint"]) == plan["checkpoint_sha256"]
    print("FORENSIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
