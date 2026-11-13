"""Execute the predeclared Gate B pilot; stop on invalid physical or replay checks."""

import argparse
import json
import math
import subprocess
from pathlib import Path

from failure2fix.clean import sha256
from failure2fix.clean_dp import load_dp
from failure2fix.envs import make_env
from failure2fix.gate_b import (
    compare_branches,
    compare_schedule,
    json_default,
    metrics,
    run_branch,
    save_run,
    validate_impulse_api,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - half), min(1, center + half)]


def level_summary(impulse, runs, clean):
    n = len(runs)
    k = sum(r["frames"][-1].success for r in runs)
    regressions = sum(
        clean[r["seed"]]["frames"][-1].success and not r["frames"][-1].success for r in runs
    )
    preserved = sum(
        clean[r["seed"]]["frames"][-1].success and r["frames"][-1].success for r in runs
    )
    improvements = sum(
        not clean[r["seed"]]["frames"][-1].success and r["frames"][-1].success for r in runs
    )
    return dict(
        impulse_Ns=impulse,
        episodes=n,
        successes=k,
        success_rate=k / n,
        wilson95=wilson(k, n),
        paired_regressions=regressions,
        paired_improvements=improvements,
        preserved_successes=preserved,
        eligible=0 < k < n and regressions > 0 and preserved > 0,
    )


def choose(levels):
    eligible = [r for r in levels if r["eligible"]]
    return (
        min(eligible, key=lambda r: (abs(r["success_rate"] - 0.5), r["impulse_Ns"]))
        if eligible
        else None
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="artifacts/gate-b/plan.json")
    p.add_argument("--upstream", default=".cache/clean-ManiSkill")
    p.add_argument("--output", default="runs/gate-b/pilot-v1")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    seeds = plan["seeds"]
    horizon = plan["horizon"]
    levels = plan["coarse_impulses_Ns"]
    if len(set(seeds)) != len(seeds) or any(
        s < 0 or 20000 <= s < 20050 or 40000 <= s < 40100 for s in seeds
    ):
        raise ValueError("Invalid or locked seeds")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="RUNNING",
        gate_pass=False,
        levels=[],
        prefix_checks=[],
        schedule_checks=[],
        sham_checks=[],
        repeat_checks=[],
        video_checks=[],
    )
    write(
        out / "config.json",
        dict(
            plan=plan,
            plan_sha256=sha256(args.plan),
            arguments=vars(args),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_diff=subprocess.check_output(["git", "diff", "HEAD"], text=True),
        ),
    )
    try:
        if sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint checksum mismatch")
        result["physical_validation"] = validate_impulse_api(levels)
        write(out / "physical-validation.json", result["physical_validation"])
        agent = load_dp(args.upstream, plan["checkpoint"])
        agent.requires_grad_(False)

        def factory():
            return make_env(
                "PickCube-v1",
                obs_mode="state",
                control_mode="pd_ee_delta_pos",
                sim_backend="cpu",
                render_backend="cpu",
                render_mode="rgb_array",
                max_episode_steps=horizon,
            )

        all_metrics = []
        clean = {}
        sham = {}
        actual = {}

        def persist(run, tag):
            path = out / "traces" / f"{tag}-{run['seed']}.jsonl.gz"
            save_run(path, run)
            row = metrics(run)
            row["regrasp_latency_s"] = (
                None
                if row["regrasp_latency_steps"] is None
                else row["regrasp_latency_steps"] * 0.05
            )
            row["post_event_goal_latency_s"] = (
                None
                if row["post_event_goal_latency_steps"] is None
                else row["post_event_goal_latency_steps"] * 0.05
            )
            if run["seed"] in clean and row["stable_goal_step"] is not None:
                baseline = metrics(clean[run["seed"]])["stable_goal_step"]
                row["paired_goal_delay_steps"] = (
                    None if baseline is None else row["stable_goal_step"] - baseline
                )
            row["trace"] = str(path.relative_to(out))
            all_metrics.append(row)
            write(out / "episodes.json", all_metrics)
            print(
                json.dumps(
                    dict(
                        seed=run["seed"],
                        branch=run["branch"],
                        J=run["impulse_Ns"],
                        trigger=run["trigger"],
                        success_at_200=row["success_at_200"],
                    )
                ),
                flush=True,
            )

        # All negative controls must pass before any nonzero in-contact impulse.
        for seed in seeds:
            clean[seed] = run_branch(factory, agent, seed, "clean", horizon=horizon)
            persist(clean[seed], "clean")
            sham[seed] = run_branch(
                factory, agent, seed, "sham", trigger=clean[seed]["trigger"], horizon=horizon
            )
            persist(sham[seed], "sham")
            check = dict(seed=seed, **compare_branches(clean[seed], sham[seed]))
            result["sham_checks"].append(check)
            write(out / "summary.json", result)
            if not check["pass"]:
                raise RuntimeError(f"STOP: sham divergence on seed {seed}: {check}")
        result["clean"] = dict(
            episodes=len(seeds), successes=sum(c["frames"][-1].success for c in clean.values())
        )

        def evaluate_level(impulse):
            runs = []
            for seed in seeds:
                run = run_branch(
                    factory, agent, seed, "disturbance", impulse, clean[seed]["trigger"], horizon
                )
                persist(run, f"impulse-{impulse:g}")
                check = dict(
                    seed=seed,
                    impulse_Ns=impulse,
                    **compare_branches(clean[seed], run, prefix_only=True),
                )
                result["prefix_checks"].append(check)
                if not check["pass"]:
                    raise RuntimeError(f"STOP: prefix divergence: {check}")
                schedule = dict(seed=seed, impulse_Ns=impulse, **compare_schedule(clean[seed], run))
                result["schedule_checks"].append(schedule)
                if not schedule["pass"]:
                    raise RuntimeError(f"STOP: altered policy inference/RNG schedule: {schedule}")
                actual[(seed, impulse)] = run
                runs.append(run)
            result["levels"].append(level_summary(impulse, runs, clean))
            write(out / "summary.json", result)

        for impulse in levels:
            evaluate_level(impulse)
        selected = choose(result["levels"])
        if selected is None:
            ordered = sorted(result["levels"], key=lambda r: r["impulse_Ns"])
            brackets = [
                (a, b)
                for a, b in zip(ordered, ordered[1:])
                if a["successes"] == len(seeds) and b["successes"] == 0
            ]
            if brackets:
                a, b = brackets[0]
                evaluate_level(math.sqrt(a["impulse_Ns"] * b["impulse_Ns"]))
                selected = choose(result["levels"])
        result["selected"] = selected
        if selected is None:
            result["status"] = "INCONCLUSIVE: no nontrivial severity in the predeclared pilot"
            raise SystemExit(2)
        impulse = selected["impulse_Ns"]
        for seed in seeds:
            for branch, reference in [
                ("clean", clean[seed]),
                ("sham", sham[seed]),
                ("disturbance", actual[(seed, impulse)]),
            ]:
                run = run_branch(
                    factory,
                    agent,
                    seed,
                    branch,
                    impulse if branch == "disturbance" else 0.0,
                    clean[seed]["trigger"],
                    horizon,
                )
                persist(run, f"repeat-{branch}-{impulse:g}")
                check = dict(seed=seed, branch=branch, **compare_branches(reference, run))
                result["repeat_checks"].append(check)
                if not check["pass"]:
                    raise RuntimeError(f"STOP: nonreproducible configuration: {check}")
        representatives = [seeds[0]]
        for regression in [True, False]:
            match = next(
                (
                    s
                    for s in seeds
                    if clean[s]["frames"][-1].success
                    and bool(not actual[(s, impulse)]["frames"][-1].success) == regression
                ),
                None,
            )
            if match is not None and match not in representatives:
                representatives.append(match)
        for seed in representatives:
            for branch, reference in [
                ("clean", clean[seed]),
                ("sham", sham[seed]),
                ("disturbance", actual[(seed, impulse)]),
            ]:
                video = out / "videos" / f"{seed}-{branch}.mp4"
                run = run_branch(
                    factory,
                    agent,
                    seed,
                    branch,
                    impulse if branch == "disturbance" else 0.0,
                    clean[seed]["trigger"],
                    horizon,
                    video,
                )
                check = dict(
                    seed=seed,
                    branch=branch,
                    video=str(video.relative_to(out)),
                    **compare_branches(reference, run),
                )
                result["video_checks"].append(check)
                if not check["pass"]:
                    raise RuntimeError(f"STOP: rendered trajectory differs: {check}")
        if sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        result.update(status="PASS", gate_pass=True)
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(out / "summary.json", result)
        print(
            json.dumps(
                dict(
                    status=result["status"], gate_pass=result["gate_pass"], levels=result["levels"]
                ),
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
