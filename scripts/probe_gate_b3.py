"""Fixed 12-seed exploratory impulse feasibility pilot; never resumes B2 confirmation."""

import argparse
import gzip
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

from failure2fix.clean import sha256
from failure2fix.clean_dp import load_dp
from failure2fix.gate_b import run_branch, validate_impulse_api
from failure2fix.gate_b2 import canonical
from failure2fix.gate_b2b import digest, save_failure
from failure2fix.gate_b3 import (
    SpatialRecorder,
    compare,
    episode,
    response,
    select_level,
    timestep_case,
)


def write(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/gate-b3/pilot-v1")
    p.add_argument("--plan", default="artifacts/gate-b3/plan.json")
    p.add_argument("--upstream", default=".cache/clean-ManiSkill")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    seeds = plan["seeds"]
    impulses = plan["impulses_Ns"]
    if seeds != list(range(80000, 80012)) or impulses != [0.064, 0.128, 0.192]:
        raise ValueError("Only the fixed preregistered development pilot is supported")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "records").mkdir()
    (out / "raw").mkdir()
    config = dict(
        plan=plan,
        plan_sha256=sha256(args.plan),
        protocol_sha256=sha256("docs/GATE_B3_PLAN.md"),
        arguments=vars(args),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_diff=subprocess.check_output(["git", "diff", "HEAD"], text=True),
        packages=subprocess.check_output([".venv/bin/pip", "freeze"], text=True),
    )
    write(out / "config.json", config)
    status = dict(
        status="RUNNING",
        checks=[],
        episodes=[],
        timestep_cases=[],
        historical_B2="STOPPED",
        historical_B2b="STOPPED",
    )
    last_attempt, last_reference = None, None
    tags = [("clean", 0.0), ("sham", 0.0), *[(f"J{J:.3f}", J) for J in impulses]]
    rows = []

    def require(ok, **detail):
        status["checks"].append(dict(pass_=bool(ok), **detail))
        if not ok:
            raise RuntimeError(f"STOP: {detail}")

    def persist(pair, tag, freq, reference):
        run, audit = pair
        key = f"{freq}-{tag}-{run['seed']}"
        with gzip.open(out / "raw" / f"{key}.pkl.gz", "wb") as f:
            pickle.dump(pair, f, protocol=5)
        public = {k: v for k, v in audit.items() if not k.startswith("_")}
        n = audit["event_index"]
        m = round(0.05 / audit["dt_s"])
        public["exact_window"] = audit["_exact"][max(0, n - 2 * m) : n + 10 * m]
        public["spatial_timeline"] = audit["_spatial"]
        frames = [
            dict(
                step=f.step,
                action=f.action,
                grasped=f.grasped,
                success=f.success,
                observation=f.state["obs"],
                cube=f.state["env/actors/cube"],
                state_sha256=digest(f.state),
            )
            for f in run["frames"]
        ]
        write_value = dict(
            event=run["event"], inference_steps=run["inference_steps"], frames=frames, audit=public
        )
        (out / "records" / f"{key}.json.gz").write_bytes(
            gzip.compress(canonical(write_value), mtime=0)
        )
        row = episode(run, audit)
        row.update(
            tag=tag,
            freq=freq,
            physical_valid=audit["valid"],
            clean_success=run["frames"][-1].success
            if reference is None
            else reference[0]["frames"][-1].success,
            response_50ms=response(run, audit, audit if reference is None else reference[1]),
            physical={
                k: v
                for k, v in public.items()
                if k
                not in [
                    "exact_window",
                    "spatial_timeline",
                    "exact_hashes",
                    "spatial_hashes",
                    "window",
                    "logical_shape_inventory",
                    "collision_geometry",
                ]
            },
        )
        rows.append(row)
        status["episodes"] = rows
        write(out / "summary.json", status)
        print(
            json.dumps(
                dict(
                    seed=run["seed"],
                    tag=tag,
                    freq=freq,
                    success=row["success_at_200"],
                    departed=row["departed"],
                    verified_loss=row["verified_loss_step"],
                    physical_valid=audit["valid"],
                )
            ),
            flush=True,
        )
        return row

    def execute(seed, tag, J, reference=None, freq=100, record=True, video=None):
        nonlocal last_attempt, last_reference
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        recorder = SpatialRecorder()
        last_reference = reference
        partial = dict(seed=seed, branch=tag, impulse_Ns=J, frames=[], incomplete=True)
        partial_audit = dict(
            _substeps=recorder.rows,
            _exact=recorder.exact,
            _spatial=recorder.spatial_rows,
            collision_geometry=recorder.collision_geometry,
        )
        last_attempt = (partial, partial_audit)

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
                sim_config=dict(sim_freq=freq, control_freq=20),
            )
            recorder.attach(env.unwrapped)
            return env

        if video is None and record and seed == seeds[0] and freq == 100:
            video = out / "videos" / f"{seed}-{tag}.mp4"
        trigger = None if reference is None else reference[0]["trigger"]
        branch = tag if tag in ["clean", "sham"] else "disturbance"
        run = run_branch(factory, agent, seed, branch, J, trigger, 200, video, capture=partial)
        last_attempt = (run, partial_audit)
        audit = recorder.evidence(run)
        pair = (run, audit)
        last_attempt = pair
        if record:
            persist(pair, tag, freq, reference)
        require(
            audit["valid"],
            kind="physical-validity",
            seed=seed,
            tag=tag,
            freq=freq,
            native_min_m=audit["min_native_separation_all_m"],
            table_depth_m=audit["max_table_penetration_m"],
            floor_depth_m=audit["max_floor_penetration_m"],
            friction_excess_Ns=audit["friction_bound_excess_Ns"],
        )
        if reference is not None:
            check = compare(reference, pair, prefix=tag != "sham")
            require(
                check["pass_"],
                kind="sham-exact" if tag == "sham" else "prefix-exact",
                seed=seed,
                tag=tag,
                freq=freq,
                detail=check,
            )
        return pair

    def load(seed, tag, freq=100):
        with gzip.open(out / "raw" / f"{freq}-{tag}-{seed}.pkl.gz", "rb") as f:
            return pickle.load(f)

    def repeat(seed, tag, J):
        nonlocal last_reference
        original = load(seed, tag)
        # Compare the same disturbance branch through its entire trajectory, not to clean.
        clean = load(seed, "clean")
        repeated = execute(seed, tag, J, clean if tag != "clean" else None, record=False)
        last_reference = original
        check = compare(original, repeated)
        require(check["pass_"], kind="full-repeat", seed=seed, tag=tag, detail=check)

    def audit_seed(seed, subset):
        for freq in [200, 400]:
            clean = execute(seed, "clean", 0.0, freq=freq)
            for tag, J in subset:
                execute(seed, tag, J, clean, freq)

    try:
        require(not config["source_diff"], kind="committed-source")
        require(sha256(plan["checkpoint"]) == plan["checkpoint_sha256"], kind="frozen-checkpoint")
        status["free_body_calibration"] = validate_impulse_api(impulses)
        agent = load_dp(args.upstream, plan["checkpoint"])
        agent.requires_grad_(False)
        for seed in seeds:
            clean = execute(seed, "clean", 0.0)
            execute(seed, "sham", 0.0, clean)
        require(
            sum(r["success_at_200"] for r in rows if r["tag"] == "clean") >= 11,
            kind="credible-clean-baseline",
        )
        for tag, J in tags[2:]:
            for seed in seeds:
                execute(seed, tag, J, load(seed, "clean"))
        levels, selected = select_level(rows)
        status.update(levels=levels, selected=selected)
        write(out / "summary.json", status)
        for seed in seeds[:3]:
            for tag, J in tags:
                repeat(seed, tag, J)
        for seed in seeds[:3]:
            audit_seed(seed, tags[1:])
        unstable = []
        for seed in seeds[:3]:
            for tag, J in tags[2:]:
                triple = [
                    next(
                        r
                        for r in rows
                        if r["seed"] == seed and r["tag"] == tag and r["freq"] == freq
                    )
                    for freq in [100, 200, 400]
                ]
                check = dict(seed=seed, tag=tag, **timestep_case(triple))
                status["timestep_cases"].append(check)
                if check["unstable"]:
                    unstable.append(check)
        controls_unstable = any(
            not r["success_at_200"] for r in rows if r["tag"] == "clean" and r["freq"] != 100
        )
        sensitive = len(unstable) >= 2 or controls_unstable
        if selected is not None and not sensitive:
            J = selected["impulse_Ns"]
            tag = f"J{J:.3f}"
            keyseeds = sorted(set([selected["recovered"][0], selected["noncompleted"][0]]))
            status["key_seeds"] = keyseeds
            for seed in keyseeds:
                if seed not in seeds[:3]:
                    for t, j in [tags[0], tags[1], (tag, J)]:
                        repeat(seed, t, j)
                    audit_seed(seed, [tags[1], (tag, J)])
                triple = [
                    next(
                        r
                        for r in rows
                        if r["seed"] == seed and r["tag"] == tag and r["freq"] == freq
                    )
                    for freq in [100, 200, 400]
                ]
                check = dict(seed=seed, tag=tag, key_case=True, **timestep_case(triple))
                status["timestep_cases"].append(check)
                sensitive = sensitive or check["unstable"]
        sensitive = sensitive or any(
            not r["success_at_200"] for r in rows if r["tag"] == "clean" and r["freq"] != 100
        )
        outcome = "C" if sensitive else ("A" if selected is not None else "B")
        status["level_descriptions"] = [
            dict(
                impulse_Ns=J,
                n=12,
                categories=dict(
                    Counter(
                        r["category"] for r in rows if r["freq"] == 100 and r["impulse_Ns"] == J
                    )
                ),
                verified_losses=sum(
                    r["verified_loss_step"] is not None
                    for r in rows
                    if r["freq"] == 100 and r["impulse_Ns"] == J
                ),
            )
            for J in impulses
        ]
        require(
            sha256(plan["checkpoint"]) == plan["checkpoint_sha256"], kind="frozen-checkpoint-final"
        )
        status.update(
            status="COMPLETE",
            outcome=outcome,
            recommendation="CONTINUE" if outcome == "A" else "PIVOT",
            abandon_current_impulse_configuration=outcome != "A",
            rationale="timestep sensitivity"
            if sensitive
            else (
                "mixed stable in-workspace failures"
                if selected
                else "no suitable mixed in-workspace fault regime"
            ),
        )
    except Exception as exc:
        status.update(
            status="STOP",
            outcome="C",
            recommendation="PIVOT",
            abandon_current_impulse_configuration=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        try:
            save_failure(out / "failure-pair.json.gz", last_reference, last_attempt)
        except Exception as save_error:
            status["failure_capture_error"] = str(save_error)
        raise
    finally:
        status["episodes"] = rows
        write(out / "summary.json", status)
        print(
            json.dumps(
                dict(
                    status=status["status"],
                    outcome=status.get("outcome"),
                    recommendation=status.get("recommendation"),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
