"""Execute the fixed Gate B2 confirmation; stop on any validity failure."""

import argparse
import gzip
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from failure2fix.clean import sha256
from failure2fix.clean_dp import load_dp
from failure2fix.gate_b import (
    compare_branches,
    compare_schedule,
    metrics,
    run_branch,
    validate_impulse_api,
)
from failure2fix.gate_b2 import (
    ContactRecorder,
    canonical,
    chunk_association,
    compare_contact,
    contact_mismatch,
    latency_summary,
    paired_effect,
    retain_failure,
)


def write(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - half), min(1, center + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="artifacts/gate-b2/plan.json")
    parser.add_argument("--output", default="runs/gate-b2/confirmation-v1")
    parser.add_argument("--upstream", default=".cache/clean-ManiSkill")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    seeds = list(range(plan["seeds"]["first"], plan["seeds"]["first"] + plan["seeds"]["count"]))
    if len(seeds) != 50 or any(s < 60000 for s in seeds):
        raise ValueError("Protocol requires 50 unused seeds >=60000")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "records").mkdir()
    (out / "raw").mkdir()
    status = dict(status="RUNNING", checks=[], episodes=[], resolution=[])
    config = dict(
        plan=plan,
        plan_sha256=sha256(args.plan),
        arguments=vars(args),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_diff=subprocess.check_output(["git", "diff", "HEAD"], text=True),
        packages=subprocess.check_output([".venv/bin/pip", "freeze"], text=True),
    )
    write(out / "config.json", config)
    rows = []
    last_attempt, last_reference = None, None
    tags = [("clean", 0.0), ("sham", 0.0), ("J0.256", 0.256), ("J0.064", 0.064), ("J0.128", 0.128)]

    def require(ok, **detail):
        status["checks"].append(dict(pass_=bool(ok), **detail))
        if not ok:
            raise RuntimeError(f"STOP: {detail}")

    def execute(seed, tag, impulse, trigger=None, freq=100, video=None):
        nonlocal last_attempt, last_reference
        last_attempt, last_reference = None, None
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        audit = ContactRecorder()

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
            audit.attach(env.unwrapped)
            return env

        branch = tag if tag in ["clean", "sham"] else "disturbance"
        run = run_branch(factory, agent, seed, branch, impulse, trigger, 200, video)
        contact = audit.evidence(run)
        contact["_substeps"] = audit.rows
        last_attempt = (run, contact)
        require(
            contact["valid"],
            kind="contact-validity",
            seed=seed,
            tag=tag,
            freq=freq,
            residual_Ns=contact["momentum_residual_inf_Ns"],
            min_bottom_m=contact["min_cube_bottom_m"],
            min_separation_m=contact["min_contact_separation_m"],
        )
        return run, contact

    def paired(clean, actual, tag, seed, freq=100):
        nonlocal last_reference
        last_reference = clean
        c, ca = clean
        a, aa = actual
        sham = tag == "sham"
        check = compare_branches(c, a, prefix_only=not sham)
        require(
            check["pass"],
            kind="sham" if sham else "prefix",
            tag=tag,
            seed=seed,
            freq=freq,
            detail=check,
        )
        require(
            compare_contact(ca, aa, prefix=not sham),
            kind="native-contact-equivalence",
            mismatch=contact_mismatch(ca, aa, prefix=not sham),
            tag=tag,
            seed=seed,
            freq=freq,
        )
        require(
            compare_schedule(c, a)["pass"],
            kind="RNG-and-inference-schedule",
            tag=tag,
            seed=seed,
            freq=freq,
        )
        # Extra pre-impulse contact query must also agree.
        before_c = ca["window"][min(2 * round(0.05 / ca["dt_s"]), ca["event_index"])]
        before_a = aa["window"][min(2 * round(0.05 / aa["dt_s"]), aa["event_index"])]
        require(
            canonical(before_c["before_contacts"]) == canonical(before_a["before_contacts"]),
            kind="pre-impulse-contact",
            tag=tag,
            seed=seed,
            freq=freq,
        )
        return check

    def persist(pair, tag, freq=100):
        run, audit = pair
        seed = run["seed"]
        key = f"{freq}-{tag}-{seed}"
        with gzip.open(out / "raw" / f"{key}.pkl.gz", "wb") as f:
            pickle.dump(pair, f, protocol=5)
        compact = []
        for frame in run["frames"]:
            compact.append(
                dict(
                    step=frame.step,
                    action=frame.action,
                    grasped=frame.grasped,
                    success=frame.success,
                    observation=frame.state["obs"],
                    cube=frame.state["env/actors/cube"],
                    queue_remaining=int(len(frame.state["policy/queue"])),
                    state_sha256=hashlib.sha256(canonical(frame.state)).hexdigest(),
                )
            )
        public_audit = {
            k: v for k, v in audit.items() if k not in ["full_fingerprints", "_substeps"]
        }
        public_audit["full_contact_trace_sha256"] = hashlib.sha256(
            canonical(audit["full_fingerprints"])
        ).hexdigest()
        payload = dict(
            event=run["event"],
            inference_steps=run["inference_steps"],
            frames=compact,
            audit=public_audit,
        )
        (out / "records" / f"{key}.json.gz").write_bytes(gzip.compress(canonical(payload), mtime=0))
        row = metrics(run)
        del row["event"]
        row.update(
            tag=tag,
            freq=freq,
            queue_remaining=run["event"]["queue_remaining"],
            early_grasp_loss=any(
                not f.grasped for f in run["frames"][run["trigger"] + 1 : run["trigger"] + 11]
            ),
            audit={k: v for k, v in public_audit.items() if k != "window"},
        )
        if freq == 100:
            rows.append(row)
        else:
            status["resolution"].append(row)
        status["episodes"] = rows
        write(out / "summary.json", status)
        print(
            json.dumps(
                dict(
                    seed=seed,
                    tag=tag,
                    freq=freq,
                    success=row["success_at_200"],
                    residual_Ns=audit["momentum_residual_inf_Ns"],
                    bottom_m=audit["min_cube_bottom_m"],
                )
            ),
            flush=True,
        )
        return row

    def load(seed, tag, freq=100):
        with gzip.open(out / "raw" / f"{freq}-{tag}-{seed}.pkl.gz", "rb") as f:
            return pickle.load(f)

    try:
        require(sha256(plan["checkpoint"]) == plan["checkpoint_sha256"], kind="frozen-checkpoint")
        status["free_body_calibration"] = validate_impulse_api([0.064, 0.128, 0.256])
        agent = load_dp(args.upstream, plan["checkpoint"])
        agent.requires_grad_(False)
        # Controls first, across the entire fixed seed set.
        for seed in seeds:
            clean = execute(seed, "clean", 0.0)
            persist(clean, "clean")
            sham = execute(seed, "sham", 0.0, clean[0]["trigger"])
            paired(clean, sham, "sham", seed)
            persist(sham, "sham")
        for tag, impulse in tags[2:]:
            for seed in seeds:
                clean = load(seed, "clean")
                actual = execute(seed, tag, impulse, clean[0]["trigger"])
                paired(clean, actual, tag, seed)
                # Same-time displacement relative to paired clean, first 0.5 s.
                a, c = actual[1], clean[1]
                start = min(2 * round(0.05 / a["dt_s"]), a["event_index"])
                delta = [
                    np.linalg.norm(x["after"][:3] - y["after"][:3])
                    for x, y in zip(a["window"][start:], c["window"][start:])
                ]
                a["max_paired_displacement_0_5s_m"] = float(max(delta))
                a["first_paired_delta_velocity_mps"] = (
                    a["first_velocity_mps"] - c["first_velocity_mps"]
                )
                persist(actual, tag)
        require(
            any(
                r["audit"]["max_paired_displacement_0_5s_m"] > 1e-6
                for r in rows
                if r["tag"] == "J0.256"
            ),
            kind="measurable-primary-displacement",
        )
        # Resolution audit has its own matched controls; never pooled with the 50 pairs.
        for freq in [200, 400]:
            for seed in seeds[:3]:
                clean = execute(seed, "clean", 0.0, freq=freq)
                persist(clean, "clean", freq)
                for tag, impulse in [tags[1], tags[2]]:
                    actual = execute(seed, tag, impulse, clean[0]["trigger"], freq)
                    paired(clean, actual, tag, seed, freq)
                    persist(actual, tag, freq)
        for seed in seeds[:5]:
            for tag, impulse in tags:
                reference = load(seed, tag)
                repeated = execute(seed, tag, impulse, reference[0]["trigger"])
                last_reference = reference
                check = compare_branches(reference[0], repeated[0])
                require(
                    check["pass"] and compare_contact(reference[1], repeated[1]),
                    kind="repeat",
                    seed=seed,
                    tag=tag,
                    detail=check,
                )
        primary = [r for r in rows if r["tag"] == "J0.256"]
        clean_rows = [r for r in rows if r["tag"] == "clean"]
        reps = [seeds[0]]
        for predicate in [
            lambda r: not r["success_at_200"],
            lambda r: r["success_at_200"],
            lambda r: r["grasp_loss_step"] is not None and r["success_at_200"],
        ]:
            reps.extend([r["seed"] for r in primary if predicate(r)][:1])
        status["video_seeds"] = sorted(set(reps))
        for seed in status["video_seeds"]:
            for tag, impulse in [tags[0], tags[1], tags[2]]:
                reference = load(seed, tag)
                repeated = execute(
                    seed,
                    tag,
                    impulse,
                    reference[0]["trigger"],
                    video=out / "videos" / f"{seed}-{tag}.mp4",
                )
                last_reference = reference
                require(
                    compare_branches(reference[0], repeated[0])["pass"]
                    and compare_contact(reference[1], repeated[1]),
                    kind="video",
                    seed=seed,
                    tag=tag,
                )
        summaries = {}
        baseline = [r["success_at_200"] for r in clean_rows]
        for tag, _ in tags:
            group = [r for r in rows if r["tag"] == tag]
            successes = sum(r["success_at_200"] for r in group)
            losses = sum(r["grasp_loss_step"] is not None for r in group)
            summaries[tag] = dict(
                successes=successes,
                success_wilson95=wilson(successes, 50),
                grasp_losses=losses,
                loss_wilson95=wilson(losses, 50),
                early_grasp_losses=sum(r["early_grasp_loss"] for r in group),
                paired=paired_effect(baseline, [r["success_at_200"] for r in group]),
                regrasp=latency_summary(group, "regrasp"),
                completion=latency_summary(group, "completion"),
                chunk_association=chunk_association(group),
            )
        effect = summaries["J0.256"]["paired"]
        go = (
            effect["conservative95"][1] < 0
            and effect["regressions"] >= 5
            and effect["preserved_successes"] >= 5
        )
        require(
            sha256(plan["checkpoint"]) == plan["checkpoint_sha256"], kind="frozen-checkpoint-final"
        )
        status.update(
            status="COMPLETE",
            validity_pass=True,
            gate_C="GO" if go else "NO-GO",
            branch_summaries=summaries,
        )
    except Exception as exc:
        status.update(
            status="STOP", validity_pass=False, gate_C="NO-GO", error=f"{type(exc).__name__}: {exc}"
        )
        if last_attempt is not None:
            try:
                retain_failure(out / "failure-pair.json.gz", last_reference, last_attempt)
            except Exception as capture_error:
                status["failure_capture_error"] = str(capture_error)
        raise
    finally:
        write(out / "summary.json", status)
        print(json.dumps(dict(status=status["status"], gate_C=status.get("gate_C"))), flush=True)


if __name__ == "__main__":
    main()
