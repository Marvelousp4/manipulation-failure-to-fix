"""Clean PickCube audits; no distribution shifts or test-driven checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from failure2fix.envs import make_env
from failure2fix.policies import as_numpy


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_bc(upstream):
    directory = Path(upstream) / "examples/baselines/bc"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("f2f_official_bc", directory / "bc.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_actor(upstream, checkpoint, device="cuda"):
    import torch

    actor = official_bc(upstream).Actor(42, 4).to(device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["actor"])
    actor.eval()
    return actor


def offline_mse(actor, path, device="cuda"):
    import h5py
    import torch

    errors, shifted_errors = [], []
    with h5py.File(path) as data, torch.inference_mode():
        for name in sorted(data):
            obs = data[name]["obs"][:]
            actions = data[name]["actions"][:]
            if len(obs) != len(actions) + 1:
                raise ValueError("Expected T+1 observations and T actions")
            prediction = actor(torch.as_tensor(obs[:-1], device=device)).cpu().numpy()
            errors.append((prediction - actions) ** 2)
            shifted = actor(torch.as_tensor(obs[1:], device=device)).cpu().numpy()
            shifted_errors.append((shifted - actions) ** 2)
    error = np.concatenate(errors)
    return dict(
        frames=len(error),
        mse=float(error.mean()),
        per_action_mse=error.mean(0).tolist(),
        deliberately_misaligned_next_obs_mse=float(np.concatenate(shifted_errors).mean()),
    )


def phase_label(grasped, reached_goal, success, approached):
    if success:
        return "success"
    if not approached:
        return "control/reach"
    if not grasped:
        return "grasp"
    if not reached_goal:
        return "transport"
    return "placement/settling"


def rollout(sample_fn, seeds, horizon=200, video_dir=None, video_seeds=()):
    """Fixed horizon, ignoring success termination as in official IL evaluation."""
    import torch

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        env = make_env(
            "PickCube-v1",
            control_mode="pd_ee_delta_pos",
            sim_backend="cpu",
            render_backend="cpu" if video_dir else "none",
            render_mode="rgb_array" if video_dir else None,
            max_episode_steps=horizon,
        )
        records = []
        try:
            for seed in seeds:
                torch.manual_seed(seed)
                if hasattr(sample_fn, "reset"):
                    sample_fn.reset(seed)
                obs, _ = env.reset(seed=seed)
                initial = as_numpy(obs).copy()
                cube_z = float(as_numpy(env.unwrapped.cube.pose.p)[0, 2])
                writer = None
                if video_dir and seed in video_seeds:
                    import imageio.v2 as imageio

                    Path(video_dir).mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(Path(video_dir) / f"{seed}.mp4", fps=20)
                trace, success, grasped, reached = [], False, False, False
                min_reach, lifted, clipped = float("inf"), False, 0
                try:
                    with torch.inference_mode():
                        for step in range(1, horizon + 1):
                            action = as_numpy(sample_fn(as_numpy(obs))).reshape(
                                env.action_space.shape
                            )
                            if not np.isfinite(action).all():
                                raise ValueError("Nonfinite policy action")
                            clipped += int(np.any(np.abs(action) > 1))
                            obs, _, _, truncated, info = env.step(action.astype(np.float32))
                            p = as_numpy(env.unwrapped.cube.pose.p)[0]
                            tcp = as_numpy(env.unwrapped.agent.tcp_pose.p)[0]
                            goal = as_numpy(env.unwrapped.goal_site.pose.p)[0]
                            distance = float(np.linalg.norm(p - goal))
                            reach = float(np.linalg.norm(p - tcp))
                            ok, grip = (
                                bool(as_numpy(info["success"]).item()),
                                bool(as_numpy(info["is_grasped"]).item()),
                            )
                            success, grasped = success or ok, grasped or grip
                            reached = reached or bool(as_numpy(info["is_obj_placed"]).item())
                            min_reach = min(min_reach, reach)
                            lifted = lifted or p[2] > cube_z + 0.04
                            trace.append(
                                dict(
                                    step=step,
                                    success=ok,
                                    grasped=grip,
                                    goal_distance=distance,
                                    tcp_distance=reach,
                                    cube_z=float(p[2]),
                                    robot_static=bool(as_numpy(info["is_robot_static"]).item()),
                                )
                            )
                            if writer:
                                frame = as_numpy(env.render())
                                writer.append_data(frame[0] if frame.ndim == 4 else frame)
                            if bool(as_numpy(truncated).item()) and step != horizon:
                                raise ValueError("Unexpected early time limit")
                finally:
                    if writer:
                        writer.close()
                at100 = trace[min(99, len(trace) - 1)]
                records.append(
                    dict(
                        seed=seed,
                        success_once=success,
                        success_at_end=trace[-1]["success"],
                        success_once_100=any(t["success"] for t in trace[:100]),
                        success_at_100=at100["success"],
                        grasped_once=grasped,
                        lifted=bool(lifted),
                        reached_goal_once=reached,
                        min_tcp_distance=min_reach,
                        clipped_action_steps=clipped,
                        steps=horizon,
                        initial_observation_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                        failure_phase=phase_label(grasped, reached, success, min_reach < 0.05),
                        trace=trace,
                    )
                )
        finally:
            env.close()
        return records


def rollup(records):
    n = len(records)
    return dict(
        episodes=n,
        success_once=sum(r["success_once"] for r in records) / n,
        success_at_end=sum(r["success_at_end"] for r in records) / n,
        success_once_100=sum(r["success_once_100"] for r in records) / n,
        phases=dict(Counter(r["failure_phase"] for r in records)),
    )


def replay_audit(path, count=50):
    import h5py
    from mani_skill.trajectory.utils import dict_to_list_of_dicts

    meta = json.loads(Path(path).with_suffix(".json").read_text())
    if meta["env_info"]["env_kwargs"]["control_mode"] != "pd_ee_delta_pos":
        raise ValueError("Unexpected demonstration controller")
    env = make_env(
        "PickCube-v1", control_mode="pd_ee_delta_pos", render_backend="none", max_episode_steps=200
    )
    records, lengths = [], []
    try:
        with h5py.File(path) as data:
            lengths = [len(data[k]["actions"]) for k in data]
            initial_grasp_flags = sum(bool(data[k]["obs"][0, 18]) for k in data)
            selected = np.linspace(0, len(meta["episodes"]) - 1, count, dtype=int)
            for index in selected:
                episode = meta["episodes"][index]
                traj = data[f"traj_{episode['episode_id']}"]
                actions, recorded = traj["actions"][:], traj["obs"][:]
                assert recorded.shape == (len(actions) + 1, 42) and actions.shape[1] == 4
                assert np.isfinite(recorded).all() and np.isfinite(actions).all()
                assert np.abs(actions).max() <= 1.000001
                obs, _ = env.reset(seed=episode["episode_seed"])
                states = dict_to_list_of_dicts(traj["env_states"])
                env.unwrapped.set_state_dict(states[0])
                obs = env.unwrapped.get_obs()
                initial_delta = np.abs(as_numpy(obs).reshape(42) - recorded[0])
                differences, success = [], False
                for t, action in enumerate(actions):
                    delta = np.abs(as_numpy(obs).reshape(42) - recorded[t])
                    differences.append(delta)
                    obs, _, _, _, info = env.step(action)
                    success = success or bool(as_numpy(info["success"]).item())
                delta = np.array(differences)
                records.append(
                    dict(
                        episode_id=episode["episode_id"],
                        seed=episode["episode_seed"],
                        success=success,
                        final_success=bool(as_numpy(info["success"]).item()),
                        steps=len(actions),
                        initial_mismatched_indices=np.flatnonzero(initial_delta > 1e-5).tolist(),
                        initial_max_error=float(initial_delta.max()),
                        max_error_after_initial=float(delta[1:].max()),
                        aligned_obs_mse=float(np.mean(delta[1:] ** 2)),
                    )
                )
    finally:
        env.close()
    return dict(
        initial_grasp_flags=initial_grasp_flags,
        controller="pd_ee_delta_pos",
        backend="cpu",
        action_alignment="obs[t] -> action[t]",
        dataset_sha256=sha256(path),
        lengths=dict(
            mean=float(np.mean(lengths)),
            percentiles=np.percentile(lengths, [0, 50, 95, 100]).tolist(),
            over_100=sum(x > 100 for x in lengths),
        ),
        records=records,
        replay_success=sum(r["success"] for r in records) / len(records),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "evaluate", "mse"])
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", default="artifacts/bc-pickcube.pt")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--videos", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", vars(args))
    if args.command == "audit":
        write_json(out / "replay.json", replay_audit(args.dataset, args.episodes))
    else:
        import torch

        actor = load_actor(args.upstream, args.checkpoint)
        write_json(out / "offline.json", offline_mse(actor, args.dataset))
        if args.command == "evaluate":

            def sample(obs):
                return actor(torch.as_tensor(obs, device="cuda", dtype=torch.float32)).cpu().numpy()

            seeds = list(range(args.seed, args.seed + args.episodes))
            records = rollout(
                sample,
                seeds,
                args.horizon,
                out / "videos" if args.videos else None,
                seeds[: args.videos],
            )
            write_json(out / "episodes.json", records)
            write_json(out / "summary.json", rollup(records))
    print(out, flush=True)


if __name__ == "__main__":
    main()
