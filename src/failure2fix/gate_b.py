"""Paired post-grasp impulse feasibility with an unchanged chunked policy."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np

from failure2fix.clean_dp import ActionChunks
from failure2fix.policies import as_numpy
from failure2fix.replay_gate import _branch_step, _frame, _leaves, compare_traces


def rng_state():
    import torch

    values = {
        "python": pickle.dumps(random.getstate()),
        "numpy": pickle.dumps(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().numpy().tobytes(),
    }
    for i, state in enumerate(torch.cuda.get_rng_state_all()):
        values[f"torch_cuda_{i}"] = state.cpu().numpy().tobytes()
    return {
        k: np.frombuffer(hashlib.sha256(v).digest(), dtype=np.uint8).copy()
        for k, v in values.items()
    }


class TrackedChunks:
    def __init__(self, agent, device="cuda"):
        self.chunks = ActionChunks(agent, device=device)
        self.calls = 0
        self.inferences = []
        self.history_frames = deque(maxlen=agent.obs_horizon)

    def act(self, obs):
        self.calls += 1
        self.history_frames.append(self.calls - 1)
        while len(self.history_frames) < self.chunks.agent.obs_horizon:
            self.history_frames.appendleft(self.calls - 1)
        if not self.chunks.actions:
            self.inferences.append(self.calls)
        return self.chunks(obs)

    def state(self):
        history = (
            np.stack(self.chunks.history) if self.chunks.history else np.empty((0, 42), np.float32)
        )
        queue = (
            np.stack(self.chunks.actions) if self.chunks.actions else np.empty((0, 4), np.float32)
        )
        return {
            "history": history.copy(),
            "queue": queue.copy(),
            "calls": np.array(self.calls),
            "inferences": np.array(len(self.inferences)),
            "history_frames": np.array(self.history_frames),
            "queue_source_inference": np.array(self.inferences[-1] if self.inferences else 0),
        }


def state_frame(env, obs, action, step, info, policy):
    frame = _frame(env, obs, action, step, info)
    frame.state.update(_leaves(policy.state(), "policy"))
    frame.state.update(_leaves(rng_state(), "rng"))
    if not all(np.isfinite(v).all() for v in frame.state.values()):
        raise RuntimeError("Nonfinite simulator or policy state")
    return frame


def cube_state(body):
    pose = body.entity_pose
    return np.concatenate([pose.p, pose.q, body.linear_velocity, body.angular_velocity]).copy()


def physical_impulse(body, impulse):
    """Buffered native momentum impulse at the world COM; no state setters."""
    vector = np.array([impulse, 0, 0], dtype=np.float32)
    before = cube_state(body)
    point = (body.entity_pose * body.cmass_local_pose).p.copy()
    body.add_force_at_point(vector, point, mode="impulse")
    after = cube_state(body)
    if not np.array_equal(before, after):
        raise RuntimeError("Unexpected immediate state mutation by buffered impulse API")
    return dict(
        impulse_Ns=vector.tolist(),
        point_world=point.tolist(),
        before=before.tolist(),
        after_api=after.tolist(),
        mass_kg=float(body.mass),
    )


def validate_impulse_api(impulses):
    """Free cube: horizontal momentum, zero COM torque and one-shot application."""
    import sapien

    results = []
    for impulse in [0.0, *impulses]:
        px = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([px])
        scene.set_timestep(0.01)
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02] * 3, density=1000)
        entity = builder.build(name="calibration-cube")
        body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        body.linear_damping = 0.0
        body.angular_damping = 0.0
        event = physical_impulse(body, impulse)
        scene.step()
        v1 = body.linear_velocity.copy()
        w1 = body.angular_velocity.copy()
        scene.step()
        v2 = body.linear_velocity.copy()
        expected = impulse / body.mass
        passed = bool(
            np.allclose(v1, [expected, 0, -0.0981], atol=1e-6, rtol=1e-6)
            and np.allclose(v2, [expected, 0, -0.1962], atol=1e-6, rtol=1e-6)
            and np.allclose(w1, 0, atol=1e-7)
        )
        results.append(
            dict(
                **event,
                velocity_after_one_step=v1.tolist(),
                velocity_after_two_steps=v2.tolist(),
                angular_velocity=w1.tolist(),
                expected_free_delta_v_mps=expected,
                **{"pass": passed},
            )
        )
        if not passed:
            raise RuntimeError(f"Impulse API calibration failed: {results[-1]}")
    return results


def run_branch(
    env_factory,
    agent,
    seed,
    branch,
    impulse=0.0,
    trigger=None,
    horizon=200,
    video=None,
    capture=None,
):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_factory()
    writer = None
    try:
        if env.unwrapped.gpu_sim_enabled:
            raise RuntimeError("Gate B adapter is validated only for CPU PhysX")
        policy = TrackedChunks(agent)
        obs, _ = env.reset(seed=seed)
        body = env.unwrapped.cube._bodies[0]
        event = None
        current_step = 0
        substep = 0
        old_before = env.unwrapped._before_simulation_step
        old_after = env.unwrapped._after_simulation_step
        records = [state_frame(env, obs, None, 0, {}, policy)]
        grasp_run = 0
        if capture is not None:
            capture.update(frames=records, inference_steps=policy.inferences)

        def before():
            nonlocal event
            old_before()
            if trigger is not None and current_step == trigger + 1 and substep == 0:
                snapshot = state_frame(env, obs, action, current_step, {}, policy)
                preserved = {
                    k: v.copy()
                    for k, v in snapshot.state.items()
                    if k.startswith(("policy/", "rng/"))
                }
                event = dict(
                    trigger_frame=trigger,
                    control_step=current_step,
                    time_s=trigger * env.unwrapped.control_timestep,
                    pre_state=snapshot.state,
                    selected_action=action.copy(),
                    queue_remaining=len(policy.chunks.actions),
                    simulation_dt_s=env.unwrapped.sim_timestep,
                    control_dt_s=env.unwrapped.control_timestep,
                    physical_api_called=branch != "clean",
                )
                if capture is not None:
                    capture["event"] = event
                if branch != "clean":
                    event["physics"] = physical_impulse(body, 0.0 if branch == "sham" else impulse)
                else:
                    event["physics"] = dict(
                        before=cube_state(body).tolist(), mass_kg=float(body.mass)
                    )
                after_state = {**_leaves(policy.state(), "policy"), **_leaves(rng_state(), "rng")}
                event["policy_rng_preserved"] = all(
                    np.array_equal(v, after_state[k]) for k, v in preserved.items()
                )
                if not event["policy_rng_preserved"]:
                    raise RuntimeError("Intervention modified policy queue/history/RNG")

        def after():
            nonlocal substep
            old_after()
            if event is not None and current_step == trigger + 1 and substep == 0:
                event["cube_after_first_physics_step"] = cube_state(body).tolist()
            substep += 1

        env.unwrapped._before_simulation_step = before
        env.unwrapped._after_simulation_step = after
        if video:
            import imageio.v2 as imageio

            Path(video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video, fps=env.unwrapped.control_freq)
        with torch.inference_mode():
            for current_step in range(1, horizon + 1):
                action = (
                    as_numpy(policy.act(obs)).astype(np.float32).reshape(env.action_space.shape)
                )
                if not np.isfinite(action).all():
                    raise RuntimeError("Nonfinite policy action")
                substep = 0
                if capture is not None:
                    capture.update(
                        control_step=current_step,
                        selected_action=action.copy(),
                        policy_state=policy.state(),
                        rng_state=rng_state(),
                    )
                obs, _, _, truncated, info = env.step(action)
                frame = state_frame(env, obs, action, current_step, info, policy)
                records.append(frame)
                grasp_run = grasp_run + 1 if frame.grasped else 0
                if branch == "clean" and trigger is None and grasp_run >= 3:
                    trigger = current_step
                    if capture is not None:
                        capture["trigger"] = trigger
                    if horizon - trigger < 8:
                        raise RuntimeError("Insufficient continuation after sustained grasp")
                if writer:
                    image = as_numpy(env.render())
                    writer.append_data(image[0] if image.ndim == 4 else image)
                if bool(as_numpy(truncated).item()) and current_step != horizon:
                    raise RuntimeError("Unexpected early truncation")
        if trigger is None or event is None:
            raise RuntimeError("No sustained grasp with an intervention point")
        if trigger != _branch_step(records):
            raise RuntimeError("Branch trigger differs from the clean grasp rule")
        event["first_updated_observation_policy_call"] = trigger + 2
        event["event_command_used_new_inference"] = (trigger + 1) in policy.inferences
        event["first_post_event_inference"] = next(
            (s for s in policy.inferences if s > trigger + 1), None
        )
        event["first_inference_with_updated_observation"] = next(
            (s for s in policy.inferences if s >= trigger + 2), None
        )
        return dict(
            seed=seed,
            branch=branch,
            impulse_Ns=impulse,
            trigger=trigger,
            event=event,
            inference_steps=policy.inferences,
            frames=records,
        )
    finally:
        if writer:
            writer.close()
        env.close()


def compare_branches(reference, repeated, prefix_only=False, atol=1e-5):
    left, right = reference["frames"], repeated["frames"]
    branch = reference["trigger"]
    if prefix_only:
        left, right = left[: branch + 1], right[: branch + 1]
    result = compare_traces(left, right, min(branch, len(left) - 2), atol)
    # Policy inputs, queues and RNG must be identical, even if physics uses a tolerance.
    for a, b in zip(left, right):
        for key in a.state:
            if key.startswith(("policy/", "rng/")) and not np.array_equal(
                a.state[key], b.state[key]
            ):
                return {"pass": False, "first_mismatch": {"step": a.step, "field": key}}
    for key in reference["event"]["pre_state"]:
        a = reference["event"]["pre_state"][key]
        b = repeated["event"]["pre_state"][key]
        exact = key.startswith(("policy/", "rng/"))
        if not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=atol, atol=atol)):
            return {
                "pass": False,
                "first_mismatch": {"step": branch + 1, "field": "pre_impulse/" + key},
            }
    if not np.array_equal(
        reference["event"]["selected_action"], repeated["event"]["selected_action"]
    ):
        return {
            "pass": False,
            "first_mismatch": {"step": branch + 1, "field": "pre_impulse/action"},
        }
    return result


def consecutive_start(frames, key, count, start):
    for i in range(start, len(frames) - count + 1):
        if all(getattr(f, key) for f in frames[i : i + count]):
            return i
    return None


def metrics(run):
    frames = run["frames"]
    event = run["trigger"]
    n = len(frames) - 1
    loss = next((i for i in range(event + 1, n + 1) if not frames[i].grasped), None)
    regrasp = consecutive_start(frames, "grasped", 3, loss + 1) if loss is not None else None
    goal = consecutive_start(frames, "success", 5, event + 1)
    return dict(
        seed=run["seed"],
        branch=run["branch"],
        impulse_Ns=run["impulse_Ns"],
        trigger=event,
        success_once=any(f.success for f in frames),
        success_at_200=frames[-1].success,
        grasp_loss_step=loss,
        regrasp_step=regrasp,
        regrasp_latency_steps=None if regrasp is None else regrasp - loss,
        regrasp_right_censored=loss is not None and regrasp is None,
        stable_goal_step=goal,
        post_event_goal_latency_steps=None if goal is None else goal - event,
        goal_right_censored=goal is None,
        success_before_event=frames[event].success,
        first_updated_observation_policy_call=run["event"]["first_updated_observation_policy_call"],
        first_inference_with_updated_observation=run["event"][
            "first_inference_with_updated_observation"
        ],
        event=run["event"],
    )


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_run(path, run):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame in run["frames"]:
        state = {
            k: dict(dtype=str(v.dtype), shape=list(v.shape), data=v.tolist())
            for k, v in frame.state.items()
        }
        lines.append(
            json.dumps(
                dict(
                    step=frame.step,
                    action=frame.action,
                    state=state,
                    grasped=frame.grasped,
                    success=frame.success,
                ),
                default=json_default,
                allow_nan=False,
            )
        )
    with path.open("xb") as out:
        out.write(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))


def compare_schedule(reference, actual):
    if reference["inference_steps"] != actual["inference_steps"]:
        return {"pass": False, "reason": "inference schedule changed"}
    first_feedback = actual["event"]["first_inference_with_updated_observation"]
    first_difference = None
    for a, b in zip(reference["frames"], actual["frames"]):
        for name in a.state:
            if name.startswith("rng/") and not np.array_equal(a.state[name], b.state[name]):
                return {"pass": False, "reason": f"common RNG changed at {a.step}: {name}"}
        if a.action is not None and not np.array_equal(a.action, b.action):
            if first_difference is None:
                first_difference = a.step
            if first_feedback is None or a.step < first_feedback:
                return {
                    "pass": False,
                    "reason": "queued action changed before new-observation inference",
                }
    return {
        "pass": True,
        "first_action_difference_step": first_difference,
        "first_inference_with_updated_observation": first_feedback,
    }
