"""Validate deterministic mid-contact replay before counterfactual experiments."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from failure2fix.policies import as_numpy


def _leaves(value: Any, prefix: str) -> dict[str, np.ndarray]:
    """Flatten numeric states; reject unknown fields instead of hiding differences."""
    if value is None:
        return {f"{prefix}/__none__": np.array([0], dtype=np.int8)}
    if isinstance(value, Mapping):
        out: dict[str, np.ndarray] = {}
        for key in sorted(value, key=str):
            out.update(_leaves(value[key], f"{prefix}/{key}"))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for i, item in enumerate(value):
            out.update(_leaves(item, f"{prefix}/{i}"))
        return out
    if hasattr(value, "p") and hasattr(value, "q"):
        return {**_leaves(value.p, f"{prefix}/p"), **_leaves(value.q, f"{prefix}/q")}
    arr = np.asarray(as_numpy(value))
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"Unsupported state leaf {prefix}: {arr.dtype}")
    return {prefix: arr.copy()}


def _flag(info: dict, key: str) -> bool:
    arr = np.asarray(as_numpy(info[key]))
    if arr.size != 1:
        raise ValueError(f"Expected single-env {key}, got {arr.shape}")
    return bool(arr.item())


@dataclass
class Frame:
    step: int
    action: np.ndarray | None
    state: dict[str, np.ndarray]
    grasped: bool
    success: bool


def _frame(env: Any, obs: Any, action: np.ndarray | None, step: int, info: dict) -> Frame:
    raw = env.unwrapped
    state = _leaves(obs, "obs")
    state.update(_leaves(raw.get_state_dict(), "env"))
    state.update(_leaves(raw.agent.get_controller_state(), "controller"))
    if hasattr(raw, "elapsed_steps"):
        state.update(_leaves(raw.elapsed_steps, "elapsed_steps"))
    return Frame(
        step, None if action is None else action.copy(), state,
        _flag(info, "is_grasped") if "is_grasped" in info else False,
        _flag(info, "success") if "success" in info else False,
    )


def run_trace(
    env_factory: Callable[[], Any],
    policy_factory: Callable[[], Any],
    seed: int,
    horizon: int,
) -> list[Frame]:
    """Recreate the policy's actions and the physical rollout from a fresh reset."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_factory()
    try:
        policy = policy_factory()
        policy.reset(seed)
        obs, _ = env.reset(seed=seed)
        records = [_frame(env, obs, None, 0, {})]
        with torch.inference_mode():
            for step in range(1, horizon + 1):
                action = np.asarray(as_numpy(policy.act(obs)), dtype=np.float32)
                if action.size != int(np.prod(env.action_space.shape)) or not np.isfinite(action).all():
                    raise ValueError(f"Invalid action at step {step}: {action.shape}")
                action = action.reshape(env.action_space.shape)
                obs, _, _, truncated, info = env.step(action)
                records.append(_frame(env, obs, action, step, info))
                if step < horizon and bool(np.asarray(as_numpy(truncated)).item()):
                    raise RuntimeError(f"Early truncation at step {step}")
        return records
    finally:
        env.close()


def _branch_step(trace: list[Frame], min_suffix: int = 8, consecutive: int = 3) -> int:
    for step in range(consecutive, len(trace) - min_suffix):
        if all(trace[k].grasped for k in range(step - consecutive + 1, step + 1)):
            return step
    raise RuntimeError("No sustained grasp phase with sufficient remaining horizon")


def compare_traces(reference: list[Frame], replay: list[Frame], branch_step: int,
                   atol: float = 1e-5) -> dict:
    if len(reference) != len(replay) or branch_step >= len(reference) - 1:
        raise ValueError("Invalid replay length or branch")
    worst = {"prefix": {"max_error": 0.0}, "suffix": {"max_error": 0.0}}
    for original, repeated in zip(reference, replay):
        phase = "prefix" if original.step <= branch_step else "suffix"
        bucket = worst[phase]
        if (original.step != repeated.step or original.grasped != repeated.grasped
                or original.success != repeated.success):
            return {"pass": False, "branch_step": branch_step,
                    "first_mismatch": {"step": original.step, "field": "phase_or_success"},
                    **worst}
        left, right = dict(original.state), dict(repeated.state)
        if (original.action is None) != (repeated.action is None):
            return {"pass": False, "branch_step": branch_step,
                    "first_mismatch": {"step": original.step, "field": "action_present"},
                    **worst}
        if original.action is not None:
            left["action"], right["action"] = original.action, repeated.action
        if left.keys() != right.keys():
            return {"pass": False, "branch_step": branch_step,
                    "first_mismatch": {"step": original.step, "field": "state_keys"},
                    **worst}
        for name in sorted(left):
            a, b = left[name], right[name]
            if a.shape != b.shape:
                return {"pass": False, "branch_step": branch_step,
                        "first_mismatch": {"step": original.step, "field": name}, **worst}
            error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
            bucket["max_error"] = max(bucket["max_error"], error)
            if not np.allclose(a, b, rtol=atol, atol=atol, equal_nan=False):
                return {"pass": False, "branch_step": branch_step,
                        "first_mismatch": {"step": original.step, "field": name,
                                           "max_error": error}, **worst}
    return {"pass": True, "branch_step": branch_step,
            "first_mismatch": None, **worst}


def replay_gate(
    env_factory: Callable[[], Any],
    policy_factory: Callable[[], Any],
    seeds: list[int],
    horizon: int = 120,
    atol: float = 1e-5,
) -> dict:
    """Fail closed on state/action/contact divergence or missing contact phase."""
    if not seeds or len(set(seeds)) != len(seeds) or horizon < 16 or atol <= 0:
        raise ValueError("Invalid seeds, horizon or tolerance")
    cases = []
    for seed in seeds:
        reference = run_trace(env_factory, policy_factory, seed, horizon)
        branch = _branch_step(reference)
        replay = run_trace(env_factory, policy_factory, seed, horizon)
        comparison = compare_traces(reference, replay, branch, atol)
        cases.append({
            "seed": seed, **comparison,
            "reference_success_once": any(frame.success for frame in reference),
            "reference_success_at_end": reference[-1].success,
        })
    return {
        "pass": all(case["pass"] for case in cases),
        "seeds": seeds, "horizon": horizon, "atol": atol,
        "cases": cases,
        "method": "independent reset plus deterministic policy/action prefix replay",
        "limitation": "Does not verify set_state_dict restore or cross-machine reproducibility",
    }
