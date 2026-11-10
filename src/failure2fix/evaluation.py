"""Paired-seed evaluation with explicit unknown-success handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from failure2fix.perturbations import ActionDelay, noisy_observation
from failure2fix.policies import as_numpy, build_policy


@dataclass(frozen=True)
class Condition:
    name: str
    obs_noise: float = 0.0
    action_delay: int = 0


@dataclass
class Episode:
    env_id: str
    policy: str
    condition: str
    seed: int
    success: bool | None
    steps: int
    total_reward: float
    end_reason: str


def scalar_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        arr = as_numpy(value)
        if arr.size != 1:
            return None
        return bool(arr.item())
    except (TypeError, ValueError, RuntimeError):
        return None


def scalar_float(value: Any) -> float:
    arr = as_numpy(value)
    if arr.size != 1:
        raise ValueError("Expected scalar reward; use a single-env simulator configuration")
    return float(arr.item())


def evaluate_condition(
    env_factory: Callable[[], Any],
    env_id: str,
    policy_name: str,
    condition: Condition,
    seeds: list[int],
    max_steps: int,
    factory: str | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> list[Episode]:
    env = env_factory()
    try:
        action_shape = tuple(env.action_space.shape)
        policy = build_policy(policy_name, env.action_space, env_id, factory=factory)
        delay = ActionDelay(condition.action_delay, action_shape)
        results = []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            policy.reset(seed)
            delay.reset()
            rng = np.random.default_rng(seed + 10901)
            cumulative_reward = 0.0
            success: bool | None = None
            end_reason = "evaluation_timeout"
            for step in range(1, max_steps + 1):
                observed = noisy_observation(obs, condition.obs_noise, rng)
                proposed_action = as_numpy(policy.act(observed)).astype(np.float32)
                executed_action = delay.apply(proposed_action)
                obs, reward, terminated, truncated, info = env.step(executed_action)
                reward_val = scalar_float(reward)
                cumulative_reward += reward_val
                reported = scalar_bool(info.get("success")) if isinstance(info, dict) else None
                if reported is not None:
                    success = reported
                terminal = scalar_bool(terminated)
                trunc = scalar_bool(truncated)
                if terminal is None or trunc is None:
                    raise ValueError("Only non-vectorized single environments are supported")
                if on_step is not None:
                    on_step({
                        "condition": condition.name,
                        "seed": seed,
                        "step": step,
                        "reward": reward_val,
                        "success_reported": reported,
                        "proposed_action_norm": float(np.linalg.norm(proposed_action)),
                        "executed_action_norm": float(np.linalg.norm(executed_action)),
                        "distance_to_goal": (
                            scalar_float(info["distance_to_goal"])
                            if isinstance(info, dict) and "distance_to_goal" in info
                            else None
                        ),
                    })
                if terminal or trunc:
                    end_reason = "terminated" if terminal else "truncated"
                    break
            else:
                step = max_steps
            results.append(Episode(env_id, policy_name, condition.name, seed,
                                   success, step, cumulative_reward, end_reason))
        return results
    finally:
        env.close()


def summarize(results: list[Episode]) -> list[dict]:
    names = list(dict.fromkeys(r.condition for r in results))
    summary = []
    for name in names:
        group = [r for r in results if r.condition == name]
        known = [r for r in group if r.success is not None]
        summary.append({
            "condition": name,
            "episodes": len(group),
            "success_known": len(known),
            "success_unknown": len(group) - len(known),
            "success_rate": (sum(r.success is True for r in known) / len(known)) if known else None,
            "mean_steps": float(np.mean([r.steps for r in group])) if group else None,
            "mean_return": float(np.mean([r.total_reward for r in group])) if group else None,
        })
    return summary


def to_dict(result: Episode) -> dict:
    return asdict(result)
