"""ToyReach smoke test plus optional ManiSkill environment adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoxLike:
    low: np.ndarray
    high: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape


class ToyReachEnv:
    def __init__(self, max_steps: int = 32) -> None:
        self.action_space = BoxLike(-np.ones(2), np.ones(2))
        self.max_steps = max_steps
        self.position = np.zeros(2)
        self.target = np.zeros(2)
        self.steps = 0

    def _obs(self) -> dict[str, np.ndarray]:
        return {"state": np.concatenate([self.position, self.target]).astype(np.float32)}

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        rng = np.random.default_rng(seed)
        self.position = rng.uniform(-0.15, 0.15, size=2)
        self.target = rng.uniform(-0.7, 0.7, size=2)
        self.steps = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=float)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        self.position += 0.12 * np.clip(action, -1.0, 1.0)
        self.steps += 1
        distance = float(np.linalg.norm(self.target - self.position))
        success = distance <= 0.075
        truncated = self.steps >= self.max_steps and not success
        return self._obs(), -distance, success, truncated, {
            "success": success,
            "distance_to_goal": distance,
        }

    def close(self) -> None:
        pass


def make_env(env_id: str, obs_mode: str = "state", control_mode: str = "pd_joint_delta_pos"):
    if env_id == "toy":
        return ToyReachEnv()
    try:
        import mani_skill.envs  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "ManiSkill is optional: pip install -e '.[sim]'. Check platform/GPU requirements."
        ) from exc
    try:
        return gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize {env_id!r}; check simulator, observation and control modes."
        ) from exc
