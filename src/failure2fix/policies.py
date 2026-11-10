"""Policy contract and smoke-test policies; not trained manipulation baselines."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ToyOracle:
    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Any) -> np.ndarray:
        state = as_numpy(observation["state"]).reshape(-1)
        if state.shape != (4,):
            raise ValueError("ToyOracle only supports the 'toy' environment")
        return np.clip((state[2:] - state[:2]) / 0.12, -1, 1)


class ZeroPolicy:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Any) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


class RandomPolicy:
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Any) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return self.rng.uniform(lo, hi).astype(np.float32)


def build_policy(policy_name: str, action_space: Any, env_id: str, factory: str | None = None):
    shape = tuple(action_space.shape)
    if policy_name == "toy-oracle":
        if env_id != "toy":
            raise ValueError("toy-oracle is only valid for --env toy")
        return ToyOracle()
    if policy_name == "zero":
        return ZeroPolicy(shape)
    if policy_name == "random":
        return RandomPolicy(as_numpy(action_space.low), as_numpy(action_space.high))
    if policy_name == "custom":
        if not factory or ":" not in factory:
            raise ValueError("Custom policies require --factory module.path:create_policy")
        module_name, fn_name = factory.rsplit(":", 1)
        fn = getattr(importlib.import_module(module_name), fn_name)
        policy = fn(action_space=action_space, env_id=env_id)
        if not callable(getattr(policy, "act", None)):
            raise TypeError("Custom policy must implement act(observation)")
        if not callable(getattr(policy, "reset", None)):
            raise TypeError("Custom policy must implement reset(seed)")
        return policy
    raise ValueError(f"Unknown policy: {policy_name}")
