"""Observation noise and action delay; not physical dynamics shifts."""

from __future__ import annotations

from typing import Any

import numpy as np

from failure2fix.policies import as_numpy


def noisy_observation(obs: Any, sigma: float, rng: np.random.Generator) -> Any:
    if sigma <= 0:
        return obs
    if isinstance(obs, dict):
        return {key: noisy_observation(val, sigma, rng) for key, val in obs.items()}
    if isinstance(obs, tuple):
        return tuple(noisy_observation(v, sigma, rng) for v in obs)
    if isinstance(obs, list):
        return [noisy_observation(v, sigma, rng) for v in obs]
    x = as_numpy(obs)
    if x.dtype.kind != "f":
        return obs
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(x.dtype)


class ActionDelay:
    def __init__(self, delay: int, action_shape: tuple[int, ...]) -> None:
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.delay = delay
        self.action_shape = action_shape
        self.buffer: list[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        self.buffer = [np.zeros(self.action_shape, dtype=np.float32) for _ in range(self.delay)]

    def apply(self, action: np.ndarray) -> np.ndarray:
        arr = as_numpy(action).astype(np.float32)
        if arr.shape != self.action_shape:
            raise ValueError(f"Expected action shape {self.action_shape}, got {arr.shape}")
        if self.delay == 0:
            return arr
        self.buffer.append(arr.copy())
        return self.buffer.pop(0)
