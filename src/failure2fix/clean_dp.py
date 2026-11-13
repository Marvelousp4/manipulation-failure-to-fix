"""Official Diffusion Policy adapter and fixed-seed CPU-vector evaluation."""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path

import numpy as np

from failure2fix.policies import as_numpy


def official_dp(upstream):
    directory = Path(upstream) / "examples/baselines/diffusion_policy"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("f2f_official_dp", directory / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ActionChunks:
    def __init__(self, agent, device="cuda"):
        self.agent, self.device = agent, device
        self.reset(0)

    def reset(self, seed):
        self.history = deque(maxlen=self.agent.obs_horizon)
        self.actions = deque()

    def __call__(self, observation):
        import torch

        obs = as_numpy(observation).reshape(-1).copy()
        self.history.append(obs)
        while len(self.history) < self.agent.obs_horizon:
            self.history.appendleft(obs)
        if not self.actions:
            batch = torch.as_tensor(
                np.stack(self.history)[None], device=self.device, dtype=torch.float32
            )
            actions = as_numpy(self.agent.get_action(batch))[0]
            self.actions.extend(actions)
        return self.actions.popleft()


def load_dp(upstream, checkpoint, device="cuda"):
    from types import SimpleNamespace

    import gymnasium as gym
    import torch

    torch.backends.cudnn.deterministic = True
    module = official_dp(upstream)
    args = module.Args()
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(2, 42)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(4,)),
    )
    agent = module.Agent(env, args).to(device)
    agent.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)["ema_agent"]
    )
    agent.eval()
    return agent


def vector_evaluate(agent, envs, seeds, horizon=200, device="cuda"):
    """Use official FrameStack/CPUGymWrapper metrics. Fixed batches of ten seeds."""
    import torch

    n = envs.num_envs
    if len(seeds) % n:
        raise ValueError("Episode count must divide the fixed vector batch size")
    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    records = []
    was_training = agent.training
    agent.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for start in range(0, len(seeds), n):
                batch = seeds[start : start + n]
                torch.manual_seed(batch[0])
                obs, _ = envs.reset(seed=batch)
                success100 = None
                for offset in range(0, horizon, agent.act_horizon):
                    action_seq = as_numpy(
                        agent.get_action(torch.as_tensor(obs, device=device, dtype=torch.float32))
                    )
                    for i in range(min(agent.act_horizon, horizon - offset)):
                        obs, _, _, truncated, info = envs.step(action_seq[:, i])
                        step = offset + i + 1
                        if step == 100:
                            episode = info["episode"]
                            success100 = (
                                np.asarray(episode["success_once"]).copy()
                                if isinstance(episode, dict)
                                else np.array([e["success_once"] for e in episode])
                            )
                        if step < horizon and np.any(truncated):
                            raise ValueError("Unexpected early truncation")
                if not np.all(truncated):
                    raise ValueError("Expected fixed-horizon truncation")
                for j, seed in enumerate(batch):
                    ep = info["final_info"][j]["episode"]
                    records.append(
                        dict(
                            seed=seed,
                            success_once=bool(ep["success_once"]),
                            success_at_end=bool(ep["success_at_end"]),
                            success_once_100=bool(success100[j]),
                            steps=horizon,
                            vector_batch_seed=batch[0],
                        )
                    )
    finally:
        agent.train(was_training)
    return records
