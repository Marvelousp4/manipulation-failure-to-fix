"""Load the official state BC actor without duplicating its implementation."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from failure2fix.policies import as_numpy


def official_bc_loader(args):
    if args.obs_mode != "state" or not all([args.checkpoint, args.provenance, args.upstream]):
        raise ValueError(
            "official-bc requires state observations, checkpoint, provenance and upstream"
        )
    checkpoint = Path(args.checkpoint)
    provenance = json.loads(Path(args.provenance).read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "-C", args.upstream, "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_diff = subprocess.check_output(["git", "-C", args.upstream, "diff", "HEAD"])
    expected = dict(
        upstream_diff_sha256=hashlib.sha256(upstream_diff).hexdigest(),
        checkpoint_sha256=digest,
        upstream_commit=revision,
        env_id=args.env,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
    )
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Provenance mismatch: {key}")
    if not provenance.get("training_command") or not provenance.get("dataset_sha256"):
        raise ValueError(
            "Missing training/dataset provenance; untrained weights are not a baseline"
        )
    if provenance.get("normalize_states") is not False:
        raise ValueError("Only unnormalized official state BC checkpoints are supported")
    if provenance.get("total_updates", 0) < 1:
        raise ValueError("Checkpoint must record a positive training budget")
    import torch

    upstream = Path(args.upstream) / "examples/baselines/bc"
    import sys

    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("official_maniskill_bc", upstream / "bc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class BCPolicy:
        def __init__(self, env):
            self.action_shape = env.action_space.shape
            state_dim = env.observation_space.shape[-1]
            action_dim = env.action_space.shape[-1]
            self.actor = module.Actor(state_dim, action_dim).to(args.device)
            self.actor.load_state_dict(
                torch.load(checkpoint, map_location=args.device, weights_only=True)["actor"]
            )
            self.actor.eval()

        def reset(self, seed):
            torch.manual_seed(seed)

        def act(self, observation):
            with torch.inference_mode():
                obs = torch.as_tensor(
                    as_numpy(observation), dtype=torch.float32, device=args.device
                )
                return self.actor(obs).cpu().numpy().reshape(self.action_shape)

    return BCPolicy, provenance
