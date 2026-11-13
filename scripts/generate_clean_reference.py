"""Official motion planner, fresh held-out seeds, then official controller conversion."""

import argparse

import gymnasium as gym
from mani_skill.examples.motionplanning.panda.run import _main

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=11000)
p.add_argument("--episodes", type=int, default=50)
p.add_argument("--output", required=True)
a = p.parse_args()
make = gym.make


def configured(*args, **kwargs):
    kwargs.update(render_backend="none", reconfiguration_freq=1, max_episode_steps=200)
    return make(*args, **kwargs)


gym.make = configured
args = argparse.Namespace(
    env_id="PickCube-v1",
    obs_mode="state",
    num_traj=a.episodes,
    only_count_success=False,
    reward_mode=None,
    sim_backend="cpu",
    render_mode=None,
    vis=False,
    save_video=False,
    traj_name="trajectory",
    shader="default",
    record_dir=a.output,
    num_procs=1,
)
_main(args, start_seed=a.seed)
