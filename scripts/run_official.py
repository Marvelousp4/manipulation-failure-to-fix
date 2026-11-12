"""Run unchanged official scripts with an explicit renderer override for WSL."""

import argparse
import runpy
import sys

import gymnasium as gym

p = argparse.ArgumentParser()
p.add_argument("--render-backend", default="none")
p.add_argument("--module")
p.add_argument("--script")
a, remainder = p.parse_known_args()
make = gym.make


def configured_make(*args, **kwargs):
    kwargs["render_backend"] = a.render_backend
    return make(*args, **kwargs)


gym.make = configured_make
sys.argv = [a.script or a.module, *remainder]
if a.script:
    from pathlib import Path

    sys.path.insert(0, str(Path(a.script).resolve().parent))
    runpy.run_path(a.script, run_name="__main__")
else:
    runpy.run_module(a.module, run_name="__main__")
