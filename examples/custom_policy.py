"""ToyReach custom-policy example only."""

import numpy as np


class Controller:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def act(self, observation):
        s = np.asarray(observation["state"]).reshape(-1)
        return np.clip((s[2:4] - s[0:2]) / 0.12, -1, 1)


def create_policy(action_space, env_id):
    if env_id != "toy":
        raise ValueError("This example controller only supports ToyReach")
    return Controller()
