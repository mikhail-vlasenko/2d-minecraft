import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, spaces

from python_wrapper.ffi_elements import NUM_ACTIONS


class PastActionsWrapper(Wrapper):
    """
    Adds the last N actions to the observation.
    Must be placed before ActionSimplificationWrapper to receive original action indices.
    """
    NUM_PAST_ACTIONS = 4

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.past_actions = np.zeros(self.NUM_PAST_ACTIONS, dtype=np.int32)

        # Extend observation space to include past actions
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "past_actions": spaces.Box(
                low=0, high=NUM_ACTIONS - 1,
                shape=(self.NUM_PAST_ACTIONS,), dtype=np.int32
            ),
        })

    def reset(self, **kwargs):
        self.past_actions = np.zeros(self.NUM_PAST_ACTIONS, dtype=np.int32)
        obs, info = self.env.reset(**kwargs)
        obs["past_actions"] = self.past_actions.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Shift past actions and add the new one
        self.past_actions = np.roll(self.past_actions, 1)
        self.past_actions[0] = action

        obs["past_actions"] = self.past_actions.copy()
        return obs, reward, terminated, truncated, info

    def get_action_name(self, action: int) -> str:
        return self.env.get_action_name(action)

    def get_actions_mask(self):
        return self.env.get_actions_mask()

