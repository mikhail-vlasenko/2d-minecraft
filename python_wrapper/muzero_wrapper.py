from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, ObservationWrapper

from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper


class MuZeroWrapper(ObservationWrapper):
    """
    Additionally adapts the environment to the MuZero algorithm.
    """
    def __init__(self, env: Union[Minecraft2dEnv, ActionSimplificationWrapper]):
        Wrapper.__init__(self, env)
        self.env: Minecraft2dEnv = env
        sample_obs = self.observation(self.sample_observation())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )

    def observation(self, obs):
        # this shape is required by the MuZero model
        return obs.reshape(1, 1, -1)

    def action_to_string(self, action_number):
        return self.env.get_action_name(action_number)

    def to_play(self):
        """
        Returns:
            The current player
        """
        return 0

    def legal_actions(self):
        """
        Returns the legal actions at each turn

        Returns:
            An array of integers, subset of the action space.
        """
        mask = self.env.get_actions_mask()
        return np.where(mask)[0].tolist()
