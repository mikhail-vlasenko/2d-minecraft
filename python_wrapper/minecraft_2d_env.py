import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from python_wrapper.ffi_elements import (
    init_lib, reset_one, step_one, num_actions, set_batch_size,
    set_record_replays, connect_env, close_one, c_lib, get_batch_size
)
from python_wrapper.observation import get_processed_observation, NUM_MATERIALS, get_actions_mask


def initialize_minecraft_connection(num_envs=1, lib_path='./target/release/ffi.dll', record_replays=False):
    init_lib(lib_path)
    set_batch_size(num_envs)
    set_record_replays(record_replays)


class Minecraft2dEnv(gym.Env):
    def __init__(
            self,
            render_mode: Optional[str] = None,
            discovered_actions_reward: float = 0,
            include_actions_in_obs: bool = False,
            lib_path: str = './target/release/ffi.dll',
            record_replays: bool = False,
            num_total_envs: int = 1,
    ):
        """
        Initialize the Minecraft 2D environment.

        Args:
            render_mode (str, optional): The render mode. Currently not supported.
            discovered_actions_reward (float): Reward for discovering new actions.
            include_actions_in_obs (bool): Whether to include available actions in the observation.
        """
        if c_lib is None:
            init_lib(lib_path)
            if get_batch_size() == 1:
                # so in Ray env creation seems somehow isolated, so the init_lib has to be called always,
                # but after the first time it connects to an existing FFI environment.
                # so if the batch size is not its default value (1), then the by calling set_batch_size again,
                # we would invalidate previous connections.
                print("Initializing Minecraft connection in Minecraft2dEnv init. "
                      "This should not happen twice in the same process")
                set_batch_size(num_total_envs)
                set_record_replays(record_replays)
        super().__init__()

        if render_mode is not None:
            raise ValueError("Rendering is not supported. Use the 2d-minecraft binary for replays.")

        self.render_mode = render_mode
        self.c_lib_index = connect_env()
        print(f"Connected to environment with index {self.c_lib_index}.")

        self.num_actions = num_actions()
        self.current_score = 0
        self.include_actions_in_obs = include_actions_in_obs
        self.discovered_actions_reward = discovered_actions_reward
        self.discovered_actions = np.zeros(self.num_actions, dtype=bool)
        self.reset_discovered_actions()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_actions)

        sample_obs = self.sample_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            warnings.warn('Seed is not supported in this environment. It is ignored.')
        super().reset(seed=seed)

        self.reset_discovered_actions()
        self.current_score = 0
        reset_one(self.c_lib_index)
        obs = get_processed_observation(self.c_lib_index)
        processed_obs, _, _, info = self._decode_observation(obs)

        return processed_obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_one(action, self.c_lib_index)
        obs = get_processed_observation(self.c_lib_index)
        processed_obs, reward, terminated, info = self._decode_observation(obs)

        truncated = False  # No truncation in this environment

        return processed_obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError("Use 2d-minecraft binary to watch replays instead of rendering in Python.")

    def close(self):
        close_one(self.c_lib_index)

    def _decode_observation(self, obs):
        info = {}
        reward = obs.score - self.current_score
        self.current_score = obs.score
        info['game_score'] = obs.score
        info['time'] = obs.time
        info['hp'] = obs.hp
        info['message'] = obs.message
        terminated = obs.done

        if self.discovered_actions_reward or self.include_actions_in_obs:
            available_actions = get_actions_mask(self.c_lib_index)
            info['available_actions'] = available_actions
            new_discovered_actions = np.logical_or(self.discovered_actions, available_actions)
            reward += self.discovered_actions_reward * (
                        np.sum(new_discovered_actions) - np.sum(self.discovered_actions))
            self.discovered_actions = new_discovered_actions
            info['discovered_actions'] = np.copy(self.discovered_actions)

        if self.include_actions_in_obs:
            processed_obs = self.flatted_obs(obs, info['available_actions'], info['discovered_actions'])
        else:
            processed_obs = self.flatted_obs(obs)
        return processed_obs, reward, terminated, info

    def reset_discovered_actions(self):
        self.discovered_actions = np.zeros(self.num_actions, dtype=bool)
        # movement, turning, and mining are always available
        self.discovered_actions[0:7] = True

    def sample_observation(self):
        obs, _, _, _ = self._decode_observation(get_processed_observation(self.c_lib_index))
        self.current_score = 0
        self.reset_discovered_actions()
        return obs

    @staticmethod
    def flatted_obs(obs, available_actions=None, discovered_actions=None):
        top_materials_flat = obs.top_materials.flatten()
        top_materials_one_hot = np.eye(NUM_MATERIALS)[top_materials_flat].flatten()
        tile_heights_flat = obs.tile_heights.flatten()
        player_pos_flat = np.array(obs.player_pos)
        player_rot = np.array([obs.player_rot])
        hp = np.array([obs.hp])
        time = np.array([obs.time])
        inventory_state = np.array(obs.inventory_state)
        mobs_flat = np.array(obs.mobs).flatten()
        loot_flat = np.array(obs.loot).flatten()

        processed_obs = np.concatenate([
            top_materials_one_hot, tile_heights_flat, player_pos_flat, player_rot,
            hp, time, inventory_state, mobs_flat, loot_flat
        ]).astype(np.float32)

        if available_actions is not None:
            processed_obs = np.concatenate([processed_obs, available_actions.astype(np.float32)])
        if discovered_actions is not None:
            processed_obs = np.concatenate([processed_obs, discovered_actions.astype(np.float32)])

        return processed_obs
