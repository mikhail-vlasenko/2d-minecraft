from typing import Type, List, Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size, set_record_replays
from python_wrapper.observation import get_processed_observation, NUM_MATERIALS


def flatted_obs(obs):
    top_materials_flat = obs.top_materials.flatten()
    top_materials_one_hot = np.eye(NUM_MATERIALS)[top_materials_flat].flatten()
    tile_heights_flat = obs.tile_heights.flatten()
    player_pos_flat = np.array(obs.player_pos)
    player_rot = np.array([obs.player_rot])
    hp = np.array([obs.hp])
    time = np.array([obs.time])
    inventory_state = np.array(obs.inventory_state)
    mobs_flat = np.array(obs.mobs).flatten()
    return np.concatenate(
        [top_materials_one_hot, tile_heights_flat, player_pos_flat, player_rot, hp, time, inventory_state, mobs_flat])


class Minecraft2dEnv(VecEnv):
    def __init__(self, num_envs=1, lib_path='./target/release/ffi.dll', record_replays=False):
        init_lib(lib_path)
        set_record_replays(record_replays)
        obs = flatted_obs(get_processed_observation(0))  # Obtain an example observation for space definition

        super().__init__(
            num_envs=num_envs,
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32),
            action_space=gym.spaces.Discrete(num_actions())
        )
        self.num_envs = num_envs
        set_batch_size(num_envs)
        self.current_scores = np.zeros(num_envs)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        observations = []
        rewards = []
        dones = []
        infos = [{} for _ in range(self.num_envs)]

        for i, action in enumerate(self.actions):
            step_one(action, i)
            obs = get_processed_observation(i)
            if obs.score == 0:
                # ensure a negative reward from transitioning from a previous game can't happen
                reward = 0
            else:
                reward = obs.score - self.current_scores[i]
            self.current_scores[i] = obs.score
            done = obs.done
            observations.append(flatted_obs(obs))
            rewards.append(reward)
            dones.append(done)

        return np.array(observations), np.array(rewards), np.array(dones), infos

    def reset(self):
        reset()
        observations = [flatted_obs(get_processed_observation(i)) for i in range(self.num_envs)]
        return np.array(observations)

    def close(self):
        set_batch_size(0)

    def render(self, mode='human'):
        raise NotImplementedError("Render not implemented here. "
                                  "Use the 2D-Minecraft binary to watch the replay, if it was saved.")

    def seed(self, seed=None):
        raise NotImplementedError("Seeding not implemented for this environment.")

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.num_envs)]
