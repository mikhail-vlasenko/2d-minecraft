from typing import Type, List, Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size, set_record_replays
from python_wrapper.observation import get_processed_observation, NUM_MATERIALS, get_actions_mask


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
    obs = np.concatenate(
        [top_materials_one_hot, tile_heights_flat, player_pos_flat, player_rot, hp, time, inventory_state, mobs_flat])
    if available_actions is not None:
        obs = np.concatenate([obs, available_actions.astype(np.float32)])
    if discovered_actions is not None:
        obs = np.concatenate([obs, discovered_actions.astype(np.float32)])
    return obs


class Minecraft2dEnv(VecEnv):
    def __init__(
            self, num_envs=1, lib_path='./target/release/ffi.dll',
            record_replays=False, discovered_actions_reward=0, include_actions_in_obs=False
    ):
        init_lib(lib_path)
        set_record_replays(record_replays)
        self.num_actions = num_actions()
        self.num_envs = num_envs
        set_batch_size(self.num_envs)
        self.current_scores = np.zeros(self.num_envs)
        self.include_actions_in_obs = include_actions_in_obs
        self.discovered_actions_reward = discovered_actions_reward
        self.discovered_actions = np.zeros((self.num_envs, self.num_actions), dtype=bool)
        for i in range(self.num_envs):
            self.reset_discovered_actions(i)

        obs = self.sample_observation()

        super().__init__(
            num_envs=self.num_envs,
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32),
            action_space=gym.spaces.Discrete(self.num_actions)
        )

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, action in enumerate(self.actions):
            step_one(action, i)
            obs = get_processed_observation(i)
            obs, reward, done, info = self._decode_observation(obs, i)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.array(observations), np.array(rewards), np.array(dones), infos

    def reset(self):
        reset()
        observations = []
        for i in range(self.num_envs):
            obs = get_processed_observation(i)
            obs, _, _, info = self._decode_observation(obs, i)
            observations.append(obs)
        return np.array(observations)

    def _decode_observation(self, obs, i):
        """
        Has side effects on self.current_scores and self.discovered_actions so should be called once per step
        """
        info = {}
        if obs.score == 0:
            # ensure a negative reward from transitioning from a previous game can't happen
            reward = 0
        else:
            reward = obs.score - self.current_scores[i]
        self.current_scores[i] = obs.score
        done = obs.done

        if self.discovered_actions_reward or self.include_actions_in_obs:
            available_actions = get_actions_mask(i)
            info['available_actions'] = available_actions
            # give a reward for discovering new actions within an episode
            new_discovered_actions = np.logical_or(self.discovered_actions[i], available_actions)
            reward += self.discovered_actions_reward * (
                        np.sum(new_discovered_actions) - np.sum(self.discovered_actions[i]))
            self.discovered_actions[i] = new_discovered_actions
            info['discovered_actions'] = np.copy(self.discovered_actions[i])

            if done:
                self.reset_discovered_actions(i)

        if self.include_actions_in_obs:
            obs = flatted_obs(obs, info['available_actions'], info['discovered_actions'])
        else:
            obs = flatted_obs(obs)
        return obs, reward, done, info

    def close(self):
        set_batch_size(0)

    def reset_discovered_actions(self, i):
        self.discovered_actions[i] = np.zeros(self.num_actions, dtype=bool)
        # these 7 actions (walk + turn + mine) are practically always available
        self.discovered_actions[i][0:7] = True

    def sample_observation(self):
        # Obtain an example observation for space definition
        obs, _, _, _ = self._decode_observation(get_processed_observation(0), 0)
        self.current_scores[0] = 0
        self.reset_discovered_actions(0)
        return obs

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
