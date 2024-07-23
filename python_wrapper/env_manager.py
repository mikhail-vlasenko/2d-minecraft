import random
import ctypes
import numpy as np
from typing import List

from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size, set_record_replays
from python_wrapper.observation import ProcessedObservation, get_processed_observation, get_action_name


class EnvManager:
    def __init__(self, lib_path='./target/release/ffi.dll', batch_size=2, record_replays=False):
        # Initialize the FFI library
        init_lib(lib_path)

        set_record_replays(record_replays)

        self.batch_size = batch_size
        set_batch_size(self.batch_size)

        # Get the number of actions
        self.num_actions = num_actions()

        self.current_scores = np.zeros(self.batch_size)

    def reset(self):
        """Resets all environments."""
        reset()
        return self.get_all_observations()[0]

    def step(self, actions: List[int]) -> tuple[list[ProcessedObservation], list[int], list[bool]]:
        """Performs a step in each environment with the given actions.

        Args:
            actions (List[int]): A list of actions for each environment in the batch.

        Returns:
            List[ProcessedObservation]: A list of observations after performing the actions.
        """
        for i, action in enumerate(actions):
            step_one(action, i)
        return self.get_all_observations()

    def get_all_observations(self) -> tuple[list[ProcessedObservation], list[int], list[bool]]:
        """Gets the current observations from all environments.

        Returns:
            tuple[list[ProcessedObservation], list[int]]: A tuple containing a list of observations and a list of rewards
        """
        observations = []
        rewards = []
        dones = []
        for i in range(self.batch_size):
            obs = get_processed_observation(i)
            if obs.score == 0:
                # ensure a negative reward from transitioning from a previous game can't happen
                rewards.append(0)
            else:
                rewards.append(obs.score - self.current_scores[i])
            self.current_scores[i] = obs.score
            observations.append(obs)
            dones.append(obs.done)
        return observations, rewards, dones

    def get_action_name(self, action: int) -> str:
        """Gets the name of an action.

        Args:
            action (int): The action index.

        Returns:
            str: The name of the action.
        """
        return get_action_name(action)

    def close(self):
        self.batch_size = 0
        set_batch_size(0)

    def set_batch_size(self, new_batch_size: int):
        self.batch_size = new_batch_size
        set_batch_size(new_batch_size)

    def set_record_replays(self, record_replays: bool):
        set_record_replays(record_replays)


if __name__ == "__main__":
    manager = EnvManager()
    observations = manager.reset()
    for obs in observations:
        print(obs)

    while True:
        actions = [random.randint(0, manager.num_actions - 1) for _ in range(manager.batch_size)]
        print(f'Performing actions: {[manager.get_action_name(action) for action in actions]}')
        observations, rewards, dones = manager.step(actions)
        for obs in observations:
            print(obs)
