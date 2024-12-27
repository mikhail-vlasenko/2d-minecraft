import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, ActionWrapper

from python_wrapper.ffi_elements import NUM_ACTIONS
from python_wrapper.minecraft_2d_env import Minecraft2dEnv


class ActionSimplificationWrapper(ActionWrapper):
    """
    Allows the model to work with a simplified action space.

    Removes the actions that can not be taken by the agent.
    Such as the ffi-disabled actions (i.e. ToggleMainMenu).

    All "place X" actions are combined into a single "place" action.
    The combined "place" action takes the first action index in the env.
    """
    def __init__(self, env: Minecraft2dEnv):
        Wrapper.__init__(self, env)
        self.env: Minecraft2dEnv = env
        self.disabled_actions = [
            'Place', 'Craft', 'Consume',
            'CloseInteractableMenu', 'ToggleMap', 'ToggleCraftMenu', 'ToggleMainMenu'
        ]
        self.num_new_actions = 1  # combined "place" action
        self.place_action_prefix = 'PlaceSpecificMaterial'
        self.combined_place_action_name = 'CombinedPlace'
        # when a placement action is requested, the highest priority material that can be placed, is placed
        self.placement_priority = ['CraftTable', 'Dirt', 'Stone', 'Plank', 'TreeLog', 'IronOre', 'Diamond']
        # obtain action descriptions from the environment
        self.all_descriptions = [self.env.get_action_name(i) for i in range(NUM_ACTIONS)]
        self.place_actions: list[tuple[int, int]] = []
        self.map_to_orig_idx = {}
        self.map_from_orig_idx = np.empty(NUM_ACTIONS, dtype=int)  # this one has continuous source domain
        self._build_action_mappings()
        self.action_space = gym.spaces.Discrete(len(self.map_to_orig_idx) + self.num_new_actions)

    def _build_action_mappings(self):
        num_reduced_actions = 0

        for i, description in enumerate(self.all_descriptions):
            if self.place_action_prefix in description:
                num_reduced_actions += 1
                for priority, material in enumerate(self.placement_priority):
                    if material in description:
                        self.place_actions.append((i, priority))
                        break
            elif any(action == description for action in self.disabled_actions):
                num_reduced_actions += 1
            else:
                # as some actions are grouped, we need to map the new indices to the original ones
                # to preserve continuity of indices
                self.map_to_orig_idx[i - num_reduced_actions + self.num_new_actions] = i

        for i in range(NUM_ACTIONS):
            self.map_from_orig_idx[i] = self.map_to_orig_idx.get(i, int(1e8))  # map to a non-existing index (should never be used)

        self.place_actions.sort(key=lambda x: x[1])

    def action(self, action):
        available_actions = self.env.get_actions_mask()

        if action == 0:
            # combined "place" action
            for i, priority in self.place_actions:
                if available_actions[i]:
                    return i
            # fallback to the first action of this category
            return self.place_actions[0][0]
        else:
            return self.map_to_orig_idx[action]

    def get_action_name(self, action):
        if action == 0:
            return self.combined_place_action_name
        else:
            return self.all_descriptions[self.map_to_orig_idx[action]]

    def get_actions_mask(self):
        actual_available_actions = self.env.get_actions_mask()
        mask = np.zeros(self.action_space.n, dtype=bool)
        for i in range(self.action_space.n):
            if i == 0:
                mask[i] = any(actual_available_actions[i] for i, _ in self.place_actions)
            else:
                mask[i] = actual_available_actions[self.map_from_orig_idx[i]]
        return mask
