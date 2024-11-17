import ctypes
from typing import Tuple

import numpy as np

from python_wrapper.ffi_elements import Observation, get_one_observation, action_name, valid_actions_mask, \
    reset_one_to_saved

OBSERVATION_GRID_SIZE = 17
NUM_MATERIALS = 13
INVENTORY_SIZE = 26
NUM_MOBS = 16
MOB_INFO_SIZE = 4
LOOT_INFO_SIZE = 3


class ProcessedObservation:
    def __init__(self,
                 top_materials: np.ndarray,
                 tile_heights: np.ndarray,
                 player_pos: Tuple[int, int, int],
                 player_rot: int,
                 hp: int,
                 time: float,
                 inventory_state: np.ndarray,
                 mobs: np.ndarray,
                 loot: np.ndarray,
                 score: int,
                 message: str,
                 done: bool = False):
        self.top_materials = top_materials
        self.tile_heights = tile_heights
        self.player_pos = player_pos
        self.player_rot = player_rot
        self.hp = hp
        self.time = time
        self.inventory_state = inventory_state
        self.mobs = mobs
        self.loot = loot
        self.score = score
        self.message = message
        self.done = done

    def __str__(self) -> str:
        return (f"Observation:\n"
                f"Player Position: {self.player_pos}\n"
                f"Player Rotation: {self.player_rot}\n"
                f"HP: {self.hp}\n"
                f"Time: {self.time}\n"
                f"Inventory State: {self.inventory_state}\n"
                f"Mobs: {self.mobs}\n"
                f"Loot: {self.loot}\n"
                f"Message: {self.message}\n"
                f"Top Materials:\n{self.top_materials}\n"
                f"Tile Heights:\n{self.tile_heights}\n"
                f"Score: {self.score}\n"
                f"Done: {self.done}")

    @staticmethod
    def from_c_observation(c_observation: Observation) -> 'ProcessedObservation':
        top_materials = np.ctypeslib.as_array(c_observation.top_materials).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        tile_heights = np.ctypeslib.as_array(c_observation.tile_heights).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        player_pos = tuple(c_observation.player_pos)
        player_rot = c_observation.player_rot
        hp = c_observation.hp
        time = c_observation.time
        inventory_state = np.ctypeslib.as_array(c_observation.inventory_state)
        mobs = np.ctypeslib.as_array(c_observation.mobs).reshape((NUM_MOBS, MOB_INFO_SIZE))
        loot = np.ctypeslib.as_array(c_observation.loot).reshape((NUM_MOBS, LOOT_INFO_SIZE))
        score = c_observation.score
        message = ctypes.cast(c_observation.message, ctypes.c_char_p).value.decode('utf-8') if c_observation.message else ""
        done = c_observation.done
        return ProcessedObservation(
            top_materials, tile_heights,
            player_pos, player_rot,
            hp, time, inventory_state,
            mobs, loot,
            score, message, done
        )


def get_processed_observation(idx: int) -> ProcessedObservation:
    c_observation = get_one_observation(idx)
    return ProcessedObservation.from_c_observation(c_observation)


def get_action_name(action: int) -> str:
    name = action_name(action)
    return ctypes.cast(name, ctypes.c_char_p).value.decode('utf-8') if name else ""


def get_actions_mask(idx: int) -> np.ndarray:
    c_action_mask = valid_actions_mask(idx)
    return np.ctypeslib.as_array(c_action_mask.mask).astype(bool)


def reset_one_to_saved_wrapped(idx: int, save_name: str):
    save_name_bytes = save_name.encode('utf-8')

    # Create a ctypes array from the bytes
    arr_type = ctypes.c_int8 * (len(save_name_bytes) + 1)  # +1 for null terminator
    save_name_arr = arr_type()

    # Copy the bytes into the array
    for i, b in enumerate(save_name_bytes):
        save_name_arr[i] = b
    save_name_arr[len(save_name_bytes)] = 0  # Add null terminator

    # Get pointer to the array
    save_name_ptr = ctypes.cast(save_name_arr, ctypes.POINTER(ctypes.c_int8))

    reset_one_to_saved(idx, save_name_ptr)
