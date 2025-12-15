import ctypes
from typing import Tuple, Optional

import numpy as np

from python_wrapper.ffi_elements import Observation, get_one_observation, action_name, \
    reset_one_to_saved, OBSERVATION_GRID_SIZE, MAX_MOBS, MOB_INFO_SIZE, LOOT_INFO_SIZE, INVENTORY_SIZE, \
    NUM_ACTIONS, NUM_MATERIALS, NUM_STATUS_EFFECTS


class ProcessedObservation:
    def __init__(
        self,
        top_materials: np.ndarray,
        tile_heights: np.ndarray,
        player_pos: Tuple[int, int, int],
        player_rot: int,
        hp: int,
        time: float,
        inventory_state: np.ndarray,
        mobs: np.ndarray,
        loot: np.ndarray,
        action_mask: np.ndarray,
        status_effects: np.ndarray,
        score: int,
        message: str,
        done: bool,
    ):
        self.top_materials = top_materials
        self.tile_heights = tile_heights
        self.player_pos = player_pos
        self.player_rot = player_rot
        self.hp = hp
        self.time = time
        self.inventory_state = inventory_state
        self.mobs = mobs
        self.loot = loot
        self.action_mask = action_mask
        self.status_effects = status_effects
        self.score = score
        self.message = message
        self.done = done

    def __str__(self) -> str:
        return (
            f"Observation:\n"
            f"Player Position: {self.player_pos}\n"
            f"Player Rotation: {self.player_rot}\n"
            f"HP: {self.hp}\n"
            f"Time: {self.time}\n"
            f"Inventory State: {self.inventory_state}\n"
            f"Mobs: {self.mobs}\n"
            f"Loot: {self.loot}\n"
            f"Action Mask: {self.action_mask}\n"
            f"Message: {self.message}\n"
            f"Top Materials:\n{self.top_materials}\n"
            f"Tile Heights:\n{self.tile_heights}\n"
            f"Status Effects: {self.status_effects}\n"
            f"Score: {self.score}\n"
            f"Done: {self.done}\n"
        )

    @staticmethod
    def from_c_observation(c_observation: Observation) -> 'ProcessedObservation':
        top_materials = np.ctypeslib.as_array(c_observation.top_materials).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        tile_heights = np.ctypeslib.as_array(c_observation.tile_heights).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        player_pos = tuple(c_observation.player_pos)
        player_rot = c_observation.player_rot
        hp = c_observation.hp
        time = c_observation.time
        inventory_state = np.ctypeslib.as_array(c_observation.inventory_state)
        mobs = np.ctypeslib.as_array(c_observation.mobs).reshape((MAX_MOBS, MOB_INFO_SIZE))
        loot = np.ctypeslib.as_array(c_observation.loot).reshape((MAX_MOBS, LOOT_INFO_SIZE))
        actions_mask = np.ctypeslib.as_array(c_observation.action_mask).astype(bool)
        status_effects = np.ctypeslib.as_array(c_observation.status_effects)
        score = c_observation.score
        message = ctypes.cast(c_observation.message, ctypes.c_char_p).value.decode('utf-8') if c_observation.message else ""
        done = c_observation.done
        return ProcessedObservation(
            top_materials, tile_heights,
            player_pos, player_rot,
            hp, time, inventory_state,
            mobs, loot,
            actions_mask,
            status_effects,
            score,
            message, 
            done,
        )

    @staticmethod
    def default():
        return ProcessedObservation(
            np.zeros((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE)),
            np.zeros((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE)),
            (0, 0, 0), 0, 0, 0, np.zeros(INVENTORY_SIZE),
            np.zeros((MAX_MOBS, MOB_INFO_SIZE)), np.zeros((MAX_MOBS, LOOT_INFO_SIZE)),
            np.zeros(NUM_ACTIONS),
            np.zeros(NUM_STATUS_EFFECTS),
            0, 
            "", 
            False,
        )


def get_processed_observation(idx: int) -> ProcessedObservation:
    c_observation = get_one_observation(idx)
    return ProcessedObservation.from_c_observation(c_observation)


def get_action_name(action: int) -> str:
    name = action_name(action)
    return ctypes.cast(name, ctypes.c_char_p).value.decode('utf-8') if name else ""


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
