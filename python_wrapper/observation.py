import ctypes
from typing import Tuple, List

import numpy as np

from python_wrapper.ffi_elements import Observation, get_one_observation, action_name

OBSERVATION_GRID_SIZE = 17


class ProcessedObservation:
    def __init__(self,
                 top_materials: np.ndarray,
                 tile_heights: np.ndarray,
                 player_pos: Tuple[int, int, int],
                 player_rot: int,
                 hp: int,
                 time: float,
                 inventory_state: List[int],
                 mobs: List[List[int]],
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
                f"Message: {self.message}\n"
                f"Top Materials:\n{self.top_materials}\n"
                f"Tile Heights:\n{self.tile_heights}\n"
                f"Done: {self.done}")

    @staticmethod
    def from_c_observation(c_observation: Observation) -> 'ProcessedObservation':
        top_materials = np.ctypeslib.as_array(c_observation.top_materials).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        tile_heights = np.ctypeslib.as_array(c_observation.tile_heights).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        player_pos = tuple(c_observation.player_pos)
        player_rot = c_observation.player_rot
        hp = c_observation.hp
        time = c_observation.time
        inventory_state = np.ctypeslib.as_array(c_observation.inventory_state).tolist()
        mobs = np.ctypeslib.as_array(c_observation.mobs).reshape((16, 4)).tolist()
        message = ctypes.cast(c_observation.message, ctypes.c_char_p).value.decode('utf-8') if c_observation.message else ""
        done = c_observation.done
        return ProcessedObservation(top_materials, tile_heights, player_pos, player_rot, hp, time, inventory_state, mobs, message, done)


def get_processed_observation(idx: int) -> ProcessedObservation:
    c_observation = get_one_observation(idx)
    return ProcessedObservation.from_c_observation(c_observation)


def get_action_name(action: int) -> str:
    name = action_name(action)
    return ctypes.cast(name, ctypes.c_char_p).value.decode('utf-8') if name else ""
