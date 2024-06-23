import ctypes

import numpy as np

OBSERVATION_GRID_SIZE = 17


class CObservation(ctypes.Structure):
    _fields_ = [
        ("top_materials", ctypes.c_int * OBSERVATION_GRID_SIZE * OBSERVATION_GRID_SIZE),
        ("tile_heights", ctypes.c_int * OBSERVATION_GRID_SIZE * OBSERVATION_GRID_SIZE),
        ("player_pos", ctypes.c_int * 3)  # Define the player_pos as an array of 3 integers
    ]

class Observation:
    def __init__(self, top_materials, tile_heights, player_pos):
        self.top_materials = top_materials
        self.tile_heights = tile_heights
        self.player_pos = player_pos

    def __str__(self):
        return (f"Observation with player position: {self.player_pos}\n"
                f"top_materials:\n{self.top_materials}\n"
                f"tile_heights:\n{self.tile_heights}\n")

    @staticmethod
    def from_c_observation(c_observation):
        top_materials = np.ctypeslib.as_array(c_observation.top_materials).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        tile_heights = np.ctypeslib.as_array(c_observation.tile_heights).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
        player_pos = tuple(c_observation.player_pos)  # Convert the player_pos array to a tuple
        return Observation(top_materials, tile_heights, player_pos)
