import ctypes


OBSERVATION_GRID_SIZE = 17


class Observation(ctypes.Structure):
    _fields_ = [
        ("top_materials", ctypes.c_int * OBSERVATION_GRID_SIZE * OBSERVATION_GRID_SIZE),
        ("tile_depths", ctypes.c_int * OBSERVATION_GRID_SIZE * OBSERVATION_GRID_SIZE)
    ]